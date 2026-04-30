"""
metrics.py — Day 2: Parse trace.jsonl into structured metrics

This module is the data layer for the dashboard.
Everything reads from the Day 1 trace.jsonl — no new API calls.

What it computes:
  - Per-trace summaries (cost, latency, step count, LLM calls)
  - Per-model cost breakdown
  - Latency distribution buckets for histogram
  - Running cost total over time (for trend line)
  - Slowest spans ranked by duration

Import this from dashboard.py and agent_with_metrics.py.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Path to Day 1 trace file ──────────────────────────────────────────────────
# Day 2 sits next to day1-otel-tracing in the same phase-A folder.
# Adjust this if you move files around.
DAY1_DIR   = Path(__file__).parent.parent / "day1-otel-tracing"
TRACE_FILE = DAY1_DIR / "traces" / "trace.jsonl"


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class SpanRecord:
    trace_id:    str
    span_id:     str
    name:        str
    duration_ms: int
    status:      str
    attributes:  dict
    start_ms:    int

    # Convenience accessors
    @property
    def cost_usd(self) -> float:
        return self.attributes.get("llm.cost_usd", 0.0)

    @property
    def input_tokens(self) -> int:
        return self.attributes.get("llm.input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.attributes.get("llm.output_tokens", 0)

    @property
    def model(self) -> str:
        return self.attributes.get("llm.model", "")

    @property
    def is_llm_call(self) -> bool:
        return self.name.startswith("llm.")

    @property
    def is_tool_call(self) -> bool:
        return self.name.startswith("agent.tool.")

    @property
    def is_agent_root(self) -> bool:
        return self.name == "agent.run"


@dataclass
class TraceMetrics:
    """Aggregated metrics for one full agent run (one trace_id)."""
    trace_id:         str
    start_ms:         int
    total_cost_usd:   float
    total_latency_ms: int          # wall-clock of the agent.run span
    llm_calls:        int
    tool_calls:       int
    step_count:       int
    input_tokens:     int
    output_tokens:    int
    models_used:      list[str]
    question:         str
    spans:            list[SpanRecord] = field(repr=False)


@dataclass
class DashboardMetrics:
    """Aggregated view across ALL traces."""
    total_cost_usd:     float
    total_traces:       int
    total_llm_calls:    int
    total_input_tokens: int
    total_output_tokens:int
    by_model:           dict[str, float]    # model → total cost
    latency_buckets:    dict[str, int]      # bucket label → count of LLM calls
    cost_over_time:     list[tuple[int, float]]  # (start_ms, cumulative_cost)
    slowest_spans:      list[SpanRecord]    # top 10 by duration_ms
    traces:             list[TraceMetrics]


# ── Loader ────────────────────────────────────────────────────────────────────
def load_spans(trace_file: Path = TRACE_FILE) -> list[SpanRecord]:
    """Load all spans from trace.jsonl."""
    if not trace_file.exists():
        return []
    records = []
    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(SpanRecord(
                    trace_id=d["trace_id"],
                    span_id=d["span_id"],
                    name=d["name"],
                    duration_ms=d["duration_ms"],
                    status=d["status"],
                    attributes=d.get("attributes", {}),
                    start_ms=d.get("start_ms", 0),
                ))
            except (json.JSONDecodeError, KeyError):
                continue   # skip malformed lines
    return records


# ── Grouping ──────────────────────────────────────────────────────────────────
def group_by_trace(spans: list[SpanRecord]) -> dict[str, list[SpanRecord]]:
    groups: dict[str, list[SpanRecord]] = {}
    for s in spans:
        groups.setdefault(s.trace_id, []).append(s)
    return groups


def build_trace_metrics(trace_id: str, spans: list[SpanRecord]) -> TraceMetrics:
    """Compute per-trace metrics from its spans."""
    llm_spans  = [s for s in spans if s.is_llm_call]
    tool_spans = [s for s in spans if s.is_tool_call]
    root_spans = [s for s in spans if s.is_agent_root]

    # Wall-clock from the agent.run root span if present, else sum of steps
    if root_spans:
        total_latency = root_spans[0].duration_ms
        start_ms      = root_spans[0].start_ms
    else:
        total_latency = sum(s.duration_ms for s in spans)
        start_ms      = min((s.start_ms for s in spans), default=0)

    # Step count from attributes if available
    step_count = 0
    if root_spans:
        step_count = root_spans[0].attributes.get("agent.step_count", 0)
    if not step_count:
        step_count = len([s for s in spans if s.name.startswith("agent.step.")])

    # Question from root span
    question = ""
    if root_spans:
        question = root_spans[0].attributes.get("agent.question", "")

    return TraceMetrics(
        trace_id=trace_id,
        start_ms=start_ms,
        total_cost_usd=sum(s.cost_usd for s in llm_spans),
        total_latency_ms=total_latency,
        llm_calls=len(llm_spans),
        tool_calls=len(tool_spans),
        step_count=step_count,
        input_tokens=sum(s.input_tokens for s in llm_spans),
        output_tokens=sum(s.output_tokens for s in llm_spans),
        models_used=list({s.model for s in llm_spans if s.model}),
        question=question,
        spans=spans,
    )


# ── Dashboard aggregation ─────────────────────────────────────────────────────
LATENCY_BUCKETS = [
    ("< 500ms",   0,    500),
    ("500-1000ms", 500, 1000),
    ("1-2s",     1000, 2000),
    ("2-5s",     2000, 5000),
    ("> 5s",     5000, 999_999),
]


def compute_dashboard(trace_file: Path = TRACE_FILE) -> Optional[DashboardMetrics]:
    """
    Main entry point. Returns None if no traces exist yet.
    Reads trace.jsonl fresh each call — no caching — so the dashboard
    always reflects the latest runs.
    """
    spans = load_spans(trace_file)
    if not spans:
        return None

    groups = group_by_trace(spans)
    traces = [build_trace_metrics(tid, slist) for tid, slist in groups.items()]
    traces.sort(key=lambda t: t.start_ms)

    llm_spans = [s for s in spans if s.is_llm_call]

    # Cost by model
    by_model: dict[str, float] = {}
    for s in llm_spans:
        if s.model:
            by_model[s.model] = by_model.get(s.model, 0.0) + s.cost_usd

    # Latency distribution (LLM call durations)
    buckets = {label: 0 for label, _, _ in LATENCY_BUCKETS}
    for s in llm_spans:
        ms = s.duration_ms
        for label, lo, hi in LATENCY_BUCKETS:
            if lo <= ms < hi:
                buckets[label] += 1
                break

    # Running cumulative cost over time
    cost_over_time: list[tuple[int, float]] = []
    running = 0.0
    for t in traces:
        running += t.total_cost_usd
        cost_over_time.append((t.start_ms, round(running, 8)))

    # Slowest spans (top 10, any type)
    slowest = sorted(spans, key=lambda s: -s.duration_ms)[:10]

    return DashboardMetrics(
        total_cost_usd=sum(t.total_cost_usd for t in traces),
        total_traces=len(traces),
        total_llm_calls=len(llm_spans),
        total_input_tokens=sum(s.input_tokens for s in llm_spans),
        total_output_tokens=sum(s.output_tokens for s in llm_spans),
        by_model=by_model,
        latency_buckets=buckets,
        cost_over_time=cost_over_time,
        slowest_spans=slowest,
        traces=traces,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = compute_dashboard()
    if not m:
        print("No traces found. Run day1-otel-tracing/agent.py first.")
        sys.exit(1)

    print(f"Total cost:    ${m.total_cost_usd:.6f}")
    print(f"Total traces:  {m.total_traces}")
    print(f"LLM calls:     {m.total_llm_calls}")
    print(f"Input tokens:  {m.total_input_tokens}")
    print(f"Output tokens: {m.total_output_tokens}")
    print(f"By model:      {m.by_model}")
    print(f"Latency dist:  {m.latency_buckets}")
    print(f"Slowest span:  {m.slowest_spans[0].name} — {m.slowest_spans[0].duration_ms}ms")