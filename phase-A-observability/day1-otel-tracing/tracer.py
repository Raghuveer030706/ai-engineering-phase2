"""
tracer.py — Day 1: OpenTelemetry tracing foundation

What this does:
  - Sets up a global TracerProvider backed by a local JSONL file exporter
  - Every LLM call gets a span: model, tokens, latency, cost_usd
  - Spans written to traces/trace.jsonl — one JSON object per line
  - trace_id follows the full request across all agents and tool calls

Why file export instead of Jaeger/Zipkin:
  - Zero infra needed — works on Windows right now with no Docker
  - JSONL is queryable with grep, Get-Content, or pandas
  - Swap to OTLP exporter for production with ONE line change

Run to verify setup (no API key needed):
  python tracer.py
"""

import json
import time
from pathlib import Path
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode


# ── Where traces land ─────────────────────────────────────────────────────────
# Path(__file__).parent always resolves to the day1-otel-tracing folder
# regardless of where PowerShell is when you run the script.
TRACE_DIR  = Path(__file__).parent / "traces"
TRACE_FILE = TRACE_DIR / "trace.jsonl"


# ── File-based span exporter ──────────────────────────────────────────────────
class JSONLFileExporter(SpanExporter):
    """
    Writes completed spans to a JSONL file.
    One JSON object per line — easy to tail or load into pandas.

    Why JSONL not JSON:
      - Append-only — no need to read + rewrite the whole file per span
      - Each line is valid JSON — works with streaming readers
      - grep/Get-Content -Tail works on it directly
    """

    def __init__(self, path: Path = TRACE_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans):
        with open(self.path, "a", encoding="utf-8") as f:
            for span in spans:
                f.write(json.dumps(self._span_to_dict(span)) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def _span_to_dict(self, span) -> dict:
        ctx = span.get_span_context()
        return {
            "trace_id":    format(ctx.trace_id, "032x"),
            "span_id":     format(ctx.span_id,  "016x"),
            "name":        span.name,
            "start_ms":    span.start_time // 1_000_000,
            "end_ms":      span.end_time   // 1_000_000,
            "duration_ms": (span.end_time - span.start_time) // 1_000_000,
            "status":      span.status.status_code.name,
            "attributes":  dict(span.attributes or {}),
            "events": [
                {
                    "name":       e.name,
                    "timestamp":  e.timestamp // 1_000_000,
                    "attributes": dict(e.attributes or {}),
                }
                for e in span.events
            ],
        }


# ── Global provider — initialised once ───────────────────────────────────────
_provider: Optional[TracerProvider] = None


def setup_tracing(service_name: str = "ai-engineering") -> TracerProvider:
    """
    Call once at application startup.
    Idempotent — safe to call multiple times, only initialises once.
    """
    global _provider
    if _provider is not None:
        return _provider

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(JSONLFileExporter()))
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def get_tracer(name: str = "ai-engineering") -> trace.Tracer:
    """Get a tracer. Always call setup_tracing() before this."""
    return trace.get_tracer(name)


def get_trace_id() -> str:
    """
    Returns the current trace_id as a 32-char hex string.
    Returns 'no-trace' if called outside a span context.
    Attach this to every log line so logs and traces correlate.
    """
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else "no-trace"


def get_span_id() -> str:
    """Returns the current span_id as a 16-char hex string."""
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.span_id, "016x") if ctx.is_valid else "no-span"


# ── Read back helpers ─────────────────────────────────────────────────────────
def load_traces() -> list[dict]:
    """Load every span from trace.jsonl."""
    if not TRACE_FILE.exists():
        return []
    with open(TRACE_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_traces_for_id(trace_id: str) -> list[dict]:
    """Load all spans that share a specific trace_id."""
    return [s for s in load_traces() if s["trace_id"] == trace_id]


# ── Smoke test — run this first, no API key needed ────────────────────────────
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold yellow]Day 1 — OpenTelemetry Tracer Setup[/bold yellow]\n")

    setup_tracing("ai-engineering-day1")
    tracer = get_tracer()

    # Parent span with two children — simulates agent root + two tool calls
    with tracer.start_as_current_span("smoke-test-root") as root:
        root.set_attribute("test.name", "day1-smoke-test")
        console.print(f"[green]✓[/green] trace_id = [cyan]{get_trace_id()}[/cyan]")
        console.print(f"[green]✓[/green] span_id  = [cyan]{get_span_id()}[/cyan]")

        with tracer.start_as_current_span("child-span-a") as child_a:
            child_a.set_attribute("child", "a")
            time.sleep(0.05)
            child_a.add_event("did-something", {"key": "value"})

        with tracer.start_as_current_span("child-span-b") as child_b:
            child_b.set_attribute("child", "b")
            time.sleep(0.03)

        root.set_status(StatusCode.OK)

    # BatchSpanProcessor flushes asynchronously — wait briefly
    time.sleep(0.5)

    spans = load_traces()
    console.print(f"\n[bold]Trace file:[/bold] {TRACE_FILE}")
    console.print(f"[bold]Spans written:[/bold] {len(spans)}\n")

    table = Table(title="Spans (last 3)", show_lines=True)
    table.add_column("name",        style="cyan")
    table.add_column("trace_id",    style="dim")
    table.add_column("duration_ms", justify="right")
    table.add_column("status")

    for s in spans[-3:]:
        table.add_row(
            s["name"],
            s["trace_id"][:16] + "...",
            str(s["duration_ms"]),
            "[green]OK[/green]" if s["status"] == "OK" else s["status"],
        )

    console.print(table)
    console.print("\n[bold green]✓ Tracer working. Run llm_client.py next.[/bold green]")