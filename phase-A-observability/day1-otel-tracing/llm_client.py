"""
llm_client.py — Day 1: Traced LLM client

What this does:
  - Wraps anthropic.Anthropic so every call automatically creates a span
  - Records: model, input_tokens, output_tokens, cost_usd, latency_ms
  - cost_usd is always real — pulled from response.usage, never estimated
  - This fixes the "estimated_cost_usd shows 0.0" bug from Phase 1

Cost table used (claude-haiku-4-5-20251001 pricing):
  Input:  $0.80  per 1M tokens  →  $0.0000008  per token
  Output: $4.00  per 1M tokens  →  $0.000004   per token

Run:
  python llm_client.py
"""

import os
import time
from dataclasses import dataclass

from pathlib import Path

import anthropic
from dotenv import load_dotenv
from opentelemetry.trace import StatusCode
from rich.console import Console
from rich.panel import Panel

from tracer import get_trace_id, get_span_id, get_tracer, load_traces_for_id, setup_tracing

# .env lives at repo root — four levels up from this file:
# day1-otel-tracing → phase-A-observability → ai-engineering-phase2 → .env
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")


# ── Pricing table ─────────────────────────────────────────────────────────────
COST_PER_TOKEN = {
    "claude-haiku-4-5-20251001": {
        "input":  0.80  / 1_000_000,
        "output": 4.00  / 1_000_000,
    },
    "claude-sonnet-4-20250514": {
        "input":  3.00  / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
}

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _cost(model: str, inp: int, out: int) -> float:
    p = COST_PER_TOKEN.get(model, COST_PER_TOKEN[DEFAULT_MODEL])
    return inp * p["input"] + out * p["output"]


# ── Response dataclass ────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    content:       str
    model:         str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    latency_ms:    int
    trace_id:      str
    span_id:       str

    def summary(self) -> str:
        return (
            f"[{self.trace_id[:8]}...] "
            f"{self.input_tokens}in / {self.output_tokens}out tokens | "
            f"${self.cost_usd:.6f} | {self.latency_ms}ms"
        )


# ── Traced client ─────────────────────────────────────────────────────────────
class TracedLLMClient:
    """
    Drop-in wrapper around anthropic.Anthropic.

    Every call to .ask() or .ask_with_history() creates an OTel span
    with full metrics attached as span attributes.

    The span name is configurable — use descriptive names like
    'llm.synthesize' or 'llm.route' so your traces are readable.

    Usage:
        client = TracedLLMClient()
        r = client.ask("What is 2+2?")
        print(r.cost_usd)   # always real, never 0.0
        print(r.trace_id)   # 32-char hex — correlates to trace.jsonl
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model  = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.tracer = get_tracer("llm-client")

    def ask(
        self,
        prompt:     str,
        system:     str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        span_name:  str = "llm.call",
    ) -> LLMResponse:
        """Single-turn LLM call, fully traced."""
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("llm.model",     self.model)
            span.set_attribute("llm.prompt_len", len(prompt))

            t0 = time.perf_counter()
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

            ms  = int((time.perf_counter() - t0) * 1000)
            inp = msg.usage.input_tokens
            out = msg.usage.output_tokens
            usd = _cost(self.model, inp, out)

            span.set_attribute("llm.input_tokens",  inp)
            span.set_attribute("llm.output_tokens", out)
            span.set_attribute("llm.cost_usd",      round(usd, 8))
            span.set_attribute("llm.latency_ms",    ms)
            span.set_status(StatusCode.OK)

            return LLMResponse(
                content=msg.content[0].text,
                model=self.model,
                input_tokens=inp,
                output_tokens=out,
                cost_usd=usd,
                latency_ms=ms,
                trace_id=get_trace_id(),
                span_id=get_span_id(),
            )

    def ask_with_history(
        self,
        messages:   list[dict],
        system:     str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        span_name:  str = "llm.call.history",
    ) -> LLMResponse:
        """
        Multi-turn call.
        messages = full conversation so far:
          [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("llm.model",         self.model)
            span.set_attribute("llm.history_turns",  len(messages))

            t0 = time.perf_counter()
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

            ms  = int((time.perf_counter() - t0) * 1000)
            inp = msg.usage.input_tokens
            out = msg.usage.output_tokens
            usd = _cost(self.model, inp, out)

            span.set_attribute("llm.input_tokens",  inp)
            span.set_attribute("llm.output_tokens", out)
            span.set_attribute("llm.cost_usd",      round(usd, 8))
            span.set_attribute("llm.latency_ms",    ms)
            span.set_status(StatusCode.OK)

            return LLMResponse(
                content=msg.content[0].text,
                model=self.model,
                input_tokens=inp,
                output_tokens=out,
                cost_usd=usd,
                latency_ms=ms,
                trace_id=get_trace_id(),
                span_id=get_span_id(),
            )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    console = Console()
    console.print("\n[bold yellow]Day 1 — Traced LLM Client[/bold yellow]\n")

    setup_tracing("ai-engineering-day1")
    client = TracedLLMClient()
    tracer = get_tracer("smoke-test")

    with tracer.start_as_current_span("llm-client-smoke-test"):
        trace_id = get_trace_id()
        console.print(f"[dim]trace_id: {trace_id}[/dim]\n")

        # Call 1
        console.print("[bold]Call 1:[/bold] Simple factual question")
        r1 = client.ask(
            "What is the capital of France? One sentence.",
            span_name="llm.factual",
        )
        console.print(Panel(
            f"[green]{r1.content}[/green]\n\n[dim]{r1.summary()}[/dim]",
            title="Response 1", border_style="cyan",
        ))

        # Call 2
        console.print("\n[bold]Call 2:[/bold] Why observability matters")
        r2 = client.ask(
            "In 2 sentences, why does observability matter in AI agent systems?",
            span_name="llm.reasoning",
        )
        console.print(Panel(
            f"[green]{r2.content}[/green]\n\n[dim]{r2.summary()}[/dim]",
            title="Response 2", border_style="cyan",
        ))

    time.sleep(0.5)  # flush BatchSpanProcessor

    spans    = load_traces_for_id(trace_id)
    tot_cost = sum(s["attributes"].get("llm.cost_usd", 0) for s in spans)
    tot_ms   = sum(s["attributes"].get("llm.latency_ms", 0) for s in spans)

    console.print(f"\n[bold]Trace summary[/bold] ({trace_id[:16]}...)")
    console.print(f"  Spans:         {len(spans)}")
    console.print(f"  Total cost:    [yellow]${tot_cost:.6f}[/yellow]")
    console.print(f"  Total latency: {tot_ms}ms")
    console.print("\n[bold green]✓ LLM client tracing working. Run agent.py next.[/bold green]")