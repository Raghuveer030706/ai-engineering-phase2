"""
async_llm_client.py — Day 3: Async LLM client

What changed from Day 1 llm_client.py:
  - Uses anthropic.AsyncAnthropic instead of anthropic.Anthropic
  - ask() and ask_with_history() are now `async def`
  - Every call site must `await` them
  - Tracing is identical — same spans, same attributes, same cost tracking

What did NOT change:
  - LLMResponse dataclass — identical
  - Cost calculation — identical
  - Span names and attributes — identical
  - System prompt handling — identical

Why AsyncAnthropic:
  The standard anthropic.Anthropic client uses httpx in synchronous mode
  internally. anthropic.AsyncAnthropic uses httpx in async mode — it
  returns an awaitable so the event loop can suspend the coroutine while
  the HTTP request is in flight, instead of blocking the whole thread.

Run:
  python async_llm_client.py
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from opentelemetry.trace import StatusCode
from rich.console import Console
from rich.panel import Panel

# Day 1 tracer lives next door — add to path (no dotted imports on Windows)
DAY1_DIR = Path(__file__).parent.parent / "day1-otel-tracing"
sys.path.insert(0, str(DAY1_DIR))

from tracer import get_trace_id, get_span_id, get_tracer, load_traces_for_id, setup_tracing

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

console = Console()


# ── Pricing (same as Day 1) ───────────────────────────────────────────────────
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


# ── Response dataclass (identical to Day 1) ───────────────────────────────────
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
            f"{self.input_tokens}in / {self.output_tokens}out | "
            f"${self.cost_usd:.6f} | {self.latency_ms}ms"
        )


# ── Async traced client ───────────────────────────────────────────────────────
class AsyncTracedLLMClient:
    """
    Async version of TracedLLMClient from Day 1.

    Key difference: uses anthropic.AsyncAnthropic.
    Every method is `async def` and must be awaited by the caller.

    The tracing logic is byte-for-byte identical to Day 1.
    Only the anthropic call and the method signatures changed.

    Usage:
        client = AsyncTracedLLMClient()
        response = await client.ask("What is 2+2?")
        #          ^^^^^ must await
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model  = model
        # AsyncAnthropic — this is the only line different from Day 1
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.tracer = get_tracer("async-llm-client")

    async def ask(
        self,
        prompt:     str,
        system:     str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        span_name:  str = "llm.call",
    ) -> LLMResponse:
        """
        Single-turn async LLM call.
        `await` this — it suspends until Anthropic responds,
        freeing the event loop to do other work during the wait.
        """
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("llm.model",      self.model)
            span.set_attribute("llm.prompt_len", len(prompt))

            t0 = time.perf_counter()
            try:
                # This is the await point — coroutine suspends here
                # Event loop can run other coroutines during this ~800ms
                msg = await self.client.messages.create(
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

            # Tracing attributes — identical to Day 1
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

    async def ask_with_history(
        self,
        messages:   list[dict],
        system:     str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        span_name:  str = "llm.call.history",
    ) -> LLMResponse:
        """
        Multi-turn async call.
        messages = full conversation history so far.
        """
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("llm.model",         self.model)
            span.set_attribute("llm.history_turns",  len(messages))

            t0 = time.perf_counter()
            try:
                msg = await self.client.messages.create(
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
async def main():
    console.print("\n[bold yellow]Day 3 — Async LLM Client[/bold yellow]\n")

    setup_tracing("ai-engineering-day3")
    client = AsyncTracedLLMClient()
    tracer = get_tracer("smoke-test")

    with tracer.start_as_current_span("async-llm-smoke-test"):
        trace_id = get_trace_id()
        console.print(f"[dim]trace_id: {trace_id}[/dim]\n")

        # Sequential awaits — same as Day 1 behaviour
        # Day 4 will run these concurrently with asyncio.gather()
        console.print("[bold]Call 1:[/bold] awaiting...")
        r1 = await client.ask(
            "What is the capital of Japan? One sentence.",
            span_name="llm.call.1",
        )
        console.print(Panel(
            f"[green]{r1.content}[/green]\n\n[dim]{r1.summary()}[/dim]",
            title="Response 1", border_style="cyan",
        ))

        console.print("\n[bold]Call 2:[/bold] awaiting...")
        r2 = await client.ask(
            "In 2 sentences, what is an event loop in asyncio?",
            span_name="llm.call.2",
        )
        console.print(Panel(
            f"[green]{r2.content}[/green]\n\n[dim]{r2.summary()}[/dim]",
            title="Response 2", border_style="cyan",
        ))

    import time as _time
    _time.sleep(0.5)  # flush BatchSpanProcessor

    spans    = load_traces_for_id(trace_id)
    tot_cost = sum(s["attributes"].get("llm.cost_usd", 0) for s in spans)
    tot_ms   = sum(s["attributes"].get("llm.latency_ms", 0) for s in spans)

    console.print(f"\n[bold]Trace summary[/bold] ({trace_id[:16]}...)")
    console.print(f"  Spans:         {len(spans)}")
    console.print(f"  Total cost:    [yellow]${tot_cost:.6f}[/yellow]")
    console.print(f"  Total latency: {tot_ms}ms")
    console.print(
        "\n[bold green]✓ Async client working. "
        "Run async_agent.py next.[/bold green]"
    )


if __name__ == "__main__":
    # asyncio.run() starts the event loop and runs main() inside it
    # This is the entry point for all async programs
    asyncio.run(main())