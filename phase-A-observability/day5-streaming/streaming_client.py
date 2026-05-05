"""
streaming_client.py — Day 5: Streaming LLM responses

The problem with Days 1-4:
  Every LLM call waits for the FULL response before returning anything.
  A 2-second call shows nothing for 2 seconds, then dumps everything at once.
  Users experience this as "the app is frozen."

What streaming does:
  Anthropic sends tokens as they are generated — typically 10-30ms apart.
  You display each token as it arrives.
  First token appears in <500ms. User sees the answer forming in real time.

Two streaming modes built here:
  1. stream_to_console() — prints tokens live as they arrive (demo mode)
  2. stream_collect()    — collects the full text + usage stats (agent mode)

Why two modes:
  Console streaming is for demos and CLI tools.
  Collect mode is for agents — you still need the full text to parse
  Thought/Action/Final Answer, but you get real usage stats mid-stream.

Run:
  python streaming_client.py
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

DAY1_DIR = Path(__file__).parent.parent / "day1-otel-tracing"
sys.path.insert(0, str(DAY1_DIR))

from tracer import get_trace_id, get_span_id, get_tracer, setup_tracing
from opentelemetry.trace import StatusCode

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
console = Console()


# ── Pricing (same as previous days) ──────────────────────────────────────────
COST_PER_TOKEN = {
    "claude-haiku-4-5-20251001": {
        "input":  0.80  / 1_000_000,
        "output": 4.00  / 1_000_000,
    },
}
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _cost(model: str, inp: int, out: int) -> float:
    p = COST_PER_TOKEN.get(model, COST_PER_TOKEN[DEFAULT_MODEL])
    return inp * p["input"] + out * p["output"]


# ── Response dataclass ────────────────────────────────────────────────────────
@dataclass
class StreamedResponse:
    content:            str
    model:              str
    input_tokens:       int
    output_tokens:      int
    cost_usd:           float
    latency_ms:         int       # time to FULL response
    time_to_first_token_ms: int   # time to FIRST token — the new metric
    trace_id:           str
    span_id:            str

    def summary(self) -> str:
        return (
            f"[{self.trace_id[:8]}...] "
            f"TTFT: {self.time_to_first_token_ms}ms | "
            f"total: {self.latency_ms}ms | "
            f"{self.input_tokens}in/{self.output_tokens}out | "
            f"${self.cost_usd:.6f}"
        )


# ── Streaming client ──────────────────────────────────────────────────────────
class StreamingLLMClient:
    """
    Async LLM client with streaming support.

    Key new concept — Time To First Token (TTFT):
      Non-streaming: you measure total latency only.
      Streaming:     you can also measure when the FIRST token arrives.
      TTFT is what determines whether the UI feels responsive.
      A 3-second response with 300ms TTFT feels fast.
      A 1-second response with 950ms TTFT feels slow.

    Usage:
        client = StreamingLLMClient()

        # Stream to console (shows tokens live)
        await client.stream_to_console("Tell me about asyncio")

        # Collect full response (for agents that need to parse output)
        response = await client.stream_collect("What is 2+2?")
        print(response.time_to_first_token_ms)
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model  = model
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.tracer = get_tracer("streaming-client")

    async def stream_to_console(
        self,
        prompt:     str,
        system:     str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        title:      str = "Streaming Response",
    ) -> StreamedResponse:
        """
        Stream tokens to the console as they arrive.
        Uses Rich Live to update a panel in place.
        """
        with self.tracer.start_as_current_span("llm.stream.console") as span:
            span.set_attribute("llm.model",      self.model)
            span.set_attribute("llm.prompt_len", len(prompt))

            t0              = time.perf_counter()
            ttft_ms         = 0
            first_token_seen = False
            accumulated     = ""
            input_tokens    = 0
            output_tokens   = 0

            with Live(
                Panel("", title=title, border_style="cyan"),
                console=console,
                refresh_per_second=15,
            ) as live:
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    async for text in stream.text_stream:
                        if not first_token_seen:
                            ttft_ms          = int((time.perf_counter() - t0) * 1000)
                            first_token_seen = True

                        accumulated += text
                        live.update(Panel(
                            accumulated,
                            title=f"{title} [dim](streaming...)[/dim]",
                            border_style="cyan",
                        ))

                    # Final message has usage stats
                    final = await stream.get_final_message()
                    input_tokens  = final.usage.input_tokens
                    output_tokens = final.usage.output_tokens

            latency_ms = int((time.perf_counter() - t0) * 1000)
            cost_usd   = _cost(self.model, input_tokens, output_tokens)

            span.set_attribute("llm.ttft_ms",        ttft_ms)
            span.set_attribute("llm.latency_ms",     latency_ms)
            span.set_attribute("llm.input_tokens",   input_tokens)
            span.set_attribute("llm.output_tokens",  output_tokens)
            span.set_attribute("llm.cost_usd",       round(cost_usd, 8))
            span.set_status(StatusCode.OK)

            return StreamedResponse(
                content=accumulated,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
                trace_id=get_trace_id(),
                span_id=get_span_id(),
            )

    async def stream_collect(
        self,
        prompt:     str,
        system:     str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        span_name:  str = "llm.stream.collect",
    ) -> StreamedResponse:
        """
        Stream internally but return the complete collected response.
        Used by the streaming agent — it needs the full text to parse
        Thought/Action/Final Answer, but still gets TTFT measurement
        and real usage stats from the stream.
        """
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("llm.model",      self.model)
            span.set_attribute("llm.prompt_len", len(prompt))

            t0               = time.perf_counter()
            ttft_ms          = 0
            first_token_seen = False
            accumulated      = ""
            input_tokens     = 0
            output_tokens    = 0

            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    if not first_token_seen:
                        ttft_ms          = int((time.perf_counter() - t0) * 1000)
                        first_token_seen = True
                    accumulated += text

                final         = await stream.get_final_message()
                input_tokens  = final.usage.input_tokens
                output_tokens = final.usage.output_tokens

            latency_ms = int((time.perf_counter() - t0) * 1000)
            cost_usd   = _cost(self.model, input_tokens, output_tokens)

            span.set_attribute("llm.ttft_ms",       ttft_ms)
            span.set_attribute("llm.latency_ms",    latency_ms)
            span.set_attribute("llm.input_tokens",  input_tokens)
            span.set_attribute("llm.output_tokens", output_tokens)
            span.set_attribute("llm.cost_usd",      round(cost_usd, 8))
            span.set_status(StatusCode.OK)

            return StreamedResponse(
                content=accumulated,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
                trace_id=get_trace_id(),
                span_id=get_span_id(),
            )

    async def stream_with_history(
        self,
        messages:   list[dict],
        system:     str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        span_name:  str = "llm.stream.history",
    ) -> StreamedResponse:
        """Multi-turn streaming — used by the streaming agent."""
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("llm.model",         self.model)
            span.set_attribute("llm.history_turns",  len(messages))

            t0               = time.perf_counter()
            ttft_ms          = 0
            first_token_seen = False
            accumulated      = ""
            input_tokens     = 0
            output_tokens    = 0

            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    if not first_token_seen:
                        ttft_ms          = int((time.perf_counter() - t0) * 1000)
                        first_token_seen = True
                    accumulated += text

                final         = await stream.get_final_message()
                input_tokens  = final.usage.input_tokens
                output_tokens = final.usage.output_tokens

            latency_ms = int((time.perf_counter() - t0) * 1000)
            cost_usd   = _cost(self.model, input_tokens, output_tokens)

            span.set_attribute("llm.ttft_ms",       ttft_ms)
            span.set_attribute("llm.latency_ms",    latency_ms)
            span.set_attribute("llm.input_tokens",  input_tokens)
            span.set_attribute("llm.output_tokens", output_tokens)
            span.set_attribute("llm.cost_usd",      round(cost_usd, 8))
            span.set_status(StatusCode.OK)

            return StreamedResponse(
                content=accumulated,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
                trace_id=get_trace_id(),
                span_id=get_span_id(),
            )


# ── Smoke test ────────────────────────────────────────────────────────────────
async def main():
    console.print("\n[bold yellow]Day 5 — Streaming LLM Client[/bold yellow]\n")
    setup_tracing("ai-engineering-day5")

    client = StreamingLLMClient()
    tracer = get_tracer("smoke-test")

    with tracer.start_as_current_span("streaming-smoke-test"):

        # Demo 1: visible streaming to console
        console.print("[bold]Demo 1:[/bold] Streaming to console (watch tokens arrive)\n")
        r1 = await client.stream_to_console(
            "Explain what asyncio.gather() does in 3 sentences.",
            title="asyncio.gather() explanation",
        )
        console.print(f"\n[dim]{r1.summary()}[/dim]\n")

        await asyncio.sleep(0.5)

        # Demo 2: collect mode (what agent uses internally)
        console.print("\n[bold]Demo 2:[/bold] Collect mode (streaming internally, returns full text)")
        r2 = await client.stream_collect(
            "What is Time To First Token and why does it matter for user experience?",
        )
        console.print(Panel(r2.content, title="Collected Response", border_style="green"))
        console.print(f"[dim]{r2.summary()}[/dim]")

    await asyncio.sleep(0.5)

    # TTFT comparison
    console.print(f"\n[bold]TTFT Comparison[/bold]")
    console.print(f"  Demo 1 — TTFT: [green]{r1.time_to_first_token_ms}ms[/green]  "
                  f"total: {r1.latency_ms}ms")
    console.print(f"  Demo 2 — TTFT: [green]{r2.time_to_first_token_ms}ms[/green]  "
                  f"total: {r2.latency_ms}ms")
    console.print(
        f"\n[dim]TTFT is what the user feels. "
        f"Total latency is what the logs show.[/dim]"
    )
    console.print("\n[bold green]✓ Streaming client working. Run streaming_agent.py next.[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())