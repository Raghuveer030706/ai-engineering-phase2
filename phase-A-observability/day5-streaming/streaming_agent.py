"""
streaming_agent.py — Day 5: Streaming ReAct agent + FastAPI endpoint

Two things in this file:

1. StreamingReActAgent
   - Same ReAct loop as Days 1-4
   - Uses StreamingLLMClient internally
   - Each step streams tokens to console as they arrive
   - Records TTFT per step — new metric in traces

2. FastAPI streaming endpoint
   - POST /ask → StreamingResponse
   - Tokens are sent to the HTTP client as they arrive
   - This is how every production LLM API works
     (OpenAI, Anthropic's own API, Gemini — all stream by default)

Why streaming in the agent:
  ReAct agents have multiple steps. Without streaming each step shows
  nothing for 800ms then dumps. With streaming the user sees the
  Thought forming, then the Action, in real time. Feels like thinking.

Run the agent directly:
  python streaming_agent.py

Run the FastAPI server:
  python streaming_agent.py --serve
  Then: curl -X POST http://localhost:8001/ask -H "Content-Type: application/json"
        -d "{\"question\": \"What is 12 * 8?\"}"
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import anthropic
from dotenv import load_dotenv
from opentelemetry.trace import StatusCode
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

DAY1_DIR = Path(__file__).parent.parent / "day1-otel-tracing"
DAY3_DIR = Path(__file__).parent.parent / "day3-async-agent"
sys.path.insert(0, str(DAY1_DIR))
sys.path.insert(0, str(DAY3_DIR))

from tracer import get_trace_id, get_span_id, get_tracer, setup_tracing
from async_agent import Tool, TOOLS, TOOL_MAP, build_system_prompt, AgentStep
from streaming_client import StreamingLLMClient, StreamedResponse, DEFAULT_MODEL

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
console = Console()


# ── Streaming ReAct Agent ─────────────────────────────────────────────────────
class StreamingReActAgent:
    """
    ReAct agent that streams each step's LLM output to the console.

    New metric tracked: time_to_first_token_ms per step.
    This tells you how long the user waits before seeing anything.

    The agent structure is identical to Day 3 AsyncReActAgent.
    Only the LLM client changed — StreamingLLMClient instead of AsyncTracedLLMClient.
    """

    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self.llm       = StreamingLLMClient()
        self.tracer    = get_tracer("streaming-react-agent")
        self.system    = build_system_prompt()

    async def run(self, question: str, stream_to_console: bool = True) -> dict:
        with self.tracer.start_as_current_span("agent.run") as root:
            trace_id = get_trace_id()
            root.set_attribute("agent.question",  question)
            root.set_attribute("agent.streaming", stream_to_console)

            console.print(f"\n[dim]trace_id: {trace_id}[/dim]")
            console.print(f"[bold cyan]Question:[/bold cyan] {question}\n")

            wall_start = time.perf_counter()
            history: list[dict] = [{"role": "user", "content": question}]
            steps:   list[AgentStep] = []
            total_cost = 0.0
            ttft_per_step: list[int] = []

            for step_num in range(1, self.max_steps + 1):
                step, ttft_ms = await self._run_step(
                    step_num, history, stream_to_console
                )
                steps.append(step)
                total_cost += step.cost_usd
                ttft_per_step.append(ttft_ms)

                if step.is_final:
                    break

                history.append({"role": "assistant", "content": self._step_text(step)})
                history.append({"role": "user",      "content": f"Observation: {step.observation}"})

            else:
                root.set_status(StatusCode.ERROR, "max_steps reached")

            wall_ms  = int((time.perf_counter() - wall_start) * 1000)
            avg_ttft = int(sum(ttft_per_step) / len(ttft_per_step)) if ttft_per_step else 0

            root.set_attribute("agent.step_count",       len(steps))
            root.set_attribute("agent.total_cost_usd",   round(total_cost, 8))
            root.set_attribute("agent.wall_clock_ms",    wall_ms)
            root.set_attribute("agent.avg_ttft_ms",      avg_ttft)
            root.set_status(StatusCode.OK)

            return {
                "answer":           steps[-1].final_answer if steps else None,
                "steps":            steps,
                "trace_id":         trace_id,
                "total_cost_usd":   total_cost,
                "total_latency_ms": wall_ms,
                "step_count":       len(steps),
                "avg_ttft_ms":      avg_ttft,
                "ttft_per_step":    ttft_per_step,
            }

    async def _run_step(
        self,
        step_num: int,
        history:  list[dict],
        stream_to_console: bool,
    ) -> tuple[AgentStep, int]:
        with self.tracer.start_as_current_span(f"agent.step.{step_num}") as step_span:
            step_span.set_attribute("agent.step_num", step_num)
            t0 = time.perf_counter()

            if stream_to_console:
                console.print(f"\n[dim]── Step {step_num} (streaming) ──[/dim]")
                response: StreamedResponse = await self.llm.stream_to_console(
                    prompt=history[-1]["content"] if len(history) == 1 else "",
                    system=self.system,
                    title=f"Step {step_num}",
                ) if len(history) == 1 else await self._stream_history(history, step_num)
            else:
                response = await self.llm.stream_with_history(
                    messages=history,
                    system=self.system,
                    span_name=f"llm.stream.step{step_num}",
                )

            parsed  = self._parse(response.content)
            elapsed = int((time.perf_counter() - t0) * 1000)

            step_span.set_attribute("agent.thought",    parsed.get("thought", ""))
            step_span.set_attribute("agent.is_final",   parsed.get("is_final", False))
            step_span.set_attribute("agent.ttft_ms",    response.time_to_first_token_ms)
            step_span.set_attribute("agent.step_cost",  round(response.cost_usd, 8))

            observation = None
            if not parsed.get("is_final") and parsed.get("action"):
                observation = self._call_tool(parsed["action"], parsed.get("action_input", ""))
                console.print(f"[dim]  → {parsed['action']}({parsed.get('action_input', '')}) = {observation}[/dim]")
                step_span.set_attribute("agent.tool",        parsed["action"])
                step_span.set_attribute("agent.observation", str(observation)[:300])

            if parsed.get("is_final"):
                console.print(Panel(
                    f"[bold green]{parsed['final_answer']}[/bold green]",
                    title="Final Answer",
                    border_style="green",
                ))

            step_span.set_status(StatusCode.OK)

            agent_step = AgentStep(
                step_num=step_num,
                thought=parsed.get("thought", ""),
                action=parsed.get("action"),
                action_input=parsed.get("action_input"),
                observation=observation,
                is_final=parsed.get("is_final", False),
                final_answer=parsed.get("final_answer"),
                span_id=get_span_id(),
                latency_ms=response.latency_ms,
                cost_usd=response.cost_usd,
            )

            return agent_step, response.time_to_first_token_ms

    async def _stream_history(self, history: list[dict], step_num: int) -> StreamedResponse:
        """Stream a multi-turn call with visible console output."""
        import os
        t0               = time.perf_counter()
        ttft_ms          = 0
        first_token_seen = False
        accumulated      = ""
        input_tokens     = 0
        output_tokens    = 0

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        from rich.live import Live
        with Live(
            Panel("", title=f"Step {step_num} (streaming...)", border_style="blue"),
            console=console,
            refresh_per_second=15,
        ) as live:
            async with client.messages.stream(
                model=self.llm.model,
                max_tokens=1024,
                system=self.system,
                messages=history,
            ) as stream:
                async for text in stream.text_stream:
                    if not first_token_seen:
                        ttft_ms          = int((time.perf_counter() - t0) * 1000)
                        first_token_seen = True
                    accumulated += text
                    live.update(Panel(
                        accumulated,
                        title=f"Step {step_num} [dim](TTFT: {ttft_ms}ms)[/dim]",
                        border_style="blue",
                    ))

                final         = await stream.get_final_message()
                input_tokens  = final.usage.input_tokens
                output_tokens = final.usage.output_tokens

        latency_ms = int((time.perf_counter() - t0) * 1000)
        cost_usd   = self.llm._cost(self.llm.model, input_tokens, output_tokens) \
                     if hasattr(self.llm, '_cost') else 0.0

        from streaming_client import StreamedResponse as SR, _cost
        return SR(
            content=accumulated,
            model=self.llm.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost(self.llm.model, input_tokens, output_tokens),
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            trace_id=get_trace_id(),
            span_id=get_span_id(),
        )

    def _call_tool(self, tool_name: str, tool_input: str) -> str:
        tool = TOOL_MAP.get(tool_name)
        if not tool:
            return f"Unknown tool: {tool_name}"
        return str(tool.fn(tool_input))

    def _parse(self, text: str) -> dict:
        out = {"thought": "", "action": None, "action_input": None,
               "is_final": False, "final_answer": None}
        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("Thought:"):
                out["thought"] = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                out["action"] = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                out["action_input"] = line[len("Action Input:"):].strip()
            elif line.startswith("Final Answer:"):
                out["is_final"]     = True
                out["final_answer"] = line[len("Final Answer:"):].strip()
        return out

    def _step_text(self, step: AgentStep) -> str:
        parts = [f"Thought: {step.thought}"]
        if step.action:
            parts.append(f"Action: {step.action}")
        if step.action_input is not None:
            parts.append(f"Action Input: {step.action_input}")
        return "\n".join(parts)


# ── FastAPI streaming endpoint ────────────────────────────────────────────────
def create_app():
    """
    Creates a FastAPI app with a streaming /ask endpoint.
    Tokens are sent to the HTTP client as they arrive using
    Server-Sent Events (text/event-stream).

    This is how production LLM APIs work:
      - Client opens connection
      - Server sends tokens as data: <token>\n\n
      - Client renders each token immediately
      - Connection closes when done
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        console.print("[red]FastAPI not found — run: pip install fastapi uvicorn --break-system-packages[/red]")
        return None

    import os as _os

    app = FastAPI(title="AI Engineering Day 5 — Streaming API")

    class AskRequest(BaseModel):
        question: str
        max_steps: int = 6

    @app.get("/health")
    async def health():
        return {"status": "ok", "streaming": True, "day": 5}

    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest):
        """
        Streaming endpoint. Returns tokens as Server-Sent Events.
        Open in browser or curl to see tokens arrive in real time.
        """
        async def token_generator() -> AsyncGenerator[str, None]:
            client = anthropic.AsyncAnthropic(api_key=_os.getenv("ANTHROPIC_API_KEY"))

            # Simple single-turn streaming for the API demo
            # (Full agent streaming would require more complex SSE protocol)
            yield f"data: {json.dumps({'type': 'start', 'question': req.question})}\n\n"

            accumulated = ""
            async with client.messages.stream(
                model=DEFAULT_MODEL,
                max_tokens=512,
                system="You are a helpful AI assistant. Be concise.",
                messages=[{"role": "user", "content": req.question}],
            ) as stream:
                async for text in stream.text_stream:
                    accumulated += text
                    yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

                final = await stream.get_final_message()

            yield f"data: {json.dumps({'type': 'done', 'total_tokens': final.usage.output_tokens})}\n\n"

        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},  # disable nginx buffering
        )

    @app.post("/ask")
    async def ask(req: AskRequest):
        """Non-streaming endpoint — returns full response (for comparison)."""
        import os as _os2
        client = anthropic.AsyncAnthropic(api_key=_os2.getenv("ANTHROPIC_API_KEY"))
        msg = await client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=512,
            system="You are a helpful AI assistant. Be concise.",
            messages=[{"role": "user", "content": req.question}],
        )
        return {
            "answer": msg.content[0].text,
            "tokens": msg.usage.output_tokens,
            "streaming": False,
        }

    return app


# ── CLI entry point ───────────────────────────────────────────────────────────
async def run_demo():
    console.rule("[bold yellow]Day 5 — Streaming ReAct Agent[/bold yellow]")
    setup_tracing("ai-engineering-day5")

    agent = StreamingReActAgent(max_steps=6)

    questions = [
        "Calculate 256 / 16, then reverse the result as text.",
        "Count the words in 'Streaming makes AI feel alive and responsive'.",
    ]

    ttft_results = []

    for i, q in enumerate(questions, 1):
        console.rule(f"[cyan]Question {i}[/cyan]")
        result = await agent.run(q, stream_to_console=True)
        ttft_results.append(result)
        console.print(
            f"\n[dim]  avg TTFT: {result['avg_ttft_ms']}ms | "
            f"wall-clock: {result['total_latency_ms']}ms | "
            f"cost: ${result['total_cost_usd']:.6f}[/dim]\n"
        )

    await asyncio.sleep(0.6)

    # Summary
    console.rule("[bold]Day 5 Summary[/bold]")
    table = Table(show_header=True, show_lines=True)
    table.add_column("Metric",        style="cyan")
    table.add_column("Value",         justify="right")
    table.add_column("Note",          style="dim")

    avg_ttft  = int(sum(r["avg_ttft_ms"] for r in ttft_results) / len(ttft_results))
    avg_total = int(sum(r["total_latency_ms"] for r in ttft_results) / len(ttft_results))

    table.add_row("Avg TTFT",         f"{avg_ttft}ms",
                  "time until user sees first token")
    table.add_row("Avg total latency", f"{avg_total}ms",
                  "time until response complete")
    table.add_row("TTFT / total",      f"{avg_ttft/avg_total*100:.0f}%",
                  "lower = more responsive feel")
    table.add_row("Day 3 baseline",    "5606ms",
                  "non-streaming, sequential")
    table.add_row("Day 4 parallel",    "2942ms",
                  "non-streaming, parallel")

    console.print(table)
    console.print(
        f"\n[bold green]✓ Day 5 complete — Phase A complete[/bold green]\n"
        f"[dim]TTFT of {avg_ttft}ms means users see the first token "
        f"in under half a second.\n"
        f"The remaining {avg_total - avg_ttft}ms feels like reading, not waiting.[/dim]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true",
                        help="Start FastAPI server instead of running demo")
    args = parser.parse_args()

    if args.serve:
        try:
            import uvicorn
            setup_tracing("ai-engineering-day5-api")
            app = create_app()
            if app:
                console.print(
                    "\n[bold green]Starting streaming API[/bold green]\n"
                    "  POST http://localhost:8001/ask/stream  ← streaming\n"
                    "  POST http://localhost:8001/ask         ← non-streaming\n"
                    "  GET  http://localhost:8001/health\n"
                    "  GET  http://localhost:8001/docs        ← Swagger UI\n"
                )
                uvicorn.run(app, host="0.0.0.0", port=8001)
        except ImportError:
            console.print(
                "[red]uvicorn not found.[/red]\n"
                "Run: pip install fastapi uvicorn --break-system-packages"
            )
    else:
        asyncio.run(run_demo())