"""
async_agent.py — Day 3: Async ReAct agent

What changed from Day 1 agent.py:
  - Agent class is now AsyncReActAgent
  - run(), _run_step(), _call_tool() are all `async def`
  - Every internal call uses `await`
  - Entry point uses asyncio.run()

What did NOT change:
  - Tool definitions (calculator, word_counter, text_reverser)
  - System prompt
  - Parser logic (_parse method)
  - Tracing — same span names, same attributes
  - Step printing — identical output

Why same wall-clock time today:
  Steps still run sequentially — await step1, then await step2, then await step3.
  The event loop CAN interleave other work during each await, but we're not
  giving it anything else to interleave yet.
  Day 4 gives it multiple things to run at once with asyncio.gather().

Benchmark to note:
  Run this and record the wall-clock time per question.
  Day 4 will compare against these numbers to prove the speedup.

Run:
  python async_agent.py
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from opentelemetry.trace import StatusCode
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Day 1 tracer — add to path
DAY1_DIR = Path(__file__).parent.parent / "day1-otel-tracing"
sys.path.insert(0, str(DAY1_DIR))

from tracer import get_trace_id, get_span_id, get_tracer, load_traces_for_id, setup_tracing
from async_llm_client import AsyncTracedLLMClient

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
console = Console()


# ── Tools (identical to Day 1) ────────────────────────────────────────────────
@dataclass
class Tool:
    name:        str
    description: str
    fn:          Callable


def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return f"Error: unsafe characters in '{expression}'"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def word_counter(text: str) -> str:
    words     = len(text.split())
    chars     = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return json.dumps({"words": words, "characters": chars, "sentences": sentences})


def text_reverser(text: str) -> str:
    return text[::-1]


TOOLS = [
    Tool("calculator",    "Evaluate arithmetic. Input: math expression.", calculator),
    Tool("word_counter",  "Count words/chars/sentences. Input: text.",    word_counter),
    Tool("text_reverser", "Reverse a string. Input: any text.",           text_reverser),
]
TOOL_MAP = {t.name: t for t in TOOLS}


# ── System prompt (identical to Day 1) ───────────────────────────────────────
def build_system_prompt() -> str:
    tool_list = "\n".join(f"  - {t.name}: {t.description}" for t in TOOLS)
    return f"""You are a ReAct agent. Solve tasks step by step.

Format each step EXACTLY like this:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to the tool>

When you have the final answer:
Thought: I now have the answer.
Final Answer: <your answer>

Available tools:
{tool_list}

Rules:
- Always start with Thought:
- Action must be exactly one of: {', '.join(TOOL_MAP.keys())}
- Never put Action and Final Answer in the same response
- Stop immediately once you write Final Answer
"""


# ── Step dataclass (identical to Day 1) ──────────────────────────────────────
@dataclass
class AgentStep:
    step_num:     int
    thought:      str
    action:       Optional[str]
    action_input: Optional[str]
    observation:  Optional[str]
    is_final:     bool
    final_answer: Optional[str]
    span_id:      str
    latency_ms:   int
    cost_usd:     float


# ── Async ReAct Agent ─────────────────────────────────────────────────────────
class AsyncReActAgent:
    """
    Async version of the Day 1 ReAct agent.

    Every method that touches the LLM is async def.
    The event loop suspends the coroutine at each `await` point,
    freeing the thread to do other work.

    Today: steps still sequential (await step1, await step2...).
    Day 4: asyncio.gather() will run independent sub-agents in parallel.

    Compare the wall-clock times here vs Day 1 — should be identical.
    That proves the async rewrite didn't break anything.
    Day 4 will show the actual speedup.
    """

    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self.llm       = AsyncTracedLLMClient()   # ← async client
        self.tracer    = get_tracer("async-react-agent")
        self.system    = build_system_prompt()

    async def run(self, question: str) -> dict:
        """
        Async entry point. Must be awaited:
            result = await agent.run("your question")
        """
        with self.tracer.start_as_current_span("agent.run") as root:
            trace_id = get_trace_id()
            root.set_attribute("agent.question",  question)
            root.set_attribute("agent.max_steps", self.max_steps)
            root.set_attribute("agent.mode",      "sequential-async")  # Day 4 will say "parallel"

            console.print(f"\n[dim]trace_id: {trace_id}[/dim]")
            console.print(f"[bold cyan]Question:[/bold cyan] {question}\n")

            wall_start = time.perf_counter()
            history: list[dict] = [{"role": "user", "content": question}]
            steps:   list[AgentStep] = []
            total_cost = 0.0

            for step_num in range(1, self.max_steps + 1):
                # Sequential await — one step at a time
                # This is what Day 4 changes to parallel
                step = await self._run_step(step_num, history)
                steps.append(step)
                total_cost += step.cost_usd
                self._print_step(step)

                if step.is_final:
                    break

                # Observations as user messages — Phase 3 gotcha preserved
                history.append({"role": "assistant", "content": self._step_text(step)})
                history.append({"role": "user",      "content": f"Observation: {step.observation}"})

            else:
                root.set_status(StatusCode.ERROR, "max_steps reached")

            wall_ms = int((time.perf_counter() - wall_start) * 1000)

            root.set_attribute("agent.step_count",       len(steps))
            root.set_attribute("agent.total_cost_usd",   round(total_cost, 8))
            root.set_attribute("agent.wall_clock_ms",    wall_ms)
            root.set_status(StatusCode.OK)

            return {
                "answer":           steps[-1].final_answer if steps else None,
                "steps":            steps,
                "trace_id":         trace_id,
                "total_cost_usd":   total_cost,
                "total_latency_ms": wall_ms,
                "step_count":       len(steps),
            }

    async def _run_step(self, step_num: int, history: list[dict]) -> AgentStep:
        with self.tracer.start_as_current_span(f"agent.step.{step_num}") as step_span:
            step_span.set_attribute("agent.step_num", step_num)
            t0 = time.perf_counter()

            # await — coroutine suspends here until LLM responds
            response = await self.llm.ask_with_history(
                messages=history,
                system=self.system,
                span_name=f"llm.call.step{step_num}",
            )

            parsed  = self._parse(response.content)
            elapsed = int((time.perf_counter() - t0) * 1000)

            step_span.set_attribute("agent.thought",  parsed.get("thought", ""))
            step_span.set_attribute("agent.is_final", parsed.get("is_final", False))

            observation = None
            if not parsed.get("is_final") and parsed.get("action"):
                # Tool calls are sync (CPU-bound, sub-millisecond)
                # No need to make them async — await is for I/O waits
                observation = self._call_tool(parsed["action"], parsed.get("action_input", ""))
                step_span.set_attribute("agent.tool",        parsed["action"])
                step_span.set_attribute("agent.observation", str(observation)[:300])

            step_span.set_attribute("agent.step_cost_usd",   round(response.cost_usd, 8))
            step_span.set_attribute("agent.step_latency_ms", elapsed)
            step_span.set_status(StatusCode.OK)

            return AgentStep(
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

    def _call_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Tools are synchronous — they're CPU-bound (arithmetic, string ops).
        No need to make them async. await is for I/O waits, not CPU work.
        This is an important distinction — don't async-ify everything.
        """
        with self.tracer.start_as_current_span(f"agent.tool.{tool_name}") as ts:
            ts.set_attribute("tool.name",  tool_name)
            ts.set_attribute("tool.input", str(tool_input)[:500])

            tool = TOOL_MAP.get(tool_name)
            if not tool:
                result = f"Unknown tool: {tool_name}"
                ts.set_status(StatusCode.ERROR, result)
                return result

            t0     = time.perf_counter()
            result = tool.fn(tool_input)
            ms     = int((time.perf_counter() - t0) * 1000)

            ts.set_attribute("tool.result",     str(result)[:500])
            ts.set_attribute("tool.latency_ms", ms)
            ts.set_status(StatusCode.OK)
            return str(result)

    # ── Parser and helpers (identical to Day 1) ───────────────────────────────
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

    def _print_step(self, step: AgentStep):
        color = "green" if step.is_final else "blue"
        label = f"Step {step.step_num}" + (" — FINAL" if step.is_final else "")
        body  = f"[bold]Thought:[/bold] {step.thought}\n"
        if step.action:
            body += f"[bold]Action:[/bold]  {step.action}({step.action_input})\n"
        if step.observation:
            body += f"[bold]Observe:[/bold] {step.observation}\n"
        if step.final_answer:
            body += f"\n[bold green]Answer:[/bold green] {step.final_answer}\n"
        body += (
            f"\n[dim]cost: ${step.cost_usd:.6f} | "
            f"latency: {step.latency_ms}ms | "
            f"span: {step.span_id[:8]}[/dim]"
        )
        console.print(Panel(body, title=label, border_style=color))


# ── Benchmark: async sequential vs Day 1 sync ─────────────────────────────────
async def run_benchmark():
    """
    Runs 2 questions and records wall-clock time.
    Save these numbers — Day 4 will beat them with parallel execution.
    """
    console.rule("[bold yellow]Day 3 — Async ReAct Agent[/bold yellow]")
    console.print(
        "[dim]Sequential async — same speed as Day 1 sync.\n"
        "Record these times. Day 4 will be faster.[/dim]\n"
    )

    setup_tracing("ai-engineering-day3")
    agent = AsyncReActAgent(max_steps=6)

    questions = [
        "Calculate 17 * 23 + 45, then count the words in "
        "'The quick brown fox jumps over the lazy dog'.",
        "Reverse the text 'Hello AI Engineering' and give me the result.",
    ]

    total_wall = 0
    for i, q in enumerate(questions, 1):
        console.rule(f"[cyan]Question {i}[/cyan]")
        t0     = time.perf_counter()
        result = await agent.run(q)
        wall   = int((time.perf_counter() - t0) * 1000)
        total_wall += wall

        console.print(
            f"\n[dim]  wall-clock: {wall}ms | "
            f"cost: ${result['total_cost_usd']:.6f} | "
            f"steps: {result['step_count']}[/dim]"
        )

    await asyncio.sleep(0.5)  # flush BatchSpanProcessor

    # Summary table
    console.rule("[bold]Day 3 Benchmark[/bold]")
    table = Table(show_header=True, show_lines=True)
    table.add_column("Metric",  style="cyan")
    table.add_column("Value",   justify="right")
    table.add_column("Note",    style="dim")

    table.add_row("Questions run",    str(len(questions)),    "")
    table.add_row("Total wall-clock", f"{total_wall}ms",      "← Day 4 will beat this")
    table.add_row("Avg per question", f"{total_wall//len(questions)}ms", "sequential async")
    table.add_row("Mode", "sequential", "await step1 → await step2 → ...")

    console.print(table)
    console.print(
        "\n[bold green]✓ Day 3 complete[/bold green]\n"
        "[dim]Note the total wall-clock time above.\n"
        "Day 4 uses asyncio.gather() to run questions in parallel\n"
        "and will show a real speedup.[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(run_benchmark())