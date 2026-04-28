"""
agent.py — Day 1: Fully traced ReAct agent

What this does:
  - ReAct agent (same pattern as your Phase 3 Day 10) rebuilt with tracing
  - Every step (think → act → observe) is its own child span
  - Parent span wraps the entire run — one trace_id for the whole thing
  - Span hierarchy makes the agent's decision tree visible in trace.jsonl

Span structure per run:
  agent.run                        ← root (whole question)
    agent.step.1                   ← one span per ReAct step
      llm.call.step1               ← LLM call (tokens, cost, latency)
      agent.tool.calculator        ← tool call (input, output, latency)
    agent.step.2
      llm.call.step2
      agent.tool.word_counter

Before today: wrong answer → no idea which step failed.
After today:  grep trace.jsonl for trace_id → see every decision.

Run:
  python agent.py
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from opentelemetry.trace import StatusCode
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tracer import get_trace_id, get_span_id, get_tracer, load_traces_for_id, setup_tracing
from llm_client import TracedLLMClient

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
console = Console()


# ── Tool definitions ──────────────────────────────────────────────────────────
@dataclass
class Tool:
    name:        str
    description: str
    fn:          Callable


def calculator(expression: str) -> str:
    """Safe arithmetic — only digits and operators allowed."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return f"Error: unsafe characters in '{expression}'"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def word_counter(text: str) -> str:
    """Count words, characters, and sentences."""
    words     = len(text.split())
    chars     = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return json.dumps({"words": words, "characters": chars, "sentences": sentences})


def text_reverser(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


TOOLS = [
    Tool("calculator",    "Evaluate arithmetic. Input: math expression as string.", calculator),
    Tool("word_counter",  "Count words/chars/sentences. Input: text string.",       word_counter),
    Tool("text_reverser", "Reverse a string. Input: any text string.",              text_reverser),
]
TOOL_MAP = {t.name: t for t in TOOLS}


# ── System prompt ─────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    tool_list = "\n".join(f"  - {t.name}: {t.description}" for t in TOOLS)
    return f"""You are a ReAct agent. Solve tasks step by step.

Format each step EXACTLY like this:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to the tool>

When you have the final answer, use:
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


# ── Step dataclass ────────────────────────────────────────────────────────────
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


# ── Agent ─────────────────────────────────────────────────────────────────────
class TracedReActAgent:
    """
    ReAct agent where every step creates a traced span.

    Key design decisions carried forward from Phase 3:
    - Observations go in as user messages (not assistant) — per your gotcha
    - Fresh agent instance per question — avoids context staleness
    - Greedy parser handles multi-line LLM responses safely
    """

    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self.llm       = TracedLLMClient()
        self.tracer    = get_tracer("react-agent")
        self.system    = build_system_prompt()

    def run(self, question: str) -> dict:
        """
        Run agent on a question. Returns:
          answer, steps, trace_id, total_cost_usd, total_latency_ms, step_count
        """
        with self.tracer.start_as_current_span("agent.run") as root:
            trace_id = get_trace_id()
            root.set_attribute("agent.question",  question)
            root.set_attribute("agent.max_steps", self.max_steps)

            console.print(f"\n[dim]trace_id: {trace_id}[/dim]")
            console.print(f"[bold cyan]Question:[/bold cyan] {question}\n")

            history: list[dict] = [{"role": "user", "content": question}]
            steps:   list[AgentStep] = []
            total_cost = 0.0

            for step_num in range(1, self.max_steps + 1):
                step = self._run_step(step_num, history)
                steps.append(step)
                total_cost += step.cost_usd
                self._print_step(step)

                if step.is_final:
                    root.set_attribute("agent.step_count",     step_num)
                    root.set_attribute("agent.total_cost_usd", round(total_cost, 8))
                    root.set_status(StatusCode.OK)
                    break

                # Observations as user messages — not assistant (Phase 3 gotcha)
                history.append({"role": "assistant", "content": self._step_text(step)})
                history.append({"role": "user",      "content": f"Observation: {step.observation}"})

            else:
                root.set_status(StatusCode.ERROR, "max_steps reached without Final Answer")

            return {
                "answer":           steps[-1].final_answer if steps else None,
                "steps":            steps,
                "trace_id":         trace_id,
                "total_cost_usd":   total_cost,
                "total_latency_ms": sum(s.latency_ms for s in steps),
                "step_count":       len(steps),
            }

    # ── Internal ──────────────────────────────────────────────────────────────
    def _run_step(self, step_num: int, history: list[dict]) -> AgentStep:
        with self.tracer.start_as_current_span(f"agent.step.{step_num}") as step_span:
            step_span.set_attribute("agent.step_num", step_num)
            t0 = time.perf_counter()

            response = self.llm.ask_with_history(
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

    def _parse(self, text: str) -> dict:
        """Parse Thought / Action / Action Input / Final Answer from LLM output."""
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
        if step.action:       parts.append(f"Action: {step.action}")
        if step.action_input is not None: parts.append(f"Action Input: {step.action_input}")
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


# ── Trace viewer ──────────────────────────────────────────────────────────────
def print_trace_summary(trace_id: str):
    spans = load_traces_for_id(trace_id)
    if not spans:
        console.print("[red]No spans found.[/red]")
        return

    table = Table(title=f"Full Trace — {trace_id[:16]}...", show_lines=True)
    table.add_column("span name",   style="cyan",   min_width=26)
    table.add_column("duration_ms", justify="right")
    table.add_column("cost_usd",    justify="right", style="yellow")
    table.add_column("tokens out",  justify="right")
    table.add_column("status",      justify="center")

    for s in spans:
        a      = s.get("attributes", {})
        cost   = a.get("llm.cost_usd", a.get("agent.total_cost_usd", 0))
        out    = a.get("llm.output_tokens", "")
        status = "[green]OK[/green]" if s["status"] == "OK" else f"[red]{s['status']}[/red]"
        table.add_row(
            s["name"],
            str(s["duration_ms"]),
            f"${cost:.6f}" if cost else "—",
            str(out) if out else "—",
            status,
        )

    console.print(table)

    tot_cost    = sum(s["attributes"].get("llm.cost_usd", 0) for s in spans)
    tot_latency = sum(s["duration_ms"] for s in spans if s["name"].startswith("agent.step"))
    llm_calls   = sum(1 for s in spans if s["name"].startswith("llm."))
    console.print(f"\n  Total LLM cost:   [yellow]${tot_cost:.6f}[/yellow]")
    console.print(f"  Total step time:  {tot_latency}ms")
    console.print(f"  LLM calls:        {llm_calls}")
    console.print(f"  Total spans:      {len(spans)}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    console.rule("[bold yellow]Day 1 — Traced ReAct Agent[/bold yellow]")
    setup_tracing("ai-engineering-day1")

    agent = TracedReActAgent(max_steps=6)

    # Run 1 — multi-tool question
    r1 = agent.run(
        "Calculate 17 * 23 + 45, then count the words in "
        "'The quick brown fox jumps over the lazy dog'."
    )

    console.rule()

    # Run 2 — single tool, tests fast path
    r2 = agent.run(
        "Reverse the text 'Hello AI Engineering' and give me the result."
    )

    # Show full trace for run 2
    console.rule("[bold]Trace Viewer[/bold]")
    print_trace_summary(r2["trace_id"])

    console.print(
        "\n[bold green]✓ Day 1 complete — "
        "traces written to traces/trace.jsonl[/bold green]"
    )
    console.print(
        "[dim]Next: python inspect_traces.py --cost[/dim]"
    )