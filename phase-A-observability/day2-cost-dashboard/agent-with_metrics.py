"""
agent_with_metrics.py — Day 2: Run more agent questions, then show dashboard

What this does:
  - Runs 3 new agent questions (adding to your existing trace.jsonl)
  - Each run produces richer span attributes than Day 1
  - After all runs complete, renders the full dashboard inline
  - Shows you how costs accumulate across a real session

This is the "integration test" for Day 2.
After running this you'll have 7+ traces to analyse.

Run:
  python agent_with_metrics.py
"""

import sys
import time
from pathlib import Path

# Day 1 code lives next door — add to path (Windows-safe, no dotted imports)
DAY1_DIR = Path(__file__).parent.parent / "day1-otel-tracing"
sys.path.insert(0, str(DAY1_DIR))

from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule

from tracer import setup_tracing
from agent import TracedReActAgent          # reuse Day 1 agent directly
from metrics import compute_dashboard
from dashboard import render

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
console = Console()


QUESTIONS = [
    # Tests calculator + multi-step reasoning
    "What is (144 / 12) * 7 + 33? Show your calculation.",

    # Tests word_counter on a longer string
    (
        "Count the words and characters in this sentence: "
        "'Observability is the ability to understand a system from its outputs.'"
    ),

    # Tests text_reverser then calculator in sequence
    "Reverse the word 'dashboard' and then calculate 256 / 16.",
]


def main():
    console.print(Rule("[bold yellow]Day 2 — Agent with Metrics[/bold yellow]"))
    console.print(
        "[dim]Running 3 new agent questions to build up trace data...[/dim]\n"
    )

    setup_tracing("ai-engineering-day2")
    agent = TracedReActAgent(max_steps=6)

    results = []
    for i, question in enumerate(QUESTIONS, 1):
        console.print(Rule(f"[cyan]Question {i} of {len(QUESTIONS)}[/cyan]"))
        result = agent.run(question)
        results.append(result)
        console.print(
            f"\n[dim]  → cost: ${result['total_cost_usd']:.6f} | "
            f"steps: {result['step_count']} | "
            f"latency: {result['total_latency_ms']}ms[/dim]\n"
        )
        time.sleep(0.3)   # small pause between runs

    # Session summary
    console.print(Rule("[bold]Session Summary[/bold]"))
    session_cost    = sum(r["total_cost_usd"]   for r in results)
    session_latency = sum(r["total_latency_ms"] for r in results)
    session_steps   = sum(r["step_count"]       for r in results)

    console.print(f"  Questions answered:  {len(results)}")
    console.print(f"  Total steps taken:   {session_steps}")
    console.print(f"  Session cost:        [yellow]${session_cost:.6f}[/yellow]")
    console.print(f"  Session latency:     {session_latency}ms total")
    console.print()

    # Flush traces then render dashboard
    time.sleep(0.6)
    console.print(Rule("[bold green]Full Dashboard (all traces including Day 1)[/bold green]"))

    m = compute_dashboard()
    if m:
        render(m)
    else:
        console.print("[red]Could not load trace data.[/red]")

    console.print(
        "\n[bold green]✓ Day 2 complete[/bold green]\n"
        "[dim]Run: python dashboard.py --live 5   "
        "(open this in a second PowerShell pane while agent.py runs)[/dim]"
    )


if __name__ == "__main__":
    main()