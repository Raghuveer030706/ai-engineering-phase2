"""
parallel_agent.py — Day 4: Parallel sub-tasks with asyncio.gather()

The problem with Day 3:
  Sequential async still waits for each question to finish before starting
  the next one. Two questions = 2 × avg_latency = 5606ms.

What asyncio.gather() does:
  Launches multiple coroutines at the same time.
  The event loop interleaves them — while question 1 is waiting for
  Anthropic to respond, question 2's request is already in flight.
  Two questions = max(q1_latency, q2_latency) ≈ 2803ms.

When this helps vs when it doesn't:
  ✓ Multiple INDEPENDENT questions to the same agent
  ✓ Multiple specialist agents running in parallel (supervisor pattern)
  ✓ Fetching context from multiple sources simultaneously
  ✗ Steps within a single question (each step depends on previous observation)
  ✗ Anything with shared mutable state (race conditions)

Real-world use case this unlocks:
  Your Phase 5 planner → orchestrator → [specialist1, specialist2, specialist3]
  Today specialists run serially. After Day 4 they run in parallel.
  That's the actual production win.

Run:
  python parallel_agent.py
"""

import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

DAY1_DIR = Path(__file__).parent.parent / "day1-otel-tracing"
DAY3_DIR = Path(__file__).parent.parent / "day3-async-agent"
sys.path.insert(0, str(DAY1_DIR))
sys.path.insert(0, str(DAY3_DIR))

from tracer import setup_tracing, get_tracer, get_trace_id
from async_agent import AsyncReActAgent

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
console = Console()


# ── Sequential runner (Day 3 approach — for comparison) ──────────────────────
async def run_sequential(agent: AsyncReActAgent, questions: list[str]) -> dict:
    """
    Run questions one after another.
    This is identical to Day 3 behaviour.
    We run it first to get a fair same-session comparison baseline.
    """
    t0      = time.perf_counter()
    results = []
    for q in questions:
        r = await agent.run(q)
        results.append(r)
    wall_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "mode":       "sequential",
        "wall_ms":    wall_ms,
        "results":    results,
        "question_count": len(questions),
    }


# ── Parallel runner (Day 4 — the new thing) ───────────────────────────────────
async def run_parallel(agent: AsyncReActAgent, questions: list[str]) -> dict:
    """
    Run all questions simultaneously with asyncio.gather().

    asyncio.gather(*coroutines) does three things:
      1. Schedules all coroutines to start immediately
      2. Suspends at every `await` inside each coroutine
      3. Interleaves them — while one waits for Anthropic, others make progress
      4. Returns when ALL are done, results in the same order as input

    The agent instance is shared safely because:
      - AsyncReActAgent has no mutable state between runs
      - Each call to agent.run() creates its own local `history` list
      - The LLM client (AsyncAnthropic) is thread/coroutine safe
    """
    t0 = time.perf_counter()

    # Build coroutines — note: calling agent.run(q) does NOT start execution yet
    # It returns a coroutine object. gather() starts them all.
    coroutines = [agent.run(q) for q in questions]

    # This is the key line — all coroutines run concurrently
    results = await asyncio.gather(*coroutines)

    wall_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "mode":           "parallel",
        "wall_ms":        wall_ms,
        "results":        list(results),
        "question_count": len(questions),
    }


# ── Parallel specialists pattern (real-world use case) ────────────────────────
async def run_specialist_pattern(questions: list[str]) -> dict:
    """
    Simulates the supervisor → [specialist1, specialist2, specialist3] pattern
    from your Phase 3 multi-agent system.

    Each specialist gets a different question (its sub-task).
    All run in parallel. Supervisor collects results.

    This is the actual production pattern Day 4 unlocks.
    """
    tracer = get_tracer("supervisor")

    with tracer.start_as_current_span("supervisor.run") as span:
        span.set_attribute("supervisor.specialist_count", len(questions))
        span.set_attribute("supervisor.mode", "parallel")

        console.print(
            f"\n[bold magenta]Supervisor[/bold magenta] dispatching "
            f"{len(questions)} specialists in parallel...\n"
        )

        t0 = time.perf_counter()

        # Each specialist is a fresh agent — no shared state
        specialists = [AsyncReActAgent(max_steps=6) for _ in questions]
        coroutines  = [spec.run(q) for spec, q in zip(specialists, questions)]
        results     = await asyncio.gather(*coroutines)

        wall_ms = int((time.perf_counter() - t0) * 1000)

        # Collect answers
        answers = [r["answer"] for r in results]
        total_cost = sum(r["total_cost_usd"] for r in results)

        span.set_attribute("supervisor.wall_clock_ms", wall_ms)
        span.set_attribute("supervisor.total_cost_usd", round(total_cost, 8))

        console.print(Panel(
            "\n".join(
                f"[cyan]Specialist {i+1}:[/cyan] {a}"
                for i, a in enumerate(answers)
            ),
            title="[bold magenta]Supervisor — Collected Results[/bold magenta]",
            border_style="magenta",
        ))

        return {
            "wall_ms":    wall_ms,
            "answers":    answers,
            "total_cost": total_cost,
        }


# ── Benchmark: sequential vs parallel ────────────────────────────────────────
async def benchmark():
    console.rule("[bold yellow]Day 4 — Parallel Agents with asyncio.gather()[/bold yellow]")

    setup_tracing("ai-engineering-day4")

    # Same 2 questions as Day 3 — fair comparison
    questions = [
        "Calculate 17 * 23 + 45, then count the words in "
        "'The quick brown fox jumps over the lazy dog'.",
        "Reverse the text 'Hello AI Engineering' and give me the result.",
    ]

    agent = AsyncReActAgent(max_steps=6)

    # ── Round 1: Sequential (Day 3 baseline, measured fresh) ─────────────────
    console.rule("[cyan]Round 1 — Sequential (Day 3 approach)[/cyan]")
    console.print("[dim]Running questions one after another...[/dim]\n")
    seq = await run_sequential(agent, questions)
    console.print(f"\n[dim]Sequential total: {seq['wall_ms']}ms[/dim]")

    console.print("\n[dim]Waiting 2s before parallel run...[/dim]")
    await asyncio.sleep(2)

    # ── Round 2: Parallel (Day 4) ─────────────────────────────────────────────
    console.rule("[green]Round 2 — Parallel (asyncio.gather)[/green]")
    console.print("[dim]Running questions simultaneously...[/dim]\n")
    par = await run_parallel(agent, questions)
    console.print(f"\n[dim]Parallel total: {par['wall_ms']}ms[/dim]")

    await asyncio.sleep(2)

    # ── Round 3: Specialist pattern ───────────────────────────────────────────
    console.rule("[magenta]Round 3 — Supervisor + Specialists Pattern[/magenta]")
    specialist_questions = [
        "What is 144 / 12 * 7?",
        "Reverse the word 'parallel'.",
        "Count words in 'asyncio gather runs coroutines concurrently'.",
    ]
    spec = await run_specialist_pattern(specialist_questions)

    await asyncio.sleep(0.6)  # flush BatchSpanProcessor

    # ── Results table ─────────────────────────────────────────────────────────
    console.rule("[bold]Benchmark Results[/bold]")

    speedup    = seq["wall_ms"] / par["wall_ms"] if par["wall_ms"] else 0
    time_saved = seq["wall_ms"] - par["wall_ms"]

    table = Table(show_header=True, show_lines=True)
    table.add_column("Metric",          style="cyan",  min_width=22)
    table.add_column("Sequential",      justify="right")
    table.add_column("Parallel",        justify="right", style="green")
    table.add_column("Delta",           justify="right", style="yellow")

    table.add_row(
        "Wall-clock (2 questions)",
        f"{seq['wall_ms']}ms",
        f"{par['wall_ms']}ms",
        f"-{time_saved}ms",
    )
    table.add_row(
        "Speedup factor",
        "1.00×",
        f"{speedup:.2f}×",
        f"+{speedup-1:.2f}×",
    )
    table.add_row(
        "Day 3 baseline",
        "5606ms",
        f"{par['wall_ms']}ms",
        f"-{5606 - par['wall_ms']}ms vs Day 3",
    )
    table.add_row(
        "Specialist pattern (3 agents)",
        "~8400ms est.",
        f"{spec['wall_ms']}ms",
        f"~{8400 - spec['wall_ms']}ms saved",
    )

    console.print(table)

    console.print(
        f"\n[bold green]✓ Day 4 complete[/bold green]\n"
        f"[dim]asyncio.gather() ran {par['question_count']} questions in "
        f"{par['wall_ms']}ms instead of {seq['wall_ms']}ms.\n"
        f"Speedup: {speedup:.2f}× — the event loop interleaved LLM waits.[/dim]"
    )

    console.print(
        "\n[bold]What this unlocks for your Phase 5 system:[/bold]\n"
        "[dim]Your planner → orchestrator → [spec1, spec2, spec3] pattern\n"
        "currently runs specialists serially. Wire asyncio.gather() at the\n"
        "orchestrator level and specialists run in parallel automatically.[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(benchmark())