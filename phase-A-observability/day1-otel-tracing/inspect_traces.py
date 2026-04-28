"""
inspect_traces.py — Day 1: Query and analyse your trace file

Your debug companion. Run after agent.py to explore what happened.

Usage (PowerShell):
  python inspect_traces.py                    # all traces summary
  python inspect_traces.py --trace <id>       # all spans for one trace
  python inspect_traces.py --cost             # cost breakdown
  python inspect_traces.py --slow 500         # spans slower than 500ms
  python inspect_traces.py --last             # most recent trace only
"""

import argparse
import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

TRACE_FILE = Path(__file__).parent / "traces" / "trace.jsonl"
console    = Console()


def load_all() -> list[dict]:
    if not TRACE_FILE.exists():
        return []
    with open(TRACE_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_by_trace(spans: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list] = {}
    for s in spans:
        groups.setdefault(s["trace_id"], []).append(s)
    return groups


# ── Views ─────────────────────────────────────────────────────────────────────
def show_all(spans: list[dict]):
    groups = group_by_trace(spans)
    table  = Table(
        title=f"All Traces ({len(groups)} runs, {len(spans)} total spans)",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("trace_id",  style="cyan",   width=20)
    table.add_column("spans",     justify="right")
    table.add_column("total_ms",  justify="right")
    table.add_column("llm_cost",  justify="right", style="yellow")
    table.add_column("llm_calls", justify="right")
    table.add_column("root span")

    for tid, slist in groups.items():
        total_ms  = sum(s["duration_ms"] for s in slist)
        llm_cost  = sum(s["attributes"].get("llm.cost_usd", 0) for s in slist)
        llm_calls = sum(1 for s in slist if s["name"].startswith("llm."))
        root      = next(
            (s["name"] for s in slist if "run" in s["name"] or "root" in s["name"]),
            slist[0]["name"],
        )
        table.add_row(
            tid[:18] + "..",
            str(len(slist)),
            str(total_ms),
            f"${llm_cost:.6f}",
            str(llm_calls),
            root,
        )

    console.print(table)


def show_trace(spans: list[dict], trace_prefix: str):
    matches = [s for s in spans if s["trace_id"].startswith(trace_prefix)]
    if not matches:
        console.print(f"[red]No spans found for prefix: {trace_prefix}[/red]")
        return

    table = Table(
        title=f"Trace: {trace_prefix}",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
    )
    table.add_column("span name",   style="cyan")
    table.add_column("span_id",     style="dim",   width=10)
    table.add_column("duration_ms", justify="right")
    table.add_column("cost_usd",    justify="right", style="yellow")
    table.add_column("tokens i/o",  justify="right")
    table.add_column("status")

    for s in matches:
        a      = s.get("attributes", {})
        cost   = a.get("llm.cost_usd", 0)
        inp    = a.get("llm.input_tokens", "")
        out    = a.get("llm.output_tokens", "")
        tok    = f"{inp}/{out}" if inp else "—"
        status = "[green]OK[/green]" if s["status"] == "OK" else f"[red]{s['status']}[/red]"
        table.add_row(
            s["name"],
            s["span_id"][:10],
            str(s["duration_ms"]),
            f"${cost:.6f}" if cost else "—",
            tok,
            status,
        )

    console.print(table)


def show_cost(spans: list[dict]):
    groups    = group_by_trace(spans)
    total     = sum(s["attributes"].get("llm.cost_usd", 0) for s in spans)
    by_model: dict[str, float] = {}
    for s in spans:
        m = s.get("attributes", {}).get("llm.model", "")
        c = s.get("attributes", {}).get("llm.cost_usd", 0)
        if m:
            by_model[m] = by_model.get(m, 0) + c

    console.print(f"\n[bold]Cost across all traces[/bold]")
    console.print(f"  Total:       [yellow]${total:.6f}[/yellow]")
    console.print(f"  Traces:      {len(groups)}")
    console.print(f"  LLM calls:   {sum(1 for s in spans if s['name'].startswith('llm.'))}\n")
    for model, cost in sorted(by_model.items(), key=lambda x: -x[1]):
        console.print(f"  {model}: [yellow]${cost:.6f}[/yellow]")


def show_slow(spans: list[dict], threshold_ms: int):
    slow = [s for s in spans if s["duration_ms"] >= threshold_ms]
    if not slow:
        console.print(f"[green]No spans >= {threshold_ms}ms[/green]")
        return

    table = Table(title=f"Slow Spans (>= {threshold_ms}ms)", box=box.SIMPLE_HEAVY)
    table.add_column("span name",   style="cyan")
    table.add_column("duration_ms", justify="right", style="red")
    table.add_column("trace_id",    style="dim")

    for s in sorted(slow, key=lambda x: -x["duration_ms"]):
        table.add_row(s["name"], str(s["duration_ms"]), s["trace_id"][:18] + "..")

    console.print(table)


def show_last(spans: list[dict]):
    groups = group_by_trace(spans)
    if not groups:
        return
    last_tid   = list(groups.keys())[-1]
    last_spans = groups[last_tid]
    console.print(f"\n[bold]Most recent trace:[/bold] {last_tid[:16]}...")
    show_trace(spans, last_tid[:8])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace inspector for Day 1")
    parser.add_argument("--trace", help="Show all spans for a trace_id prefix")
    parser.add_argument("--cost",  action="store_true", help="Cost breakdown")
    parser.add_argument("--slow",  type=int, metavar="MS", help="Spans >= MS milliseconds")
    parser.add_argument("--last",  action="store_true", help="Most recent trace only")
    args = parser.parse_args()

    spans = load_all()
    if not spans:
        console.print("[yellow]No traces yet. Run agent.py first.[/yellow]")
        raise SystemExit(0)

    console.print(f"\n[dim]File: {TRACE_FILE} | Total spans: {len(spans)}[/dim]\n")

    if args.trace:
        show_trace(spans, args.trace)
    elif args.cost:
        show_cost(spans)
    elif args.slow is not None:
        show_slow(spans, args.slow)
    elif args.last:
        show_last(spans)
    else:
        show_all(spans)