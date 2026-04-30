"""
dashboard.py — Day 2: Cost & latency terminal dashboard

What this does:
  - Reads traces/trace.jsonl from Day 1 (no new API calls)
  - Renders a live terminal dashboard with:
      - Running cost total + projection
      - Per-model cost breakdown bar chart
      - LLM call latency histogram
      - Per-trace table (cost, latency, steps, tokens)
      - Top 5 slowest spans
  - Live mode: refreshes every N seconds watching for new traces
  - Snapshot mode: one-shot render and exit

Why terminal, not a web UI:
  You can leave this running in a split pane while agent.py runs.
  No browser, no server, no port. See costs update in real time.

Usage:
  python dashboard.py              # snapshot — render once and exit
  python dashboard.py --live 5     # refresh every 5 seconds
  python dashboard.py --live 2     # refresh every 2 seconds
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import plotext as plt
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from metrics import compute_dashboard, DashboardMetrics, TRACE_FILE

console = Console()


# ── Formatters ────────────────────────────────────────────────────────────────
def fmt_cost(usd: float) -> str:
    if usd == 0:
        return "$0.000000"
    if usd < 0.001:
        return f"${usd:.6f}"
    return f"${usd:.4f}"


def fmt_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms/1000:.1f}s"


# ── Render sections ───────────────────────────────────────────────────────────
def render_header(m: DashboardMetrics) -> Panel:
    """Top KPI bar."""
    cost_per_trace = m.total_cost_usd / m.total_traces if m.total_traces else 0

    # Project monthly cost if you ran 100 agent queries/day
    monthly_projection = cost_per_trace * 100 * 30

    kpis = [
        ("Total cost",         fmt_cost(m.total_cost_usd), "yellow"),
        ("Traces",             str(m.total_traces),         "cyan"),
        ("LLM calls",          str(m.total_llm_calls),      "cyan"),
        ("Avg cost/trace",     fmt_cost(cost_per_trace),    "green"),
        ("Input tokens",       f"{m.total_input_tokens:,}", "blue"),
        ("Output tokens",      f"{m.total_output_tokens:,}","blue"),
        ("100 q/day × 30d",    fmt_cost(monthly_projection),"magenta"),
    ]

    parts = []
    for label, value, color in kpis:
        parts.append(f"[dim]{label}[/dim]\n[bold {color}]{value}[/bold {color}]")

    text = "    ".join(parts)
    return Panel(text, title="[bold]Cost & Usage Overview[/bold]", border_style="yellow")


def render_model_breakdown(m: DashboardMetrics) -> Panel:
    """Bar chart: cost per model."""
    if not m.by_model:
        return Panel("No model data", title="By Model")

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("Model",   style="cyan")
    table.add_column("Cost",    justify="right", style="yellow")
    table.add_column("Share",   justify="right")
    table.add_column("Bar")

    total = m.total_cost_usd or 1
    max_w = 30  # bar width

    for model, cost in sorted(m.by_model.items(), key=lambda x: -x[1]):
        pct      = cost / total
        bar_len  = max(1, int(pct * max_w))
        bar      = "█" * bar_len + "░" * (max_w - bar_len)
        table.add_row(
            model.split("-")[1] if "-" in model else model,  # shorten name
            fmt_cost(cost),
            f"{pct*100:.1f}%",
            f"[yellow]{bar}[/yellow]",
        )

    return Panel(table, title="[bold]Cost by Model[/bold]", border_style="green")


def render_latency_histogram(m: DashboardMetrics) -> Panel:
    """ASCII histogram of LLM call latencies using plotext."""
    buckets = m.latency_buckets
    labels  = list(buckets.keys())
    values  = list(buckets.values())

    if sum(values) == 0:
        return Panel("No latency data", title="Latency Distribution")

    # plotext renders to string — capture it
    plt.clf()
    plt.theme("dark")
    plt.bar(labels, values, width=0.6)
    plt.title("LLM Call Latency Distribution")
    plt.xlabel("Bucket")
    plt.ylabel("Count")
    plt.plotsize(60, 12)
    chart_str = plt.build()

    return Panel(chart_str, title="[bold]Latency Histogram[/bold]", border_style="blue")


def render_trace_table(m: DashboardMetrics) -> Panel:
    """Per-trace breakdown table."""
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False, padding=(0, 1))
    table.add_column("#",         style="dim",    width=3)
    table.add_column("trace_id",  style="cyan",   width=12)
    table.add_column("cost",      justify="right", style="yellow")
    table.add_column("latency",   justify="right")
    table.add_column("steps",     justify="right")
    table.add_column("llm",       justify="right")
    table.add_column("in tok",    justify="right", style="blue")
    table.add_column("out tok",   justify="right", style="blue")
    table.add_column("question",  style="dim",     max_width=40)

    for i, t in enumerate(m.traces, 1):
        # Colour the cost: green if cheap, yellow if medium, red if expensive
        avg = m.total_cost_usd / m.total_traces if m.total_traces else 0
        cost_color = "green" if t.total_cost_usd < avg else "yellow"

        table.add_row(
            str(i),
            t.trace_id[:10] + "..",
            f"[{cost_color}]{fmt_cost(t.total_cost_usd)}[/{cost_color}]",
            fmt_ms(t.total_latency_ms),
            str(t.step_count),
            str(t.llm_calls),
            f"{t.input_tokens:,}",
            f"{t.output_tokens:,}",
            (t.question[:38] + "..") if len(t.question) > 40 else t.question,
        )

    return Panel(table, title="[bold]Per-Trace Breakdown[/bold]", border_style="cyan")


def render_slowest_spans(m: DashboardMetrics) -> Panel:
    """Top 5 slowest spans."""
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("span",       style="cyan")
    table.add_column("duration",   justify="right", style="red")
    table.add_column("trace",      style="dim")

    for s in m.slowest_spans[:5]:
        table.add_row(
            s.name,
            fmt_ms(s.duration_ms),
            s.trace_id[:10] + "..",
        )

    return Panel(table, title="[bold]Top 5 Slowest Spans[/bold]", border_style="red")


def render_cost_trend(m: DashboardMetrics) -> Panel:
    """Cumulative cost over traces as a sparkline."""
    if len(m.cost_over_time) < 2:
        return Panel("Need 2+ traces for trend", title="Cost Trend")

    costs = [c for _, c in m.cost_over_time]
    xs    = list(range(1, len(costs) + 1))

    plt.clf()
    plt.theme("dark")
    plt.plot(xs, costs, marker="braille")
    plt.title("Cumulative Cost ($)")
    plt.xlabel("Trace #")
    plt.ylabel("Total $")
    plt.plotsize(50, 10)
    chart_str = plt.build()

    return Panel(chart_str, title="[bold]Cumulative Cost Trend[/bold]", border_style="magenta")


# ── Full render ───────────────────────────────────────────────────────────────
def render(m: DashboardMetrics, refresh_in: Optional[int] = None):
    """Clear screen and render all panels."""
    os.system("cls" if os.name == "nt" else "clear")

    timestamp = time.strftime("%H:%M:%S")
    mode_note = f"  [dim]auto-refresh every {refresh_in}s — Ctrl+C to stop[/dim]" if refresh_in else ""
    console.print(
        f"\n[bold yellow]AI Engineering — Day 2 Dashboard[/bold yellow]"
        f"  [dim]{timestamp}[/dim]{mode_note}\n"
    )

    console.print(render_header(m))
    console.print()
    console.print(Columns([render_model_breakdown(m), render_cost_trend(m)]))
    console.print()
    console.print(render_latency_histogram(m))
    console.print()
    console.print(render_trace_table(m))
    console.print()
    console.print(render_slowest_spans(m))
    console.print()
    console.print(f"[dim]Reading from: {TRACE_FILE}[/dim]")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cost & latency dashboard")
    parser.add_argument(
        "--live", type=int, metavar="SECONDS",
        help="Refresh interval in seconds (omit for snapshot mode)",
    )
    args = parser.parse_args()

    if args.live:
        console.print(f"[bold]Live mode[/bold] — refreshing every {args.live}s. Ctrl+C to stop.\n")
        try:
            while True:
                m = compute_dashboard()
                if m:
                    render(m, refresh_in=args.live)
                else:
                    console.print("[yellow]Waiting for traces... run agent.py in another window.[/yellow]")
                time.sleep(args.live)
        except KeyboardInterrupt:
            console.print("\n[dim]Dashboard stopped.[/dim]")
    else:
        m = compute_dashboard()
        if not m:
            console.print(
                "[yellow]No traces found.[/yellow]\n"
                "Run [cyan]python ../day1-otel-tracing/agent.py[/cyan] first."
            )
            sys.exit(1)
        render(m)