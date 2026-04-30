# Day 1 — OpenTelemetry Tracing

## What you'll build
A tracing layer that wraps every LLM call and agent step with an
OpenTelemetry span. After today, every request has a `trace_id` you
can use to see exactly what happened, how long each step took,
and what it cost — down to the individual tool call.

## Why this matters
Before today: agent gives a wrong answer → no idea which step failed.
After today:  grep `trace.jsonl` for `trace_id` → see every decision.

This also fixes the **"estimated_cost_usd shows 0.0"** bug from Phase 1.
`response.usage` is now wired properly. Every call shows real cost.

## Files
```
day1-otel-tracing/
├── tracer.py           ← OTel setup + JSONL file exporter (foundation)
├── llm_client.py       ← Traced Anthropic wrapper
├── agent.py            ← ReAct agent with full span hierarchy
├── inspect_traces.py   ← CLI to query traces/trace.jsonl
├── requirements.txt    ← only NEW packages not already in ai-journey
└── traces/             ← created at runtime, gitignored
    └── trace.jsonl
```

## Setup

### 1 — Create the new repo on GitHub
Go to github.com → New repository → name: `ai-engineering-phase2`
Leave it empty (no README, no .gitignore).

### 2 — Set up local repo (PowerShell)
```powershell
# Create folder wherever you keep your projects
mkdir C:\Raghu\ai-engineering-phase2
cd C:\Raghu\ai-engineering-phase2

git init
git remote add origin https://github.com/Raghuveer030706/ai-engineering-phase2.git
```

### 3 — Copy files in
Copy the downloaded files into:
```
C:\Raghu\ai-engineering-phase2\
├── .gitignore
├── README.md
└── phase-A-observability\
    └── day1-otel-tracing\
        ├── tracer.py
        ├── llm_client.py
        ├── agent.py
        ├── inspect_traces.py
        ├── requirements.txt
        └── README.md
```

### 4 — Create .env at repo root
```
C:\Raghu\ai-engineering-phase2\.env
```
Contents:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 5 — Install the only new packages
```powershell
conda activate ai-journey
cd C:\Raghu\ai-engineering-phase2\phase-A-observability\day1-otel-tracing
pip install opentelemetry-api==1.27.0 opentelemetry-sdk==1.27.0 --break-system-packages
```
Everything else (anthropic, rich, python-dotenv) is already in your env.

---

## Run it — in order

### Step 1: Verify tracer (no API key needed)
```powershell
python tracer.py
```
**Expected:** Table with 3 spans, all status OK.
File `traces\trace.jsonl` is created.

### Step 2: Verify LLM client tracing
```powershell
python llm_client.py
```
**Expected:** 2 LLM calls. Real `cost_usd` printed (not 0.0). Trace summary.

### Step 3: Full traced agent run
```powershell
python agent.py
```
**Expected:** 2 agent runs with step panels. Trace viewer table at the end.

### Step 4: Inspect your traces
```powershell
# Summary of all runs
python inspect_traces.py

# Full span breakdown for a specific trace (paste first 8 chars of trace_id)
python inspect_traces.py --trace a3f9c2b1

# Total cost across all runs
python inspect_traces.py --cost

# Find slow spans (anything over 1 second)
python inspect_traces.py --slow 1000

# Most recent run only
python inspect_traces.py --last
```

---

## What to look for in trace.jsonl

After running `agent.py`, open `traces\trace.jsonl`.
Each line is one span. A multi-step run produces something like:

```
agent.run          trace_id=a3f9c2... duration_ms=4821
agent.step.1       trace_id=a3f9c2... duration_ms=1203
llm.call.step1     trace_id=a3f9c2... cost_usd=0.000312  tokens=234in/89out
agent.tool.calculator  trace_id=a3f9c2... duration_ms=1
agent.step.2       trace_id=a3f9c2... duration_ms=987
llm.call.step2     trace_id=a3f9c2... cost_usd=0.000198
agent.tool.word_counter  trace_id=a3f9c2... duration_ms=0
```

All spans share the same `trace_id` — that's the point.

---

## PowerShell tips for reading traces

```powershell
# Tail the last 5 spans
Get-Content traces\trace.jsonl -Tail 5

# Pretty-print a single span
Get-Content traces\trace.jsonl -Tail 1 | python -m json.tool

# Count total spans written
(Get-Content traces\trace.jsonl).Count
```
## What you proved today
1. `cost_usd` is real — from `response.usage`, never estimated or zero
2. Every LLM call has a `trace_id` — logs and traces now correlate
3. Span hierarchy shows the agent's full decision tree
4. You can answer: "which step was slowest?" and "what did this run cost?"

# Day 2 — Cost & Latency Dashboard

## What you'll build
A terminal dashboard that reads `traces/trace.jsonl` from Day 1
and renders live cost totals, latency histograms, per-trace breakdowns,
and a cumulative cost trend — all without making a single new API call.

## Why no new API calls?
The data is already there. Day 1 captured everything.
This is the point of structured traces — you can answer new questions
about your system without re-running it.

## Files
```
day2-cost-dashboard/
├── metrics.py              ← data layer: parse trace.jsonl into structs
├── dashboard.py            ← render panels (snapshot + live mode)
├── agent_with_metrics.py   ← run 3 more questions, then show dashboard
└── requirements.txt        ← only plotext is new
```

## What the dashboard shows
- **KPI bar** — total cost, avg cost/trace, 30-day projection at 100 q/day
- **Cost by model** — bar chart with percentage share
- **Latency histogram** — distribution of LLM call durations
- **Per-trace table** — cost, latency, steps, token counts, question preview
- **Top 5 slowest spans** — where time is actually going
- **Cumulative cost trend** — cost accumulation over your session

## Setup

### Install the one new package
pip install plotext==5.2.8 --break-system-packages

## Run it — in order

### Step 1: Verify metrics layer reads Day 1 traces correctly
```powershell
python metrics.py
```
**Expected:** prints total cost, traces, LLM calls, latency distribution.
Should match your Day 1 `inspect_traces.py --cost` output.

### Step 2: Snapshot dashboard (one-shot render)
```powershell
python dashboard.py
```
**Expected:** full dashboard rendered in terminal. All panels visible.

### Step 3: Add more trace data + show dashboard
```powershell
python agent_with_metrics.py
```
**Expected:** 3 new agent runs, then full dashboard with 7+ traces.
The latency histogram and cost trend will now have meaningful shapes.

### Step 4: Live mode (open in a second PowerShell pane)
```powershell
# Pane 1 — live dashboard
python dashboard.py --live 3

# Pane 2 — run more agent questions (reuse Day 1 agent)
cd ..\day1-otel-tracing
python agent.py
```
Watch the dashboard update as new traces arrive.

## What you proved today
1. Structured traces are queryable — no re-running needed to get new metrics
2. Cost projection: avg cost/trace × queries/day × 30 days = real budget number
3. Latency histogram shows where time goes (almost always the LLM call, not tools)
4. Live mode: dashboard + agent in split panes = production monitoring feel