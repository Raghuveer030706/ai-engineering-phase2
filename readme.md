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

# Day 3 — Async Agent Core

## What you built
The Day 1 ReAct agent rewritten with `asyncio`.
Same behaviour. Same tools. Same traces. Same answers.
Different architecture — now async/await native.

## What changed vs Day 1

| Thing | Day 1 | Day 3 |
|-------|-------|-------|
| Anthropic client | `anthropic.Anthropic` | `anthropic.AsyncAnthropic` |
| `run()` | `def run()` | `async def run()` |
| `_run_step()` | `def _run_step()` | `async def _run_step()` |
| LLM call | `self.llm.ask(...)` | `await self.llm.ask(...)` |
| Entry point | `agent.run(q)` | `asyncio.run(agent.run(q))` |
| Tool calls | sync | sync (still — tools are CPU, not I/O) |
| Tracing | identical | identical |
| Wall-clock time | ~2.7s | ~2.7s (same — steps still sequential) |

## What did NOT change
- Tool implementations (calculator, word_counter, text_reverser)
- System prompt
- Parser logic
- Span names and attributes
- Cost tracking

## Why same speed today
Steps are still sequential:
```
await step1  →  await step2  →  await step3
```
The event loop CAN interleave work during each await,
but we're not giving it anything else to run yet.

Day 4 changes this to:
```
asyncio.gather(question1, question2, question3)  # all at once
```
## Key concepts proved today
1. `async def` + `await` = pauseable functions, not blocking ones
2. `asyncio.AsyncAnthropic` is the only client change needed
3. Tools stay synchronous — async is for I/O waits, not CPU work
4. Wall-clock time is identical to Day 1 — async alone doesn't speed things up
5. `asyncio.run()` is the entry point for all async programs

## Tomorrow — Day 4
`asyncio.gather()` — run multiple agent questions simultaneously.
The event loop will interleave the LLM waits across all questions.
Expected speedup: 2-3x on a batch of independent questions.
The benchmark numbers from today are the baseline to beat.

# Day 4 — Parallel Sub-Tasks with asyncio.gather()

## What you'll prove today
Running 2 questions sequentially = 5606ms (Day 3 baseline).
Running 2 questions in parallel  = ~2800ms (Day 4 target).
Speedup: ~2×. The event loop interleaves LLM waits instead of stacking them.

## The one new concept: asyncio.gather()

```python
# Sequential — total time = sum of all latencies
result1 = await agent.run(q1)   # wait 2800ms
result2 = await agent.run(q2)   # wait 2800ms
# Total: 5600ms

# Parallel — total time = max of all latencies
results = await asyncio.gather(
    agent.run(q1),   # both start immediately
    agent.run(q2),   # event loop interleaves their waits
)
# Total: ~2800ms
```

## When parallel helps vs when it doesn't

| Situation | Use parallel? | Why |
|-----------|--------------|-----|
| Multiple independent questions | ✓ Yes | No dependency between them |
| Supervisor → multiple specialists | ✓ Yes | Each specialist is independent |
| Fetch context from multiple sources | ✓ Yes | Pure I/O, no dependency |
| Steps within one question | ✗ No | Step 2 needs step 1's observation |
| Shared mutable state | ✗ No | Race conditions |

## Files
```
day4-parallel-agents/
├── parallel_agent.py   ← sequential vs parallel benchmark + specialist pattern
└── requirements.txt    ← no new packages
```

## Setup
No new packages needed.

## Run it

The script runs three rounds automatically:
- Round 1: Sequential (fresh Day 3 baseline)
- Round 2: Parallel with asyncio.gather()
- Round 3: Supervisor + 3 specialists pattern

**Expected output at the end:**

```
Wall-clock (2 questions)   5400ms    2800ms    -2600ms
Speedup factor             1.00×     ~2.00×    +1.00×
Specialist pattern (3)     ~8400ms   ~2900ms   ~5500ms saved
```

## What you proved today
1. asyncio.gather() runs independent coroutines concurrently
2. Speedup ≈ N× for N independent questions (bounded by slowest)
3. The supervisor → specialists pattern is the real production use case
4. Steps within a single question cannot be parallelised (data dependency)
5. Shared AsyncReActAgent instance is safe — no mutable state between runs

# Day 5 — Streaming Responses

## What you'll build
A streaming LLM client and agent where tokens appear in the terminal
as they are generated. First token arrives in under 500ms.
Plus a FastAPI endpoint that streams tokens over HTTP using Server-Sent Events.

## The new metric: Time To First Token (TTFT)

Non-streaming: user waits 2 seconds, sees nothing, then gets everything at once.
Streaming:     user sees first token in ~300ms, reads as the rest arrives.

Same total latency. Completely different user experience.

```
Non-streaming: [=====2000ms of silence=====] DUMP
Streaming:     [~300ms] token token token token token token ...
```

TTFT is the metric that determines whether an AI product feels responsive.

## Files
```
day5-streaming/
├── streaming_client.py   ← async streaming wrapper, TTFT measurement
├── streaming_agent.py    ← streaming ReAct agent + FastAPI /ask/stream
└── requirements.txt      ← fastapi + uvicorn if not already installed
```

## Setup

FastAPI and uvicorn may already be in your env from Phase 1. Check:
```powershell
pip show fastapi uvicorn
```

If not installed:
```powershell
pip install fastapi uvicorn --break-system-packages
```

## Run it — in order

### Step 1: Streaming client demo
```powershell
conda activate ai-journey
cd <your-repo>\phase-A-observability\day5-streaming
python streaming_client.py
```
Watch tokens appear character by character in the terminal.
Note the TTFT printed at the end — should be under 500ms.

### Step 2: Streaming agent demo
```powershell
python streaming_agent.py
```
Each ReAct step streams its output live.
Summary table at the end shows avg TTFT vs total latency.

### Step 3: FastAPI streaming server
```powershell
python streaming_agent.py --serve
```
Server starts at http://localhost:8001

Test the streaming endpoint (new PowerShell window):
```powershell
$body = '{"question": "What is 12 * 8?"}'
Invoke-WebRequest -Uri http://localhost:8001/ask/stream `
  -Method POST `
  -ContentType "application/json" `
  -Body $body | Select-Object -ExpandProperty Content
```

Or open http://localhost:8001/docs in browser → try /ask/stream interactively.

## What to observe

**TTFT vs total latency ratio:**
A healthy streaming endpoint has TTFT < 20% of total latency.
If TTFT is 400ms and total is 2000ms, ratio is 20% — good.
If TTFT is 1800ms and total is 2000ms, something is buffering your stream.

**Server-Sent Events format:**
Each token arrives as:
```
data: {"type": "token", "text": "Hello"}

data: {"type": "token", "text": " world"}

data: {"type": "done", "total_tokens": 42}
```
This is the same format used by OpenAI, Anthropic's public API, and Gemini.

## Phase A Complete

| Day | Built | Key metric |
|-----|-------|-----------|
| 1 | OTel tracing | trace_id on every LLM call |
| 2 | Cost dashboard | $0.004598 across 6 traces |
| 3 | Async agent | sequential async baseline: 5606ms |
| 4 | Parallel agents | asyncio.gather() → 2942ms (1.47×) |
| 5 | Streaming | TTFT < 500ms |

# Phase B — Retrieval Mastery

**Goal:** Close the context recall gap from Phase 1 (0.583) and beat the overall RAGAS capstone score of 0.827.

**Corpus:** Attention Is All You Need (Vaswani et al., 2017)  
**Embeddings:** all-MiniLM-L6-v2 (local, no API cost)  
**LLM:** claude-haiku-4-5-20251001  
**Eval:** RAGAS 0.4.3, 6 fixed questions, same set every day

---

## Running Scoreboard

| Day | Strategy | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall | Overall |
|-----|----------|-------------|-------------|---------------|------------|---------|
| Phase 1 capstone | Hybrid RAG | — | — | — | 0.583 | 0.827 |
| B-D1 baseline | Raw chunks | 0.9208 | 0.9736 | 0.6972 | 1.0000 | 0.8979 |
| B-D1 contextual | Claude context prepend | 0.9552 | 0.9755 | 0.8056 | 1.0000 | **0.9341** |
| B-D2 parent-doc | Small search → large return | 0.9554 | 0.8025 | 0.6250 | 0.8333 | 0.8041 |

---

## ChromaDB Collections

All collections share one `chroma_db/` folder under `day1-contextual-retrieval/`:

| Collection | Day | Purpose |
|------------|-----|---------|
| `baseline_attention` | D1 | Raw 400-token chunks, no context |
| `contextual_attention` | D1 | Claude-summary prepended chunks |
| `small_chunks_attention` | D2 | 128-token search index |
| `parent_chunks_attention` | D2 | 512-token retrieval store (ID fetch only) |

---

## Day 1 — Contextual Retrieval

**Folder:** `day1-contextual-retrieval/`

### The problem
Isolated chunks lose referential context. A chunk saying *"it increased by 20%"* means nothing without knowing what *"it"* refers to. Cosine search finds the right location but the embedding carries no document-level signal.

### The fix
Before indexing, ask Claude to write 1–2 sentences describing where each chunk sits in the document. Prepend that summary inside `<context>` tags. The embedding now carries both local meaning and document position.

```
<context>
This chunk is from Section 3.2 describing the encoder sublayer
connection and layer normalization step.
</context>

...raw chunk text...
```

### Files

| File | What it does |
|------|-------------|
| `setup_baseline.py` | Chunks PDF (400 tokens, 80 overlap), embeds with MiniLM, indexes into `baseline_attention` |
| `contextual_retrieval.py` | Generates Claude context per chunk, prepends it, re-embeds, indexes into `contextual_attention` |
| `eval.py` | RAGAS 0.4.3 on both collections, prints side-by-side comparison |

### Run order

```powershell
conda activate ai-journey
cd phase-B-retrieval\day1-contextual-retrieval

python setup_baseline.py
python contextual_retrieval.py
python eval.py
```

### Results

```
Metric                  Baseline   Contextual      Delta
────────────────────────────────────────────────────────
faithfulness              0.9208       0.9552   ▲ 0.0344
answer_relevancy          0.9736       0.9755   ▲ 0.0019
context_precision         0.6972       0.8056   ▲ 0.1083
context_recall            1.0000       1.0000   ─ 0.0000
OVERALL (mean)            0.8979       0.9341   ▲ 0.0362
```

**Key win:** `context_precision` +0.1083. Claude-generated summaries make embeddings location-aware — retrieval returns more relevant chunks for the same query. Overall 0.9341 beats Phase 1 capstone (0.827) by +0.107.

### Gotchas
- `contextual_retrieval.py` makes one Claude call per chunk (~31 calls). `RATE_DELAY=0.3s` avoids bursts.
- RAGAS hit 429 rate limit at 19/24 calls on the contextual eval run — recovered and finished. `raise_exceptions=False` added to Day 2+ evals.
- `chroma_db/` lands inside `day1-contextual-retrieval/` — point all Day 2–5 scripts at this path explicitly.

---

## Day 2 — Parent-Document Retrieval

**Folder:** `day2-parent-document/`

### The idea
Index small chunks (128 tokens) for precise embedding search. Each small chunk's metadata stores a `parent_id` pointing to its 512-token parent. When a small chunk matches, fetch and return the parent instead — tighter search, richer context.

```
Query → embed → search small_chunks → get parent_id → fetch parent → return to LLM
```

### Why it seemed promising
Small chunks have tighter, more specific embeddings — better cosine match. But small chunks often cut off mid-sentence. Returning the parent gives the full surrounding passage.

### Files

| File | What it does |
|------|-------------|
| `parent_document_retrieval.py` | Builds `small_chunks_attention` (128-token, with embeddings) and `parent_chunks_attention` (512-token, ID-fetch only, no embeddings) |
| `eval_day2.py` | RAGAS comparison: Day 1 contextual vs Day 2 parent-document, running scoreboard |

### Run order

```powershell
conda activate ai-journey
cd phase-B-retrieval\day2-parent-document

python parent_document_retrieval.py
python eval_day2.py
```

### Results

```
Metric                  Day1-Ctx     Day2-PDR        Delta
──────────────────────────────────────────────────────────
faithfulness              0.9539       0.9554   ▲ 0.0015
answer_relevancy          0.9701       0.8025   ▼ 0.1676
context_precision         0.8056       0.6250   ▼ 0.1806
context_recall            1.0000       0.8333   ▼ 0.1667
OVERALL (mean)            0.9324       0.8041   ▼ 0.1283
```

### Why it regressed

512-token parents are too coarse for a dense 15-page paper. Each parent window crosses multiple concepts — noise goes up, precision drops. One question's answer also split across two parent boundaries, so the second half wasn't retrieved in top-5 (recall drop). Day 1 contextual at 400 tokens with Claude summaries was better calibrated for this corpus.

**Faithfulness held (+0.0015)** — answers stayed grounded in whatever was retrieved, even when retrieval was noisy.

### Gotchas
- `ChromaDB InternalError: Nothing found on disk` — triggered when parent collection was indexed with embeddings. Fixed by calling `parent_col.add()` without embeddings. Collections used only for ID-based `get()` must not have embeddings.
- `CHROMA_PATH` must point to `../day1-contextual-retrieval/chroma_db` — not a local `../chroma_db`.

---

## Setup Notes (apply to all days)

```powershell
# Always run from the day subfolder
cd phase-B-retrieval\dayN-folder-name
python script.py

# PDF lives at repo root
ai-engineering-phase2\data\attention-is-all-you-need.pdf

# PDF path in each script
PDF_PATH = Path("../../data/attention-is-all-you-need.pdf")

# Shared chroma_db path in Day 2–5 scripts
CHROMA_PATH = Path("../day1-contextual-retrieval/chroma_db")
```

**RAGAS 0.4.3 reminders:**
- Always pass `llm=` and `embeddings=` explicitly — OpenAI is the default
- `EvaluationResult` is not a dict — use `to_pandas().select_dtypes(include="number").mean()`
- Add `raise_exceptions=False` to survive rate-limit blips mid-eval

# Phase B — Retrieval Mastery

**Goal:** Close the context recall gap from Phase 1 (0.583) and beat the overall RAGAS capstone score of 0.827.

**Corpus:** Attention Is All You Need (Vaswani et al., 2017)  
**Embeddings:** all-MiniLM-L6-v2 (local, no API cost)  
**LLM:** claude-haiku-4-5-20251001  
**Eval:** RAGAS 0.4.3, 6 fixed questions, same set every day

---

## Running Scoreboard

| Day | Strategy | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall | Overall |
|-----|----------|-------------|-------------|---------------|------------|---------|
| Phase 1 capstone | Hybrid RAG | — | — | — | 0.583 | 0.827 |
| B-D1 baseline | Raw chunks | 0.9208 | 0.9736 | 0.6972 | 1.0000 | 0.8979 |
| B-D1 contextual | Claude context prepend | 0.9552 | 0.9755 | 0.8056 | 1.0000 | **0.9341** |
| B-D2 parent-doc | Small search → large return | 0.9554 | 0.8025 | 0.6250 | 0.8333 | 0.8041 |
| B-D3 late chunking | Full-doc token attention | 1.0000 ★ | 0.9659 ★ | 0.7389 | 1.0000 | 0.9262 |

---

## ChromaDB Collections

All collections share one `chroma_db/` folder under `day1-contextual-retrieval/`:

| Collection | Day | Purpose |
|------------|-----|---------|
| `baseline_attention` | D1 | Raw 400-token chunks, no context |
| `contextual_attention` | D1 | Claude-summary prepended chunks |
| `small_chunks_attention` | D2 | 128-token search index |
| `parent_chunks_attention` | D2 | 512-token retrieval store (ID fetch only) |
| `late_chunking_attention` | D3 | Jina token-span embeddings, full-doc context |

---

## Day 1 — Contextual Retrieval

**Folder:** `day1-contextual-retrieval/`

### The problem
Isolated chunks lose referential context. A chunk saying *"it increased by 20%"* means nothing without knowing what *"it"* refers to. Cosine search finds the right location but the embedding carries no document-level signal.

### The fix
Before indexing, ask Claude to write 1–2 sentences describing where each chunk sits in the document. Prepend that summary inside `<context>` tags. The embedding now carries both local meaning and document position.

```
<context>
This chunk is from Section 3.2 describing the encoder sublayer
connection and layer normalization step.
</context>

...raw chunk text...
```

### Files

| File | What it does |
|------|-------------|
| `setup_baseline.py` | Chunks PDF (400 tokens, 80 overlap), embeds with MiniLM, indexes into `baseline_attention` |
| `contextual_retrieval.py` | Generates Claude context per chunk, prepends it, re-embeds, indexes into `contextual_attention` |
| `eval.py` | RAGAS 0.4.3 on both collections, prints side-by-side comparison |

### Run order

```powershell
conda activate ai-journey
cd phase-B-retrieval\day1-contextual-retrieval

python setup_baseline.py
python contextual_retrieval.py
python eval.py
```

### Results

```
Metric                  Baseline   Contextual      Delta
────────────────────────────────────────────────────────
faithfulness              0.9208       0.9552   ▲ 0.0344
answer_relevancy          0.9736       0.9755   ▲ 0.0019
context_precision         0.6972       0.8056   ▲ 0.1083
context_recall            1.0000       1.0000   ─ 0.0000
OVERALL (mean)            0.8979       0.9341   ▲ 0.0362
```

**Key win:** `context_precision` +0.1083. Claude-generated summaries make embeddings location-aware — retrieval returns more relevant chunks for the same query. Overall 0.9341 beats Phase 1 capstone (0.827) by +0.107.

### Gotchas
- `contextual_retrieval.py` makes one Claude call per chunk (~31 calls). `RATE_DELAY=0.3s` avoids bursts.
- RAGAS hit 429 rate limit at 19/24 calls on the contextual eval run — recovered and finished. `raise_exceptions=False` added to Day 2+ evals.
- `chroma_db/` lands inside `day1-contextual-retrieval/` — point all Day 2–5 scripts at this path explicitly.

---

## Day 2 — Parent-Document Retrieval

**Folder:** `day2-parent-document/`

### The idea
Index small chunks (128 tokens) for precise embedding search. Each small chunk's metadata stores a `parent_id` pointing to its 512-token parent. When a small chunk matches, fetch and return the parent instead — tighter search, richer context.

```
Query → embed → search small_chunks → get parent_id → fetch parent → return to LLM
```

### Why it seemed promising
Small chunks have tighter, more specific embeddings — better cosine match. But small chunks often cut off mid-sentence. Returning the parent gives the full surrounding passage.

### Files

| File | What it does |
|------|-------------|
| `parent_document_retrieval.py` | Builds `small_chunks_attention` (128-token, with embeddings) and `parent_chunks_attention` (512-token, ID-fetch only, no embeddings) |
| `eval_day2.py` | RAGAS comparison: Day 1 contextual vs Day 2 parent-document, running scoreboard |

### Run order

```powershell
conda activate ai-journey
cd phase-B-retrieval\day2-parent-document

python parent_document_retrieval.py
python eval_day2.py
```

### Results

```
Metric                  Day1-Ctx     Day2-PDR        Delta
──────────────────────────────────────────────────────────
faithfulness              0.9539       0.9554   ▲ 0.0015
answer_relevancy          0.9701       0.8025   ▼ 0.1676
context_precision         0.8056       0.6250   ▼ 0.1806
context_recall            1.0000       0.8333   ▼ 0.1667
OVERALL (mean)            0.9324       0.8041   ▼ 0.1283
```

### Why it regressed

512-token parents are too coarse for a dense 15-page paper. Each parent window crosses multiple concepts — noise goes up, precision drops. One question's answer also split across two parent boundaries, so the second half wasn't retrieved in top-5 (recall drop). Day 1 contextual at 400 tokens with Claude summaries was better calibrated for this corpus.

**Faithfulness held (+0.0015)** — answers stayed grounded in whatever was retrieved, even when retrieval was noisy.

### Gotchas
- `ChromaDB InternalError: Nothing found on disk` — triggered when parent collection was indexed with embeddings. Fixed by calling `parent_col.add()` without embeddings. Collections used only for ID-based `get()` must not have embeddings.
- `CHROMA_PATH` must point to `../day1-contextual-retrieval/chroma_db` — not a local `../chroma_db`.

---

## Day 3 — Late Chunking

**Folder:** `day3-late-chunking/`

### The idea
Standard chunking splits text first, then embeds each chunk independently — cross-sentence context is lost at boundaries. Late chunking flips the order: embed the *full document* in one transformer forward pass, then slice the resulting token embeddings into chunks post-hoc. Every chunk's embedding carries the full document's context baked in.

```
Standard:  split text → embed each chunk independently   (context lost at boundaries)
Late:      embed full doc → slice token embeddings        (context preserved everywhere)
```

### Model
`jinaai/jina-embeddings-v2-base-en` — 8192 token context window vs 256 for MiniLM. Required for late chunking. First run downloads ~500MB (cached after).

```powershell
pip install transformers==4.40.0 --break-system-packages   # pin before running
pip install einops --break-system-packages
```

### Files

| File | What it does |
|------|-------------|
| `late_chunking.py` | Tokenizes full PDF, runs one Jina forward pass, mean-pools 256-token spans into embeddings, indexes into `late_chunking_attention` |
| `eval_day3.py` | Three-way RAGAS comparison: D1 contextual vs D2 parent-doc vs D3 late chunking. Loads both MiniLM (D1/D2) and Jina (D3) — each collection queried with its own model |

### Run order

```powershell
conda activate ai-journey
cd phase-B-retrieval\day3-late-chunking

python late_chunking.py
python eval_day3.py
```

### Results

```
Metric                  D1-Ctx     D2-PDR    D3-Late
─────────────────────────────────────────────────────
faithfulness            0.9537     0.9375    1.0000 ★
answer_relevancy        0.9647     0.8045    0.9659 ★
context_precision       0.8056 ★   0.6250    0.7389
context_recall          1.0000 ★   0.8333    1.0000 ★
OVERALL (mean)          0.9310 ★   0.8001    0.9262
```

### Reading the results

Late chunking wins on faithfulness (1.0) and answer relevancy — full-document token attention produces complete, clean chunk boundaries with no dangling references. Claude has unambiguous material to answer from.

Day 1 contextual still leads on precision (0.8056 vs 0.7389) — Claude-generated summaries add an explicit semantic label that cosine search locks onto. Late chunking's context is implicit in the math; better grounding, but diffuse signal for retrieval.

| Strategy | Strength | Weakness |
|----------|----------|----------|
| D1 Contextual | Precision via explicit LLM label | Costs Claude calls per chunk |
| D2 Parent-doc | Full passage returned | Too wide — noise kills precision |
| D3 Late chunking | Faithfulness=1.0, no LLM cost | Precision below D1 |

Day 4 multi-vector stores both a summary embedding and a full-text embedding per chunk — combining D1's precision signal with D3's faithfulness.

### Gotchas
- `transformers.onnx` removed in newer transformers — pin `transformers==4.40.0` before loading Jina
- Each collection must be queried with the model it was indexed with — MiniLM for D1/D2, Jina for D3
- Jina processes document in segments if total tokens exceed 8000 — still far better than MiniLM's 256-token limit per chunk

---

## Setup Notes (apply to all days)

```powershell
# Always run from the day subfolder
cd phase-B-retrieval\dayN-folder-name
python script.py

# PDF lives at repo root
ai-engineering-phase2\data\attention-is-all-you-need.pdf

# PDF path in each script
PDF_PATH = Path("../../data/attention-is-all-you-need.pdf")

# Shared chroma_db path in Day 2–5 scripts
CHROMA_PATH = Path("../day1-contextual-retrieval/chroma_db")
```

**RAGAS 0.4.3 reminders:**
- Always pass `llm=` and `embeddings=` explicitly — OpenAI is the default
- `EvaluationResult` is not a dict — use `to_pandas().select_dtypes(include="number").mean()`
- Add `raise_exceptions=False` to survive rate-limit blips mid-eval

# Phase B — Retrieval Mastery

**Goal:** Close the context recall gap from Phase 1 (0.583) and beat the overall RAGAS capstone score of 0.827.

**Corpus:** Attention Is All You Need (Vaswani et al., 2017)  
**Embeddings:** all-MiniLM-L6-v2 (local, no API cost)  
**LLM:** claude-haiku-4-5-20251001  
**Eval:** RAGAS 0.4.3, 6 fixed questions, same set every day

---

## Running Scoreboard

| Day | Strategy | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall | Overall |
|-----|----------|-------------|-------------|---------------|------------|---------|
| Phase 1 capstone | Hybrid RAG | — | — | — | 0.583 | 0.827 |
| B-D1 baseline | Raw chunks | 0.9208 | 0.9736 | 0.6972 | 1.0000 | 0.8979 |
| B-D1 contextual | Claude context prepend | 0.9552 | 0.9755 | 0.8056 | 1.0000 | **0.9341** |
| B-D2 parent-doc | Small search → large return | 0.9554 | 0.8025 | 0.6250 | 0.8333 | 0.8041 |
| B-D3 late chunking | Full-doc token attention | 1.0000 ★ | 0.9659 ★ | 0.7389 | 1.0000 | 0.9262 |
| B-D4 multi-vector | Summary + fulltext union | 0.9542 | 0.9646 | 0.7162 | 1.0000 | 0.9087 |

---

## ChromaDB Collections

All collections share one `chroma_db/` folder under `day1-contextual-retrieval/`:

| Collection | Day | Purpose |
|------------|-----|---------|
| `baseline_attention` | D1 | Raw 400-token chunks, no context |
| `contextual_attention` | D1 | Claude-summary prepended chunks |
| `small_chunks_attention` | D2 | 128-token search index |
| `parent_chunks_attention` | D2 | 512-token retrieval store (ID fetch only) |
| `late_chunking_attention` | D3 | Jina token-span embeddings, full-doc context |
| `summary_vectors_attention` | D4 | MiniLM embeddings of Claude summaries |
| `fulltext_vectors_attention` | D4 | Jina embeddings of raw chunks |

---

## Day 1 — Contextual Retrieval

**Folder:** `day1-contextual-retrieval/`

### The problem
Isolated chunks lose referential context. A chunk saying *"it increased by 20%"* means nothing without knowing what *"it"* refers to. Cosine search finds the right location but the embedding carries no document-level signal.

### The fix
Before indexing, ask Claude to write 1–2 sentences describing where each chunk sits in the document. Prepend that summary inside `<context>` tags. The embedding now carries both local meaning and document position.

```
<context>
This chunk is from Section 3.2 describing the encoder sublayer
connection and layer normalization step.
</context>

...raw chunk text...
```

### Files

| File | What it does |
|------|-------------|
| `setup_baseline.py` | Chunks PDF (400 tokens, 80 overlap), embeds with MiniLM, indexes into `baseline_attention` |
| `contextual_retrieval.py` | Generates Claude context per chunk, prepends it, re-embeds, indexes into `contextual_attention` |
| `eval.py` | RAGAS 0.4.3 on both collections, prints side-by-side comparison |

### Run order

```powershell
conda activate ai-journey
cd phase-B-retrieval\day1-contextual-retrieval

python setup_baseline.py
python contextual_retrieval.py
python eval.py
```

### Results

```
Metric                  Baseline   Contextual      Delta
────────────────────────────────────────────────────────
faithfulness              0.9208       0.9552   ▲ 0.0344
answer_relevancy          0.9736       0.9755   ▲ 0.0019
context_precision         0.6972       0.8056   ▲ 0.1083
context_recall            1.0000       1.0000   ─ 0.0000
OVERALL (mean)            0.8979       0.9341   ▲ 0.0362
```

**Key win:** `context_precision` +0.1083. Claude-generated summaries make embeddings location-aware — retrieval returns more relevant chunks for the same query. Overall 0.9341 beats Phase 1 capstone (0.827) by +0.107.

### Gotchas
- `contextual_retrieval.py` makes one Claude call per chunk (~31 calls). `RATE_DELAY=0.3s` avoids bursts.
- RAGAS hit 429 rate limit at 19/24 calls on the contextual eval run — recovered and finished. `raise_exceptions=False` added to Day 2+ evals.
- `chroma_db/` lands inside `day1-contextual-retrieval/` — point all Day 2–5 scripts at this path explicitly.

---

## Day 2 — Parent-Document Retrieval

**Folder:** `day2-parent-document/`

### The idea
Index small chunks (128 tokens) for precise embedding search. Each small chunk's metadata stores a `parent_id` pointing to its 512-token parent. When a small chunk matches, fetch and return the parent instead — tighter search, richer context.

```
Query → embed → search small_chunks → get parent_id → fetch parent → return to LLM
```

### Why it seemed promising
Small chunks have tighter, more specific embeddings — better cosine match. But small chunks often cut off mid-sentence. Returning the parent gives the full surrounding passage.

### Files

| File | What it does |
|------|-------------|
| `parent_document_retrieval.py` | Builds `small_chunks_attention` (128-token, with embeddings) and `parent_chunks_attention` (512-token, ID-fetch only, no embeddings) |
| `eval_day2.py` | RAGAS comparison: Day 1 contextual vs Day 2 parent-document, running scoreboard |

### Run order

```powershell
conda activate ai-journey
cd phase-B-retrieval\day2-parent-document

python parent_document_retrieval.py
python eval_day2.py
```

### Results

```
Metric                  Day1-Ctx     Day2-PDR        Delta
──────────────────────────────────────────────────────────
faithfulness              0.9539       0.9554   ▲ 0.0015
answer_relevancy          0.9701       0.8025   ▼ 0.1676
context_precision         0.8056       0.6250   ▼ 0.1806
context_recall            1.0000       0.8333   ▼ 0.1667
OVERALL (mean)            0.9324       0.8041   ▼ 0.1283
```

### Why it regressed

512-token parents are too coarse for a dense 15-page paper. Each parent window crosses multiple concepts — noise goes up, precision drops. One question's answer also split across two parent boundaries, so the second half wasn't retrieved in top-5 (recall drop). Day 1 contextual at 400 tokens with Claude summaries was better calibrated for this corpus.

**Faithfulness held (+0.0015)** — answers stayed grounded in whatever was retrieved, even when retrieval was noisy.

### Gotchas
- `ChromaDB InternalError: Nothing found on disk` — triggered when parent collection was indexed with embeddings. Fixed by calling `parent_col.add()` without embeddings. Collections used only for ID-based `get()` must not have embeddings.
- `CHROMA_PATH` must point to `../day1-contextual-retrieval/chroma_db` — not a local `../chroma_db`.

---

## Day 3 — Late Chunking

**Folder:** `day3-late-chunking/`

### The idea
Standard chunking splits text first, then embeds each chunk independently — cross-sentence context is lost at boundaries. Late chunking flips the order: embed the *full document* in one transformer forward pass, then slice the resulting token embeddings into chunks post-hoc. Every chunk's embedding carries the full document's context baked in.

```
Standard:  split text → embed each chunk independently   (context lost at boundaries)
Late:      embed full doc → slice token embeddings        (context preserved everywhere)
```

### Model
`jinaai/jina-embeddings-v2-base-en` — 8192 token context window vs 256 for MiniLM. Required for late chunking. First run downloads ~500MB (cached after).

```powershell
pip install transformers==4.40.0 --break-system-packages   # pin before running
pip install einops --break-system-packages
```

### Files

| File | What it does |
|------|-------------|
| `late_chunking.py` | Tokenizes full PDF, runs one Jina forward pass, mean-pools 256-token spans into embeddings, indexes into `late_chunking_attention` |
| `eval_day3.py` | Three-way RAGAS comparison: D1 contextual vs D2 parent-doc vs D3 late chunking. Loads both MiniLM (D1/D2) and Jina (D3) — each collection queried with its own model |

### Run order

```powershell
conda activate ai-journey
cd phase-B-retrieval\day3-late-chunking

python late_chunking.py
python eval_day3.py
```

### Results

```
Metric                  D1-Ctx     D2-PDR    D3-Late
─────────────────────────────────────────────────────
faithfulness            0.9537     0.9375    1.0000 ★
answer_relevancy        0.9647     0.8045    0.9659 ★
context_precision       0.8056 ★   0.6250    0.7389
context_recall          1.0000 ★   0.8333    1.0000 ★
OVERALL (mean)          0.9310 ★   0.8001    0.9262
```

### Reading the results

Late chunking wins on faithfulness (1.0) and answer relevancy — full-document token attention produces complete, clean chunk boundaries with no dangling references. Claude has unambiguous material to answer from.

Day 1 contextual still leads on precision (0.8056 vs 0.7389) — Claude-generated summaries add an explicit semantic label that cosine search locks onto. Late chunking's context is implicit in the math; better grounding, but diffuse signal for retrieval.

| Strategy | Strength | Weakness |
|----------|----------|----------|
| D1 Contextual | Precision via explicit LLM label | Costs Claude calls per chunk |
| D2 Parent-doc | Full passage returned | Too wide — noise kills precision |
| D3 Late chunking | Faithfulness=1.0, no LLM cost | Precision below D1 |

Day 4 multi-vector stores both a summary embedding and a full-text embedding per chunk — combining D1's precision signal with D3's faithfulness.

### Gotchas
- `transformers.onnx` removed in newer transformers — pin `transformers==4.40.0` before loading Jina
- Each collection must be queried with the model it was indexed with — MiniLM for D1/D2, Jina for D3
- Jina processes document in segments if total tokens exceed 8000 — still far better than MiniLM's 256-token limit per chunk

---

## Day 4 — Multi-Vector Retrieval

**Folder:** `day4-multi-vector/`

### The idea
One chunk, two embeddings stored in two separate ChromaDB collections. Query searches both, deduplicates by chunk index, returns the union up to top-K.

```
Query
  ├── MiniLM(question) → search summary_vectors_attention  → chunk_ids
  ├── Jina(question)   → search fulltext_vectors_attention → chunk_ids
  └── deduplicate → fetch raw chunks from fulltext col → return top-K
```

| Collection | Embedding model | What's stored |
|------------|----------------|---------------|
| `summary_vectors_attention` | MiniLM 384-dim | Claude-generated summaries |
| `fulltext_vectors_attention` | Jina 768-dim | Raw chunk text |

### Hypothesis
D1 contextual led on precision via explicit Claude summary labels. D3 late chunking led on faithfulness via Jina's full-doc token attention. Multi-vector was designed to get both — summary vectors for precise retrieval, fulltext vectors for faithful answers.

### Files

| File | What it does |
|------|-------------|
| `multi_vector_retrieval.py` | Generates Claude summary per chunk, embeds summary with MiniLM + raw chunk with Jina, indexes into two separate collections |
| `eval_day4.py` | Four-way RAGAS comparison D1/D2/D3/D4, D4 vs D1 delta breakdown |

### Run order

```powershell
conda activate ai-journey
cd phase-B-retrieval\day4-multi-vector

python multi_vector_retrieval.py
Start-Sleep -Seconds 3
python eval_day4.py
```

### Results

```
Metric                  D1-Ctx     D2-PDR    D3-Late      D4-MV
────────────────────────────────────────────────────────────────
faithfulness            0.9148     0.9024    1.0000 ★    0.9542
answer_relevancy        0.9717 ★   0.8064       nan     0.9646
context_precision       0.8056 ★   0.6250       nan     0.7162
context_recall          1.0000 ★   0.8333    1.0000 ★   1.0000 ★
OVERALL (mean)          0.9230 ★   0.7918       nan     0.9087
```

> **Note on D3 nan:** Rate limit hit mid-eval on the four-way run. D3 authoritative scores are from its dedicated Day 3 eval: `answer_relevancy: 0.9659`, `context_precision: 0.7389`, `overall: 0.9262`.

### Why the hypothesis didn't hold

Union retrieval adds chunks from both summary search and Jina fulltext search. When the two searches return different chunks, top-K fills with more candidates — but the extra Jina candidates are less precise than what summary search alone returns. More surface area, more noise. Context precision dropped from 0.8056 to 0.7162 (-0.0894), which wasn't compensated by the faithfulness gain (+0.0394).

**The pattern across all four days:**

| Strategy | Strength | Weakness |
|----------|----------|----------|
| D1 Contextual | Precision via explicit LLM label | Costs Claude calls per chunk |
| D2 Parent-doc | Full passage width | Too wide — noise kills precision |
| D3 Late chunking | Faithfulness = 1.0, no LLM cost | Precision below D1 |
| D4 Multi-vector | Faithfulness improved | Union adds retrieval noise |

**D1 contextual wins the phase.** Claude-generated chunk summaries are the single most effective intervention on this corpus.

### Gotchas
- `Start-Sleep -Seconds 3` between index build and eval — ChromaDB HNSW index needs time to flush before querying
- D3 and D4 both use Jina (768-dim). D1 and D2 use MiniLM (384-dim). Never cross-query a collection with the wrong model
- Four-way eval makes ~96 RAGAS LLM calls — rate limit blips produce `nan`. Use `raise_exceptions=False` and re-run dedicated evals for affected strategies if needed

---

## Setup Notes (apply to all days)

```powershell
# Always run from the day subfolder
cd phase-B-retrieval\dayN-folder-name
python script.py

# PDF lives at repo root
ai-engineering-phase2\data\attention-is-all-you-need.pdf

# PDF path in each script
PDF_PATH = Path("../../data/attention-is-all-you-need.pdf")

# Shared chroma_db path in Day 2–5 scripts
CHROMA_PATH = Path("../day1-contextual-retrieval/chroma_db")
```

**RAGAS 0.4.3 reminders:**
- Always pass `llm=` and `embeddings=` explicitly — OpenAI is the default
- `EvaluationResult` is not a dict — use `to_pandas().select_dtypes(include="number").mean()`
- Add `raise_exceptions=False` to survive rate-limit blips mid-eval

## Day 5 — A/B Retrieval Harness

**Folder:** `day5-ab-harness/`

### The deliverable
One script that runs any registered retrieval strategy against the
fixed eval set and outputs a ranked table automatically.
Default run costs zero API calls — loads cached scores from Days 1–4.

### Usage
python ab_eval.py                        # free, instant — ranked table
python ab_eval.py --new my-strategy      # score only new strategies
python ab_eval.py --rerun d1-contextual  # force re-score one strategy
python ab_eval.py --rerun all            # full re-run (expensive)

### Adding a new strategy
1. Build a ChromaDB collection for it
2. Write a retrieve_fn(question) -> list[str] closure
3. Add it to build_strategy_registry()
4. Run: python ab_eval.py --new your-strategy-key

### Phase B Final Results
Winner: d1-contextual (0.9341) — +0.1071 vs Phase 1 capstone (0.827)