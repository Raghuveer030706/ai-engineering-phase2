"""
Phase B Day 5 — A/B Retrieval Harness (cost-optimised)

Default run: zero LLM calls — loads cached scores from previous evals,
prints ranked table and Phase B final summary instantly.

Adding a new strategy:
  1. Write a retrieve_fn(question: str) -> list[str] closure
  2. Register it in build_strategy_registry()
  3. Run: python ab_eval.py --new your-strategy-key

Flags:
  python ab_eval.py                          # ranked table, zero API calls
  python ab_eval.py --new my-strategy        # score only new strategies
  python ab_eval.py --rerun d1-contextual    # force re-run one strategy
  python ab_eval.py --rerun all              # re-run everything (expensive)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import chromadb
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import anthropic

load_dotenv()

# ── RAGAS imports ─────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms        import LangchainLLMWrapper
from ragas.embeddings  import LangchainEmbeddingsWrapper
from langchain_anthropic            import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH         = Path("../day1-contextual-retrieval/chroma_db")
RESULTS_PATH        = Path("results")
MINILM_MODEL        = "all-MiniLM-L6-v2"
JINA_MODEL          = "jinaai/jina-embeddings-v2-base-en"
LLM_MODEL           = "claude-haiku-4-5-20251001"
TOP_K               = 5
METRICS             = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Collection names
COL_BASELINE    = "baseline_attention"
COL_CTX         = "contextual_attention"
COL_PDR_SMALL   = "small_chunks_attention"
COL_PDR_PARENT  = "parent_chunks_attention"
COL_LATE        = "late_chunking_attention"
COL_MV_SUMMARY  = "summary_vectors_attention"
COL_MV_FULL     = "fulltext_vectors_attention"

# ── Cached scores from dedicated per-day evals ───────────────────────────────
# Authoritative — measured in dedicated single-strategy evals.
# Default run loads these directly. No API calls needed.
KNOWN_SCORES = {
    "d1-baseline": {
        "faithfulness":      0.9208,
        "answer_relevancy":  0.9736,
        "context_precision": 0.6972,
        "context_recall":    1.0000,
    },
    "d1-contextual": {
        "faithfulness":      0.9552,
        "answer_relevancy":  0.9755,
        "context_precision": 0.8056,
        "context_recall":    1.0000,
    },
    "d2-parent-doc": {
        "faithfulness":      0.9554,
        "answer_relevancy":  0.8025,
        "context_precision": 0.6250,
        "context_recall":    0.8333,
    },
    "d3-late-chunk": {
        "faithfulness":      1.0000,
        "answer_relevancy":  0.9659,
        "context_precision": 0.7389,
        "context_recall":    1.0000,
    },
    "d4-multi-vec": {
        "faithfulness":      0.9542,
        "answer_relevancy":  0.9646,
        "context_precision": 0.7162,
        "context_recall":    1.0000,
    },
}

# ── Fixed eval set — never change ────────────────────────────────────────────
EVAL_QA = [
    {
        "question": "What is the purpose of the multi-head attention mechanism in the Transformer?",
        "ground_truth": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, improving the model's ability to focus on various aspects of the input simultaneously.",
    },
    {
        "question": "How does the Transformer avoid recurrence, and what does it use instead?",
        "ground_truth": "The Transformer replaces recurrent layers entirely with self-attention layers that relate all positions in a sequence to each other directly, without sequential processing.",
    },
    {
        "question": "What are the three types of attention used in the Transformer architecture?",
        "ground_truth": "The Transformer uses encoder self-attention, decoder self-attention, and encoder-decoder (cross) attention.",
    },
    {
        "question": "Why is positional encoding necessary in the Transformer?",
        "ground_truth": "Since the Transformer contains no recurrence or convolution, positional encodings are added to give the model information about the relative or absolute position of each token in the sequence.",
    },
    {
        "question": "What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation?",
        "ground_truth": "The Transformer achieved 28.4 BLEU on the WMT 2014 English-to-German translation task, outperforming existing best models including ensembles.",
    },
    {
        "question": "How does scaled dot-product attention work?",
        "ground_truth": "Scaled dot-product attention computes attention weights by taking the dot product of queries and keys, scaling by the square root of the key dimension, applying softmax, then multiplying by the values.",
    },
]

# ── Embedding helpers ─────────────────────────────────────────────────────────

def embed_minilm(minilm: SentenceTransformer, text: str) -> list[float]:
    return minilm.encode([text])[0].tolist()


def embed_jina(jina_tok, jina_mod, text: str) -> list[float]:
    inputs = jina_tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=8000,
        padding=False,
    )
    with torch.no_grad():
        outputs = jina_mod(**inputs)
    token_embs = outputs.last_hidden_state[0]
    pooled     = token_embs.mean(dim=0)
    normed     = (pooled / (pooled.norm() + 1e-9)).cpu().numpy()
    return normed.tolist()


# ── Strategy registry ─────────────────────────────────────────────────────────

def build_strategy_registry(chroma, minilm, jina_tok, jina_mod) -> dict:
    """
    Returns dict of strategy_key -> retrieve_fn.
    retrieve_fn: (question: str) -> list[str]

    To add a new strategy:
      1. Load or create its ChromaDB collection
      2. Write a retrieve_fn closure below
      3. Add it to the returned dict
      4. Run: python ab_eval.py --new your-strategy-key
    """
    baseline_col   = chroma.get_collection(COL_BASELINE)
    ctx_col        = chroma.get_collection(COL_CTX)
    small_col      = chroma.get_collection(COL_PDR_SMALL)
    parent_col     = chroma.get_collection(COL_PDR_PARENT)
    late_col       = chroma.get_collection(COL_LATE)
    mv_summary_col = chroma.get_collection(COL_MV_SUMMARY)
    mv_full_col    = chroma.get_collection(COL_MV_FULL)

    def retrieve_baseline(question: str) -> list[str]:
        emb = embed_minilm(minilm, question)
        return baseline_col.query(query_embeddings=[emb], n_results=TOP_K)["documents"][0]

    def retrieve_contextual(question: str) -> list[str]:
        emb = embed_minilm(minilm, question)
        return ctx_col.query(query_embeddings=[emb], n_results=TOP_K)["documents"][0]

    def retrieve_parent_doc(question: str) -> list[str]:
        emb     = embed_minilm(minilm, question)
        results = small_col.query(
            query_embeddings=[emb], n_results=TOP_K, include=["metadatas"]
        )
        seen = []
        for meta in results["metadatas"][0]:
            pid = meta["parent_id"]
            if pid not in seen:
                seen.append(pid)
        if not seen:
            return []
        return parent_col.get(ids=seen, include=["documents"])["documents"]

    def retrieve_late_chunking(question: str) -> list[str]:
        emb = embed_jina(jina_tok, jina_mod, question)
        return late_col.query(query_embeddings=[emb], n_results=TOP_K)["documents"][0]

    def retrieve_multi_vector(question: str) -> list[str]:
        s_emb     = embed_minilm(minilm, question)
        s_results = mv_summary_col.query(
            query_embeddings=[s_emb], n_results=TOP_K, include=["metadatas"]
        )
        j_emb     = embed_jina(jina_tok, jina_mod, question)
        j_results = mv_full_col.query(
            query_embeddings=[j_emb], n_results=TOP_K, include=["metadatas"]
        )
        seen_ids = []
        for meta in s_results["metadatas"][0]:
            cid = f"chunk_{meta['chunk_index']}"
            if cid not in seen_ids:
                seen_ids.append(cid)
        for meta in j_results["metadatas"][0]:
            cid = f"chunk_{meta['chunk_index']}"
            if cid not in seen_ids:
                seen_ids.append(cid)
        return mv_full_col.get(
            ids=seen_ids[:TOP_K], include=["documents"]
        )["documents"]

    return {
        "d1-baseline":   retrieve_baseline,
        "d1-contextual": retrieve_contextual,
        "d2-parent-doc": retrieve_parent_doc,
        "d3-late-chunk": retrieve_late_chunking,
        "d4-multi-vec":  retrieve_multi_vector,
    }


# ── Dataset + RAGAS ───────────────────────────────────────────────────────────

def build_dataset(retrieve_fn, llm_client: anthropic.Anthropic, label: str) -> Dataset:
    rows = []
    for i, qa in enumerate(EVAL_QA):
        contexts     = retrieve_fn(qa["question"])
        context_text = "\n\n---\n\n".join(contexts)
        prompt = (
            f"Answer the question based ONLY on the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {qa['question']}\n\nAnswer:"
        )
        response = llm_client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        rows.append({
            "question":     qa["question"],
            "answer":       response.content[0].text.strip(),
            "contexts":     contexts,
            "ground_truth": qa["ground_truth"],
        })
        print(f"    [{label}] Q{i+1}/{len(EVAL_QA)} answered")
    return Dataset.from_list(rows)


def run_ragas(dataset: Dataset, ragas_llm, ragas_embed) -> dict:
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embed,
        raise_exceptions=False,
    )
    return result.to_pandas().select_dtypes(include="number").mean().to_dict()


# ── Output ────────────────────────────────────────────────────────────────────

def compute_overall(scores: dict) -> float:
    vals = [scores.get(m, float("nan")) for m in METRICS]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else 0.0


def print_ranked_table(all_scores: dict[str, dict]) -> None:
    ranked   = sorted(all_scores.items(), key=lambda x: compute_overall(x[1]), reverse=True)
    labels   = [r[0] for r in ranked]
    col_w    = 14
    header   = f"{'Metric':<22}" + "".join(f"{l:>{col_w}}" for l in labels)
    div      = "=" * len(header)

    print(f"\n{div}")
    print("  Phase B Day 5 — A/B Retrieval Harness — Ranked Results")
    print(div)
    print(header)
    print("-" * len(header))

    for m in METRICS:
        vals  = [ranked[i][1].get(m, float("nan")) for i in range(len(ranked))]
        valid = [v for v in vals if v == v]
        best  = max(valid) if valid else None
        row   = f"  {m:<20}"
        for v in vals:
            if v != v:
                cell = "nan"
            else:
                cell = f"{v:.4f}" + ("★" if best and abs(v - best) < 1e-6 else " ")
            row += f"{cell:>{col_w}}"
        print(row)

    print("-" * len(header))
    overall_vals = [compute_overall(s) for _, s in ranked]
    best_overall = max(overall_vals) if overall_vals else 0.0
    row = f"  {'OVERALL (mean)':<20}"
    for ov in overall_vals:
        cell = f"{ov:.4f}" + ("★" if abs(ov - best_overall) < 1e-6 else " ")
        row += f"{cell:>{col_w}}"
    print(row)
    print(div)

    winner_label, _ = ranked[0]
    print(f"\n  🏆 Winner: {winner_label}  (overall {best_overall:.4f})")
    print(f"\n  📊 Delta vs winner:")
    for label, scores in ranked[1:]:
        delta = compute_overall(scores) - best_overall
        print(f"     {label:<22} {delta:+.4f}")
    print()


def print_phase_b_summary(all_scores: dict[str, dict]) -> None:
    ranked          = sorted(all_scores.items(), key=lambda x: compute_overall(x[1]), reverse=True)
    winner_label, _ = ranked[0]
    winner_overall  = compute_overall(ranked[0][1])
    medals          = ["🥇", "🥈", "🥉", "  4.", "  5.", "  6.", "  7.", "  8."]
    descriptions    = {
        "d1-baseline":   "Raw chunks, no enrichment",
        "d1-contextual": "Claude-generated context prepended per chunk",
        "d2-parent-doc": "Small chunks for search, parent chunks returned",
        "d3-late-chunk": "Full-doc Jina token embeddings chunked post-hoc",
        "d4-multi-vec":  "Summary (MiniLM) + fulltext (Jina) union retrieval",
    }

    print("=" * 62)
    print("  Phase B — Retrieval Mastery — Final Summary")
    print("=" * 62)
    print(f"\n  Corpus  : Attention Is All You Need (Vaswani et al. 2017)")
    print(f"  Eval    : {len(EVAL_QA)} fixed questions, RAGAS 0.4.3")
    print(f"  Baseline: Phase 1 capstone 0.8270\n")
    print("  Strategies ranked:")
    for i, (label, scores) in enumerate(ranked):
        ov   = compute_overall(scores)
        desc = descriptions.get(label, "")
        print(f"  {medals[i]} {label:<22} {ov:.4f}   {desc}")

    best_faith = max(s.get("faithfulness", 0)      for _, s in ranked)
    best_prec  = max(s.get("context_precision", 0) for _, s in ranked)

    print(f"\n  Winner : {winner_label}")
    print(f"  Overall: {winner_overall:.4f}  (+{winner_overall - 0.8270:.4f} vs Phase 1 capstone)")
    print(f"\n  Best faithfulness      : {best_faith:.4f}  (d3-late-chunk)")
    print(f"  Best context_precision : {best_prec:.4f}  (d1-contextual)")
    print(f"\n  Key finding:")
    print(f"  Explicit Claude-generated summaries produce the sharpest")
    print(f"  retrieval signal on a dense academic paper. Context precision")
    print(f"  0.8056 — unreached by any other strategy. Late chunking")
    print(f"  achieves faithfulness=1.0 at zero LLM cost but trails on")
    print(f"  precision. Multi-vector union adds noise at TOP_K=5.")
    print()
    print("=" * 62)


def save_results(all_scores: dict[str, dict]) -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp":      datetime.now().isoformat(),
        "corpus":         "attention-is-all-you-need",
        "eval_questions": len(EVAL_QA),
        "top_k":          TOP_K,
        "strategies": {
            label: {**scores, "overall": compute_overall(scores)}
            for label, scores in all_scores.items()
        },
    }
    out_path = RESULTS_PATH / "ab_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {out_path}")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Phase B A/B Retrieval Harness")
    parser.add_argument(
        "--new",
        nargs="+",
        metavar="STRATEGY",
        help="Score only these new strategies; all others load from cache",
        default=[],
    )
    parser.add_argument(
        "--rerun",
        nargs="+",
        metavar="STRATEGY",
        help="Force re-run RAGAS on these known strategies. Use 'all' to rerun everything.",
        default=[],
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    rerun_all  = "all" in args.rerun
    force_rerun = set(args.rerun) if not rerun_all else set()
    new_strats  = set(args.new)
    needs_api   = rerun_all or bool(force_rerun) or bool(new_strats)

    print("=== Phase B Day 5 — A/B Retrieval Harness ===\n")

    # ── Load cached scores ────────────────────────────────────────────────────
    all_scores: dict[str, dict] = {}

    if not rerun_all:
        for label, scores in KNOWN_SCORES.items():
            if label not in force_rerun and label not in new_strats:
                all_scores[label] = scores
                print(f"  ✓ {label:<22} loaded from cache")

    # ── Fast path: nothing to score live ─────────────────────────────────────
    if not needs_api:
        print("\n  No API calls needed — all scores from cache.")
        print_ranked_table(all_scores)
        print_phase_b_summary(all_scores)
        save_results(all_scores)
        print("=== Done. ===\n")
        return

    # ── Load models only when API calls are required ──────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)
    if not CHROMA_PATH.exists():
        print(f"ERROR: chroma_db not found at {CHROMA_PATH}")
        sys.exit(1)

    print("\n1. Loading models …")
    minilm   = SentenceTransformer(MINILM_MODEL)
    print(f"   MiniLM : {MINILM_MODEL}")
    jina_tok = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    jina_mod = AutoModel.from_pretrained(JINA_MODEL, trust_remote_code=True)
    jina_mod.eval()
    print(f"   Jina   : {JINA_MODEL}")
    llm_client  = anthropic.Anthropic(api_key=api_key)
    ragas_llm   = LangchainLLMWrapper(
        ChatAnthropic(model=LLM_MODEL, anthropic_api_key=api_key)
    )
    ragas_embed = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=MINILM_MODEL)
    )
    print(f"   LLM    : {LLM_MODEL}\n")

    print("2. Loading ChromaDB collections …")
    chroma   = chromadb.PersistentClient(path=str(CHROMA_PATH))
    registry = build_strategy_registry(chroma, minilm, jina_tok, jina_mod)

    to_score = set(registry.keys()) if rerun_all else (force_rerun | new_strats)
    unknown  = to_score - set(registry.keys())
    if unknown:
        print(f"ERROR: Unknown strategies: {unknown}")
        print(f"Available: {list(registry.keys())}")
        sys.exit(1)

    print(f"   Live scoring : {sorted(to_score)}")
    print(f"   From cache   : {sorted(set(all_scores.keys()))}\n")

    print("3. Building answer datasets …")
    datasets = {}
    for label in sorted(to_score):
        print(f"   {label} …")
        datasets[label] = build_dataset(registry[label], llm_client, label)
        print()

    print("4. Running RAGAS …")
    for label, dataset in datasets.items():
        print(f"   Scoring {label} …")
        all_scores[label] = run_ragas(dataset, ragas_llm, ragas_embed)
        print(f"   Overall: {compute_overall(all_scores[label]):.4f}\n")

    print_ranked_table(all_scores)
    print_phase_b_summary(all_scores)
    save_results(all_scores)
    print("=== Done. ===\n")


if __name__ == "__main__":
    main()