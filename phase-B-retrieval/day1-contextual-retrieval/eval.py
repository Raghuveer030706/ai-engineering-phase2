"""
Phase B Day 1 — RAGAS Evaluation
Runs identical RAGAS eval against:
  - baseline_attention   (raw chunks, no context)
  - contextual_attention (context-prepended chunks)

Prints a side-by-side comparison table.
Measures: faithfulness, answer_relevancy, context_precision, context_recall

RAGAS gotchas baked in:
  - ragas==0.4.3 pinned
  - llm= and embeddings= passed explicitly (no OpenAI default)
  - EvaluationResult → to_pandas().select_dtypes(include="number").mean()
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

load_dotenv()

# ── RAGAS imports (0.4.3 API) ─────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms   import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH  = Path("chroma_db")
EMBED_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "claude-haiku-4-5-20251001"
TOP_K        = 5

# ── Eval questions — same set used throughout Phase B for apples-to-apples ───
# Questions cover different types of recall challenge:
#   Q1: precise factual (should retrieve well even baseline)
#   Q2: cross-section concept (needs context to know what "it" refers to)
#   Q3: comparative (spans multiple sections)
#   Q4: mechanism question (spread across encoder/decoder sections)
#   Q5: numerical (specific values scattered in paper)
#   Q6: motivation (abstract reasoning section)

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def retrieve(collection, embed_model: SentenceTransformer, question: str, top_k: int = TOP_K) -> list[str]:
    embedding = embed_model.encode([question])[0].tolist()
    results   = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["documents"][0]


def build_ragas_dataset(collection, embed_model: SentenceTransformer, llm_client) -> Dataset:
    """Build the Dataset ragas.evaluate() expects."""
    rows = []
    for qa in EVAL_QA:
        contexts = retrieve(collection, embed_model, qa["question"])

        # Generate answer from retrieved contexts using Claude
        context_text = "\n\n---\n\n".join(contexts)
        prompt = (
            f"Answer the question based ONLY on the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {qa['question']}\n\n"
            f"Answer:"
        )
        response = llm_client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip()

        rows.append({
            "question":     qa["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": qa["ground_truth"],
        })

    return Dataset.from_list(rows)


def run_ragas(dataset: Dataset, ragas_llm, ragas_embeddings) -> dict:
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    df     = result.to_pandas().select_dtypes(include="number")
    means  = df.mean().to_dict()
    return means


def print_comparison(baseline_scores: dict, contextual_scores: dict) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    col_w   = 12

    header = f"{'Metric':<22} {'Baseline':>{col_w}} {'Contextual':>{col_w}} {'Delta':>{col_w}}"
    print("\n" + "=" * len(header))
    print("  Phase B Day 1 — RAGAS Comparison: Baseline vs Contextual Retrieval")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    overall_baseline    = 0.0
    overall_contextual  = 0.0

    for m in metrics:
        b = baseline_scores.get(m, 0.0)
        c = contextual_scores.get(m, 0.0)
        d = c - b
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "─")
        print(f"  {m:<20} {b:>{col_w}.4f} {c:>{col_w}.4f} {arrow} {abs(d):>{col_w-2}.4f}")
        overall_baseline   += b
        overall_contextual += c

    print("-" * len(header))
    ob = overall_baseline   / len(metrics)
    oc = overall_contextual / len(metrics)
    od = oc - ob
    arrow = "▲" if od > 0 else ("▼" if od < 0 else "─")
    print(f"  {'OVERALL (mean)':<20} {ob:>{col_w}.4f} {oc:>{col_w}.4f} {arrow} {abs(od):>{col_w-2}.4f}")
    print("=" * len(header))

    # Context recall highlight — this is the metric Phase B targets
    b_recall = baseline_scores.get("context_recall", 0.0)
    c_recall = contextual_scores.get("context_recall", 0.0)
    print(f"\n  🎯 Context recall target: 0.583 → 0.75+")
    print(f"     Baseline:   {b_recall:.4f}")
    print(f"     Contextual: {c_recall:.4f}")
    if c_recall > 0.75:
        print("     ✓ TARGET HIT")
    elif c_recall > b_recall:
        print(f"     Progress: +{c_recall - b_recall:.4f}")
    else:
        print("     ✗ No improvement — see Day 2 (parent-document retrieval)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    if not CHROMA_PATH.exists():
        print("ERROR: chroma_db not found. Run setup_baseline.py and contextual_retrieval.py first.")
        sys.exit(1)

    print("=== Phase B Day 1 — RAGAS Evaluation ===\n")

    # Clients
    print("1. Loading models …")
    embed_model = SentenceTransformer(EMBED_MODEL)
    llm_client  = anthropic.Anthropic(api_key=api_key)

    # RAGAS wrappers — explicit, no OpenAI default
    ragas_llm   = LangchainLLMWrapper(
        ChatAnthropic(model=LLM_MODEL, anthropic_api_key=api_key)
    )
    ragas_embed = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    print(f"   LLM: {LLM_MODEL}")
    print(f"   Embeddings: {EMBED_MODEL}\n")

    # ChromaDB
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        baseline_col    = chroma.get_collection("baseline_attention")
        contextual_col  = chroma.get_collection("contextual_attention")
    except Exception as e:
        print(f"ERROR loading collections: {e}")
        print("Run setup_baseline.py then contextual_retrieval.py first.")
        sys.exit(1)

    print(f"2. Baseline collection:    {baseline_col.count()} chunks")
    print(f"   Contextual collection:  {contextual_col.count()} chunks\n")

    # Build datasets
    print("3. Building RAGAS datasets (6 questions × 2 collections) …")
    print("   Baseline answers …")
    baseline_ds = build_ragas_dataset(baseline_col, embed_model, llm_client)
    print("   Contextual answers …")
    contextual_ds = build_ragas_dataset(contextual_col, embed_model, llm_client)

    # Run RAGAS
    print("\n4. Running RAGAS on baseline (this makes ~48 LLM calls) …")
    baseline_scores = run_ragas(baseline_ds, ragas_llm, ragas_embed)

    print("\n5. Running RAGAS on contextual …")
    contextual_scores = run_ragas(contextual_ds, ragas_llm, ragas_embed)

    # Print results
    print_comparison(baseline_scores, contextual_scores)

    print("=== Done. Run contextual_retrieval.py for Day 2 (parent-document). ===\n")


if __name__ == "__main__":
    main()