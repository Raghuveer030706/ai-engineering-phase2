"""
Phase B Day 2 — RAGAS Evaluation
Compares:
  - contextual_attention    (Day 1 best: overall 0.9341)
  - parent-document         (Day 2: small chunk search → parent chunk retrieval)

Same 6 questions. Same RAGAS 0.4.3 setup. Apples-to-apples.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
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
from langchain_anthropic          import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH       = Path("../day1-contextual-retrieval/chroma_db")
EMBED_MODEL       = "all-MiniLM-L6-v2"
LLM_MODEL         = "claude-haiku-4-5-20251001"
TOP_K             = 5

SMALL_COLLECTION  = "small_chunks_attention"
PARENT_COLLECTION = "parent_chunks_attention"
CTX_COLLECTION    = "contextual_attention"

# ── Same eval set as Day 1 — never change these ───────────────────────────────
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

# ── Retrieval helpers ─────────────────────────────────────────────────────────

def retrieve_contextual(collection, embed_model: SentenceTransformer, question: str, top_k: int) -> list[str]:
    """Standard vector search on contextual collection."""
    embedding = embed_model.encode([question])[0].tolist()
    results   = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["documents"][0]


def retrieve_parent(
    small_col,
    parent_col,
    embed_model: SentenceTransformer,
    question: str,
    top_k: int,
) -> list[str]:
    """Search small chunks, return their parent chunks (deduplicated)."""
    embedding = embed_model.encode([question])[0].tolist()
    results   = small_col.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas"],
    )

    seen = []
    for meta in results["metadatas"][0]:
        pid = meta["parent_id"]
        if pid not in seen:
            seen.append(pid)

    if not seen:
        return []

    fetched = parent_col.get(ids=seen, include=["documents"])
    return fetched["documents"]


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_dataset(retrieve_fn, llm_client: anthropic.Anthropic, label: str) -> Dataset:
    rows = []
    for i, qa in enumerate(EVAL_QA):
        contexts = retrieve_fn(qa["question"])
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
        print(f"    [{label}] Q{i+1}/6 answered")

    return Dataset.from_list(rows)


def run_ragas(dataset: Dataset, ragas_llm, ragas_embeddings) -> dict:
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,   # survive rate-limit blips
    )
    return result.to_pandas().select_dtypes(include="number").mean().to_dict()


# ── Output ────────────────────────────────────────────────────────────────────

def print_comparison(day1_scores: dict, day2_scores: dict) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    col_w   = 12

    header = f"{'Metric':<22} {'Day1-Ctx':>{col_w}} {'Day2-PDR':>{col_w}} {'Delta':>{col_w}}"
    div    = "=" * len(header)

    print(f"\n{div}")
    print("  Phase B Day 2 — Contextual (D1) vs Parent-Document Retrieval (D2)")
    print(div)
    print(header)
    print("-" * len(header))

    overall_d1 = 0.0
    overall_d2 = 0.0

    for m in metrics:
        d1 = day1_scores.get(m, 0.0)
        d2 = day2_scores.get(m, 0.0)
        delta  = d2 - d1
        arrow  = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "─")
        print(f"  {m:<20} {d1:>{col_w}.4f} {d2:>{col_w}.4f} {arrow} {abs(delta):>{col_w-2}.4f}")
        overall_d1 += d1
        overall_d2 += d2

    print("-" * len(header))
    od1   = overall_d1 / len(metrics)
    od2   = overall_d2 / len(metrics)
    odelta = od2 - od1
    arrow  = "▲" if odelta > 0.001 else ("▼" if odelta < -0.001 else "─")
    print(f"  {'OVERALL (mean)':<20} {od1:>{col_w}.4f} {od2:>{col_w}.4f} {arrow} {abs(odelta):>{col_w-2}.4f}")
    print(div)

    # Phase B running scoreboard
    print("\n  📊 Phase B Running Scoreboard")
    print(f"     Phase 1 capstone  : 0.8270")
    print(f"     Day 1 baseline    : 0.8979")
    print(f"     Day 1 contextual  : 0.9341")
    print(f"     Day 2 parent-doc  : {od2:.4f}  {'← best' if od2 > 0.9341 else ''}")
    print()

    # Precision highlight (Day 1's biggest win was precision)
    d1_p = day1_scores.get("context_precision", 0.0)
    d2_p = day2_scores.get("context_precision", 0.0)
    print(f"  🎯 Context precision: {d1_p:.4f} → {d2_p:.4f}  (parent chunks more focused?)")
    d1_r = day1_scores.get("context_recall", 0.0)
    d2_r = day2_scores.get("context_recall", 0.0)
    print(f"  🎯 Context recall  : {d1_r:.4f} → {d2_r:.4f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    if not CHROMA_PATH.exists():
        print("ERROR: chroma_db not found. Run parent_document_retrieval.py first.")
        sys.exit(1)

    print("=== Phase B Day 2 — RAGAS Evaluation ===\n")

    print("1. Loading models …")
    embed_model = SentenceTransformer(EMBED_MODEL)
    llm_client  = anthropic.Anthropic(api_key=api_key)
    ragas_llm   = LangchainLLMWrapper(
        ChatAnthropic(model=LLM_MODEL, anthropic_api_key=api_key)
    )
    ragas_embed = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    print(f"   LLM: {LLM_MODEL}  |  Embed: {EMBED_MODEL}\n")

    print("2. Loading ChromaDB collections …")
    chroma     = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ctx_col    = chroma.get_collection(CTX_COLLECTION)
    small_col  = chroma.get_collection(SMALL_COLLECTION)
    parent_col = chroma.get_collection(PARENT_COLLECTION)
    print(f"   contextual_attention : {ctx_col.count()} chunks")
    print(f"   small_chunks         : {small_col.count()} chunks")
    print(f"   parent_chunks        : {parent_col.count()} chunks\n")

    # Bind retrieval functions (no extra args needed in build_dataset)
    def retrieve_ctx(q):
        return retrieve_contextual(ctx_col, embed_model, q, TOP_K)

    def retrieve_pdr(q):
        return retrieve_parent(small_col, parent_col, embed_model, q, TOP_K)

    print("3. Building answer datasets …")
    print("   Day 1 contextual answers …")
    day1_ds = build_dataset(retrieve_ctx, llm_client, "D1-ctx")

    print("\n   Day 2 parent-document answers …")
    day2_ds = build_dataset(retrieve_pdr, llm_client, "D2-pdr")

    print("\n4. Running RAGAS on Day 1 contextual …")
    day1_scores = run_ragas(day1_ds, ragas_llm, ragas_embed)

    print("\n5. Running RAGAS on Day 2 parent-document …")
    day2_scores = run_ragas(day2_ds, ragas_llm, ragas_embed)

    print_comparison(day1_scores, day2_scores)

    print("=== Done. Run Day 3 (late_chunking.py) next. ===\n")


if __name__ == "__main__":
    main()