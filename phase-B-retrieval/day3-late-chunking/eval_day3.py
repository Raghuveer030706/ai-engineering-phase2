"""
Phase B Day 3 — RAGAS Evaluation
Three-way comparison:
  - contextual_attention     (Day 1 best: 0.9341)
  - parent-document          (Day 2:      0.8041)
  - late_chunking_attention  (Day 3:      ?)

Key difference from Day 1/2 evals:
  Late chunking uses jina-embeddings-v2-base-en for BOTH indexing and query.
  Must use the same model for retrieval that was used for indexing.
  Day 1 contextual still uses all-MiniLM-L6-v2 for its retrieval.
  Two embed models loaded — each used for its own collection.
"""

import os
import sys
from pathlib import Path
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
CHROMA_PATH      = Path("../day1-contextual-retrieval/chroma_db")
MINILM_MODEL     = "all-MiniLM-L6-v2"
JINA_MODEL       = "jinaai/jina-embeddings-v2-base-en"
LLM_MODEL        = "claude-haiku-4-5-20251001"
TOP_K            = 5

CTX_COLLECTION   = "contextual_attention"
PDR_SMALL        = "small_chunks_attention"
PDR_PARENT       = "parent_chunks_attention"
LATE_COLLECTION  = "late_chunking_attention"

# ── Same eval set — never change ──────────────────────────────────────────────
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

def retrieve_minilm(collection, embed_model: SentenceTransformer, question: str, top_k: int) -> list[str]:
    embedding = embed_model.encode([question])[0].tolist()
    results   = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["documents"][0]


def retrieve_parent(small_col, parent_col, embed_model: SentenceTransformer, question: str, top_k: int) -> list[str]:
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


def embed_question_jina(tokenizer, model, question: str) -> list[float]:
    """Embed a short question with jina — mean pool, L2 normalise."""
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    token_embs = outputs.last_hidden_state[0]
    pooled     = token_embs.mean(dim=0)
    normed     = (pooled / (pooled.norm() + 1e-9)).cpu().numpy()
    return normed.tolist()


def retrieve_late(late_col, jina_tokenizer, jina_model, question: str, top_k: int) -> list[str]:
    embedding = embed_question_jina(jina_tokenizer, jina_model, question)
    results   = late_col.query(query_embeddings=[embedding], n_results=top_k)
    return results["documents"][0]


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(retrieve_fn, llm_client: anthropic.Anthropic, label: str) -> Dataset:
    rows = []
    for i, qa in enumerate(EVAL_QA):
        contexts     = retrieve_fn(qa["question"])
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
        raise_exceptions=False,
    )
    return result.to_pandas().select_dtypes(include="number").mean().to_dict()


# ── Output ────────────────────────────────────────────────────────────────────

def print_comparison(scores: dict[str, dict]) -> None:
    metrics   = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels    = list(scores.keys())
    col_w     = 11

    # Header
    header = f"{'Metric':<22}" + "".join(f"{l:>{col_w}}" for l in labels)
    div    = "=" * len(header)

    print(f"\n{div}")
    print("  Phase B Day 3 — Retrieval Strategy Comparison")
    print(div)
    print(header)
    print("-" * len(header))

    overalls = {l: 0.0 for l in labels}

    for m in metrics:
        row = f"  {m:<20}"
        vals = [scores[l].get(m, 0.0) for l in labels]
        best = max(vals)
        for l, v in zip(labels, vals):
            marker = " ★" if v == best and len(labels) > 1 else "  "
            row += f"{v:>{col_w}.4f}"
            overalls[l] += v
        # reprint with star on best
        row = f"  {m:<20}"
        for l, v in zip(labels, vals):
            cell = f"{v:.4f}"
            if v == best:
                cell += "★"
            else:
                cell += " "
            row += f"{cell:>{col_w}}"
        print(row)

    print("-" * len(header))
    row = f"  {'OVERALL (mean)':<20}"
    overall_vals = [overalls[l] / len(metrics) for l in labels]
    best_overall = max(overall_vals)
    for l, ov in zip(labels, overall_vals):
        cell = f"{ov:.4f}"
        if ov == best_overall:
            cell += "★"
        else:
            cell += " "
        row += f"{cell:>{col_w}}"
    print(row)
    print(div)

    # Running scoreboard
    print("\n  📊 Phase B Running Scoreboard")
    print(f"     Phase 1 capstone : 0.8270")
    print(f"     Day 1 baseline   : 0.8979")
    print(f"     Day 1 contextual : 0.9341")
    print(f"     Day 2 parent-doc : 0.8041")
    best_label = labels[overall_vals.index(best_overall)]
    print(f"     Day 3 late-chunk : {best_overall:.4f}  {'← new best ✓' if best_overall > 0.9341 else '← Day 1 contextual still leads'}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    if not CHROMA_PATH.exists():
        print(f"ERROR: chroma_db not found at {CHROMA_PATH}")
        sys.exit(1)

    print("=== Phase B Day 3 — RAGAS Evaluation ===\n")

    print("1. Loading models …")
    minilm_model  = SentenceTransformer(MINILM_MODEL)
    print(f"   MiniLM loaded: {MINILM_MODEL}")

    print(f"   Loading Jina (may take a moment) …")
    jina_tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    jina_model     = AutoModel.from_pretrained(JINA_MODEL, trust_remote_code=True)
    jina_model.eval()
    print(f"   Jina loaded:   {JINA_MODEL}")

    llm_client  = anthropic.Anthropic(api_key=api_key)
    ragas_llm   = LangchainLLMWrapper(
        ChatAnthropic(model=LLM_MODEL, anthropic_api_key=api_key)
    )
    ragas_embed = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=MINILM_MODEL)
    )
    print(f"   RAGAS LLM: {LLM_MODEL}\n")

    print("2. Loading ChromaDB collections …")
    chroma     = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ctx_col    = chroma.get_collection(CTX_COLLECTION)
    small_col  = chroma.get_collection(PDR_SMALL)
    parent_col = chroma.get_collection(PDR_PARENT)
    late_col   = chroma.get_collection(LATE_COLLECTION)
    print(f"   contextual_attention     : {ctx_col.count()} chunks")
    print(f"   small_chunks_attention   : {small_col.count()} chunks")
    print(f"   parent_chunks_attention  : {parent_col.count()} chunks")
    print(f"   late_chunking_attention  : {late_col.count()} chunks\n")

    # Bind retrieve functions
    def retrieve_ctx(q):  return retrieve_minilm(ctx_col, minilm_model, q, TOP_K)
    def retrieve_pdr(q):  return retrieve_parent(small_col, parent_col, minilm_model, q, TOP_K)
    def retrieve_lc(q):   return retrieve_late(late_col, jina_tokenizer, jina_model, q, TOP_K)

    print("3. Building answer datasets …")
    print("   Day 1 contextual …")
    d1_ds = build_dataset(retrieve_ctx, llm_client, "D1-ctx")
    print("\n   Day 2 parent-document …")
    d2_ds = build_dataset(retrieve_pdr, llm_client, "D2-pdr")
    print("\n   Day 3 late chunking …")
    d3_ds = build_dataset(retrieve_lc,  llm_client, "D3-late")

    print("\n4. Running RAGAS …")
    print("   Day 1 contextual …")
    d1_scores = run_ragas(d1_ds, ragas_llm, ragas_embed)
    print("   Day 2 parent-document …")
    d2_scores = run_ragas(d2_ds, ragas_llm, ragas_embed)
    print("   Day 3 late chunking …")
    d3_scores = run_ragas(d3_ds, ragas_llm, ragas_embed)

    print_comparison({
        "D1-Ctx": d1_scores,
        "D2-PDR": d2_scores,
        "D3-Late": d3_scores,
    })

    print("=== Done. Run Day 4 (multi_vector.py) next. ===\n")


if __name__ == "__main__":
    main()