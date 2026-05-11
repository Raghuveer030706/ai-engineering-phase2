"""
Phase B Day 4 — RAGAS Evaluation
Four-way comparison:
  - contextual_attention     (D1: 0.9341 overall)
  - parent-document          (D2: 0.8041 overall)
  - late_chunking_attention  (D3: 0.9262 overall)
  - multi-vector             (D4: summary+fulltext union)

Multi-vector hypothesis:
  Summary vectors (MiniLM) catch concept-level queries.
  Fulltext vectors (Jina) catch detail-level queries.
  Union of both surfaces more relevant chunks than either alone.
  Expected: context_precision >= D1, faithfulness >= D3, overall > 0.9341.
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
CHROMA_PATH         = Path("../day1-contextual-retrieval/chroma_db")
MINILM_MODEL        = "all-MiniLM-L6-v2"
JINA_MODEL          = "jinaai/jina-embeddings-v2-base-en"
LLM_MODEL           = "claude-haiku-4-5-20251001"
TOP_K               = 5

CTX_COLLECTION      = "contextual_attention"
PDR_SMALL           = "small_chunks_attention"
PDR_PARENT          = "parent_chunks_attention"
LATE_COLLECTION     = "late_chunking_attention"
SUMMARY_COLLECTION  = "summary_vectors_attention"
FULLTEXT_COLLECTION = "fulltext_vectors_attention"

# ── Eval set — fixed across all Phase B days ──────────────────────────────────
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

def retrieve_minilm(col, minilm: SentenceTransformer, question: str, top_k: int) -> list[str]:
    emb     = minilm.encode([question])[0].tolist()
    results = col.query(query_embeddings=[emb], n_results=top_k)
    return results["documents"][0]


def retrieve_parent(small_col, parent_col, minilm: SentenceTransformer, question: str, top_k: int) -> list[str]:
    emb     = minilm.encode([question])[0].tolist()
    results = small_col.query(query_embeddings=[emb], n_results=top_k, include=["metadatas"])
    seen    = []
    for meta in results["metadatas"][0]:
        pid = meta["parent_id"]
        if pid not in seen:
            seen.append(pid)
    if not seen:
        return []
    return parent_col.get(ids=seen, include=["documents"])["documents"]


def embed_jina(tokenizer, model, text: str) -> list[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8000, padding=False)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embs = outputs.last_hidden_state[0]
    pooled     = token_embs.mean(dim=0)
    normed     = (pooled / (pooled.norm() + 1e-9)).cpu().numpy()
    return normed.tolist()


def retrieve_late(late_col, jina_tok, jina_mod, question: str, top_k: int) -> list[str]:
    emb     = embed_jina(jina_tok, jina_mod, question)
    results = late_col.query(query_embeddings=[emb], n_results=top_k)
    return results["documents"][0]


def retrieve_multi_vector(
    summary_col, fulltext_col,
    minilm: SentenceTransformer,
    jina_tok, jina_mod,
    question: str,
    top_k: int,
) -> list[str]:
    # Summary search (MiniLM)
    s_emb     = minilm.encode([question])[0].tolist()
    s_results = summary_col.query(query_embeddings=[s_emb], n_results=top_k, include=["metadatas"])

    # Fulltext search (Jina)
    j_emb     = embed_jina(jina_tok, jina_mod, question)
    j_results = fulltext_col.query(query_embeddings=[j_emb], n_results=top_k, include=["metadatas"])

    # Union — summary hits first, then fulltext additions
    seen_ids = []
    for meta in s_results["metadatas"][0]:
        cid = f"chunk_{meta['chunk_index']}"
        if cid not in seen_ids:
            seen_ids.append(cid)
    for meta in j_results["metadatas"][0]:
        cid = f"chunk_{meta['chunk_index']}"
        if cid not in seen_ids:
            seen_ids.append(cid)

    seen_ids = seen_ids[:top_k]
    return fulltext_col.get(ids=seen_ids, include=["documents"])["documents"]


# ── Dataset builder ───────────────────────────────────────────────────────────

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
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels  = list(scores.keys())
    col_w   = 11

    header = f"{'Metric':<22}" + "".join(f"{l:>{col_w}}" for l in labels)
    div    = "=" * len(header)

    print(f"\n{div}")
    print("  Phase B Day 4 — Four-Way Retrieval Strategy Comparison")
    print(div)
    print(header)
    print("-" * len(header))

    overalls = {l: 0.0 for l in labels}
    for m in metrics:
        vals = [scores[l].get(m, 0.0) for l in labels]
        best = max(vals)
        row  = f"  {m:<20}"
        for v in vals:
            cell = f"{v:.4f}" + ("★" if v == best else " ")
            row += f"{cell:>{col_w}}"
        print(row)
        for l, v in zip(labels, vals):
            overalls[l] += v

    print("-" * len(header))
    overall_vals = [overalls[l] / len(metrics) for l in labels]
    best_overall = max(overall_vals)
    row = f"  {'OVERALL (mean)':<20}"
    for ov in overall_vals:
        cell = f"{ov:.4f}" + ("★" if ov == best_overall else " ")
        row += f"{cell:>{col_w}}"
    print(row)
    print(div)

    # Full scoreboard
    d4_overall = overall_vals[labels.index("D4-MV")] if "D4-MV" in labels else 0.0
    print("\n  📊 Phase B Running Scoreboard")
    print(f"     Phase 1 capstone : 0.8270")
    print(f"     Day 1 baseline   : 0.8979")
    print(f"     Day 1 contextual : 0.9341")
    print(f"     Day 2 parent-doc : 0.8041")
    print(f"     Day 3 late-chunk : 0.9262")
    print(f"     Day 4 multi-vec  : {d4_overall:.4f}  {'← new best ✓' if d4_overall > 0.9341 else '← Day 1 contextual still leads'}")

    # Metric-level breakdown for D4
    d4 = scores.get("D4-MV", {})
    d1 = scores.get("D1-Ctx", {})
    print(f"\n  🎯 D4 vs D1 contextual (the benchmark to beat):")
    for m in metrics:
        v4 = d4.get(m, 0.0)
        v1 = d1.get(m, 0.0)
        delta = v4 - v1
        arrow = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "─")
        print(f"     {m:<22}: {v1:.4f} → {v4:.4f}  {arrow} {abs(delta):.4f}")
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

    print("=== Phase B Day 4 — RAGAS Evaluation ===\n")

    print("1. Loading models …")
    minilm = SentenceTransformer(MINILM_MODEL)
    print(f"   MiniLM: {MINILM_MODEL}")
    jina_tok = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    jina_mod = AutoModel.from_pretrained(JINA_MODEL, trust_remote_code=True)
    jina_mod.eval()
    print(f"   Jina  : {JINA_MODEL}")
    llm_client  = anthropic.Anthropic(api_key=api_key)
    ragas_llm   = LangchainLLMWrapper(ChatAnthropic(model=LLM_MODEL, anthropic_api_key=api_key))
    ragas_embed = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=MINILM_MODEL))
    print(f"   LLM   : {LLM_MODEL}\n")

    print("2. Loading ChromaDB collections …")
    chroma      = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ctx_col     = chroma.get_collection(CTX_COLLECTION)
    small_col   = chroma.get_collection(PDR_SMALL)
    parent_col  = chroma.get_collection(PDR_PARENT)
    late_col    = chroma.get_collection(LATE_COLLECTION)
    summary_col = chroma.get_collection(SUMMARY_COLLECTION)
    ft_col      = chroma.get_collection(FULLTEXT_COLLECTION)
    print(f"   contextual        : {ctx_col.count()}")
    print(f"   small / parent    : {small_col.count()} / {parent_col.count()}")
    print(f"   late_chunking     : {late_col.count()}")
    print(f"   summary / fulltext: {summary_col.count()} / {ft_col.count()}\n")

    # Bind retrieve functions
    def r_ctx(q):  return retrieve_minilm(ctx_col, minilm, q, TOP_K)
    def r_pdr(q):  return retrieve_parent(small_col, parent_col, minilm, q, TOP_K)
    def r_late(q): return retrieve_late(late_col, jina_tok, jina_mod, q, TOP_K)
    def r_mv(q):   return retrieve_multi_vector(summary_col, ft_col, minilm, jina_tok, jina_mod, q, TOP_K)

    print("3. Building answer datasets …")
    print("   D1 contextual …")
    d1_ds = build_dataset(r_ctx,  llm_client, "D1-Ctx")
    print("\n   D2 parent-document …")
    d2_ds = build_dataset(r_pdr,  llm_client, "D2-PDR")
    print("\n   D3 late chunking …")
    d3_ds = build_dataset(r_late, llm_client, "D3-Late")
    print("\n   D4 multi-vector …")
    d4_ds = build_dataset(r_mv,   llm_client, "D4-MV")

    print("\n4. Running RAGAS (4 × ~24 LLM calls) …")
    print("   D1 …")
    d1_scores = run_ragas(d1_ds, ragas_llm, ragas_embed)
    print("   D2 …")
    d2_scores = run_ragas(d2_ds, ragas_llm, ragas_embed)
    print("   D3 …")
    d3_scores = run_ragas(d3_ds, ragas_llm, ragas_embed)
    print("   D4 …")
    d4_scores = run_ragas(d4_ds, ragas_llm, ragas_embed)

    print_comparison({
        "D1-Ctx":  d1_scores,
        "D2-PDR":  d2_scores,
        "D3-Late": d3_scores,
        "D4-MV":   d4_scores,
    })

    print("=== Done. Run Day 5 (ab_eval.py) to build the automated harness. ===\n")


if __name__ == "__main__":
    main()