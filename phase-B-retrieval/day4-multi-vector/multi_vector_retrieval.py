"""
Phase B Day 4 — Multi-Vector Retrieval

One chunk, two embeddings:
  1. Summary embedding  — MiniLM on Claude-generated summary (precise, searchable)
  2. Full-text embedding — Jina on raw chunk text (contextual, faithful)

Two ChromaDB collections, same chunk_index metadata:
  summary_vectors_attention  — MiniLM embeddings of summaries
  fulltext_vectors_attention — Jina embeddings of raw chunks

Query flow:
  embed question with MiniLM → search summary collection   → candidate chunk_ids
  embed question with Jina   → search fulltext collection  → candidate chunk_ids
  deduplicate by chunk_index → fetch raw chunk texts       → return top-K unique

Why this beats either alone:
  - Summary vectors catch concept-level queries ("what is multi-head attention")
  - Full-text vectors catch detail-level queries ("28.4 BLEU on WMT 2014")
  - Union of both surfaces more of the right chunks than either alone
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import anthropic
import pypdf

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH            = Path("../../data/attention-is-all-you-need.pdf")
CHROMA_PATH         = Path("../day1-contextual-retrieval/chroma_db")
SUMMARY_COLLECTION  = "summary_vectors_attention"
FULLTEXT_COLLECTION = "fulltext_vectors_attention"

MINILM_MODEL        = "all-MiniLM-L6-v2"
JINA_MODEL          = "jinaai/jina-embeddings-v2-base-en"
LLM_MODEL           = "claude-haiku-4-5-20251001"

CHUNK_SIZE          = 400   # tokens approx — same as Day 1 for consistency
CHUNK_OVERLAP       = 80
RATE_DELAY          = 0.3   # seconds between Claude calls

# ── Text helpers ──────────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    pages  = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    char_size    = chunk_size * 4
    char_overlap = overlap * 4
    chunks = []
    start  = 0
    while start < len(text):
        end   = start + char_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += char_size - char_overlap
    return chunks

# ── Summary generation (same prompt as Day 1) ─────────────────────────────────

def get_doc_window(full_text: str, chunk: str, window_chars: int = 3000) -> str:
    pos  = full_text.find(chunk[:100])
    if pos == -1:
        return full_text[:window_chars]
    half  = window_chars // 2
    start = max(0, pos - half)
    end   = min(len(full_text), pos + half)
    return full_text[start:end]


def generate_summary(client: anthropic.Anthropic, full_text: str, chunk: str) -> str:
    doc_window = get_doc_window(full_text, chunk)
    prompt = f"""Here is a document excerpt:
<document>
{doc_window}
</document>

Here is a specific chunk from that document:
<chunk>
{chunk}
</chunk>

Write 1-2 sentences describing what this chunk is about and where it fits
in the document. Be specific — mention the section, concept, or argument.
Do NOT repeat the chunk verbatim. Output ONLY the sentences, nothing else."""

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

# ── Jina embedding helper ─────────────────────────────────────────────────────

def embed_with_jina(tokenizer, model, text: str) -> np.ndarray:
    """Embed a single text with Jina, mean-pool, L2-normalise."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=8000,
        padding=False,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    token_embs = outputs.last_hidden_state[0]
    pooled     = token_embs.mean(dim=0)
    normed     = (pooled / (pooled.norm() + 1e-9)).cpu().numpy()
    return normed

# ── Collection builder ────────────────────────────────────────────────────────

def build_multi_vector_collections(
    chunks: list[str],
    full_text: str,
    llm_client: anthropic.Anthropic,
    minilm: SentenceTransformer,
    jina_tokenizer,
    jina_model,
    chroma_client: chromadb.PersistentClient,
) -> dict:

    # Drop and recreate
    for name in [SUMMARY_COLLECTION, FULLTEXT_COLLECTION]:
        try:
            chroma_client.delete_collection(name)
            print(f"  Deleted existing '{name}'")
        except Exception:
            pass

    summary_col  = chroma_client.create_collection(
        name=SUMMARY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    fulltext_col = chroma_client.create_collection(
        name=FULLTEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    summaries         = []
    summary_embeddings = []
    fulltext_embeddings = []
    failed            = 0

    print(f"\n  Processing {len(chunks)} chunks …")
    for i, chunk in enumerate(chunks):
        # 1. Generate summary via Claude
        try:
            summary = generate_summary(llm_client, full_text, chunk)
        except Exception as e:
            print(f"    [WARN] chunk {i} summary failed: {e} — using chunk[:120]")
            summary = chunk[:120]
            failed += 1

        summaries.append(summary)

        # 2. Embed summary with MiniLM
        s_emb = minilm.encode([summary])[0]
        summary_embeddings.append(s_emb.tolist())

        # 3. Embed full chunk with Jina
        j_emb = embed_with_jina(jina_tokenizer, jina_model, chunk)
        fulltext_embeddings.append(j_emb.tolist())

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(chunks)} done …")

        time.sleep(RATE_DELAY)

    print(f"  Summary generation complete. Failures: {failed}/{len(chunks)}")

    # Build shared metadata + ids
    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    meta_base = [
        {
            "chunk_index": i,
            "source":      "attention-is-all-you-need",
        }
        for i in range(len(chunks))
    ]

    # Index summary collection — stores summary text + MiniLM embedding
    summary_meta = [{**m, "vector_type": "summary"} for m in meta_base]
    summary_col.add(
        ids=ids,
        embeddings=summary_embeddings,
        documents=summaries,        # store summary text for inspection
        metadatas=summary_meta,
    )
    print(f"  ✓ {len(chunks)} summary vectors indexed into '{SUMMARY_COLLECTION}'")

    # Index fulltext collection — stores raw chunk text + Jina embedding
    fulltext_meta = [{**m, "vector_type": "fulltext"} for m in meta_base]
    fulltext_col.add(
        ids=ids,
        embeddings=fulltext_embeddings,
        documents=chunks,           # store raw chunk text — this is what gets returned
        metadatas=fulltext_meta,
    )
    print(f"  ✓ {len(chunks)} fulltext vectors indexed into '{FULLTEXT_COLLECTION}'")

    return {
        "chunks":         len(chunks),
        "failed_summary": failed,
        "summary_dim":    len(summary_embeddings[0]) if summary_embeddings else 0,
        "fulltext_dim":   len(fulltext_embeddings[0]) if fulltext_embeddings else 0,
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_multi_vector(
    summary_col,
    fulltext_col,
    minilm: SentenceTransformer,
    jina_tokenizer,
    jina_model,
    question: str,
    top_k: int = 5,
) -> list[str]:
    """
    Search both collections, union chunk_ids (dedup by chunk_index),
    fetch raw chunk texts from fulltext collection.
    Returns up to top_k unique chunks.
    """
    # Summary search (MiniLM)
    s_emb     = minilm.encode([question])[0].tolist()
    s_results = summary_col.query(
        query_embeddings=[s_emb],
        n_results=top_k,
        include=["metadatas"],
    )

    # Fulltext search (Jina)
    j_emb     = embed_with_jina(jina_tokenizer, jina_model, question).tolist()
    j_results = fulltext_col.query(
        query_embeddings=[j_emb],
        n_results=top_k,
        include=["metadatas"],
    )

    # Union of chunk ids — preserve order, summary hits first
    seen_ids = []
    for meta in s_results["metadatas"][0]:
        cid = f"chunk_{meta['chunk_index']}"
        if cid not in seen_ids:
            seen_ids.append(cid)
    for meta in j_results["metadatas"][0]:
        cid = f"chunk_{meta['chunk_index']}"
        if cid not in seen_ids:
            seen_ids.append(cid)

    # Keep top_k unique
    seen_ids = seen_ids[:top_k]

    # Fetch raw chunk texts from fulltext collection
    fetched = fulltext_col.get(ids=seen_ids, include=["documents"])
    return fetched["documents"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    print("=== Phase B Day 4 — Multi-Vector Retrieval ===\n")

    print("1. Extracting text …")
    full_text = extract_text(PDF_PATH)
    print(f"   {len(full_text):,} characters")

    print("\n2. Chunking …")
    chunks = chunk_text(full_text)
    print(f"   {len(chunks)} chunks  (size≈{CHUNK_SIZE} tokens)")

    print("\n3. Loading models …")
    minilm = SentenceTransformer(MINILM_MODEL)
    print(f"   MiniLM loaded : {MINILM_MODEL}")

    jina_tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    jina_model_obj = AutoModel.from_pretrained(JINA_MODEL, trust_remote_code=True)
    jina_model_obj.eval()
    print(f"   Jina loaded   : {JINA_MODEL}")

    llm_client = anthropic.Anthropic(api_key=api_key)
    print(f"   LLM           : {LLM_MODEL}")

    # Sample — show both vectors for chunk 0
    print("\n4. Sample — chunk 0 vectors …")
    sample_summary  = generate_summary(llm_client, full_text, chunks[0])
    sample_s_emb    = minilm.encode([sample_summary])[0]
    sample_j_emb    = embed_with_jina(jina_tokenizer, jina_model_obj, chunks[0])
    print(f"   Summary  : {sample_summary!r}")
    print(f"   MiniLM embedding dim : {sample_s_emb.shape[0]}  (summary vector)")
    print(f"   Jina embedding dim   : {sample_j_emb.shape[0]}  (fulltext vector)")

    print("\n5. Building multi-vector collections …")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    stats  = build_multi_vector_collections(
        chunks, full_text, llm_client,
        minilm, jina_tokenizer, jina_model_obj,
        chroma,
    )

    print(f"\n=== Summary ===")
    print(f"  Chunks indexed   : {stats['chunks']}")
    print(f"  Failed summaries : {stats['failed_summary']}")
    print(f"  Summary emb dim  : {stats['summary_dim']}  (MiniLM)")
    print(f"  Fulltext emb dim : {stats['fulltext_dim']}  (Jina)")

    # Smoke test
    print("\n6. Smoke test retrieval …")
    summary_col  = chroma.get_collection(SUMMARY_COLLECTION)
    fulltext_col = chroma.get_collection(FULLTEXT_COLLECTION)
    test_q       = "What BLEU score did the Transformer achieve?"
    results      = retrieve_multi_vector(
        summary_col, fulltext_col,
        minilm, jina_tokenizer, jina_model_obj,
        test_q, top_k=3,
    )
    print(f"   Query: {test_q!r}")
    print(f"   Returned {len(results)} chunks")
    print(f"   Chunk[0] preview: {results[0][:200]!r}")

    print("\n=== Done. Run eval_day4.py to compare all four strategies. ===")


if __name__ == "__main__":
    main()