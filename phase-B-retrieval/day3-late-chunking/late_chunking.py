"""
Phase B Day 3 — Late Chunking

Standard chunking: split text → embed each chunk independently
Late chunking:     embed full document → split embeddings post-hoc

Why this matters:
  With standard chunking, a chunk saying "This mechanism" has no idea
  what "This" refers to — the reference lives in the previous chunk.
  The embedding is blind to it.

  With late chunking, the full document is processed by the transformer
  in one pass. Every token's representation attends to every other token.
  When we slice the resulting embeddings into chunks, each slice already
  carries the full document's context baked in.

Model: jinaai/jina-embeddings-v2-base-en
  - 8192 token context window (vs 256 for all-MiniLM-L6-v2)
  - Designed for late chunking use case
  - Mean pooling over token embeddings per chunk span

Architecture:
  1. Tokenize full document
  2. Run transformer forward pass → token embeddings (one per token)
  3. Define chunk spans (by token count)
  4. Mean-pool token embeddings within each span → one embedding per chunk
  5. Store (chunk_text, late_embedding) in ChromaDB
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import chromadb
import pypdf
import torch
from transformers import AutoTokenizer, AutoModel

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH         = Path("../../data/attention-is-all-you-need.pdf")
CHROMA_PATH      = Path("../day1-contextual-retrieval/chroma_db")
COLLECTION       = "late_chunking_attention"
LATE_MODEL_NAME  = "jinaai/jina-embeddings-v2-base-en"

# Chunk span size in tokens (model tokens, not word tokens)
# Smaller than previous days — late chunking compensates for small size
# via full-document context already in the embedding
CHUNK_TOKEN_SIZE = 256
CHUNK_OVERLAP    = 32   # token overlap between spans

# Max tokens the model can handle in one pass
# jina-v2 supports 8192 — we'll process in segments if doc exceeds this
MAX_MODEL_TOKENS = 8000  # leave headroom for special tokens

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    pages  = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def mean_pool(token_embeddings: torch.Tensor, span_start: int, span_end: int) -> np.ndarray:
    """Mean pool token embeddings over [span_start, span_end)."""
    span = token_embeddings[span_start:span_end]   # (span_len, hidden_dim)
    pooled = span.mean(dim=0)                       # (hidden_dim,)
    # L2 normalise — cosine search needs unit vectors
    normed = pooled / (pooled.norm() + 1e-9)
    return normed.cpu().numpy()


def get_chunk_spans(n_tokens: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """
    Return list of (start, end) token index spans.
    Each span is [start, end) — end is exclusive.
    """
    spans = []
    start = 0
    while start < n_tokens:
        end = min(start + chunk_size, n_tokens)
        spans.append((start, end))
        if end == n_tokens:
            break
        start += chunk_size - overlap
    return spans


def tokens_to_text(tokenizer, input_ids: torch.Tensor, span_start: int, span_end: int) -> str:
    """Decode a token span back to text, skipping special tokens."""
    span_ids = input_ids[span_start:span_end]
    return tokenizer.decode(span_ids, skip_special_tokens=True).strip()


def embed_segment_late_chunking(
    tokenizer,
    model,
    text_segment: str,
    chunk_token_size: int,
    chunk_overlap: int,
) -> list[tuple[str, np.ndarray]]:
    """
    Run late chunking on one text segment (fits within MAX_MODEL_TOKENS).
    Returns list of (chunk_text, embedding) pairs.
    """
    inputs = tokenizer(
        text_segment,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_TOKENS,
        padding=False,
    )

    input_ids  = inputs["input_ids"][0]   # (seq_len,)
    n_tokens   = len(input_ids)

    # Forward pass — get all token embeddings in one shot
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state: (1, seq_len, hidden_dim)
    token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

    # Skip [CLS] token at position 0 for span calculations
    # (special token — not part of content)
    content_start = 1
    content_end   = n_tokens - 1  # skip [SEP] at end if present

    spans = get_chunk_spans(
        content_end - content_start,
        chunk_token_size,
        chunk_overlap,
    )

    results = []
    for span_start_rel, span_end_rel in spans:
        abs_start = content_start + span_start_rel
        abs_end   = content_start + span_end_rel

        chunk_text = tokens_to_text(tokenizer, input_ids, abs_start, abs_end)
        if not chunk_text.strip():
            continue

        embedding = mean_pool(token_embeddings, abs_start, abs_end)
        results.append((chunk_text, embedding))

    return results


def build_late_chunking_collection(
    full_text: str,
    tokenizer,
    model,
    chroma_client: chromadb.PersistentClient,
) -> dict:
    # Drop and recreate
    try:
        chroma_client.delete_collection(COLLECTION)
        print(f"  Deleted existing '{COLLECTION}'")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Check if document fits in one model pass
    test_tokens = tokenizer(full_text, return_tensors="pt", truncation=False)
    total_tokens = test_tokens["input_ids"].shape[1]
    print(f"  Document tokens: {total_tokens:,}  (model max: {MAX_MODEL_TOKENS})")

    if total_tokens <= MAX_MODEL_TOKENS:
        # Single pass — ideal case, maximum context
        print("  Strategy: single-pass (full document context)")
        segments = [full_text]
    else:
        # Split into overlapping text segments, each within model limit
        # Use character-based split as approximation (4 chars ≈ 1 token)
        print(f"  Strategy: multi-segment (document exceeds {MAX_MODEL_TOKENS} tokens)")
        seg_char_size    = MAX_MODEL_TOKENS * 4
        seg_char_overlap = CHUNK_OVERLAP * 4 * 4   # generous overlap between segments
        segments = []
        start = 0
        while start < len(full_text):
            end = start + seg_char_size
            segments.append(full_text[start:end])
            if end >= len(full_text):
                break
            start += seg_char_size - seg_char_overlap
        print(f"  Segments: {len(segments)}")

    all_texts      = []
    all_embeddings = []

    for seg_idx, segment in enumerate(segments):
        print(f"  Processing segment {seg_idx + 1}/{len(segments)} …")
        pairs = embed_segment_late_chunking(
            tokenizer, model, segment, CHUNK_TOKEN_SIZE, CHUNK_OVERLAP
        )
        for text, emb in pairs:
            all_texts.append(text)
            all_embeddings.append(emb)

    print(f"  Total chunks produced: {len(all_texts)}")

    # Index into ChromaDB
    ids       = [f"late_{i}" for i in range(len(all_texts))]
    metadatas = [
        {
            "chunk_index": i,
            "source":      "attention-is-all-you-need",
            "strategy":    "late_chunking",
        }
        for i in range(len(all_texts))
    ]

    collection.add(
        ids=ids,
        embeddings=[e.tolist() for e in all_embeddings],
        documents=all_texts,
        metadatas=metadatas,
    )
    print(f"  ✓ Indexed {len(all_texts)} late-chunked embeddings into '{COLLECTION}'")

    return {
        "total_tokens":   total_tokens,
        "segments":       len(segments),
        "chunks_indexed": len(all_texts),
        "embedding_dim":  all_embeddings[0].shape[0] if all_embeddings else 0,
    }


def retrieve_late(
    collection,
    tokenizer,
    model,
    question: str,
    top_k: int = 5,
) -> list[str]:
    """
    Embed question using the same jina model (single sentence — no late chunking needed).
    Search against late-chunked collection.
    """
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pool all tokens (question is short — no chunking)
    token_embs = outputs.last_hidden_state[0]
    pooled     = token_embs.mean(dim=0)
    normed     = (pooled / (pooled.norm() + 1e-9)).cpu().numpy()

    results = collection.query(
        query_embeddings=[normed.tolist()],
        n_results=top_k,
    )
    return results["documents"][0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    print("=== Phase B Day 3 — Late Chunking ===\n")

    print("1. Extracting text …")
    full_text = extract_text(PDF_PATH)
    print(f"   {len(full_text):,} characters")

    print(f"\n2. Loading {LATE_MODEL_NAME} …")
    print("   (First run downloads ~500MB — subsequent runs use cache)")
    tokenizer = AutoTokenizer.from_pretrained(
        LATE_MODEL_NAME,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        LATE_MODEL_NAME,
        trust_remote_code=True,
    )
    model.eval()
    print("   Model loaded")

    print("\n3. Building late-chunking ChromaDB collection …")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    stats  = build_late_chunking_collection(full_text, tokenizer, model, chroma)

    print(f"\n=== Summary ===")
    print(f"  Document tokens : {stats['total_tokens']:,}")
    print(f"  Segments        : {stats['segments']}")
    print(f"  Chunks indexed  : {stats['chunks_indexed']}")
    print(f"  Embedding dim   : {stats['embedding_dim']}")

    # Smoke test
    print("\n4. Smoke test retrieval …")
    late_col = chroma.get_collection(COLLECTION)
    test_q   = "How does scaled dot-product attention work?"
    results  = retrieve_late(late_col, tokenizer, model, test_q, top_k=3)
    print(f"   Query: {test_q!r}")
    print(f"   Top chunk preview: {results[0][:200]!r}")

    print("\n=== Done. Run eval_day3.py to compare vs Day 1 and Day 2. ===")


if __name__ == "__main__":
    main()