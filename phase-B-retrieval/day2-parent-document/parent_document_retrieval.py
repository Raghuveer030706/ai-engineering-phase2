"""
Phase B Day 2 — Parent-Document Retrieval

Architecture:
  - Small chunks (128 tokens / ~512 chars) indexed for precise search
  - Parent chunks (512 tokens / ~2048 chars) stored for retrieval
  - Each small chunk metadata contains parent_id
  - Query flow: embed question → search small chunks → fetch parent chunk → return parent

Why this works:
  Small chunks have tighter, more specific embeddings — better cosine match.
  But small chunks often cut off mid-thought. The parent gives the full passage.
  Result: precision of small-chunk search + context richness of large-chunk retrieval.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import pypdf

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH              = Path("../../data/attention-is-all-you-need.pdf")
CHROMA_PATH           = Path("../day1-contextual-retrieval/chroma_db")
SMALL_COLLECTION      = "small_chunks_attention"
PARENT_COLLECTION     = "parent_chunks_attention"
EMBED_MODEL           = "all-MiniLM-L6-v2"

PARENT_CHUNK_SIZE     = 512    # tokens approx
PARENT_OVERLAP        = 100
SMALL_CHUNK_SIZE      = 128    # tokens approx
SMALL_OVERLAP         = 20

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    pages  = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
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


def split_parent_into_small(parent: str, small_size: int, small_overlap: int) -> list[str]:
    """Split a single parent chunk into small chunks."""
    return chunk_text(parent, small_size, small_overlap)


def build_collections(
    parent_chunks: list[str],
    model: SentenceTransformer,
    client: chromadb.PersistentClient,
) -> dict:
    # ── Drop and recreate both collections ────────────────────────────────────
    for name in [SMALL_COLLECTION, PARENT_COLLECTION]:
        try:
            client.delete_collection(name)
            print(f"  Deleted existing '{name}'")
        except Exception:
            pass

    small_col  = client.create_collection(
        name=SMALL_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    parent_col = client.create_collection(
        name=PARENT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Index parent chunks (no embedding needed — fetched by ID) ─────────────
    print(f"\n  Indexing {len(parent_chunks)} parent chunks …")
    parent_ids       = [f"parent_{i}" for i in range(len(parent_chunks))]
    parent_metadatas = [
        {"chunk_index": i, "source": "attention-is-all-you-need", "type": "parent"}
        for i in range(len(parent_chunks))
    ]
    # Parent collection stores text + dummy embedding (we fetch by ID, not search)
    parent_col.add(
    ids=parent_ids,
    documents=parent_chunks,
    metadatas=parent_metadatas,
    )   
    print(f"  ✓ {len(parent_chunks)} parent chunks indexed")

    # ── Build and index small chunks ──────────────────────────────────────────
    all_small_chunks  = []
    all_small_ids     = []
    all_small_meta    = []

    small_id_counter = 0
    for parent_idx, parent in enumerate(parent_chunks):
        small_chunks = split_parent_into_small(parent, SMALL_CHUNK_SIZE, SMALL_OVERLAP)
        for small in small_chunks:
            if not small.strip():
                continue
            all_small_chunks.append(small)
            all_small_ids.append(f"small_{small_id_counter}")
            all_small_meta.append({
                "parent_id":   f"parent_{parent_idx}",
                "parent_index": parent_idx,
                "source":      "attention-is-all-you-need",
                "type":        "small",
            })
            small_id_counter += 1

    print(f"\n  Embedding {len(all_small_chunks)} small chunks …")
    small_embeddings = model.encode(all_small_chunks, show_progress_bar=True, batch_size=32)

    small_col.add(
        ids=all_small_ids,
        embeddings=small_embeddings.tolist(),
        documents=all_small_chunks,
        metadatas=all_small_meta,
    )
    print(f"  ✓ {len(all_small_chunks)} small chunks indexed")

    return {
        "parent_count": len(parent_chunks),
        "small_count":  len(all_small_chunks),
        "ratio":        len(all_small_chunks) / len(parent_chunks),
    }


def retrieve_via_parent(
    small_col,
    parent_col,
    embed_model: SentenceTransformer,
    question: str,
    top_k: int = 5,
) -> list[str]:
    """
    Query flow:
      1. Embed question
      2. Search small_chunks_attention for top_k matches
      3. Extract parent_id from each match metadata
      4. Deduplicate parent_ids (multiple small chunks may map to same parent)
      5. Fetch parent chunks by ID
      6. Return parent chunk texts
    """
    embedding = embed_model.encode([question])[0].tolist()
    results   = small_col.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas", "documents"],
    )

    # Deduplicate parent IDs — preserve order
    seen_parent_ids = []
    for meta in results["metadatas"][0]:
        pid = meta["parent_id"]
        if pid not in seen_parent_ids:
            seen_parent_ids.append(pid)

    # Fetch parents by ID
    if not seen_parent_ids:
        return []

    parent_results = parent_col.get(
        ids=seen_parent_ids,
        include=["documents"],
    )
    return parent_results["documents"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    print("=== Phase B Day 2 — Parent-Document Retrieval ===\n")

    print("1. Extracting text …")
    full_text = extract_text(PDF_PATH)
    print(f"   {len(full_text):,} characters")

    print("\n2. Chunking into parents …")
    parent_chunks = chunk_text(full_text, PARENT_CHUNK_SIZE, PARENT_OVERLAP)
    print(f"   {len(parent_chunks)} parent chunks  (size≈{PARENT_CHUNK_SIZE} tokens)")

    # Show sample parent → small split
    sample_smalls = split_parent_into_small(parent_chunks[0], SMALL_CHUNK_SIZE, SMALL_OVERLAP)
    print(f"\n   Sample: parent[0] ({len(parent_chunks[0])} chars) → {len(sample_smalls)} small chunks")
    print(f"   Small[0] preview: {sample_smalls[0][:150]!r}")

    print("\n3. Loading embedding model …")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"   {EMBED_MODEL}")

    print("\n4. Building ChromaDB collections …")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    stats  = build_collections(parent_chunks, model, chroma)

    print(f"\n=== Summary ===")
    print(f"  Parent chunks : {stats['parent_count']}")
    print(f"  Small chunks  : {stats['small_count']}")
    print(f"  Ratio         : {stats['ratio']:.1f}x small per parent")

    # Smoke test — one retrieval
    print("\n5. Smoke test retrieval …")
    small_col  = chroma.get_collection(SMALL_COLLECTION)
    parent_col = chroma.get_collection(PARENT_COLLECTION)
    test_q     = "How does multi-head attention work?"
    parents    = retrieve_via_parent(small_col, parent_col, model, test_q, top_k=3)
    print(f"   Query: {test_q!r}")
    print(f"   Returned {len(parents)} parent chunks")
    print(f"   Parent[0] length: {len(parents[0])} chars")
    print(f"   Parent[0] preview: {parents[0][:200]!r}")

    print("\n=== Done. Run eval_day2.py to compare vs Day 1. ===")


if __name__ == "__main__":
    main()