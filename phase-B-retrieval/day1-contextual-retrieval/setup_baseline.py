"""
Phase B Day 1 — Baseline collection setup
Chunks the PDF using semantic splitting, embeds with all-MiniLM-L6-v2,
indexes into ChromaDB collection: 'baseline_attention'
No context prepended — this is the "before" state.
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
PDF_PATH       = Path("../../data/attention-is-all-you-need.pdf")
CHROMA_PATH    = Path("chroma_db")
COLLECTION     = "baseline_attention"
EMBED_MODEL    = "all-MiniLM-L6-v2"
CHUNK_SIZE     = 400    # tokens (approx chars / 4)
CHUNK_OVERLAP  = 80

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Simple fixed-size character chunking with overlap.
    chunk_size and overlap are in approximate tokens (chars/4).
    Convert to chars for slicing.
    """
    char_size    = chunk_size * 4
    char_overlap = overlap * 4
    chunks = []
    start  = 0
    while start < len(text):
        end = start + char_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += char_size - char_overlap
    return chunks


def build_baseline_collection(chunks: list[str], model: SentenceTransformer) -> None:
    client     = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Drop and recreate for clean state
    try:
        client.delete_collection(COLLECTION)
        print(f"  Deleted existing '{COLLECTION}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"  Embedding {len(chunks)} chunks …")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)

    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_index": i, "source": "attention-is-all-you-need"} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=metadatas,
    )
    print(f"  ✓ Indexed {len(chunks)} chunks into '{COLLECTION}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Copy attention-is-all-you-need.pdf into the data/ folder.")
        sys.exit(1)

    print("=== Phase B Day 1 — Baseline Setup ===\n")

    print("1. Extracting text from PDF …")
    text = extract_text(PDF_PATH)
    print(f"   Extracted {len(text):,} characters")

    print("\n2. Chunking …")
    chunks = chunk_text(text)
    print(f"   Created {len(chunks)} chunks  (size≈{CHUNK_SIZE} tokens, overlap≈{CHUNK_OVERLAP})")
    print(f"   Sample chunk[0][:200]:\n   {chunks[0][:200]!r}\n")

    print("3. Loading embedding model …")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"   Model: {EMBED_MODEL}")

    print("\n4. Building baseline ChromaDB collection …")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    build_baseline_collection(chunks, model)

    print("\n=== Baseline ready. Run contextual_retrieval.py next. ===")


if __name__ == "__main__":
    main()