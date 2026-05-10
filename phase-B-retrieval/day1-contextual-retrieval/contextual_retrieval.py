"""
Phase B Day 1 — Contextual Retrieval
For every chunk:
  1. Send (full_doc_excerpt + chunk) to Claude
  2. Get a 1-2 sentence context summary
  3. Prepend: "<context>\n{summary}\n</context>\n\n{chunk}"
  4. Embed the enriched chunk
  5. Index into ChromaDB collection: 'contextual_attention'

This is Anthropic's own technique from their Nov 2024 blog post.
The key insight: isolated chunks lose referential context.
Prepending context fixes "it increased by 20%" → the chunk now
knows what "it" refers to.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
import pypdf

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH          = Path("../../data/attention-is-all-you-need.pdf")
CHROMA_PATH       = Path("chroma_db")
COLLECTION        = "contextual_attention"
EMBED_MODEL       = "all-MiniLM-L6-v2"
LLM_MODEL         = "claude-haiku-4-5-20251001"
CHUNK_SIZE        = 400
CHUNK_OVERLAP     = 80
# Number of chars from the doc to include as context window for Claude.
# Enough to give section-level context without blowing up token costs.
DOC_CONTEXT_CHARS = 3000
# Delay between Claude calls (seconds) to avoid rate-limit bursts.
RATE_DELAY        = 0.3

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def get_doc_context_window(full_text: str, chunk: str, window_chars: int = DOC_CONTEXT_CHARS) -> str:
    """
    Return the portion of the full document surrounding the chunk.
    Gives Claude enough surrounding text to write a meaningful context summary.
    Uses the position of the chunk in the doc; falls back to doc start if not found.
    """
    pos = full_text.find(chunk[:100])   # find by first 100 chars of chunk
    if pos == -1:
        return full_text[:window_chars]
    half = window_chars // 2
    start = max(0, pos - half)
    end   = min(len(full_text), pos + half)
    return full_text[start:end]


def generate_context(client: anthropic.Anthropic, full_text: str, chunk: str) -> str:
    """
    Ask Claude to write a short context description for a chunk.
    Returns the context string (1-2 sentences).
    """
    doc_window = get_doc_context_window(full_text, chunk)

    prompt = f"""Here is a document excerpt:
<document>
{doc_window}
</document>

Here is a specific chunk from that document that will be embedded and retrieved:
<chunk>
{chunk}
</chunk>

Write 1-2 sentences that describe what this chunk is about and where it fits
in the document. Be specific — mention the section, concept, or argument this
chunk belongs to. Do NOT repeat the chunk verbatim. Do NOT add commentary.
Output ONLY the context sentences, nothing else."""

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def prepend_context(context: str, chunk: str) -> str:
    return f"<context>\n{context}\n</context>\n\n{chunk}"


def build_contextual_collection(
    chunks: list[str],
    full_text: str,
    llm_client: anthropic.Anthropic,
    embed_model: SentenceTransformer,
) -> dict:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION)
        print(f"  Deleted existing '{COLLECTION}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    enriched_chunks  = []
    context_tokens   = 0
    failed           = 0

    print(f"  Generating context for {len(chunks)} chunks via Claude …")
    for i, chunk in enumerate(chunks):
        try:
            context = generate_context(llm_client, full_text, chunk)
            enriched = prepend_context(context, chunk)
            enriched_chunks.append(enriched)
            # rough token count from response (each word ≈ 1.3 tokens)
            context_tokens += len(context.split()) * 13 // 10
        except Exception as e:
            print(f"    [WARN] chunk {i} context failed: {e} — using raw chunk")
            enriched_chunks.append(chunk)
            failed += 1

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(chunks)} done …")

        time.sleep(RATE_DELAY)

    print(f"\n  Context generation complete.")
    print(f"  Failed: {failed}/{len(chunks)}")
    print(f"  Approx context tokens generated: {context_tokens:,}")

    print(f"\n  Embedding {len(enriched_chunks)} enriched chunks …")
    embeddings = embed_model.encode(enriched_chunks, show_progress_bar=True, batch_size=32)

    ids       = [f"chunk_{i}" for i in range(len(enriched_chunks))]
    metadatas = [
        {
            "chunk_index": i,
            "source": "attention-is-all-you-need",
            "has_context": True,
        }
        for i in range(len(enriched_chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=enriched_chunks,
        metadatas=metadatas,
    )

    print(f"  ✓ Indexed {len(enriched_chunks)} contextual chunks into '{COLLECTION}'")

    return {
        "total_chunks": len(chunks),
        "failed_context": failed,
        "approx_context_tokens": context_tokens,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    print("=== Phase B Day 1 — Contextual Retrieval ===\n")

    print("1. Extracting text …")
    full_text = extract_text(PDF_PATH)
    print(f"   {len(full_text):,} characters")

    print("\n2. Chunking …")
    chunks = chunk_text(full_text)
    print(f"   {len(chunks)} chunks")

    print("\n3. Loading models …")
    embed_model = SentenceTransformer(EMBED_MODEL)
    llm_client  = anthropic.Anthropic(api_key=api_key)
    print(f"   Embed: {EMBED_MODEL}")
    print(f"   LLM:   {LLM_MODEL}")

    # Print a sample context before full run so you can sanity-check
    print("\n4. Sample context generation (chunk 0) …")
    sample_context  = generate_context(llm_client, full_text, chunks[0])
    sample_enriched = prepend_context(sample_context, chunks[0])
    print(f"   Context:  {sample_context!r}")
    print(f"   Enriched chunk preview:\n   {sample_enriched[:300]!r}\n")

    print("5. Building contextual collection …")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    stats = build_contextual_collection(chunks, full_text, llm_client, embed_model)

    print("\n=== Summary ===")
    print(f"  Chunks processed : {stats['total_chunks']}")
    print(f"  Context failures : {stats['failed_context']}")
    print(f"  Approx ctx tokens: {stats['approx_context_tokens']:,}")
    print("\nRun eval.py to compare baseline vs contextual.")


if __name__ == "__main__":
    main()