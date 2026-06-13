"""
chunk_and_embed_patch.py
========================
INSTRUCTIONS: This file shows exactly what to ADD to your existing
C:\Users\Ezeking\PromptForge\src\processing\chunk_and_embed.py

Do NOT replace chunk_and_embed.py. Open it in Notepad, find the
main() function, and add the JSONL loader block shown below.

The book ingestor writes:  data/book_all_chunks.jsonl
chunk_and_embed.py reads:  all normal sources PLUS that JSONL file.

─────────────────────────────────────────────────────────────────────
STEP 1: Add this import near the top of chunk_and_embed.py
(after existing imports, before constants)
─────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

# Path where BookIngestor writes its merged output
BOOKS_JSONL_PATH: Path = Path("data") / "book_all_chunks.jsonl"

─────────────────────────────────────────────────────────────────────
STEP 2: Add this function to chunk_and_embed.py
(paste it before the main() function)
─────────────────────────────────────────────────────────────────────
"""

# ═══════════════════════════════════════════════════════════════════
# PASTE THIS FUNCTION INTO chunk_and_embed.py
# ═══════════════════════════════════════════════════════════════════

def load_book_chunks_into_collection(collection, jsonl_path: "Path") -> int:
    """
    Load pre-chunked book data from BookIngestor JSONL output into ChromaDB.

    Books are pre-chunked by ingest_books.py (which uses the shared ForgeCore
    chunker). This function embeds them and upserts into the collection,
    exactly as chunk_and_embed.py does for all other sources.

    Args:
        collection: ChromaDB collection object (already open).
        jsonl_path: Path to data/book_all_chunks.jsonl

    Returns:
        Number of chunks added/updated.
    """
    import json
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    if not jsonl_path.exists():
        print(f"[books] No JSONL found at {jsonl_path} — skipping book ingestion.")
        return 0

    # Load embedding model (same model as rest of pipeline)
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    print(f"[books] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    if not records:
        print("[books] JSONL file is empty — nothing to embed.")
        return 0

    print(f"[books] Embedding {len(records)} book chunks...")

    # Process in batches (matches existing BATCH_SIZE constant in chunk_and_embed.py)
    BATCH_SIZE = 32  # must match existing pipeline constant
    total_upserted = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        ids = [r["id"] for r in batch]
        metas = [r["meta"] for r in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )
        total_upserted += len(batch)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  [books] {total_upserted}/{len(records)} chunks embedded...")

    print(f"[books] ✅ {total_upserted} book chunks upserted into collection.")
    return total_upserted


# ═══════════════════════════════════════════════════════════════════
# STEP 3: In the main() function of chunk_and_embed.py, add this
# call AFTER the existing embedding loop, BEFORE compress/upload:
# ═══════════════════════════════════════════════════════════════════
"""
    # ── Load pre-chunked book data from BookIngestor ──────────────
    books_jsonl = Path("data") / "book_all_chunks.jsonl"
    book_chunks_added = load_book_chunks_into_collection(collection, books_jsonl)
    print(f"Books added to corpus: {book_chunks_added}")
"""

# ═══════════════════════════════════════════════════════════════════
# That is the complete patch. Three things total:
#   1. Two imports at the top (json, Path — likely already there)
#   2. The BOOKS_JSONL_PATH constant
#   3. The load_book_chunks_into_collection() function
#   4. One function call inside main()
# ═══════════════════════════════════════════════════════════════════
