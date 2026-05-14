"""
patch_chunk_embed.py
====================
Run this once from C:\\Users\\Ezeking\\PromptForge to overwrite
src/processing/chunk_and_embed.py with the corrected version.

    C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe patch_chunk_embed.py
"""

import os

TARGET = os.path.join("src", "processing", "chunk_and_embed.py")

NEW_CONTENT = '''"""
chunk_and_embed.py — PromptForge Layer 2 + Layer 3
===================================================
Reads raw JSON files from data/raw/, chunks them into sentence-aware
segments, embeds with bge-large-en-v1.5, and stores in ChromaDB.

Fixes (2026-05-07):
  - Handles JSON saved as list [] (Reddit) OR dict {} (GitHub/arXiv/docs)
  - Content-addressable chunk IDs (SHA-256) — eliminates duplicate ID errors
  - Exponential backoff retry on ChromaDB pool timeout
  - Reduced batch size (BATCH_SIZE=32) to avoid connection exhaustion
  - Resume-safe: chunks already in store are skipped by ID lookup

Run from project root:
    C:\\\\Users\\\\Ezeking\\\\AppData\\\\Local\\\\Programs\\\\Python\\\\Python311\\\\python.exe src/processing/chunk_and_embed.py
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE_WORDS: int = 384       # ~512 tokens at 0.75 words/token
CHUNK_OVERLAP_WORDS: int = 37     # ~50 token overlap
VECTOR_STORE_PATH: str = "data/vector_store"
COLLECTION_NAME: str = "promptforge"
EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
BATCH_SIZE: int = 32
MAX_RETRIES: int = 5
RETRY_BASE_DELAY_SECONDS: float = 2.0
MIN_TEXT_LENGTH: int = 50

RAW_DATA_DIRS: list[str] = [
    "data/raw/github",
    "data/raw/arxiv",
    "data/raw/reddit",
    "data/raw/docs",
]

TEXT_KEYS: tuple[str, ...] = (
    "text", "content", "body", "abstract", "readme", "selftext", "comment_body"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/chunk_and_embed.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_dict(item: dict[str, Any]) -> str:
    """Return the first usable text field from a JSON object.

    Args:
        item: Dict representing one document or post.

    Returns:
        Stripped text string, or empty string if nothing usable.
    """
    for key in TEXT_KEYS:
        candidate = item.get(key, "")
        if isinstance(candidate, str) and len(candidate.strip()) >= MIN_TEXT_LENGTH:
            return candidate.strip()
    return ""


def make_chunk_id(source: str, content: str, position: int) -> str:
    """Generate a collision-proof chunk ID via SHA-256.

    Args:
        source: Source identifier string.
        content: Raw text of the chunk.
        position: Chunk index within the document.

    Returns:
        String formatted as \\'prefix_hexdigest\\'.
    """
    prefix = source.split("_")[0] if "_" in source else "chunk"
    digest = hashlib.sha256(
        f"{source}::{position}::{content}".encode("utf-8")
    ).hexdigest()[:32]
    return f"{prefix}_{digest}"


def chunk_text(
    text: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split text into overlapping word-window chunks, preserving code blocks.

    Args:
        text: Raw input text.
        chunk_size_words: Maximum words per chunk.
        overlap_words: Words carried into next chunk for context continuity.

    Returns:
        List of text chunk strings.
    """
    if not text or not text.strip():
        return []

    lines = text.split("\\n")
    protected: list[str] = []
    in_fence = False
    block: list[str] = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_fence:
                block.append(line)
                protected.append(" ".join(block))
                block = []
                in_fence = False
            else:
                in_fence = True
                block = [line]
        elif in_fence:
            block.append(line)
        else:
            protected.append(line)

    if block:
        protected.append(" ".join(block))

    sentences = re.split(r"(?<=[.!?])\\s+", " ".join(protected))
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        current.extend(words)
        if len(current) >= chunk_size_words:
            chunks.append(" ".join(current))
            current = current[-overlap_words:] if overlap_words > 0 else []

    if len(current) > 10:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# ChromaDB upsert with retry
# ---------------------------------------------------------------------------

def upsert_with_retry(
    collection: chromadb.Collection,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    attempt: int = 0,
) -> bool:
    """Upsert to ChromaDB with exponential backoff on pool timeout.

    Args:
        collection: Target ChromaDB collection.
        ids: Chunk ID strings.
        embeddings: Embedding vectors.
        documents: Raw text per chunk.
        metadatas: Metadata dicts per chunk.
        attempt: Current retry count (0-indexed).

    Returns:
        True on success, False if all retries exhausted.
    """
    try:
        collection.upsert(
            ids=ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas,
        )
        return True
    except Exception as exc:
        err = str(exc).lower()
        if "pool timed out" in err or "timed out" in err:
            if attempt >= MAX_RETRIES:
                logger.error("Pool timeout: all retries exhausted — batch skipped.")
                return False
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            logger.warning("Pool timeout (attempt %d) — retrying in %.1fs.", attempt + 1, delay)
            time.sleep(delay)
            return upsert_with_retry(collection, ids, embeddings, documents, metadatas, attempt + 1)
        if "ids to be unique" in err or "duplicate" in err:
            logger.warning("Duplicate ID conflict — skipping batch of %d.", len(ids))
            return False
        raise


def get_existing_ids(collection: chromadb.Collection) -> set[str]:
    """Return IDs already stored in the collection (for resume-safe skipping).

    Args:
        collection: ChromaDB collection to query.

    Returns:
        Set of existing chunk ID strings.
    """
    try:
        return set(collection.get(include=[])["ids"])
    except Exception as exc:
        logger.warning("Could not load existing IDs: %s — proceeding without skip.", exc)
        return set()


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_json_file(
    json_path: Path,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    existing_ids: set[str],
) -> tuple[int, int]:
    """Load one raw JSON file, chunk it, embed new chunks, upsert to ChromaDB.

    Handles two shapes:
        list [] — multiple posts per file (Reddit ingestion format)
        dict {} — single document (GitHub, arXiv, docs)

    Args:
        json_path: Path to raw JSON file.
        model: Loaded SentenceTransformer.
        collection: Target ChromaDB collection.
        existing_ids: Chunk IDs already in store.

    Returns:
        Tuple (chunks_stored, chunks_skipped).
    """
    try:
        with open(json_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error in %s: %s", json_path.name, exc)
        return 0, 0
    except OSError as exc:
        logger.error("Cannot read %s: %s", json_path.name, exc)
        return 0, 0

    # --- Normalise to (metadata_dict, text_content) ---
    if isinstance(raw, list):
        # Reddit files: list of post dicts
        if not raw:
            return 0, 0
        metadata_source: dict[str, Any] = raw[0] if isinstance(raw[0], dict) else {}
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                t = extract_text_from_dict(item)
                if t:
                    parts.append(t)
        text_content = "\\n\\n".join(parts)

    elif isinstance(raw, dict):
        metadata_source = raw
        text_content = extract_text_from_dict(raw)

    else:
        logger.warning("Unexpected JSON type %s in %s — skipping.", type(raw).__name__, json_path.name)
        return 0, 0

    if not text_content:
        logger.warning("No usable text in %s — skipping.", json_path.name)
        return 0, 0

    source: str = metadata_source.get("source", json_path.stem)
    chunks = chunk_text(text_content)
    if not chunks:
        return 0, 0

    chunk_ids = [make_chunk_id(source, c, i) for i, c in enumerate(chunks)]
    new_idx = [i for i, cid in enumerate(chunk_ids) if cid not in existing_ids]
    skipped = len(chunks) - len(new_idx)

    if not new_idx:
        return 0, skipped

    base_meta: dict[str, Any] = {
        "source": source,
        "source_type": metadata_source.get("source_type", "unknown"),
        "credibility_tier": metadata_source.get("credibility_tier", "unknown"),
        "date": str(metadata_source.get("date", metadata_source.get("published", metadata_source.get("created_utc", "")))),
        "author": str(metadata_source.get("author", metadata_source.get("authors", ""))),
        "url": metadata_source.get("url", metadata_source.get("html_url", metadata_source.get("permalink", ""))),
        "title": metadata_source.get("title", metadata_source.get("name", json_path.stem)),
        "file": json_path.name,
    }

    new_chunks = [chunks[i] for i in new_idx]
    new_ids = [chunk_ids[i] for i in new_idx]
    new_metas = [{**base_meta, "chunk_position": i, "chunk_total": len(chunks)} for i in new_idx]

    stored = 0
    for start in range(0, len(new_chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        btexts = new_chunks[start:end]
        bids = new_ids[start:end]
        bmetas = new_metas[start:end]

        try:
            embs = model.encode(btexts, normalize_embeddings=True, show_progress_bar=False).tolist()
        except Exception as exc:
            logger.error("Embedding failed for batch %d-%d in %s: %s", start, end, json_path.name, exc)
            continue

        if upsert_with_retry(collection, bids, embs, btexts, bmetas):
            stored += len(btexts)
            time.sleep(0.1)

    return stored, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full chunk-and-embed pipeline across all raw data directories."""
    os.makedirs("logs", exist_ok=True)
    logger.info("=" * 70)
    logger.info("PromptForge chunk_and_embed.py — starting pipeline")

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    except Exception as exc:
        logger.critical("Failed to load embedding model: %s", exc)
        raise

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    try:
        client = chromadb.PersistentClient(
            path=VECTOR_STORE_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB connected. Collection has %d chunks.", collection.count())
    except Exception as exc:
        logger.critical("ChromaDB connection failed: %s", exc)
        raise

    existing_ids = get_existing_ids(collection)
    logger.info("Found %d existing chunks — will be skipped.", len(existing_ids))

    total_stored = total_skipped = total_files = 0

    for raw_dir in RAW_DATA_DIRS:
        dir_path = Path(raw_dir)
        if not dir_path.exists():
            logger.warning("Directory not found, skipping: %s", raw_dir)
            continue

        json_files = sorted(dir_path.glob("*.json"))
        logger.info("Processing %d files in %s", len(json_files), raw_dir)

        for json_path in json_files:
            stored, skipped = process_json_file(json_path, model, collection, existing_ids)
            total_stored += stored
            total_skipped += skipped
            total_files += 1
            if stored > 0:
                logger.info("%s: %d chunks stored, %d skipped.", json_path.name, stored, skipped)
            elif skipped > 0:
                logger.info("%s: all %d chunks already embedded — skipping.", json_path.name, skipped)

    final_count = collection.count()
    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info("  Files processed : %d", total_files)
    logger.info("  Chunks stored   : %d (this run)", total_stored)
    logger.info("  Chunks skipped  : %d (already in store)", total_skipped)
    logger.info("  Collection total: %d chunks", final_count)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
'''

# Write the file
with open(TARGET, "w", encoding="utf-8") as fh:
    fh.write(NEW_CONTENT)

print(f"SUCCESS — wrote corrected file to: {TARGET}")
print("Now run:")
print(r"  C:\Users\Ezeking\AppData\Local\Programs\Python\Python311\python.exe src/processing/chunk_and_embed.py")
