"""
build_bm25_cache.py — Pre-build BM25 index and corpus cache for PromptForge.

Reads all chunks from ChromaDB, builds the BM25Okapi index, and saves
three objects to a pickle file:
    - bm25:          BM25Okapi index (ready for keyword search)
    - corpus_chunks: list of document strings (parallel to bm25)
    - corpus_metas:  list of metadata dicts (parallel to corpus_chunks)

The Space loads this pickle on startup instead of rebuilding from scratch.
Effect: cold-start time drops by 25-35 seconds on a 15,000+ chunk corpus.

Run this AFTER chunk_and_embed.py and BEFORE upload_vectorstore.py.

Usage:
    python deploy/build_bm25_cache.py
"""

import logging
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PATH: str = "data/vector_store"
COLLECTION_NAME: str = "promptforge"
CACHE_OUTPUT_PATH: Path = Path("deploy/bm25_cache.pkl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_bm25_cache() -> None:
    """Load corpus from ChromaDB, build BM25 index, save to pickle.

    Raises:
        RuntimeError: If ChromaDB collection is empty.
        OSError: If pickle file cannot be written.
    """
    logger.info("Connecting to ChromaDB at %s ...", CHROMA_PATH)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Diagnose what collections actually exist in the downloaded DB
    existing = client.list_collections()
    logger.info("Collections found in DB: %s", [c.name for c in existing])

    # If our target collection doesn't exist but another one does, use that
    if existing and not any(c.name == COLLECTION_NAME for c in existing):
        actual_name = existing[0].name
        logger.warning(
            "Collection '%s' not found. Using '%s' instead.",
            COLLECTION_NAME, actual_name
        )
        collection = client.get_collection(actual_name)
    else:
        collection = client.get_or_create_collection(COLLECTION_NAME)

    logger.info("Fetching all documents from collection...")
    all_docs = collection.get(include=["documents", "metadatas"])
    corpus_chunks: list[str] = all_docs["documents"]
    corpus_metas: list[dict] = all_docs["metadatas"]

    if not corpus_chunks:
        raise RuntimeError(
            "ChromaDB collection is empty. "
            "Run chunk_and_embed.py before building the BM25 cache."
        )

    logger.info("Corpus loaded: %d chunks.", len(corpus_chunks))
    logger.info("Tokenising corpus for BM25...")

    tokenised: list[list[str]] = [doc.lower().split() for doc in corpus_chunks]

    logger.info("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenised)
    logger.info("BM25 index built.")

    cache_payload = {
        "bm25": bm25,
        "corpus_chunks": corpus_chunks,
        "corpus_metas": corpus_metas,
    }

    CACHE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving cache to %s ...", CACHE_OUTPUT_PATH)
    try:
        with open(CACHE_OUTPUT_PATH, "wb") as fh:
            pickle.dump(cache_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as exc:
        raise OSError(f"Failed to write BM25 cache to {CACHE_OUTPUT_PATH}: {exc}") from exc

    size_mb = round(CACHE_OUTPUT_PATH.stat().st_size / (1024 * 1024), 2)
    logger.info("Cache saved. File size: %.2f MB", size_mb)
    logger.info(
        "Next steps: run upload_vectorstore.py (it will upload this cache "
        "alongside ChromaDB), then update app.py to load from pickle."
    )


if __name__ == "__main__":
    build_bm25_cache()