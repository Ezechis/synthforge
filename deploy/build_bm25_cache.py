"""
build_bm25_cache.py — Pre-build BM25 index and corpus cache for SynthForge.

Reads all chunks from ChromaDB, builds the BM25Okapi index, and saves
three objects to a pickle file:
    - bm25:          BM25Okapi index (ready for keyword search)
    - corpus_chunks: list of document strings (parallel to bm25)
    - corpus_metas:  list of metadata dicts (parallel to corpus_chunks)

The Space loads this pickle on startup instead of rebuilding from scratch.
Effect: cold-start time drops by 25-35 seconds on a 15,000+ chunk corpus.

Run this AFTER chunk_and_embed.py and BEFORE upload_vectorstore.py.

Local path:          data/vector_store/chroma.sqlite3
GitHub Actions path: data/vector_store/vector_store/chroma.sqlite3
This script auto-detects which path is present.

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

# Local environment: ChromaDB lives directly at data/vector_store/
# GitHub Actions: snapshot_download puts files at data/vector_store/vector_store/
# We auto-detect which is present so this script works in both environments.
_BASE_PATH = Path("data/vector_store")
_ACTIONS_PATH = _BASE_PATH / "vector_store"

if (_ACTIONS_PATH / "chroma.sqlite3").exists():
    CHROMA_PATH: str = str(_ACTIONS_PATH)
    print(f"[PATH] GitHub Actions environment detected — using {CHROMA_PATH}")
elif (_BASE_PATH / "chroma.sqlite3").exists():
    CHROMA_PATH = str(_BASE_PATH)
    print(f"[PATH] Local environment detected — using {CHROMA_PATH}")
else:
    raise FileNotFoundError(
        "Cannot find chroma.sqlite3 in either data/vector_store/ "
        "or data/vector_store/vector_store/. "
        "Run download_vectorstore.py first."
    )

COLLECTION_NAME: str = "synthforge"  # FIXED: was "promptforge" — do not change
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
        RuntimeError: If collection integrity check fails.
        RuntimeError: If ChromaDB collection is empty.
        OSError: If pickle file cannot be written.
    """
    logger.info("Connecting to ChromaDB at %s ...", CHROMA_PATH)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # -------------------------------------------------------------------
    # GUARD: Assert exactly one collection exists named 'synthforge'.
    # This permanently prevents the two-UUID-folder bug from causing
    # silent failures where the wrong collection is selected.
    # If this raises: delete synthforge-vectorstore on HF and re-upload
    # from local using upload_vectorstore.py.
    # -------------------------------------------------------------------
    existing_collections = [c.name for c in client.list_collections()]
    if existing_collections != ["synthforge"]:
        raise RuntimeError(
            "COLLECTION INTEGRITY CHECK FAILED. "
            "Expected: ['synthforge'] "
            "Found: " + str(existing_collections) + " "
            "Fix: Delete synthforge-vectorstore on HF, re-upload from local."
        )
    logger.info("Collection integrity check passed: %s confirmed.", existing_collections)

    # -------------------------------------------------------------------
    # Fetch all documents from the verified collection
    # -------------------------------------------------------------------
    collection = client.get_collection(COLLECTION_NAME)

    logger.info("Fetching all documents from collection '%s'...", COLLECTION_NAME)
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
        raise OSError(
            "Failed to write BM25 cache to {}: {}".format(CACHE_OUTPUT_PATH, exc)
        ) from exc

    size_mb = round(CACHE_OUTPUT_PATH.stat().st_size / (1024 * 1024), 2)
    logger.info("Cache saved. File size: %.2f MB", size_mb)
    logger.info(
        "Next steps: run upload_vectorstore.py to upload this cache "
        "alongside ChromaDB, then verify Space loads correctly."
    )


if __name__ == "__main__":
    build_bm25_cache()
