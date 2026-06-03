"""
deploy/download_vectorstore.py
================================
Downloads the ChromaDB vectorstore and BM25 cache from the Hugging Face
Dataset repo onto the local filesystem (used by GitHub Actions runners
at the start of every ingestion job).

On GitHub Actions, this script runs BEFORE any ingestion so the runner
starts with the current production corpus and only adds new chunks.

Environment variables:
    HF_TOKEN         — Hugging Face read/write token
    HF_DATASET_REPO  — Dataset repo ID, e.g. ezechinnabugwu/synthforge-vectorstore
    VECTOR_STORE_PATH — Local destination path (default: data/vector_store)

Usage:
    python deploy/download_vectorstore.py

Author: Ezechinyere Nnabugwu / DeepForge
"""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_STORE_PATH: Path = Path(
    os.environ.get("VECTOR_STORE_PATH", "data/vector_store")
)
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO: str = os.environ.get(
    "HF_DATASET_REPO", "ezechinnabugwu/synthforge-vectorstore"
)
IGNORE_PATTERNS: list[str] = ["*.md", ".gitattributes", "*.json"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download vectorstore from HF Dataset to VECTOR_STORE_PATH."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    if not HF_DATASET_REPO:
        logger.error("HF_DATASET_REPO environment variable not set.")
        sys.exit(1)

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading vectorstore from: %s", HF_DATASET_REPO)
    logger.info("Destination: %s", VECTOR_STORE_PATH.resolve())

    try:
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            local_dir=str(VECTOR_STORE_PATH),
            token=HF_TOKEN or None,
            ignore_patterns=IGNORE_PATTERNS,
        )
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)

    # Verify ChromaDB presence
    sqlite_path = VECTOR_STORE_PATH / "chroma.sqlite3"
    if sqlite_path.exists():
        size_mb = sqlite_path.stat().st_size / (1024 * 1024)
        logger.info("ChromaDB verified: %.1f MB", size_mb)
    else:
        # Some setups store chroma.sqlite3 in a subdirectory
        nested = VECTOR_STORE_PATH / "vector_store" / "chroma.sqlite3"
        if nested.exists():
            size_mb = nested.stat().st_size / (1024 * 1024)
            logger.info("ChromaDB verified (nested): %.1f MB", size_mb)
        else:
            nested_dir = VECTOR_STORE_PATH / "vector_store"
            if nested_dir.is_dir():
                import shutil
                logger.info("Nested vectorstore found at %s — flattening.", nested_dir)
                for item in nested_dir.iterdir():
                    dest = VECTOR_STORE_PATH / item.name
                    if not dest.exists():
                        shutil.move(str(item), str(dest))
                        logger.info("  Moved: %s", item.name)
                nested_dir.rmdir()
                logger.info("Flatten complete.")
            else:
                logger.warning(
                    "chroma.sqlite3 not found after download. "
                    "Ingestion will create a new vectorstore."
                )

    # Verify BM25 cache
    bm25_path = VECTOR_STORE_PATH / "bm25_cache.pkl"
    if bm25_path.exists():
        size_kb = bm25_path.stat().st_size / 1024
        logger.info("BM25 cache verified: %.1f KB", size_kb)
    else:
        logger.warning("bm25_cache.pkl not found. Will be rebuilt after embedding.")

    logger.info("Vectorstore download complete.")


if __name__ == "__main__":
    main()
