"""
compress_vectorstore.py — Compact ChromaDB before upload to HuggingFace.

ChromaDB uses SQLite internally. Over time, deletions and updates leave
dead space in the database file. SQLite VACUUM reclaims that space and
rewrites the file compactly. This reduces upload size and Space cold-start
download time.

Run this BEFORE upload_vectorstore.py, not after.

Usage:
    python deploy/compress_vectorstore.py
"""

import logging
import os
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PATH: str = "data/vector_store"
SQLITE_FILENAME: str = "chroma.sqlite3"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_size_mb(path: Path) -> float:
    """Return total size of a directory in megabytes.

    Args:
        path: Directory path to measure.

    Returns:
        Total size in MB, rounded to 2 decimal places.
    """
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def compress_vectorstore() -> None:
    """Run SQLite VACUUM on the ChromaDB database file.

    Measures size before and after, logs the reduction achieved.

    Raises:
        FileNotFoundError: If ChromaDB path or SQLite file does not exist.
        sqlite3.Error: If VACUUM operation fails.
    """
    chroma_dir = Path(CHROMA_PATH)
    sqlite_path = chroma_dir / SQLITE_FILENAME

    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {chroma_dir}. "
            "Run chunk_and_embed.py first."
        )

    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"SQLite file not found: {sqlite_path}. "
            "ChromaDB may not have been initialised yet."
        )

    size_before = get_size_mb(chroma_dir)
    logger.info("ChromaDB size before compression: %.2f MB", size_before)
    logger.info("Running SQLite VACUUM on %s ...", sqlite_path)

    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.execute("VACUUM")
        conn.close()
    except sqlite3.Error as exc:
        raise sqlite3.Error(f"VACUUM failed on {sqlite_path}: {exc}") from exc

    size_after = get_size_mb(chroma_dir)
    reduction = size_before - size_after
    reduction_pct = (reduction / size_before * 100) if size_before > 0 else 0

    logger.info("ChromaDB size after compression:  %.2f MB", size_after)
    logger.info(
        "Space saved: %.2f MB (%.1f%% reduction)", reduction, reduction_pct
    )
    logger.info("Ready for upload.")


if __name__ == "__main__":
    compress_vectorstore()