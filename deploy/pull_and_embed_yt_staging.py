"""
pull_and_embed_yt_staging.py
────────────────────────────
Pulls YouTube transcript JSONL files from the HF staging dataset
(ezechinnabugwu/synthforge-yt-staging) and embeds them into the local
ChromaDB collection 'synthforge'.

Run this as part of your normal corpus update sequence whenever you want
to absorb the latest batch of YouTube transcripts that GitHub Actions produced.

Usage (Window 1 — after setting HF_HUB_OFFLINE=0):
  set HF_TOKEN=hf_your_token_here
  set HF_HUB_OFFLINE=0
  C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe deploy/pull_and_embed_yt_staging.py

After this script completes, run the standard corpus update sequence:
  compress_vectorstore.py → build_bm25_cache.py → upload_vectorstore.py → restart Space

Author: DeepForge / Claude (Sonnet 4.6)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants — all adjustable without touching logic
# ---------------------------------------------------------------------------

STAGING_REPO_ID: str = "ezechinnabugwu/synthforge-yt-staging"
STAGING_TRANSCRIPTS_SUBDIR: str = "transcripts"

LOCAL_VECTORSTORE_PATH: str = "data/vector_store"
COLLECTION_NAME: str = "synthforge"

# Must match the embedding model used by chunk_and_embed.py everywhere
EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

# Match the batch size used in chunk_and_embed.py to prevent ChromaDB pool exhaustion
CHROMA_BATCH_SIZE: int = 32

# Track which staging files have already been embedded
EMBEDDED_TRACKER_PATH: Path = Path("data/yt_staging_embedded.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedded tracker (which staging files have already been absorbed)
# ---------------------------------------------------------------------------

def load_embedded_tracker() -> set[str]:
    """Load the set of staging filenames already embedded.

    Returns:
        Set of already-embedded filenames (e.g. {'abc123.jsonl', 'def456.jsonl'}).
    """
    if EMBEDDED_TRACKER_PATH.exists():
        try:
            data = json.loads(EMBEDDED_TRACKER_PATH.read_text(encoding="utf-8"))
            already_done: set[str] = set(data.get("embedded_files", []))
            logger.info("Embedded tracker loaded: %d files already done", len(already_done))
            return already_done
        except json.JSONDecodeError as exc:
            logger.warning("Tracker file corrupt, starting fresh: %s", exc)
    return set()


def save_embedded_tracker(embedded_files: set[str]) -> None:
    """Persist the embedded tracker.

    Args:
        embedded_files: Complete set of embedded filenames.
    """
    EMBEDDED_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDED_TRACKER_PATH.write_text(
        json.dumps({"embedded_files": sorted(embedded_files)}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Staging download
# ---------------------------------------------------------------------------

def list_staging_files(hf_token: str) -> list[str]:
    """List all transcript JSONL filenames in the HF staging dataset.

    Args:
        hf_token: HuggingFace token with read access.

    Returns:
        List of filenames (not full paths) in the transcripts/ subdir.
    """
    from huggingface_hub import list_repo_tree

    filenames: list[str] = []
    try:
        items = list_repo_tree(
            repo_id=STAGING_REPO_ID,
            repo_type="dataset",
            path_in_repo=STAGING_TRANSCRIPTS_SUBDIR,
            token=hf_token,
        )
        for item in items:
            if hasattr(item, "path") and item.path.endswith(".jsonl"):
                # item.path is like 'transcripts/abc123.jsonl' — extract filename only
                filenames.append(Path(item.path).name)
        logger.info("Staging repo contains %d transcript files", len(filenames))
    except Exception as exc:
        logger.error("Failed to list staging files: %s", exc)
    return filenames


def download_staging_file(filename: str, local_dir: Path, hf_token: str) -> Path | None:
    """Download one transcript JSONL from the staging dataset.

    Args:
        filename: Filename inside the transcripts/ subdir.
        local_dir: Local directory to download into.
        hf_token: HuggingFace token.

    Returns:
        Local Path to the downloaded file, or None on failure.
    """
    from huggingface_hub import hf_hub_download
    import shutil

    try:
        cached_path = hf_hub_download(
            repo_id=STAGING_REPO_ID,
            filename=f"{STAGING_TRANSCRIPTS_SUBDIR}/{filename}",
            repo_type="dataset",
            token=hf_token,
        )
        local_path = local_dir / filename
        shutil.copy(cached_path, local_path)
        return local_path
    except Exception as exc:
        logger.warning("Failed to download %s: %s", filename, exc)
        return None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: SentenceTransformer,
) -> int:
    """Embed a list of chunks and upsert them into ChromaDB.

    Processes in batches of CHROMA_BATCH_SIZE to prevent pool exhaustion.

    Args:
        chunks: List of chunk dicts with keys: chunk_id, text, metadata.
        collection: ChromaDB collection object.
        model: SentenceTransformer embedding model.

    Returns:
        Number of chunks successfully embedded.
    """
    total_added = 0
    for batch_start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + CHROMA_BATCH_SIZE]
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            total_added += len(batch)
        except Exception as exc:
            logger.error(
                "Embedding batch %d-%d failed: %s",
                batch_start,
                batch_start + len(batch),
                exc,
            )
    return total_added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Pull staged YouTube transcripts and embed them into ChromaDB."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set. Aborting.")
        sys.exit(1)

    # Guard: this script must NOT run with HF_HUB_OFFLINE=1
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        logger.error(
            "HF_HUB_OFFLINE is set to 1. This script needs network access. "
            "Run: set HF_HUB_OFFLINE=0  — then retry."
        )
        sys.exit(1)

    # Load what's already embedded
    embedded_files = load_embedded_tracker()

    # List staging files
    staging_filenames = list_staging_files(hf_token)
    pending_filenames = [f for f in staging_filenames if f not in embedded_files]

    if not pending_filenames:
        logger.info("No new staging files to embed. Corpus is already up to date.")
        return

    logger.info("%d new staging files to embed", len(pending_filenames))

    # Load embedding model
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=LOCAL_VECTORSTORE_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        logger.error("Could not open ChromaDB collection '%s': %s", COLLECTION_NAME, exc)
        sys.exit(1)

    count_before = collection.count()
    logger.info("ChromaDB collection '%s' has %d chunks before this run", COLLECTION_NAME, count_before)

    # Process each pending staging file
    tmp_dir = Path("data/yt_staging_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    total_chunks_added = 0

    for i, filename in enumerate(pending_filenames, 1):
        logger.info("[%d/%d] Processing staging file: %s", i, len(pending_filenames), filename)

        local_path = download_staging_file(filename, tmp_dir, hf_token)
        if local_path is None:
            logger.warning("  Skipping (download failed): %s", filename)
            continue

        # Read chunks from JSONL
        chunks: list[dict] = []
        try:
            for line in local_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("  Skipping (parse error): %s — %s", filename, exc)
            continue

        if not chunks:
            logger.warning("  No chunks in file: %s", filename)
            embedded_files.add(filename)  # mark done anyway
            save_embedded_tracker(embedded_files)
            continue

        # Embed and store
        added = embed_and_store(chunks, collection, model)
        total_chunks_added += added
        logger.info("  ✓ Added %d chunks from %s", added, filename)

        # Mark as embedded
        embedded_files.add(filename)
        save_embedded_tracker(embedded_files)

        # Clean up tmp file
        local_path.unlink(missing_ok=True)
        time.sleep(0.2)

    # Cleanup tmp directory
    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # not empty — fine

    count_after = collection.count()
    logger.info(
        "Done. Added %d chunks. Collection now has %d chunks (was %d).",
        total_chunks_added,
        count_after,
        count_before,
    )
    logger.info(
        "Next step: run the standard corpus update sequence "
        "(compress → build_bm25_cache → upload_vectorstore → restart Space)"
    )


if __name__ == "__main__":
    main()
