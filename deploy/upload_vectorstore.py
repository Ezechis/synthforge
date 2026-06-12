"""
upload_vectorstore.py — PromptForge Deployment Step 1
======================================================
Uploads the local ChromaDB vector store to a Hugging Face Dataset repo
so HF Spaces can download it on cold start.

Works in both environments:
    Local:          data/vector_store/chroma.sqlite3
    GitHub Actions: data/vector_store/vector_store/chroma.sqlite3
                    (snapshot_download nests the repo's vector_store/ folder)
The correct directory is auto-detected, same as compress_vectorstore.py
and build_bm25_cache.py.

Auth: uses the HF_TOKEN environment variable when set (GitHub Actions),
otherwise the cached `huggingface-cli login` token (local).

Prerequisites:
    pip install huggingface_hub
"""

import logging
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_USERNAME: str = "ezechinnabugwu"
DATASET_REPO_NAME: str = "synthforge-vectorstore"

# Auto-detect the directory that actually contains chroma.sqlite3.
_BASE_STORE_PATH = Path("data/vector_store")
_ACTIONS_STORE_PATH = _BASE_STORE_PATH / "vector_store"
if (_ACTIONS_STORE_PATH / "chroma.sqlite3").exists():
    VECTOR_STORE_LOCAL_PATH: str = str(_ACTIONS_STORE_PATH)
else:
    VECTOR_STORE_LOCAL_PATH = str(_BASE_STORE_PATH)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _safe_upload_folder(api: HfApi, max_retries: int = 3, **upload_kwargs) -> None:
    """Upload a folder to HuggingFace with retry on 429 Too Many Requests.

    Args:
        api: HfApi instance.
        max_retries: Number of retry attempts on 429 errors.
        **upload_kwargs: Passed straight through to api.upload_folder
            (folder_path, repo_id, repo_type, path_in_repo, commit_message...).
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(**upload_kwargs)
            logger.info("Upload succeeded on attempt %d.", attempt)
            return
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "Too Many Requests" in err_str:
                last_exc = exc
                wait = 30 * attempt  # 30s, 60s, 90s
                logger.warning(
                    "429 Too Many Requests on attempt %d/%d. "
                    "Waiting %ds before retry...",
                    attempt, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Upload to {upload_kwargs.get('repo_id')} failed after "
        f"{max_retries} attempts (429 rate limit)."
    ) from last_exc


def main() -> None:
    """Upload local ChromaDB directory to HF Dataset repo."""

    vector_store_path = Path(VECTOR_STORE_LOCAL_PATH)
    if not (vector_store_path / "chroma.sqlite3").exists():
        logger.error(
            "chroma.sqlite3 not found at: %s — run chunk_and_embed.py "
            "(or download_vectorstore.py) first.",
            VECTOR_STORE_LOCAL_PATH,
        )
        sys.exit(1)
    logger.info("Uploading vector store directory: %s", VECTOR_STORE_LOCAL_PATH)

    repo_id = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
    api = HfApi()

    # Create the dataset repo if it does not exist
    logger.info("Creating dataset repo (if not exists): %s", repo_id)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,   # Keep private — your corpus, your keys
            exist_ok=True,
        )
        logger.info("Repo ready: https://huggingface.co/datasets/%s", repo_id)
    except Exception as exc:
        logger.error("Failed to create repo: %s", exc)
        sys.exit(1)

    # Upload the entire vector store directory
    logger.info(
        "Uploading vector store from %s — this may take several minutes...",
        VECTOR_STORE_LOCAL_PATH,
    )
    try:
        _safe_upload_folder(
            api,
            folder_path=str(vector_store_path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="vector_store",   # Stored at root/vector_store/ in the repo
            commit_message="Update ChromaDB vector store",
        )
        logger.info("Upload complete.")
        logger.info("Dataset URL: https://huggingface.co/datasets/%s", repo_id)
    except Exception as exc:
        logger.error("Upload failed: %s", exc)
        sys.exit(1)

    # Upload BM25 cache pickle if it exists
    bm25_cache_path = Path("deploy/bm25_cache.pkl")
    if bm25_cache_path.exists():
        logger.info("Uploading BM25 cache (%s)...", bm25_cache_path)
        try:
            api.upload_file(
                path_or_fileobj=str(bm25_cache_path),
                path_in_repo="bm25_cache.pkl",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Update BM25 cache",
            )
            logger.info("BM25 cache uploaded.")
        except Exception as exc:
            logger.warning("BM25 cache upload failed (non-fatal): %s", exc)
    else:
        logger.info(
            "No BM25 cache found at %s — skipping. "
            "Run deploy/build_bm25_cache.py to generate it.",
            bm25_cache_path,
        )

    logger.info("")
    logger.info("All uploads complete.")
    logger.info("HF_DATASET_REPO = %s", repo_id)


if __name__ == "__main__":
    main()
