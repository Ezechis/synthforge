import time
"""
upload_vectorstore.py — PromptForge Deployment Step 1
======================================================
Uploads the local ChromaDB vector store to a Hugging Face Dataset repo
so HF Spaces can download it on cold start.

Run ONCE from C:\\Users\\Ezeking\\PromptForge after embedding is complete:
    C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe deploy/upload_vectorstore.py

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   (paste your HF token when prompted)
"""

import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# ---------------------------------------------------------------------------
# Configuration — edit these two values before running
# ---------------------------------------------------------------------------

HF_USERNAME: str = "ezechinnabugwu"          # e.g. "ezeking"
DATASET_REPO_NAME: str = "synthforge-vectorstore"  # will be created if absent
VECTOR_STORE_LOCAL_PATH: str = "data/vector_store"  # relative to project root

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



def _safe_upload_folder(
    api,
    folder_path: str,
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    max_retries: int = 3,
) -> None:
    """Upload a folder to HuggingFace with retry on 429 Too Many Requests.

    Args:
        api: HfApi instance.
        folder_path: Local folder to upload.
        repo_id: HuggingFace repository ID.
        path_in_repo: Target path inside the repository.
        repo_type: Repository type (default: dataset).
        max_retries: Number of retry attempts on 429 errors.
    """
    import logging
    log = logging.getLogger(__name__)

    # Ensure repo exists (safe — does not fail if already exists)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True,
            private=True,
        )
    except Exception as exc:
        log.warning("create_repo warning (non-fatal): %s", exc)

    for attempt in range(1, max_retries + 1):
        try:
            _safe_upload_folder(api, 
                folder_path=folder_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
            )
            log.info("Upload succeeded on attempt %d.", attempt)
            return
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "Too Many Requests" in err_str:
                wait = 30 * attempt  # 30s, 60s, 90s
                log.warning(
                    "429 Too Many Requests on attempt %d/%d. "
                    "Waiting %ds before retry...",
                    attempt, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Upload to {repo_id} failed after {max_retries} attempts (429 rate limit)."
    )

def main() -> None:
    """Upload local ChromaDB directory to HF Dataset repo."""

    if HF_USERNAME == "YOUR_HF_USERNAME":
        logger.error(
            "Edit HF_USERNAME in this script before running. "
            "Set it to your actual Hugging Face username."
        )
        sys.exit(1)

    vector_store_path = Path(VECTOR_STORE_LOCAL_PATH)
    if not vector_store_path.exists():
        logger.error(
            "Vector store not found at: %s — run chunk_and_embed.py first.",
            VECTOR_STORE_LOCAL_PATH,
        )
        sys.exit(1)

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
        _safe_upload_folder(api, 
            folder_path=str(vector_store_path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="vector_store",   # Stored at root/vector_store/ in the repo
            commit_message="Upload ChromaDB vector store",
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
                commit_message="Upload BM25 cache",
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
