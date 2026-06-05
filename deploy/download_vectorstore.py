"""
deploy/download_vectorstore.py
Improved version with retry logic for Hugging Face rate limits (429).
"""

import os
import time
import random
from pathlib import Path

from huggingface_hub import snapshot_download, HfHubHTTPError

# Configuration
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "ezechinnabugwu/synthforge-vectorstore")
LOCAL_VECTORSTORE_PATH = Path("data/vector_store")
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 10


def download_vectorstore_with_retry():
    """Download vectorstore with exponential backoff + jitter on rate limits."""

    LOCAL_VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Downloading vectorstore from {HF_DATASET_REPO}...")

            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                local_dir=str(LOCAL_VECTORSTORE_PATH),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            print("✅ Vectorstore downloaded successfully.")
            return True

        except HfHubHTTPError as e:
            if e.response.status_code == 429:
                if attempt == MAX_RETRIES:
                    print(f"❌ Rate limit hit {MAX_RETRIES} times. Giving up.")
                    raise

                backoff = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
                jitter = random.uniform(0, backoff * 0.3)
                sleep_time = backoff + jitter

                print(f"⚠️  Rate limited (429). Sleeping for {sleep_time:.1f} seconds before retry...")
                time.sleep(sleep_time)
            else:
                print(f"❌ Hugging Face HTTP error: {e}")
                raise

        except Exception as e:
            print(f"❌ Unexpected error while downloading: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5)

    return False


if __name__ == "__main__":
    # Only download if the database doesn't already exist (helps with caching)
    if not (LOCAL_VECTORSTORE_PATH / "chroma.sqlite3").exists():
        download_vectorstore_with_retry()
    else:
        print("✅ Vectorstore already exists (using cache). Skipping download.")