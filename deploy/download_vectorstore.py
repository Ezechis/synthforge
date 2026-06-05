"""
deploy/download_vectorstore.py
Strong retry version for heavy rate limiting.
"""

import os
import time
import random
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "ezechinnabugwu/synthforge-vectorstore")
LOCAL_VECTORSTORE_PATH = Path("data/vector_store")


def download_vectorstore_with_retry(max_retries=6, initial_backoff=30):
    LOCAL_VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Attempt {attempt}/{max_retries}] Downloading from {HF_DATASET_REPO}...")

            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                local_dir=str(LOCAL_VECTORSTORE_PATH),
                local_dir_use_symlinks=False,
            )
            print("✅ Download successful.")
            return True

        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429:
                if attempt == max_retries:
                    print("❌ Still rate limited after all retries. Giving up for now.")
                    return False

                # Much longer backoff for 429
                backoff = initial_backoff * (2 ** (attempt - 1))
                jitter = random.uniform(0, backoff * 0.4)
                sleep_time = backoff + jitter

                print(f"⚠️  Rate limited (429). Waiting {sleep_time:.0f} seconds before retry...")
                time.sleep(sleep_time)
            else:
                print(f"❌ HTTP error {status}: {e}")
                return False

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if attempt == max_retries:
                return False
            time.sleep(10)

    return False


if __name__ == "__main__":
    if not (LOCAL_VECTORSTORE_PATH / "chroma.sqlite3").exists():
        success = download_vectorstore_with_retry()
        if not success:
            print("⚠️ Download failed due to rate limits. Workflow will continue with existing cache if available.")
    else:
        print("✅ Using cached vectorstore. Skipping download.")