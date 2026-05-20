"""Download YouTube progress file from HF Dataset. Used by GitHub Actions."""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

token = os.environ["HF_TOKEN"]
repo  = os.environ.get("HF_DATASET_REPO", "ezechinnabugwu/promptforge-vectorstore")
dest  = Path("data/youtube_progress.json")
dest.parent.mkdir(parents=True, exist_ok=True)

try:
    hf_hub_download(repo_id=repo, repo_type="dataset",
                    filename="youtube_progress.json",
                    token=token, local_dir=str(dest.parent))
    print("Progress file downloaded.")
except EntryNotFoundError:
    print("No progress file yet — starting fresh.")
except Exception as exc:
    print(f"Warning: {exc}")