"""Upload YouTube progress file to HF Dataset. Used by GitHub Actions."""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
repo  = os.environ.get("HF_DATASET_REPO", "ezechinnabugwu/promptforge-vectorstore")
f     = Path("data/youtube_progress.json")

if not f.exists():
    print("No progress file to upload.")
    sys.exit(0)

HfApi(token=token).upload_file(
    path_or_fileobj=str(f),
    path_in_repo="youtube_progress.json",
    repo_id=repo, repo_type="dataset",
    commit_message="Update YouTube transcription progress"
)
print("Progress file uploaded.")