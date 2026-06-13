"""Inspect the current state of the HF Dataset vector_store folder."""
from huggingface_hub import HfApi

api = HfApi(token='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')
repo = 'ezechinnabugwu/synthforge-vectorstore'

print(f"\n=== Root of {repo} ===")
for f in api.list_repo_tree(repo_id=repo, repo_type='dataset'):
    print(f"  {f.path}")

print(f"\n=== vector_store/ contents ===")
for f in api.list_repo_tree(repo_id=repo, repo_type='dataset', path_in_repo='vector_store'):
    print(f"  {f.path}")
