from huggingface_hub import HfApi

api = HfApi(token="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")
repo = "ezechinnabugwu/synthforge-vectorstore"

print("=== ALL FILES IN REPO ===")
files = api.list_repo_files(repo, repo_type="dataset")
for f in sorted(files):
    if "vector_store" in f or len(f) > 30:
        print(f)
print("\n=== Look for vector_store/ followed by long UUID folders ===")