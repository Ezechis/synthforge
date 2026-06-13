from huggingface_hub import HfApi

api = HfApi(token="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")
repo = "ezechinnabugwu/synthforge-vectorstore"

print("Deleting old vector_store folder from HF...")
api.delete_folder("vector_store", repo_id=repo, repo_type="dataset")
print("✅ Old vector_store folder deleted successfully!")