from huggingface_hub import HfApi
files = list(HfApi().list_repo_files(repo_id='ezechinnabugwu/synthforge-vectorstore', repo_type='dataset'))
print(f"Total files: {len(files)}")
for f in files[:40]:
    print(f)
