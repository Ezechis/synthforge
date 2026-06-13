from huggingface_hub import HfApi 
api = HfApi(token='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') 
files = list(api.list_repo_tree(repo_id='ezechinnabugwu/synthforge-vectorstore', repo_type='dataset', path_in_repo='vector_store')) 
[print(f.path) for f in files if hasattr(f, 'path')] 
