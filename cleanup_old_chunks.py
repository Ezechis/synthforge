# Deletes old arXiv and Reddit chunks from ChromaDB before Batch B re-embed.
# GitHub and docs chunks are correct — they stay untouched.

import chromadb
from chromadb.config import Settings
from pathlib import Path

VECTOR_STORE_PATH = r"C:\Users\Ezeking\PromptForge\data\vector_store"

client = chromadb.PersistentClient(
    path=VECTOR_STORE_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_collection("promptforge")
print(f"Before cleanup: {collection.count()} chunks")

for source in ("arxiv", "reddit"):
    results = collection.get(where={"source": {"$eq": source}})
    ids = results["ids"]
    if ids:
        collection.delete(ids=ids)
        print(f"Deleted {len(ids)} {source} chunks")
    else:
        print(f"No {source} chunks found to delete")

print(f"After cleanup:  {collection.count()} chunks")
print("Ready for re-embed.")