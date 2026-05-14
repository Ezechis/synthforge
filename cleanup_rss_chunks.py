import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="data/vector_store",
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_collection("promptforge")
print(f"Before: {collection.count()} chunks")

results = collection.get(where={"file": {"$eq": "rss_feeds.json"}})
if results["ids"]:
    collection.delete(ids=results["ids"])
    print(f"Deleted {len(results['ids'])} old RSS chunks")
else:
    print("No RSS chunks found")

print(f"After: {collection.count()} chunks")