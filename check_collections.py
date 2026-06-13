from chromadb import PersistentClient

client = PersistentClient(path="data/vector_store")
collections = client.list_collections()

print("Collections found:", [c.name for c in collections])