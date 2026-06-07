"""
deploy/build_bm25_cache.py
Improved version with better collection handling and logging.
"""

import os
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from chromadb import PersistentClient

VECTORSTORE_PATH = Path("data/vector_store")
BM25_CACHE_PATH = Path("deploy/bm25_cache.pkl")
TARGET_COLLECTION = "synthforge"


def build_bm25_cache():
    print("Connecting to ChromaDB at data/vector_store ...")

    client = PersistentClient(path=str(VECTORSTORE_PATH))
    collections = client.list_collections()

    print(f"Collections found in DB: {[c.name for c in collections]}")

    # Try to get the target collection
    try:
        collection = client.get_collection(name=TARGET_COLLECTION)
    except Exception:
        print(f"❌ Collection '{TARGET_COLLECTION}' not found!")
        print("Available collections:", [c.name for c in collections])
        raise RuntimeError(f"Collection '{TARGET_COLLECTION}' does not exist. Please check your vectorstore.")

    print(f"Fetching all documents from collection '{TARGET_COLLECTION}'...")

    results = collection.get(include=["documents", "metadatas"])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    if not documents:
        raise RuntimeError(
            f"ChromaDB collection '{TARGET_COLLECTION}' is empty. "
            "Run chunk_and_embed.py before building the BM25 cache."
        )

    print(f"Corpus loaded: {len(documents)} chunks.")

    # Tokenize documents
    print("Tokenising corpus for BM25...")
    tokenized_corpus = [doc.split() for doc in documents]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Save cache
    print("Saving cache to deploy/bm25_cache.pkl ...")
    BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "corpus_chunks": documents,
            "corpus_metas": metadatas
        }, f)

    print(f"✅ BM25 cache built successfully with {len(documents)} documents.")


if __name__ == "__main__":
    build_bm25_cache()