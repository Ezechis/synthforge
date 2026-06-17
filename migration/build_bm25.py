"""Rebuild the BM25 cache for the FRESH 2,247-record clean store.

The legacy deploy/bm25_cache.pkl was built for the 21,527-record corpus and is
invalid for the migrated corpus. This mirrors deploy/build_bm25_cache.py exactly
(same payload keys, same `doc.lower().split()` tokenisation, same integrity
guard) but reads the fresh store and writes the cache *alongside* it so the
clean store + its BM25 cache travel together to the cutover.

Does NOT touch deploy/bm25_cache.pkl (the production artifact) -- that swap is
part of the gated Phase-6 cutover, not this step.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

REPO = Path(__file__).resolve().parents[1]
CLEAN_STORE = REPO / "data" / "vector_store_clean"
CACHE_OUT = CLEAN_STORE / "bm25_cache.pkl"
COLLECTION = "synthforge"


def build() -> Path:
    client = chromadb.PersistentClient(path=str(CLEAN_STORE))
    names = [c.name for c in client.list_collections()]
    if names != [COLLECTION]:
        raise RuntimeError(f"integrity: expected ['{COLLECTION}'], found {names}")

    coll = client.get_collection(COLLECTION, embedding_function=None)
    all_docs = coll.get(include=["documents", "metadatas"])
    corpus_chunks = all_docs["documents"]
    corpus_metas = all_docs["metadatas"]
    if not corpus_chunks:
        raise RuntimeError("clean store is empty")

    tokenised = [doc.lower().split() for doc in corpus_chunks]
    bm25 = BM25Okapi(tokenised)

    payload = {
        "bm25": bm25,
        "corpus_chunks": corpus_chunks,
        "corpus_metas": corpus_metas,
    }
    with open(CACHE_OUT, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = CACHE_OUT.stat().st_size / (1024 * 1024)
    print(f"BM25 cache rebuilt over {len(corpus_chunks)} chunks -> {CACHE_OUT} "
          f"({size_mb:.2f} MB)")
    return CACHE_OUT


if __name__ == "__main__":
    build()
