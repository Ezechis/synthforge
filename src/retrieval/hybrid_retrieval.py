"""
SynthForge - Hybrid Retrieval System
Layer 4: Combines dense vector search (ChromaDB) with sparse BM25
keyword search, then reranks results using a cross-encoder model.
Query routing weights retrieval based on detected query type.

Usage: py src/retrieval/hybrid_retrieval.py
"""

import logging
import time
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from config.settings import (
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    LOG_DIR,
)

# -- Logging setup -------------------------------------------------------------
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "retrieval.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
DENSE_TOP_K: int = 20
BM25_TOP_K: int = 20
RERANK_TOP_N: int = 8
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

ARXIV_SIGNALS: list[str] = [
    "paper", "study", "research", "evidence", "empirical",
    "published", "findings", "theorem", "proof", "theoretical",
    "authors", "journal", "cite", "citation", "literature"
]
GITHUB_SIGNALS: list[str] = [
    "code", "implement", "library", "example", "how to",
    "tutorial", "repository", "function", "class", "install",
    "usage", "api", "script", "notebook", "demo"
]
REDDIT_SIGNALS: list[str] = [
    "experience", "practice", "community", "people say",
    "practitioners", "real world", "production", "tips",
    "tricks", "advice", "opinion", "recommend", "tried"
]


class QueryRouter:
    """Detects query intent and returns source weighting."""

    def detect_intent(self, query: str) -> dict[str, float]:
        """Analyse query and return source weights.

        Args:
            query: Raw user query string.

        Returns:
            Dict of source weights.
        """
        query_lower = query.lower()
        arxiv_score = sum(1 for sig in ARXIV_SIGNALS if sig in query_lower)
        github_score = sum(1 for sig in GITHUB_SIGNALS if sig in query_lower)
        reddit_score = sum(1 for sig in REDDIT_SIGNALS if sig in query_lower)
        total = arxiv_score + github_score + reddit_score

        if total == 0:
            return {"arxiv": 0.5, "github": 0.4, "reddit": 0.1}

        return {
            "arxiv": arxiv_score / total if arxiv_score else 0.1,
            "github": github_score / total if github_score else 0.1,
            "reddit": reddit_score / total if reddit_score else 0.05,
        }


class SynthForgeRetriever:
    """Hybrid retrieval system for SynthForge."""

    def __init__(self) -> None:
        """Initialise all retrieval components."""
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Loading cross-encoder: %s", CROSS_ENCODER_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        logger.info("Connecting to ChromaDB vector store...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR)
        )
        self.collection = self.chroma_client.get_collection("synthforge")

        logger.info("Building BM25 index...")
        self.bm25, self.bm25_docs, self.bm25_metadatas = (
            self._build_bm25_index()
        )

        self.router = QueryRouter()
        logger.info(
            "Retriever initialised. Corpus size: %d chunks",
            self.collection.count()
        )

    def _build_bm25_index(self) -> tuple:
        """Build BM25 index from all documents in ChromaDB.

        Returns:
            Tuple of BM25 index, document texts, and metadatas.
        """
        logger.info("Fetching all documents for BM25 index...")
        results = self.collection.get(include=["documents", "metadatas"])
        documents = results["documents"]
        metadatas = results["metadatas"]
        tokenised = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index built over %d documents.", len(documents))
        return bm25, documents, metadatas

    def _dense_search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """Perform dense vector similarity search.

        Args:
            query: User query string.
            top_k: Number of results to retrieve.
            source_filter: Optional source type to filter by.

        Returns:
            List of result dicts with text, metadata, and score.
        """
        try:
            query_embedding = self.embed_model.encode(
                query,
                normalize_embeddings=True,
            ).tolist()

            where_filter = (
                {"source": source_filter} if source_filter else None
            )

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
                where=where_filter,
            )

            chunks = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "dense_score": 1 - dist,
                    "retrieval_method": "dense",
                })
            return chunks

        except Exception as exc:
            logger.error("Dense search failed: %s", exc)
            time.sleep(1)
            return []

    def _bm25_search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
    ) -> list[dict]:
        """Perform BM25 sparse keyword search.

        Args:
            query: User query string.
            top_k: Number of results to retrieve.

        Returns:
            List of result dicts with text, metadata, and score.
        """
        tokenised_query = query.lower().split()
        scores = self.bm25.get_scores(tokenised_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        chunks = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunks.append({
                    "text": self.bm25_docs[idx],
                    "metadata": self.bm25_metadatas[idx],
                    "bm25_score": float(scores[idx]),
                    "retrieval_method": "bm25",
                })
        return chunks

    def _merge_results(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        source_weights: dict[str, float],
    ) -> list[dict]:
        """Merge dense and BM25 results, deduplicating by text hash.

        Args:
            dense_results: Results from dense vector search.
            bm25_results: Results from BM25 search.
            source_weights: Source type weights from QueryRouter.

        Returns:
            Deduplicated merged list of candidate chunks.
        """
        seen_hashes: set[int] = set()
        merged: list[dict] = []

        for chunk in dense_results + bm25_results:
            text_hash = hash(chunk["text"])
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)
            source = chunk["metadata"].get("source", "unknown")
            chunk["source_weight"] = source_weights.get(source, 0.1)
            merged.append(chunk)

        return merged

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = RERANK_TOP_N,
    ) -> list[dict]:
        """Rerank candidate chunks using cross-encoder model.

        Args:
            query: User query string.
            candidates: Candidate chunks from merged retrieval.
            top_n: Number of top chunks to return after reranking.

        Returns:
            Top N chunks sorted by cross-encoder relevance score.
        """
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)
            chunk["final_score"] = (
                float(score) * 0.7 + chunk.get("source_weight", 0.1) * 0.3
            )

        reranked = sorted(
            candidates,
            key=lambda x: x["final_score"],
            reverse=True,
        )
        return reranked[:top_n]

    def retrieve(
        self,
        query: str,
        top_n: int = RERANK_TOP_N,
    ) -> list[dict]:
        """Full hybrid retrieval pipeline for a single query.

        Args:
            query: User query string.
            top_n: Number of final chunks to return.

        Returns:
            Top N ranked chunks with full metadata and scores.
        """
        logger.info("Query: %s", query[:80])
        source_weights = self.router.detect_intent(query)
        logger.info("Source weights: %s", source_weights)

        dense_results = self._dense_search(query, top_k=DENSE_TOP_K)
        bm25_results = self._bm25_search(query, top_k=BM25_TOP_K)

        candidates = self._merge_results(
            dense_results, bm25_results, source_weights
        )
        logger.info("Candidates after merge: %d", len(candidates))

        final_chunks = self._rerank(query, candidates, top_n=top_n)
        logger.info("Final chunks after reranking: %d", len(final_chunks))

        return final_chunks


def test_retrieval() -> None:
    """Run a test query to verify the retrieval pipeline."""
    retriever = SynthForgeRetriever()
    test_query = (
        "What is chain-of-thought prompting and what is "
        "the empirical evidence for it?"
    )
    logger.info("Running test query: %s", test_query)
    results = retriever.retrieve(test_query)

    print("\n" + "=" * 70)
    print(f"QUERY: {test_query}")
    print("=" * 70)

    for i, chunk in enumerate(results, 1):
        meta = chunk["metadata"]
        print(f"\n[{i}] Source: {meta.get('source', 'unknown').upper()}")
        print(f"    Rerank score: {chunk.get('rerank_score', 0):.4f}")
        print(f"    Text preview: {chunk['text'][:200]}...")
    print("=" * 70)


if __name__ == "__main__":
    test_retrieval()