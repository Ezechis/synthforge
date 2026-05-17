"""
src/eval/run_evals.py
======================
Automated regression testing for PromptForge.
Runs every query in the golden eval set through retrieval + generation,
checks expected components, and writes a quality report.

Designed to run in GitHub Actions after every corpus update.
Results uploaded to HF Dataset for tracking over time.

Usage:
    python src/eval/run_evals.py

Environment variables:
    GROQ_API_KEY       -- generation model
    HF_TOKEN           -- for uploading results
    HF_DATASET_REPO    -- e.g. ezechinnabugwu/promptforge-vectorstore
    EVAL_SAMPLE_SIZE   -- how many queries to run (default: 20, full=60)
    VECTOR_STORE_PATH  -- local ChromaDB path (default: data/vector_store)

Author: Ezechinyere Nnabugwu / DeepForge
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_STORE_PATH: Path = Path(
    os.environ.get("VECTOR_STORE_PATH", "data/vector_store")
)
EVAL_SET_PATH: Path = Path("data/evals/golden_eval_set.jsonl")
RESULTS_DIR: Path   = Path("data/evals/results")
GROQ_API_URL: str   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str     = "llama-3.1-8b-instant"  # Higher RPM on Groq free tier for eval runs
EMBED_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME: str = "promptforge"
MIN_RETRIEVAL_SCORE: float = -6.5
EVAL_SAMPLE_SIZE: int = int(os.environ.get("EVAL_SAMPLE_SIZE", "20"))
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO: str = os.environ.get(
    "HF_DATASET_REPO", "ezechinnabugwu/promptforge-vectorstore"
)


# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

def load_resources():
    """Load ChromaDB, embedding model, reranker, and BM25 cache."""
    from chromadb import PersistentClient
    from chromadb.config import Settings
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from rank_bm25 import BM25Okapi

    chroma_path = str(VECTOR_STORE_PATH / "vector_store")
    if not Path(chroma_path).exists():
        chroma_path = str(VECTOR_STORE_PATH)

    client     = PersistentClient(path=chroma_path,
                                  settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker    = CrossEncoder(RERANKER_MODEL_NAME)

    bm25_cache = VECTOR_STORE_PATH / "bm25_cache.pkl"
    if bm25_cache.exists():
        with open(bm25_cache, "rb") as fh:
            cache = pickle.load(fh)
        bm25         = cache["bm25"]
        corpus_chunks = cache["corpus_chunks"]
        corpus_metas  = cache["corpus_metas"]
    else:
        all_docs     = collection.get(include=["documents", "metadatas"])
        corpus_chunks = all_docs["documents"]
        corpus_metas  = all_docs["metadatas"]
        bm25         = BM25Okapi([d.lower().split() for d in corpus_chunks])

    return collection, embed_model, reranker, bm25, corpus_chunks, corpus_metas


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    collection,
    embed_model,
    reranker,
    bm25,
    corpus_chunks: list,
    corpus_metas: list,
    n_results: int = 20,
) -> list[dict]:
    """Hybrid BM25 + dense retrieval with cross-encoder reranking."""
    query_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    dense = collection.query(
        query_embeddings=[query_vec],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25 = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:n_results]

    seen: set[str] = set()
    candidates: list[tuple] = []
    for doc, meta in (
        list(zip(dense["documents"][0], dense["metadatas"][0]))
        + [(corpus_chunks[i], corpus_metas[i]) for i in top_bm25]
    ):
        key = doc[:100]
        if key not in seen:
            seen.add(key)
            candidates.append((doc, meta))

    if not candidates:
        return []

    scores = reranker.predict([[query, doc] for doc, _ in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    results = [
        {"score": float(s), "text": doc, "metadata": meta}
        for s, (doc, meta) in ranked[:10]
        if float(s) > MIN_RETRIEVAL_SCORE
    ]
    return results if len(results) >= 3 else [
        {"score": float(s), "text": doc, "metadata": meta}
        for s, (doc, meta) in ranked[:3]
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(query: str, chunks: list[dict]) -> str:
    """Generate answer from retrieved chunks via Groq."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return ""

    parts = []
    for chunk in chunks[:8]:
        meta  = chunk["metadata"]
        label = (
            f"[{meta.get('source','?').upper()} | "
            f"{meta.get('credibility_tier','unknown')}]"
        )
        parts.append(f"{label}\n{' '.join(chunk['text'].split()[:300])}")

    context = "\n\n---\n\n".join(parts)
    user_msg = f"RETRIEVED CONTEXT:\n\n{context}\n\nQUERY: {query}"

    for attempt in range(3):
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content":
                         "You are PromptForge. Answer only from retrieved context. "
                         "Always cite original authors (e.g. Wei et al, Wang et al) when discussing research techniques. Include key technical terms. Aim for 150-250 words."},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 1200,
                    "temperature": 0.1,
                },
                timeout=45,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as exc:
            if resp.status_code == 429 and attempt < 2:
                wait = 15 * (2 ** attempt)
                logger.warning("Rate limited (429). Waiting %ds. Retry %d/3.", wait, attempt + 1)
                time.sleep(wait)
                continue
            logger.error("Generation failed: %s", exc)
            return ""
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return ""
    return ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def check_components(answer: str, expected: list[str]) -> dict:
    """
    Check what fraction of expected components appear in the answer.

    Args:
        answer: Generated answer text.
        expected: List of expected component strings (case-insensitive).

    Returns:
        Dict with component_score, found, missing.
    """
    answer_lower = answer.lower()
    found   = [c for c in expected if c.lower() in answer_lower]
    missing = [c for c in expected if c.lower() not in answer_lower]
    score   = len(found) / len(expected) if expected else 0.0
    return {"component_score": round(score, 3), "found": found, "missing": missing}


def has_tier1_source(chunks: list[dict]) -> bool:
    """Check if retrieved chunks include at least one Tier 1 source."""
    return any(
        c["metadata"].get("source", "") == "arxiv"
        or c["metadata"].get("credibility_tier", "") == "primary"
        for c in chunks
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval_set(
    eval_items: list[dict],
    collection,
    embed_model,
    reranker,
    bm25,
    corpus_chunks,
    corpus_metas,
) -> dict:
    """
    Run the evaluation loop over a list of eval items.

    Returns:
        Summary dict with aggregate metrics and per-query results.
    """
    results         = []
    total_score     = 0.0
    tier1_coverage  = 0
    answered        = 0
    failed_tier1    = []

    for i, item in enumerate(eval_items, 1):
        query_id = item["id"]
        query    = item["query"]
        expected = item["expected_components"]
        needs_t1 = item.get("tier1_required", False)

        logger.info("[%d/%d] %s — %s", i, len(eval_items), query_id, query[:60])

        # Retrieve
        chunks = retrieve(
            query, collection, embed_model, reranker, bm25, corpus_chunks, corpus_metas
        )
        retrieval_ok = len(chunks) >= 3

        # Tier 1 check
        t1_present = has_tier1_source(chunks) if chunks else False
        if needs_t1 and not t1_present:
            failed_tier1.append(query_id)

        # Generate
        answer = generate(query, chunks) if retrieval_ok else ""
        if answer:
            answered += 1

        # Evaluate
        eval_result = check_components(answer, expected)
        score = eval_result["component_score"]
        if answer:  # Only count answered queries in mean
            total_score += score

        if t1_present:
            tier1_coverage += 1

        result_row = {
            "id":              query_id,
            "query":           query,
            "category":        item.get("category", ""),
            "difficulty":      item.get("difficulty", ""),
            "component_score": score,
            "found":           eval_result["found"],
            "missing":         eval_result["missing"],
            "chunks_retrieved": len(chunks),
            "tier1_present":   t1_present,
            "answered":        bool(answer),
        }
        results.append(result_row)
        logger.info("  Score: %.2f | T1: %s | Chunks: %d",
                    score, t1_present, len(chunks))

        time.sleep(5)   # Rate limit courtesy pause — 3s avoids Groq 429s

    n = len(eval_items)
    summary = {
        "run_timestamp":       datetime.utcnow().isoformat(),
        "queries_run":         n,
        "queries_answered":    answered,
        "answer_rate":         round(answered / n, 3) if n else 0,
        "mean_component_score": round(total_score / answered, 3) if answered else 0,
        "tier1_coverage_rate": round(tier1_coverage / n, 3) if n else 0,
        "failed_tier1_queries": failed_tier1,
        "corpus_size":         collection.count(),
        "per_query_results":   results,
    }
    return summary


# ---------------------------------------------------------------------------
# Upload results
# ---------------------------------------------------------------------------

def upload_results(results_path: Path) -> None:
    """Upload eval results to HF Dataset for tracking over time."""
    if not HF_TOKEN or not HF_DATASET_REPO:
        logger.info("No HF_TOKEN — skipping results upload.")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=results_path,
            path_in_repo=f"eval_results/{results_path.name}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Eval run: {results_path.stem}",
        )
        logger.info("Results uploaded to HF Dataset: %s", results_path.name)
    except Exception as exc:
        logger.warning("Results upload failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load eval set, run queries, write report, upload results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    if not EVAL_SET_PATH.exists():
        logger.error("Eval set not found: %s", EVAL_SET_PATH)
        return

    eval_items = []
    with open(EVAL_SET_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                eval_items.append(json.loads(line))

    # Sample for CI runs — full set for weekly deep eval
    if EVAL_SAMPLE_SIZE < len(eval_items):
        import random
        random.seed(42)
        eval_items = random.sample(eval_items, EVAL_SAMPLE_SIZE)
        logger.info("Sampling %d/%d eval queries.", EVAL_SAMPLE_SIZE, len(eval_items))
    else:
        logger.info("Running full eval set: %d queries.", len(eval_items))

    logger.info("Loading resources...")
    collection, embed_model, reranker, bm25, corpus_chunks, corpus_metas = load_resources()
    logger.info("Corpus size: %d chunks.", collection.count())

    logger.info("Starting eval run...")
    summary = run_eval_set(
        eval_items, collection, embed_model, reranker, bm25, corpus_chunks, corpus_metas
    )

    # Write results
    timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"eval_{timestamp}.json"
    result_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    # Also write latest.json for quick access
    latest_file = RESULTS_DIR / "latest.json"
    latest_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    # Upload
    upload_results(result_file)
    upload_results(latest_file)

    # Print summary
    logger.info("=" * 55)
    logger.info("EVAL SUMMARY")
    logger.info("  Corpus size:          %d chunks", summary["corpus_size"])
    logger.info("  Queries run:          %d", summary["queries_run"])
    logger.info("  Answer rate:          %.1f%%", summary["answer_rate"] * 100)
    logger.info("  Mean component score: %.3f", summary["mean_component_score"])
    logger.info("  Tier 1 coverage:      %.1f%%", summary["tier1_coverage_rate"] * 100)
    if summary["failed_tier1_queries"]:
        logger.warning("  Failed Tier 1 queries: %s", summary["failed_tier1_queries"])
    logger.info("  Results: %s", result_file)
    logger.info("=" * 55)

    # Quality gate — fail CI if mean score drops below threshold
    MIN_ACCEPTABLE_SCORE = 0.10
    if summary["mean_component_score"] < MIN_ACCEPTABLE_SCORE:
        logger.error(
            "QUALITY GATE FAILED: mean score %.3f < %.2f threshold.",
            summary["mean_component_score"],
            MIN_ACCEPTABLE_SCORE,
        )
        raise SystemExit(1)

    logger.info("Quality gate passed.")


if __name__ == "__main__":
    main()
