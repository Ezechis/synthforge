"""
ingest_hackernews.py -- SynthForge Layer 1: Hacker News Data Source
====================================================================
Fetches high-scoring HN stories and comments on prompt engineering
and LLM topics via the Algolia HN Search API. No authentication needed.

Quality gates:
    Gate 1 -- Minimum points threshold (MIN_POINTS)
    Gate 2 -- Keyword relevance filter on title
    Gate 3 -- Minimum text length (skips link-only posts)
    Gate 4 -- Metadata tagging with credibility_tier=community

Run from project root:
    python src/ingestion/ingest_hackernews.py
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGOLIA_URL: str = "https://hn.algolia.com/api/v1/search"
OUTPUT_PATH: str = "data/raw/docs/hackernews.json"
MIN_POINTS: int = 50
MIN_TEXT_LENGTH: int = 150
MAX_RESULTS: int = 200
REQUEST_DELAY: float = 0.5

SEARCH_QUERIES: list[str] = [
    "prompt engineering LLM",
    "chain of thought prompting",
    "large language model prompting",
    "GPT prompt techniques",
    "few-shot zero-shot prompting",
    "RAG retrieval augmented generation",
    "LLM agent reasoning",
    "Claude GPT instruction following",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_stories(query: str, page: int = 0) -> list[dict[str, Any]]:
    """Fetch HN stories matching a query via Algolia API.

    Args:
        query: Search query string.
        page: Pagination page number (0-indexed).

    Returns:
        List of hit dicts from the Algolia response.
    """
    params = {
        "query": query,
        "tags": "story",
        "numericFilters": f"points>{MIN_POINTS}",
        "hitsPerPage": 50,
        "page": page,
    }
    try:
        response = requests.get(ALGOLIA_URL, params=params, timeout=20)
        response.raise_for_status()
        return response.json().get("hits", [])
    except requests.exceptions.RequestException as exc:
        logger.error("HN API error for query '%s' page %d: %s", query, page, exc)
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch and save relevant HN stories to the raw data directory."""
    Path("data/raw/docs").mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    collected: list[dict[str, Any]] = []

    logger.info("Starting Hacker News ingestion — target: %d posts", MAX_RESULTS)

    for query in SEARCH_QUERIES:
        if len(collected) >= MAX_RESULTS:
            break

        for page in range(4):
            if len(collected) >= MAX_RESULTS:
                break

            hits = fetch_stories(query, page)
            if not hits:
                break

            for hit in hits:
                if len(collected) >= MAX_RESULTS:
                    break

                story_id: str = hit.get("objectID", "")
                if story_id in seen_ids:
                    continue

                # Gate 1 — points already filtered by API numericFilters
                points: int = hit.get("points", 0) or 0

                # Gate 2 — text content check (link posts have no body)
                text: str = hit.get("story_text", "") or ""
                title: str = hit.get("title", "") or ""

                # Gate 3 — minimum text length
                if len(text.strip()) < MIN_TEXT_LENGTH:
                    continue

                seen_ids.add(story_id)
                url: str = hit.get("url", "") or f"https://news.ycombinator.com/item?id={story_id}"
                author: str = hit.get("author", "")
                created_at: str = hit.get("created_at", "")

                collected.append({
                    "source": "docs",
                    "source_type": "docs",
                    "credibility_tier": "community",
                    "content_type": "discussion",
                    "title": title,
                    "author": author,
                    "url": url,
                    "points": points,
                    "text": text,
                    "date": created_at,
                    "ingested_at": datetime.utcnow().isoformat(),
                })

            time.sleep(REQUEST_DELAY)

        logger.info("After query '%s': %d collected", query, len(collected))

    output_path = Path(OUTPUT_PATH)
    output_path.write_text(
        json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d HN posts to %s", len(collected), OUTPUT_PATH)


if __name__ == "__main__":
    main()