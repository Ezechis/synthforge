"""
ingest_paperswithcode.py -- SynthForge Layer 1: Papers With Code
=================================================================
Fetches top prompt engineering papers with benchmark results and
GitHub implementation links via the Papers With Code API.
No authentication required.

Run from project root:
    python src/ingestion/ingest_paperswithcode.py
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

API_BASE: str = "https://paperswithcode.com/api/v1"
OUTPUT_PATH: str = "data/raw/docs/paperswithcode.json"
REQUEST_DELAY: float = 1.0
MAX_PER_QUERY: int = 50

SEARCH_QUERIES: list[str] = [
    "prompt engineering",
    "chain of thought prompting",
    "retrieval augmented generation",
    "large language model agent",
    "few-shot learning language model",
    "instruction tuning",
    "reinforcement learning human feedback",
    "hallucination large language model",
    "in-context learning",
    "self-consistency prompting",
    "tool use language model",
    "constitutional AI",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_papers(query: str, page: int = 1) -> list[dict[str, Any]]:
    """Fetch papers from Papers With Code API for a given query.

    Args:
        query: Search query string.
        page: Page number (1-indexed).

    Returns:
        List of paper dicts from API response.
    """
    try:
        response = requests.get(
            f"{API_BASE}/papers/",
            params={"q": query, "page": page, "items_per_page": MAX_PER_QUERY},
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.RequestException as exc:
        logger.error("API error for '%s' page %d: %s", query, page, exc)
        return []


def fetch_methods(paper_id: str) -> list[str]:
    """Fetch method names associated with a paper.

    Args:
        paper_id: Papers With Code paper ID.

    Returns:
        List of method name strings.
    """
    try:
        response = requests.get(
            f"{API_BASE}/papers/{paper_id}/methods/",
            timeout=10,
        )
        response.raise_for_status()
        methods = response.json().get("results", [])
        return [m.get("name", "") for m in methods if m.get("name")]
    except Exception:
        return []


def main() -> None:
    """Fetch Papers With Code entries and save to raw data directory."""
    Path("data/raw/docs").mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    collected: list[dict[str, Any]] = []

    for query in SEARCH_QUERIES:
        logger.info("Searching: '%s'", query)

        for page in range(1, 4):
            papers = fetch_papers(query, page)
            if not papers:
                break

            for paper in papers:
                paper_id: str = paper.get("id", "")
                if paper_id in seen_ids or not paper_id:
                    continue
                seen_ids.add(paper_id)

                title: str = paper.get("title", "")
                abstract: str = paper.get("abstract", "") or ""
                arxiv_id: str = paper.get("arxiv_id", "") or ""
                github_url: str = paper.get("repository", {}).get("url", "") if paper.get("repository") else ""
                stars: int = paper.get("stars", 0) or 0
                url: str = f"https://paperswithcode.com/paper/{paper_id}"

                if len(abstract.strip()) < 100:
                    continue

                # Build rich text combining abstract + metadata
                text = (
                    f"Title: {title}\n\n"
                    f"Abstract: {abstract}\n\n"
                    f"GitHub: {github_url or 'Not available'}\n"
                    f"Stars: {stars}\n"
                    f"ArXiv: {arxiv_id or 'Not available'}"
                )

                collected.append({
                    "source": "docs",
                    "source_type": "docs",
                    "credibility_tier": "primary",
                    "content_type": "research_paper",
                    "title": title,
                    "author": "",
                    "url": url,
                    "arxiv_id": arxiv_id,
                    "github_url": github_url,
                    "stars": stars,
                    "text": text,
                    "ingested_at": datetime.utcnow().isoformat(),
                })

            logger.info(
                "Query '%s' page %d: %d total collected",
                query, page, len(collected),
            )
            time.sleep(REQUEST_DELAY)

    output_path = Path(OUTPUT_PATH)
    output_path.write_text(
        json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d Papers With Code entries to %s", len(collected), OUTPUT_PATH)


if __name__ == "__main__":
    main()