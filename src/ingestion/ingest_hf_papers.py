"""
ingest_hf_papers.py — HuggingFace Daily Papers ingestion for PromptForge.

Replaces the broken ingest_paperswithcode.py.
Source: https://huggingface.co/api/daily_papers
Returns trending AI papers with upvotes, abstracts, authors. Free, no auth.

Usage:
    python src/ingestion/ingest_hf_papers.py

Output:
    data/raw/docs/hf_papers.json
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_PAPERS_API_URL: str = "https://huggingface.co/api/daily_papers"
OUTPUT_PATH: Path = Path("data/raw/docs/hf_papers.json")

MAX_DAYS_BACK: int = 365
MIN_UPVOTES: int = 5
REQUEST_DELAY_SECONDS: float = 1.0
REQUEST_TIMEOUT_SECONDS: int = 30

RELEVANCE_KEYWORDS: tuple[str, ...] = (
    "prompt",
    "prompting",
    "chain-of-thought",
    "chain of thought",
    "few-shot",
    "few shot",
    "zero-shot",
    "zero shot",
    "in-context learning",
    "in context learning",
    "instruction tuning",
    "instruction following",
    "llm",
    "large language model",
    "language model",
    "rag",
    "retrieval augmented",
    "retrieval-augmented",
    "agent",
    "reasoning",
    "self-consistency",
    "self consistency",
    "tree of thought",
    "tree-of-thought",
    "react",
    "reflexion",
    "tool use",
    "tool-use",
    "function calling",
    "fine-tuning",
    "finetuning",
    "alignment",
    "rlhf",
    "dspy",
    "jailbreak",
    "hallucination",
    "context window",
    "attention",
    "transformer",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _is_relevant(title: str, abstract: str) -> bool:
    """Return True if the paper is relevant to prompt engineering.

    Args:
        title: Paper title string.
        abstract: Paper abstract string.

    Returns:
        True if any relevance keyword appears in title or abstract.
    """
    haystack = (title + " " + abstract).lower()
    return any(kw in haystack for kw in RELEVANCE_KEYWORDS)


def _fetch_papers_for_date(date_str: str) -> list[dict[str, Any]]:
    """Fetch HuggingFace daily papers for a specific date.

    Args:
        date_str: ISO date string, e.g. '2025-05-10'.

    Returns:
        List of raw paper dicts from the API, or empty list on failure.
    """
    try:
        response = requests.get(
            HF_PAPERS_API_URL,
            params={"date": date_str},
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "PromptForge/1.0 (research corpus builder)"},
        )
        if response.status_code == 404:
            logger.debug("No papers for %s (404)", date_str)
            return []
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        logger.warning("Unexpected API response shape for %s: %s", date_str, type(data))
        return []
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching papers for %s — skipping", date_str)
        return []
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error for %s: %s", date_str, exc)
        return []
    except requests.exceptions.RequestException as exc:
        logger.error("Network error for %s: %s", date_str, exc)
        return []


def _parse_paper(raw: dict[str, Any], fetch_date: str) -> dict[str, Any] | None:
    """Extract and normalise fields from a raw HF Papers API paper dict.

    Args:
        raw: Raw dict from HuggingFace API.
        fetch_date: The date string this paper was fetched under.

    Returns:
        Normalised dict ready for PromptForge corpus, or None if discarded.
    """
    paper_info: dict[str, Any] = raw.get("paper", raw)

    paper_id: str = paper_info.get("id", "")
    title: str = paper_info.get("title", "").strip()
    abstract: str = paper_info.get("summary", paper_info.get("abstract", "")).strip()
    authors_raw: list[dict] = paper_info.get("authors", [])
    authors: list[str] = [a.get("name", "") for a in authors_raw if a.get("name")]
    upvotes: int = raw.get("upvotes", raw.get("numComments", 0))
    published_at: str = paper_info.get("publishedAt", fetch_date)

    if not title or not abstract:
        return None
    if upvotes < MIN_UPVOTES:
        return None
    if not _is_relevant(title, abstract):
        return None

    url: str = f"https://arxiv.org/abs/{paper_id}" if paper_id else ""
    hf_url: str = f"https://huggingface.co/papers/{paper_id}" if paper_id else ""

    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "published_at": published_at,
        "fetch_date": fetch_date,
        "upvotes": upvotes,
        "url": url,
        "hf_url": hf_url,
        "source": "docs",
        "content_type": "paper_abstract",
        "text": f"Title: {title}\n\nAbstract: {abstract}",
        "credibility_tier": "primary",
    }


def ingest_hf_papers() -> None:
    """Fetch, filter, and save HuggingFace daily papers to corpus.

    Iterates backwards from today over MAX_DAYS_BACK days.
    Output written to OUTPUT_PATH as a JSON array.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now(tz=timezone.utc).date()
    papers_collected: list[dict[str, Any]] = []
    dates_fetched: int = 0
    dates_with_results: int = 0

    logger.info(
        "Starting HF Papers ingestion: %d days back, min %d upvotes",
        MAX_DAYS_BACK,
        MIN_UPVOTES,
    )

    for days_ago in range(MAX_DAYS_BACK):
        target_date = today - timedelta(days=days_ago)
        date_str = target_date.isoformat()

        raw_papers = _fetch_papers_for_date(date_str)
        dates_fetched += 1

        if raw_papers:
            dates_with_results += 1
            kept_count = 0
            for raw in raw_papers:
                parsed = _parse_paper(raw, date_str)
                if parsed:
                    papers_collected.append(parsed)
                    kept_count += 1
            logger.info(
                "%s — %d raw papers → %d kept",
                date_str, len(raw_papers), kept_count,
            )

        time.sleep(REQUEST_DELAY_SECONDS)

    # Deduplicate by arXiv ID
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for paper in papers_collected:
        pid = paper["id"]
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            deduped.append(paper)
        elif not pid:
            deduped.append(paper)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(deduped, fh, ensure_ascii=False, indent=2)

    logger.info(
        "Done. %d dates fetched, %d had papers, %d collected, %d after dedup. Saved to %s",
        dates_fetched, dates_with_results, len(papers_collected), len(deduped), OUTPUT_PATH,
    )


if __name__ == "__main__":
    ingest_hf_papers()