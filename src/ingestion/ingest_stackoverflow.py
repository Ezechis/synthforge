"""
ingest_stackoverflow.py -- PromptForge Layer 1: Stack Overflow Data Source
==========================================================================
Fetches high-voted questions and accepted answers on prompt engineering
and LLM topics via the Stack Exchange API. No API key required (300 req/day).

Quality gates:
    Gate 1 -- Minimum question score (MIN_QUESTION_SCORE)
    Gate 2 -- Tag-based topic curation
    Gate 3 -- Must have at least one answer
    Gate 4 -- Metadata tagging with credibility_tier=implementation

Run from project root:
    python src/ingestion/ingest_stackoverflow.py
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

API_BASE: str = "https://api.stackexchange.com/2.3"
OUTPUT_PATH: str = "data/raw/docs/stackoverflow.json"
MIN_QUESTION_SCORE: int = 5
PAGE_SIZE: int = 50
REQUEST_DELAY: float = 1.0

# Tags to query — each tag is a separate API call
TARGET_TAGS: list[str] = [
    "prompt-engineering",
    "large-language-model",
    "chatgpt",
    "gpt-4",
    "langchain",
    "llm",
    "openai-api",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_questions(tag: str, page: int = 1) -> list[dict[str, Any]]:
    """Fetch questions for a given tag from Stack Overflow.

    Args:
        tag: Stack Overflow tag string.
        page: Page number (1-indexed).

    Returns:
        List of question item dicts.
    """
    params = {
        "tagged": tag,
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": PAGE_SIZE,
        "page": page,
    }
    try:
        response = requests.get(
            f"{API_BASE}/questions", params=params, timeout=20
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error_id"):
            logger.error("API error: %s", data.get("error_message"))
            return []
        return data.get("items", [])
    except requests.exceptions.RequestException as exc:
        logger.error("SO API request failed for tag '%s': %s", tag, exc)
        return []


def fetch_answers(question_id: int) -> list[dict[str, Any]]:
    """Fetch answers for a specific question.

    Args:
        question_id: Stack Overflow question ID.

    Returns:
        List of answer item dicts sorted by score.
    """
    params = {
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": 5,
    }
    try:
        response = requests.get(
            f"{API_BASE}/questions/{question_id}/answers",
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to fetch answers for question %d: %s", question_id, exc)
        return []


def strip_html(text: str) -> str:
    """Remove HTML tags from a string.

    Args:
        text: Raw HTML string.

    Returns:
        Plain text with tags removed.
    """
    import re
    return re.sub(r"<[^>]+>", " ", text).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch SO questions and top answers and save to raw data directory."""
    Path("data/raw/docs").mkdir(parents=True, exist_ok=True)

    seen_ids: set[int] = set()
    collected: list[dict[str, Any]] = []

    logger.info("Starting Stack Overflow ingestion")

    for tag in TARGET_TAGS:
        logger.info("Fetching tag: %s", tag)

        for page in range(1, 4):
            questions = fetch_questions(tag, page)
            if not questions:
                break

            for q in questions:
                q_id: int = q.get("question_id", 0)
                if q_id in seen_ids:
                    continue

                # Gate 1 — minimum score
                score: int = q.get("score", 0)
                if score < MIN_QUESTION_SCORE:
                    continue

                # Gate 3 — must have answers
                answer_count: int = q.get("answer_count", 0)
                if answer_count == 0:
                    continue

                seen_ids.add(q_id)

                q_title: str = q.get("title", "")
                q_body: str = strip_html(q.get("body", ""))
                q_tags: list[str] = q.get("tags", [])
                q_url: str = q.get("link", "")
                q_owner: str = (q.get("owner") or {}).get("display_name", "")

                # Fetch top answer
                answers = fetch_answers(q_id)
                top_answer_text = ""
                if answers:
                    top_answer_text = strip_html(answers[0].get("body", ""))

                # Combine question + top answer into one document
                combined_text = (
                    f"QUESTION: {q_title}\n\n"
                    f"{q_body}\n\n"
                    f"TOP ANSWER:\n{top_answer_text}"
                )

                collected.append({
                    "source": "docs",
                    "source_type": "docs",
                    "credibility_tier": "implementation",
                    "content_type": "qa",
                    "title": q_title,
                    "author": q_owner,
                    "url": q_url,
                    "tags": q_tags,
                    "score": score,
                    "text": combined_text,
                    "ingested_at": datetime.utcnow().isoformat(),
                })

                time.sleep(REQUEST_DELAY)

            logger.info("Tag '%s' page %d: %d total collected", tag, page, len(collected))
            time.sleep(REQUEST_DELAY)

    output_path = Path(OUTPUT_PATH)
    output_path.write_text(
        json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d SO items to %s", len(collected), OUTPUT_PATH)


if __name__ == "__main__":
    main()