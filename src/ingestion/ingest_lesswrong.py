"""
ingest_lesswrong.py -- SynthForge Layer 1: LessWrong Data Source
=================================================================
Fetches high-karma LessWrong posts on AI/ML/prompt engineering topics
via the public GraphQL API. No authentication required.

Quality gates:
    Gate 1 -- Minimum karma score (MIN_KARMA)
    Gate 2 -- Keyword relevance filter on title and tags
    Gate 3 -- Minimum word count (avoids stubs)
    Gate 4 -- Metadata tagging with credibility_tier=primary

Run from project root:
    python src/ingestion/ingest_lesswrong.py
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

GRAPHQL_URL: str = "https://www.lesswrong.com/graphql"
OUTPUT_PATH: str = "data/raw/docs/lesswrong.json"
MIN_KARMA: int = 30
MIN_WORD_COUNT: int = 200
MAX_POSTS: int = 200
BATCH_SIZE: int = 50
REQUEST_DELAY: float = 1.5

RELEVANCE_KEYWORDS: set[str] = {
    "prompt", "llm", "language model", "gpt", "claude", "gemini",
    "chain of thought", "few-shot", "zero-shot", "reasoning", "alignment",
    "rlhf", "fine-tun", "transformer", "agent", "rag", "retrieval",
    "hallucin", "instruction", "in-context", "embeddings", "inference",
}

GRAPHQL_QUERY: str = """
query GetTopPosts($limit: Int!, $offset: Int!) {
  posts(input: {
    terms: {
      view: "top"
      limit: $limit
      offset: $offset
    }
  }) {
    results {
      _id
      title
      slug
      baseScore
      wordCount
      postedAt
      tags {
        name
      }
      user {
        displayName
        username
      }
      contents {
        markdown
      }
    }
  }
}
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_relevant(title: str, tags: list[str], text: str) -> bool:
    """Return True if the post is relevant to prompt engineering or LLMs.

    Args:
        title: Post title.
        tags: List of tag name strings.
        text: Post body text.

    Returns:
        True if any relevance keyword found in title, tags, or first 500 chars.
    """
    haystack = " ".join([title] + tags + [text[:500]]).lower()
    return any(kw in haystack for kw in RELEVANCE_KEYWORDS)


def fetch_batch(offset: int) -> list[dict[str, Any]]:
    """Fetch one batch of posts from the LessWrong GraphQL API.

    Args:
        offset: Pagination offset.

    Returns:
        List of raw post dicts from the API.
    """
    payload = {
        "query": GRAPHQL_QUERY,
        "variables": {"limit": BATCH_SIZE, "offset": offset},
    }
    try:
        response = requests.post(
            GRAPHQL_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("posts", {}).get("results", [])
    except requests.exceptions.RequestException as exc:
        logger.error("LessWrong API request failed at offset %d: %s", offset, exc)
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch and save relevant LessWrong posts to the raw data directory."""
    Path("data/raw/docs").mkdir(parents=True, exist_ok=True)

    collected: list[dict[str, Any]] = []
    offset = 0
    consecutive_empty = 0

    logger.info("Starting LessWrong ingestion — target: %d posts", MAX_POSTS)

    while len(collected) < MAX_POSTS:
        batch = fetch_batch(offset)

        if not batch:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                logger.info("Three empty batches — stopping.")
                break
            offset += BATCH_SIZE
            time.sleep(REQUEST_DELAY)
            continue

        consecutive_empty = 0

        for post in batch:
            if len(collected) >= MAX_POSTS:
                break

            karma: int = post.get("baseScore", 0) or 0
            word_count: int = post.get("wordCount", 0) or 0
            title: str = post.get("title", "") or ""
            tags: list[str] = [t.get("name", "") for t in (post.get("tags") or [])]
            contents = post.get("contents") or {}
            text: str = contents.get("markdown", "") or ""
            author_info = post.get("user") or {}
            author: str = author_info.get("displayName", author_info.get("username", ""))

            # Gate 1 — karma threshold
            if karma < MIN_KARMA:
                continue

            # Gate 2 — relevance filter
            if not is_relevant(title, tags, text):
                continue

            # Gate 3 — minimum content length
            if word_count < MIN_WORD_COUNT or len(text.strip()) < 200:
                continue

            slug: str = post.get("slug", post.get("_id", ""))
            url: str = f"https://www.lesswrong.com/posts/{post.get('_id', '')}/{slug}"

            collected.append({
                "source": "docs",
                "source_type": "docs",
                "credibility_tier": "primary",
                "content_type": "essay",
                "title": title,
                "author": author,
                "url": url,
                "tags": tags,
                "karma": karma,
                "text": text,
                "ingested_at": datetime.utcnow().isoformat(),
            })

        logger.info(
            "Offset %d — batch size %d — collected so far: %d",
            offset, len(batch), len(collected),
        )
        offset += BATCH_SIZE
        time.sleep(REQUEST_DELAY)

    output_path = Path(OUTPUT_PATH)
    output_path.write_text(
        json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d LessWrong posts to %s", len(collected), OUTPUT_PATH)


if __name__ == "__main__":
    main()