"""
ingest_rss_feeds.py -- PromptForge Layer 1: RSS/Atom Feed Aggregator
====================================================================
Ingests AI research newsletters and company blog posts via RSS/Atom feeds.
Covers Substack newsletters, researcher blogs, and company research blogs.
No API key required — all feeds are publicly accessible.

Quality gates:
    Gate 1 -- Minimum content length (MIN_CONTENT_LENGTH)
    Gate 2 -- Curated feed whitelist only
    Gate 3 -- HTML tag stripping for clean text
    Gate 4 -- Metadata tagging with per-feed credibility tier

Run from project root:
    python src/ingestion/ingest_rss_feeds.py
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_PATH: str = "data/raw/docs/rss_feeds.json"
MIN_CONTENT_LENGTH: int = 300
REQUEST_DELAY: float = 1.5
REQUEST_TIMEOUT: int = 20

# Curated feed list — each entry defines credibility and content type
FEEDS: list[dict[str, str]] = [
    # ── Researcher newsletters (highest epistemic quality) ────────────────
    {
        "url": "https://magazine.sebastianraschka.com/feed",
        "name": "Ahead of AI — Sebastian Raschka",
        "credibility_tier": "primary",
        "content_type": "newsletter",
    },
    {
        "url": "https://karpathy.github.io/feed.xml",
        "name": "Andrej Karpathy Blog",
        "credibility_tier": "primary",
        "content_type": "blog",
    },
    {
        "url": "https://lilianweng.github.io/index.xml",
        "name": "Lilian Weng Blog",
        "credibility_tier": "primary",
        "content_type": "blog",
    },
    # ── Company research blogs ─────────────────────────────────────────────
    {
        "url": "https://openai.com/blog/rss.xml",
        "name": "OpenAI Blog",
        "credibility_tier": "primary",
        "content_type": "research_blog",
    },
    {
        "url": "https://huggingface.co/blog/feed.xml",
        "name": "Hugging Face Blog",
        "credibility_tier": "implementation",
        "content_type": "technical_blog",
    },
    # ── AI newsletters ─────────────────────────────────────────────────────
    {
        "url": "https://importai.substack.com/feed",
        "name": "Import AI — Jack Clark",
        "credibility_tier": "primary",
        "content_type": "newsletter",
    },
    {
        "url": "https://thealgorithmicbridge.substack.com/feed",
        "name": "The Algorithmic Bridge",
        "credibility_tier": "primary",
        "content_type": "newsletter",
    },
    {
        "url": "https://lastweekinai.substack.com/feed",
        "name": "Last Week in AI",
        "credibility_tier": "community",
        "content_type": "newsletter",
    },
    {
        "url": "https://simonwillison.net/atom/everything/",
        "name": "Simon Willison Blog",
        "credibility_tier": "implementation",
        "content_type": "blog",
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_html(html: str) -> str:
    """Remove HTML tags and clean whitespace from a string.

    Args:
        html: Raw HTML or mixed text string.

    Returns:
        Plain text with tags removed and whitespace normalised.
    """
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_full_content(url: str) -> str:
    """Attempt to fetch the full article text from a URL.

    Falls back to empty string on any network or parse error.

    Args:
        url: Article URL to fetch.

    Returns:
        Extracted article text, or empty string on failure.
    """
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "PromptForge-Bot/1.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove nav, header, footer, scripts
        for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()
        # Try article body first, fall back to main, then body
        for selector in ["article", "main", ".post-content", ".entry-content", "body"]:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(separator=" ")
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) >= MIN_CONTENT_LENGTH:
                    return text
    except Exception as exc:
        logger.debug("Could not fetch full content from %s: %s", url, exc)
    return ""


def parse_feed(feed_config: dict[str, str]) -> list[dict[str, Any]]:
    """Parse one RSS/Atom feed and return cleaned post dicts.

    Args:
        feed_config: Dict with url, name, credibility_tier, content_type.

    Returns:
        List of post dicts ready for the corpus.
    """
    feed_url = feed_config["url"]
    feed_name = feed_config["name"]
    tier = feed_config["credibility_tier"]
    content_type = feed_config["content_type"]

    try:
        parsed = feedparser.parse(feed_url)
    except Exception as exc:
        logger.error("feedparser failed for %s: %s", feed_name, exc)
        return []

    if not parsed.entries:
        logger.warning("No entries found in feed: %s", feed_name)
        return []

    posts: list[dict[str, Any]] = []

    for entry in parsed.entries:
        title: str = entry.get("title", "")
        url: str = entry.get("link", "")
        author: str = entry.get("author", feed_name)
        published: str = entry.get("published", entry.get("updated", ""))

        # Extract text from feed summary first
        summary_html: str = (
            entry.get("content", [{}])[0].get("value", "")
            or entry.get("summary", "")
        )
        text: str = strip_html(summary_html)

        # If feed content is too short, fetch the full article
        if len(text) < MIN_CONTENT_LENGTH and url:
            logger.debug("Short summary for '%s' — fetching full article", title)
            full_text = fetch_full_content(url)
            if full_text:
                text = full_text
            time.sleep(REQUEST_DELAY)

        if len(text) < MIN_CONTENT_LENGTH:
            continue

        posts.append({
            "source": "docs",
            "source_type": "docs",
            "credibility_tier": tier,
            "content_type": content_type,
            "feed_name": feed_name,
            "title": title,
            "author": author,
            "url": url,
            "published": published,
            "text": text,
            "ingested_at": datetime.utcnow().isoformat(),
        })

    return posts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch all RSS feeds and save to the raw data directory."""
    Path("data/raw/docs").mkdir(parents=True, exist_ok=True)

    all_posts: list[dict[str, Any]] = []

    for feed_config in FEEDS:
        feed_name = feed_config["name"]
        logger.info("Fetching: %s", feed_name)

        posts = parse_feed(feed_config)
        all_posts.extend(posts)
        logger.info(
            "%s — %d posts collected (total: %d)",
            feed_name, len(posts), len(all_posts),
        )
        time.sleep(REQUEST_DELAY)

    output_path = Path(OUTPUT_PATH)
    output_path.write_text(
        json.dumps(all_posts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d RSS posts to %s", len(all_posts), OUTPUT_PATH)


if __name__ == "__main__":
    main()