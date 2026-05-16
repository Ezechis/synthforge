"""
src/ingestion/ingest_reddit_public.py
=======================================
Reddit ingestion using Reddit's public JSON API.
Zero credentials required — no PRAW, no OAuth, no API registration.

Reddit exposes public subreddit data at:
  https://www.reddit.com/r/{subreddit}/{sort}.json

The same 4-gate quality filter from the original design is applied:
  Gate 1 — Quantitative threshold (score >= 50 for posts, >= 20 for comments)
  Gate 2 — Subreddit curation (5 target communities)
  Gate 3 — LLM quality scoring via Groq (1-5, discard < 3)
  Gate 4 — Metadata tagging (source=reddit, credibility_tier=community)

Output: JSONL file at data/raw/reddit_public.jsonl
        (processed by chunk_and_embed.py on next run)

Environment variables:
    GROQ_API_KEY  — used for Gate 3 quality scoring (optional: skips gate if absent)
    RAW_DATA_DIR  — output directory (default: data/raw)

Author: Ezechinyere Nnabugwu / DeepForge
"""

import json
import logging
import os
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

# Gate 2: Target subreddits
TARGET_SUBREDDITS: list[str] = [
    "PromptEngineering",
    "LocalLLaMA",
    "MachineLearning",
    "LanguageModelSafety",
    "ChatGPTPro",
]

# Sorts to pull from each subreddit
SORTS: list[str] = ["hot", "top", "new"]
TIME_FILTER: str = "month"           # for "top" sort
POSTS_PER_SORT: int = 100            # max per request (Reddit API limit)

# Gate 1: Thresholds
MIN_POST_SCORE: int  = 50
MIN_COMMENT_SCORE: int = 20

# Gate 3: LLM quality scoring
MIN_QUALITY_SCORE: float = 3.0       # below this = discard
GROQ_MODEL: str = "llama-3.3-70b-versatile"

RAW_DATA_DIR: Path = Path(os.environ.get("RAW_DATA_DIR", "data/raw"))
OUTPUT_FILE: Path = RAW_DATA_DIR / "reddit_public.jsonl"
PROGRESS_FILE: Path = RAW_DATA_DIR / "reddit_seen_ids.json"

# Respectful rate limiting — Reddit asks for 1 request per 2 seconds
REQUEST_DELAY_SECONDS: float = 2.0

# User-Agent required by Reddit — identify yourself
HEADERS: dict[str, str] = {
    "User-Agent": "PromptForge/1.0 (academic research; prompt engineering knowledge base)",
}


# ---------------------------------------------------------------------------
# Progress tracking — avoid re-ingesting posts already seen
# ---------------------------------------------------------------------------

def load_seen_ids() -> set[str]:
    """Load the set of already-processed post IDs."""
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            seen = set(data.get("seen", []))
            logger.info("Loaded %d previously seen post IDs.", len(seen))
            return seen
        except Exception as exc:
            logger.warning("Could not load progress file: %s", exc)
    return set()


def save_seen_ids(seen: set[str]) -> None:
    """Persist seen post IDs."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps({"seen": list(seen), "updated": datetime.utcnow().isoformat()}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Gate 1 + 2: Fetch posts from Reddit JSON API
# ---------------------------------------------------------------------------

def fetch_subreddit_posts(subreddit: str, sort: str) -> list[dict]:
    """
    Fetch posts from a subreddit via the public Reddit JSON API.

    Args:
        subreddit: Subreddit name (without r/).
        sort:      Sort order — 'hot', 'top', or 'new'.

    Returns:
        List of raw post dicts passing Gate 1 score threshold.
    """
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params: dict[str, str | int] = {"limit": POSTS_PER_SORT}
    if sort == "top":
        params["t"] = TIME_FILTER

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
        resp.raise_for_status()
        children = resp.json()["data"]["children"]
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error fetching r/%s/%s: %s", subreddit, sort, exc)
        return []
    except Exception as exc:
        logger.error("Error fetching r/%s/%s: %s", subreddit, sort, exc)
        return []

    passing = []
    for child in children:
        post = child["data"]
        score = post.get("score", 0)
        # Gate 1: minimum score
        if score < MIN_POST_SCORE:
            continue
        # Skip deleted / removed content
        selftext = post.get("selftext", "").strip()
        if selftext in ("", "[deleted]", "[removed]"):
            continue

        passing.append({
            "id":        post.get("id", ""),
            "title":     post.get("title", "")[:500],
            "text":      selftext,
            "score":     score,
            "url":       f"https://www.reddit.com{post.get('permalink', '')}",
            "subreddit": subreddit,
            "author":    post.get("author", "unknown"),
            "created":   datetime.utcfromtimestamp(
                            post.get("created_utc", 0)
                         ).isoformat(),
        })

    logger.info("r/%s/%s: %d posts fetched, %d passed Gate 1",
                subreddit, sort, len(children), len(passing))
    time.sleep(REQUEST_DELAY_SECONDS)
    return passing


def fetch_top_comments(post_url: str, n: int = 5) -> list[dict]:
    """
    Fetch top-level comments for a post via the JSON API.

    Args:
        post_url: Reddit permalink URL.
        n:        Maximum number of comments to return.

    Returns:
        List of comment dicts passing Gate 1 score threshold.
    """
    json_url = post_url.rstrip("/") + ".json"
    try:
        resp = requests.get(
            json_url,
            headers=HEADERS,
            params={"limit": 20, "sort": "top"},
            timeout=20,
        )
        resp.raise_for_status()
        comments_data = resp.json()
        if len(comments_data) < 2:
            return []
        children = comments_data[1]["data"]["children"]
    except Exception as exc:
        logger.debug("Could not fetch comments for %s: %s", post_url, exc)
        return []

    passing = []
    for child in children:
        if child.get("kind") != "t1":
            continue
        c = child["data"]
        score = c.get("score", 0)
        body  = c.get("body", "").strip()
        if score < MIN_COMMENT_SCORE or body in ("", "[deleted]", "[removed]"):
            continue
        passing.append({
            "id":     c.get("id", ""),
            "text":   body,
            "score":  score,
            "author": c.get("author", "unknown"),
        })
        if len(passing) >= n:
            break

    time.sleep(REQUEST_DELAY_SECONDS)
    return passing


# ---------------------------------------------------------------------------
# Gate 3: LLM quality scoring via Groq
# ---------------------------------------------------------------------------

def score_chunk_quality(text: str) -> float:
    """
    Score a text chunk 1-5 for technical specificity, actionability, accuracy.

    Uses Groq llama-3.3-70b-versatile as a cheap classifier.
    Returns 3.0 (neutral pass) if GROQ_API_KEY is absent.

    Args:
        text: The chunk text to score.

    Returns:
        Float score between 1.0 and 5.0.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        logger.debug("GROQ_API_KEY absent — Gate 3 skipped, defaulting to 3.0")
        return 3.0

    prompt = (
        "Rate this text on technical specificity, actionability, and accuracy "
        "for a prompt engineering knowledge base. "
        "Reply with ONLY a single number between 1 and 5. No explanation.\n\n"
        f"TEXT:\n{text[:800]}"
    )
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,
                "temperature": 0.0,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        score = float(raw.split()[0])
        return max(1.0, min(5.0, score))
    except Exception as exc:
        logger.debug("Quality scoring failed: %s — defaulting to 3.0", exc)
        return 3.0


# ---------------------------------------------------------------------------
# Gate 4: Build metadata-tagged chunk
# ---------------------------------------------------------------------------

def build_chunk(text: str, post: dict, chunk_type: str = "post") -> dict:
    """
    Build a metadata-tagged chunk ready for chunk_and_embed.py.

    Args:
        text:       The text content of the chunk.
        post:       Parent post dict.
        chunk_type: 'post' or 'comment'.

    Returns:
        Chunk dict with all required metadata fields.
    """
    return {
        "text":             text,
        "source":           "reddit",
        "credibility_tier": "community",    # Gate 4 tag — downweighted at generation
        "subreddit":        post["subreddit"],
        "author":           post.get("author", "unknown"),
        "score":            post.get("score", 0),
        "url":              post.get("url", ""),
        "title":            post.get("title", ""),
        "chunk_type":       chunk_type,
        "created":          post.get("created", ""),
        "ingested":         datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full 4-gate Reddit ingestion pipeline."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen_ids   = load_seen_ids()
    new_chunks: list[dict] = []
    new_ids:    set[str]   = set()

    total_fetched = 0
    gate1_passed  = 0
    gate3_passed  = 0

    for subreddit in TARGET_SUBREDDITS:
        for sort in SORTS:
            posts = fetch_subreddit_posts(subreddit, sort)
            total_fetched += len(posts)

            for post in posts:
                post_id = post["id"]
                if post_id in seen_ids or post_id in new_ids:
                    continue

                gate1_passed += 1
                combined_text = f"{post['title']}\n\n{post['text']}"

                # Gate 3: quality score
                quality = score_chunk_quality(combined_text)
                if quality < MIN_QUALITY_SCORE:
                    logger.debug("Gate 3 reject (%.1f): %s", quality, post["title"][:60])
                    continue

                gate3_passed += 1

                # Gate 4: tag and save post chunk
                new_chunks.append(build_chunk(combined_text, post, chunk_type="post"))
                new_ids.add(post_id)

                # Fetch and score top comments
                if post.get("url"):
                    comments = fetch_top_comments(post["url"], n=5)
                    for comment in comments:
                        comment_quality = score_chunk_quality(comment["text"])
                        if comment_quality >= MIN_QUALITY_SCORE:
                            comment_post = {**post, "author": comment["author"],
                                            "score": comment["score"]}
                            new_chunks.append(
                                build_chunk(comment["text"], comment_post, chunk_type="comment")
                            )

    # Append to output JSONL (one chunk per line)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fh:
        for chunk in new_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    seen_ids.update(new_ids)
    save_seen_ids(seen_ids)

    logger.info("=" * 50)
    logger.info("Reddit ingestion complete.")
    logger.info("  Fetched:      %d posts", total_fetched)
    logger.info("  Gate 1 pass:  %d posts", gate1_passed)
    logger.info("  Gate 3 pass:  %d posts", gate3_passed)
    logger.info("  New chunks:   %d", len(new_chunks))
    logger.info("  Output:       %s", OUTPUT_FILE)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
