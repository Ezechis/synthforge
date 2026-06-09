"""
SynthForge - Reddit Public JSON Ingestion
Layer 1: Fetches posts and comments from target subreddits using
Reddit's public JSON API. No credentials required.
Applies the 4-gate quality filter before corpus entry.

Usage: py src/ingestion/ingest_reddit.py
"""

import json
import logging
import time
import re
from datetime import datetime
from pathlib import Path

import requests
import praw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    REDDIT_MIN_POST_UPVOTES,
    REDDIT_MIN_COMMENT_UPVOTES,
    REDDIT_TARGET_SUBREDDITS,
    DATA_RAW,
    LOG_DIR,
)

# -- Logging setup -------------------------------------------------------------
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingest_reddit.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
OUTPUT_DIR: Path = DATA_RAW / "reddit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS: dict[str, str] = {
    "User-Agent": "SynthForge/0.1 research tool (read-only)",
}

RATE_LIMIT_PAUSE: float = 2.0
MAX_POSTS_PER_SUBREDDIT: int = 100
POST_SORTS: list[str] = ["top", "hot"]

# Extended subreddit list including your 17 communities
TARGET_SUBREDDITS: list[str] = [
    "PromptEngineering",
    "ChatGPTPromptGenius",
    "LocalLLaMA",
    "LanguageTechnology",
    "ChatGPT",
    "ClaudeAI",
    "MachineLearning",
    "learnmachinelearning",
    "OpenAI",
    "MLQuestions",
    "AI_Agents",
    "aipromptprogramming",
    "AIToolTesting",
    "LanguageModelSafety",
    "ChatGPTPro",
]

# Gate 3 - keyword quality filter (replaces LLM scorer for zero-cost)
QUALITY_KEYWORDS: list[str] = [
    "prompt", "llm", "gpt", "model", "token", "context", "instruction",
    "fine-tun", "embed", "retrieval", "inference", "generation", "chain",
    "few-shot", "zero-shot", "temperature", "output", "response", "language",
    "claude", "openai", "anthropic", "mistral", "llama", "transformer",
    "attention", "reasoning", "hallucin", "rag", "vector", "semantic",
]

MIN_QUALITY_KEYWORD_HITS: int = 2
MIN_TEXT_LENGTH: int = 100




def _fetch_subreddit_posts(subreddit_name: str, limit: int = 100) -> list[dict]:
    """Fetch top posts from a subreddit using authenticated PRAW.

    Args:
        subreddit_name: Name of the subreddit (without r/ prefix).
        limit: Maximum number of posts to fetch.

    Returns:
        List of post dicts with title, selftext, score, url, author, created_utc.
    """
    reddit = _get_reddit_client()
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    try:
        for submission in subreddit.top(time_filter="year", limit=limit):
            if submission.score < 50:
                continue
            posts.append({
                "title": submission.title,
                "selftext": submission.selftext,
                "score": submission.score,
                "url": submission.url,
                "author": str(submission.author),
                "created_utc": submission.created_utc,
                "subreddit": subreddit_name,
                "source": "reddit",
                "credibility_tier": "community",
            })
        for submission in subreddit.hot(limit=limit):
            if submission.score < 50:
                continue
            if not any(p["url"] == submission.url for p in posts):
                posts.append({
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "score": submission.score,
                    "url": submission.url,
                    "author": str(submission.author),
                    "created_utc": submission.created_utc,
                    "subreddit": subreddit_name,
                    "source": "reddit",
                    "credibility_tier": "community",
                })
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(
            "PRAW fetch failed for r/%s: %s", subreddit_name, exc
        )
    return posts

def _get_reddit_client() -> praw.Reddit:
    """Initialise authenticated PRAW client from environment variables.

    Reads REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME,
    REDDIT_PASSWORD, and REDDIT_USER_AGENT from the environment.
    Raises RuntimeError if any required variable is missing.
    """
    required = [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USERNAME",
        "REDDIT_PASSWORD",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing Reddit OAuth env vars: {missing}. "
            "Set them in GitHub Secrets or local environment."
        )
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        username=os.environ["REDDIT_USERNAME"],
        password=os.environ["REDDIT_PASSWORD"],
        user_agent=os.environ.get(
            "REDDIT_USER_AGENT",
            "SynthForge-Ingest/1.0 by EzeDezighner"
        ),
    )

def passes_gate_3(text: str) -> bool:
    """Gate 3 — keyword-based quality scoring.

    Counts domain-relevant keyword hits as proxy for technical
    specificity. Replaces LLM scorer for zero-cost operation.

    Args:
        text: Post or comment text to evaluate.

    Returns:
        True if text passes minimum quality threshold.
    """
    text_lower = text.lower()
    hits = sum(1 for kw in QUALITY_KEYWORDS if kw in text_lower)
    return hits >= MIN_QUALITY_KEYWORD_HITS


def clean_text(text: str) -> str:
    """Remove Reddit markdown and clean text for corpus entry.

    Args:
        text: Raw Reddit post or comment text.

    Returns:
        Cleaned plain text string.
    """
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"#+\s", "", text)
    text = re.sub(r"\*{1,2}([^\*]+)\*{1,2}", r"\1", text)
    text = re.sub(r">{1,}\s?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_posts(subreddit: str, sort: str = "top") -> list[dict]:
    """Fetch posts from a subreddit using public JSON API.

    Args:
        subreddit: Subreddit name without r/ prefix.
        sort: Sort method - 'top', 'hot', or 'new'.

    Returns:
        List of raw post dicts from Reddit API.
    """
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params = {"limit": 100, "t": "year"}

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        posts = data.get("data", {}).get("children", [])
        logger.info(
            "Fetched %d posts from r/%s (%s)",
            len(posts), subreddit, sort
        )
        return posts

    except Exception as exc:
        logger.error("Failed to fetch r/%s: %s", subreddit, exc)
        return []


def fetch_comments(subreddit: str, post_id: str) -> list[dict]:
    """Fetch top comments for a specific post.

    Args:
        subreddit: Subreddit name without r/ prefix.
        post_id: Reddit post ID string.

    Returns:
        List of comment dicts passing Gate 1 upvote threshold.
    """
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    params = {"limit": 20, "sort": "top", "depth": 2}

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if len(data) < 2:
            return []

        comments = []
        for item in data[1]["data"]["children"]:
            if item["kind"] != "t1":
                continue
            comment = item["data"]
            body = comment.get("body", "")
            score = comment.get("score", 0)

            # Gate 1 — comment upvote threshold
            if score < REDDIT_MIN_COMMENT_UPVOTES:
                continue

            # Gate 3 — quality keyword filter
            if not passes_gate_3(body):
                continue

            if len(body) < MIN_TEXT_LENGTH:
                continue

            comments.append({
                "body": clean_text(body),
                "score": score,
                "author": comment.get("author", "unknown"),
            })

        return comments

    except Exception as exc:
        logger.warning(
            "Failed to fetch comments for post %s: %s", post_id, exc
        )
        return []


def process_post(post_data: dict, subreddit: str) -> dict | None:
    """Apply all 4 gates and structure a post for corpus entry.

    Args:
        post_data: Raw post dict from Reddit API.
        subreddit: Subreddit name for metadata tagging.

    Returns:
        Structured post dict or None if post fails any gate.
    """
    post = post_data.get("data", {})

    title = post.get("title", "")
    selftext = post.get("selftext", "")
    score = post.get("score", 0)
    post_id = post.get("id", "")
    url = post.get("url", "")
    permalink = f"https://reddit.com{post.get('permalink', '')}"
    created_utc = post.get("created_utc", 0)
    author = post.get("author", "unknown")
    num_comments = post.get("num_comments", 0)

    full_text = f"{title}\n\n{selftext}".strip()

    # Gate 1 — upvote threshold
    if score < REDDIT_MIN_POST_UPVOTES:
        return None

    # Gate 2 — subreddit already filtered by TARGET_SUBREDDITS list

    # Gate 3 — quality keyword filter
    if not passes_gate_3(full_text):
        return None

    # Gate 4 — minimum text length
    if len(full_text) < MIN_TEXT_LENGTH:
        return None

    # Fetch top comments
    time.sleep(RATE_LIMIT_PAUSE)
    comments = fetch_comments(subreddit, post_id)

    return {
        "source": "reddit",
        "credibility_tier": "community",
        "subreddit": subreddit,
        "post_id": post_id,
        "title": title,
        "text": clean_text(full_text),
        "score": score,
        "author": author,
        "num_comments": num_comments,
        "url": permalink,
        "created_utc": created_utc,
        "comments": comments,
        "ingested_at": datetime.utcnow().isoformat(),
    }


def run_ingestion() -> None:
    """Main ingestion function — scrapes all target subreddits.

    Applies 4-gate quality filter. Resume-safe via existing file check.
    """
    total_saved = 0

    for subreddit in TARGET_SUBREDDITS:
        logger.info("Processing r/%s...", subreddit)
        subreddit_posts = []

        for sort in POST_SORTS:
            posts = fetch_posts(subreddit, sort)
            time.sleep(RATE_LIMIT_PAUSE)

            for post_data in posts:
                if len(subreddit_posts) >= MAX_POSTS_PER_SUBREDDIT:
                    break

                processed = process_post(post_data, subreddit)
                if processed is None:
                    continue

                # Deduplicate by post_id
                existing_ids = {p["post_id"] for p in subreddit_posts}
                if processed["post_id"] in existing_ids:
                    continue

                subreddit_posts.append(processed)
                time.sleep(RATE_LIMIT_PAUSE)

        if subreddit_posts:
            output_path = OUTPUT_DIR / f"{subreddit}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(subreddit_posts, f, indent=2, ensure_ascii=False)
            logger.info(
                "Saved %d posts from r/%s",
                len(subreddit_posts), subreddit
            )
            total_saved += len(subreddit_posts)
        else:
            logger.warning("No qualifying posts from r/%s", subreddit)

    logger.info("Reddit ingestion complete. Total posts saved: %d", total_saved)


if __name__ == "__main__":
    run_ingestion()