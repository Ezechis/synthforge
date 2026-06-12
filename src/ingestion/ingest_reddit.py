"""
SynthForge - Reddit Ingestion (PRAW OAuth with public-JSON fallback)
Layer 1: Fetches posts and comments from target subreddits and applies
the 4-gate quality filter before corpus entry.

Fetch strategy:
    - When REDDIT_* OAuth credentials are present in the environment
      (GitHub Actions), fetching uses authenticated PRAW exclusively.
      Authentication failure aborts with a non-zero exit. It NEVER
      silently falls back to the public API: in CI that produces a
      green run with zero data.
    - When no credentials are set (local dev convenience), falls back
      to Reddit's public JSON API.

Output: data/raw/reddit/<subreddit>.json - consumed by
src/processing/chunk_and_embed.py (SHA-256 chunk IDs make the
downstream embedding resume-safe).

Usage: py src/ingestion/ingest_reddit.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import requests

if TYPE_CHECKING:  # praw is only needed at runtime when credentials are set
    import praw

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
PRAW_TOP_TIME_FILTER: str = "year"
COMMENTS_PER_POST: int = 20

# OAuth credentials read from the environment (GitHub Secrets in Actions).
CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USERNAME",
    "REDDIT_PASSWORD",
)
DEFAULT_USER_AGENT: str = "SynthForge-Ingest/1.0 by EzeDezighner"

# Fetchers share one signature so process_post works on both paths:
# (subreddit, post_id) -> list of comment dicts.
CommentFetcher = Callable[[str, str], list[dict[str, Any]]]

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


# -- Credentials / PRAW client ---------------------------------------------------
def reddit_credentials_present() -> bool:
    """Return True when any Reddit OAuth credential env var is set.

    Any credential present means the run intends to be authenticated, so
    an incomplete set is treated as a configuration error by
    _get_reddit_client rather than a reason to fall back silently.

    Returns:
        True if at least one of CREDENTIAL_ENV_VARS is non-empty.
    """
    return any(os.environ.get(var) for var in CREDENTIAL_ENV_VARS)


def _get_reddit_client() -> "praw.Reddit":
    """Initialise and verify an authenticated PRAW client.

    Reads REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME,
    REDDIT_PASSWORD (required) and REDDIT_USER_AGENT (optional, has a
    default) from the environment, then performs an eager auth check -
    PRAW is lazy and would otherwise defer failure to the first fetch.

    Returns:
        Authenticated praw.Reddit instance.

    Raises:
        RuntimeError: If praw is not installed, any required env var is
            missing, or Reddit rejects the credentials.
    """
    try:
        import praw
    except ImportError as exc:
        raise RuntimeError("praw is not installed - run: pip install praw") from exc

    missing = [var for var in CREDENTIAL_ENV_VARS if not os.environ.get(var)]
    if missing:
        raise RuntimeError(
            f"Missing Reddit OAuth env vars: {missing}. "
            "Set them in GitHub Secrets or the local environment."
        )

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        username=os.environ["REDDIT_USERNAME"],
        password=os.environ["REDDIT_PASSWORD"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", DEFAULT_USER_AGENT),
    )
    try:
        authenticated_user = reddit.user.me()
    except Exception as exc:
        raise RuntimeError(f"Reddit OAuth authentication failed: {exc}") from exc

    logger.info("Reddit authenticated as u/%s", authenticated_user)
    return reddit


# -- Quality gates / cleaning ----------------------------------------------------
def passes_gate_3(text: str) -> bool:
    """Gate 3 - keyword-based quality scoring.

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


# -- Public JSON fetchers (local dev fallback - NO credentials only) -------------
def fetch_posts_public(subreddit: str, sort: str = "top") -> list[dict[str, Any]]:
    """Fetch posts from a subreddit using the public JSON API.

    Args:
        subreddit: Subreddit name without r/ prefix.
        sort: Sort method - 'top', 'hot', or 'new'.

    Returns:
        List of raw post dicts in Reddit's listing-child shape
        ({"data": {...}}), or empty list on failure.
    """
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params = {"limit": 100, "t": "year"}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        posts = data.get("data", {}).get("children", [])
        logger.info("Fetched %d posts from r/%s (%s)", len(posts), subreddit, sort)
        return posts
    except requests.RequestException as exc:
        logger.error("Failed to fetch r/%s: %s", subreddit, exc)
        return []


def fetch_comments_public(subreddit: str, post_id: str) -> list[dict[str, Any]]:
    """Fetch top comments for a post using the public JSON API.

    Args:
        subreddit: Subreddit name without r/ prefix.
        post_id: Reddit post ID string.

    Returns:
        List of comment dicts ({body, score, author}) passing all gates.
    """
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    params = {"limit": COMMENTS_PER_POST, "sort": "top", "depth": 2}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch comments for post %s: %s", post_id, exc)
        return []

    if len(data) < 2:
        return []

    comments: list[dict[str, Any]] = []
    for item in data[1]["data"]["children"]:
        if item["kind"] != "t1":
            continue
        comment = item["data"]
        filtered = _filter_comment(
            body=comment.get("body", ""),
            score=comment.get("score", 0),
            author=comment.get("author", "unknown"),
        )
        if filtered is not None:
            comments.append(filtered)
    return comments


# -- PRAW fetchers (authenticated path) -------------------------------------------
def fetch_posts_praw(
    reddit: "praw.Reddit",
    subreddit: str,
    sort: str = "top",
) -> list[dict[str, Any]]:
    """Fetch posts via authenticated PRAW, shaped like public JSON children.

    Uses subreddit.top(time_filter='year', limit=100) for 'top' and
    subreddit.hot(limit=100) for 'hot', mirroring the public endpoint
    parameters so both paths feed process_post identically.

    Args:
        reddit: Verified PRAW Reddit instance.
        subreddit: Subreddit name without r/ prefix.
        sort: Sort method - 'top' or 'hot'.

    Returns:
        List of post dicts in the listing-child shape ({"data": {...}}).

    Raises:
        ValueError: If sort is not 'top' or 'hot'.
    """
    sub = reddit.subreddit(subreddit)
    if sort == "top":
        listing = sub.top(time_filter=PRAW_TOP_TIME_FILTER, limit=MAX_POSTS_PER_SUBREDDIT)
    elif sort == "hot":
        listing = sub.hot(limit=MAX_POSTS_PER_SUBREDDIT)
    else:
        raise ValueError(f"Unsupported sort for PRAW path: {sort!r}")

    posts: list[dict[str, Any]] = []
    try:
        for submission in listing:
            posts.append({
                "data": {
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "score": submission.score,
                    "id": submission.id,
                    "url": submission.url,
                    "permalink": submission.permalink,
                    "created_utc": submission.created_utc,
                    "author": str(submission.author),
                    "num_comments": submission.num_comments,
                },
            })
    except Exception as exc:
        # Subreddit-level failures (private/banned/quarantined) - skip the
        # subreddit but keep the run going. Auth itself was verified earlier.
        logger.error("PRAW fetch failed for r/%s (%s): %s", subreddit, sort, exc)
        return posts

    logger.info("Fetched %d posts from r/%s (%s) via PRAW", len(posts), subreddit, sort)
    return posts


def fetch_comments_praw(
    reddit: "praw.Reddit",
    subreddit: str,
    post_id: str,
) -> list[dict[str, Any]]:
    """Fetch top comments for a post via authenticated PRAW.

    Args:
        reddit: Verified PRAW Reddit instance.
        subreddit: Subreddit name (logging only - PRAW resolves by ID).
        post_id: Reddit post ID string.

    Returns:
        List of comment dicts ({body, score, author}) passing all gates.
    """
    try:
        submission = reddit.submission(id=post_id)
        submission.comment_sort = "top"
        submission.comments.replace_more(limit=0)
        top_level = list(submission.comments)[:COMMENTS_PER_POST]
    except Exception as exc:
        logger.warning(
            "PRAW comment fetch failed for r/%s post %s: %s",
            subreddit, post_id, exc,
        )
        return []

    comments: list[dict[str, Any]] = []
    for comment in top_level:
        body = getattr(comment, "body", "")
        filtered = _filter_comment(
            body=body,
            score=getattr(comment, "score", 0),
            author=str(getattr(comment, "author", "unknown")),
        )
        if filtered is not None:
            comments.append(filtered)
    return comments


def _filter_comment(body: str, score: int, author: str) -> dict[str, Any] | None:
    """Apply the comment gates shared by both fetch paths.

    Gate 1: comment upvote threshold. Gate 3: keyword quality filter.
    Plus minimum text length.

    Args:
        body: Raw comment body text.
        score: Comment upvote score.
        author: Comment author name.

    Returns:
        Structured comment dict, or None if the comment fails any gate.
    """
    if score < REDDIT_MIN_COMMENT_UPVOTES:
        return None
    if not passes_gate_3(body):
        return None
    if len(body) < MIN_TEXT_LENGTH:
        return None
    return {"body": clean_text(body), "score": score, "author": author}


# -- Post processing ---------------------------------------------------------------
def process_post(
    post_data: dict[str, Any],
    subreddit: str,
    comment_fetcher: CommentFetcher,
) -> dict[str, Any] | None:
    """Apply all 4 gates and structure a post for corpus entry.

    Args:
        post_data: Raw post dict in listing-child shape ({"data": {...}}).
        subreddit: Subreddit name for metadata tagging.
        comment_fetcher: Callable fetching comments for (subreddit, post_id)
            via whichever fetch path this run uses.

    Returns:
        Structured post dict or None if post fails any gate.
    """
    post = post_data.get("data", {})

    title = post.get("title", "")
    selftext = post.get("selftext", "")
    score = post.get("score", 0)
    post_id = post.get("id", "")
    permalink = f"https://reddit.com{post.get('permalink', '')}"
    created_utc = post.get("created_utc", 0)
    author = post.get("author", "unknown")
    num_comments = post.get("num_comments", 0)

    full_text = f"{title}\n\n{selftext}".strip()

    # Gate 1 - upvote threshold
    if score < REDDIT_MIN_POST_UPVOTES:
        return None

    # Gate 2 - subreddit already filtered by TARGET_SUBREDDITS list

    # Gate 3 - quality keyword filter
    if not passes_gate_3(full_text):
        return None

    # Gate 4 - minimum text length
    if len(full_text) < MIN_TEXT_LENGTH:
        return None

    # Fetch top comments
    time.sleep(RATE_LIMIT_PAUSE)
    comments = comment_fetcher(subreddit, post_id)

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


# -- Main ingestion ------------------------------------------------------------------
def run_ingestion() -> None:
    """Main ingestion function - scrapes all target subreddits.

    Selects the fetch path by credential presence: authenticated PRAW
    when any REDDIT_* credential env var is set (failing hard if auth
    fails), public JSON API only when none are set. Applies the 4-gate
    quality filter. Resume-safe downstream via SHA-256 chunk IDs.

    Raises:
        RuntimeError: If credentials are present but PRAW cannot
            authenticate (propagated from _get_reddit_client).
    """
    authenticated = reddit_credentials_present()
    if authenticated:
        # Raises RuntimeError on any auth problem - NEVER falls back to
        # the public API, which 403s in CI and yields green-but-empty runs.
        reddit = _get_reddit_client()
        logger.info("Fetch mode: authenticated PRAW OAuth")
        post_fetcher = partial(fetch_posts_praw, reddit)
        comment_fetcher: CommentFetcher = partial(fetch_comments_praw, reddit)
    else:
        logger.warning(
            "No REDDIT_* credentials in environment - using public JSON "
            "API (local dev fallback; expect 403s from datacenter IPs)."
        )
        post_fetcher = fetch_posts_public
        comment_fetcher = fetch_comments_public

    total_saved = 0

    for subreddit in TARGET_SUBREDDITS:
        logger.info("Processing r/%s...", subreddit)
        subreddit_posts: list[dict[str, Any]] = []

        for sort in POST_SORTS:
            posts = post_fetcher(subreddit, sort)
            time.sleep(RATE_LIMIT_PAUSE)

            for post_data in posts:
                if len(subreddit_posts) >= MAX_POSTS_PER_SUBREDDIT:
                    break

                processed = process_post(post_data, subreddit, comment_fetcher)
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

    if authenticated and total_saved == 0:
        # 15 subreddits of year-top posts can never legitimately all be
        # empty - zero output in authenticated mode means something broke.
        logger.error(
            "Authenticated run saved 0 posts across all subreddits - "
            "failing the run instead of reporting a green no-op."
        )
        sys.exit(1)


if __name__ == "__main__":
    try:
        run_ingestion()
    except RuntimeError as exc:
        logger.error("FATAL: %s", exc)
        sys.exit(1)
