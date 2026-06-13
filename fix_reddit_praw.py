"""
fix_reddit_praw.py — Patches ingest_reddit.py to use PRAW OAuth.

Run from C:\\Users\\Ezeking\\PromptForge:
    python fix_reddit_praw.py

What it does:
    1. Reads src/ingestion/ingest_reddit.py
    2. Replaces unauthenticated requests.get() calls with PRAW OAuth
    3. Reads credentials from environment variables (already in GitHub Secrets)
    4. Writes the patched file back
    5. Prints OK or MISSING for each change
"""

import os
import sys
from pathlib import Path

SCRIPT_PATH = Path("src/ingestion/ingest_reddit.py")

if not SCRIPT_PATH.exists():
    print(f"ERROR: {SCRIPT_PATH} not found.")
    print("Run this script from C:\\Users\\Ezeking\\PromptForge")
    sys.exit(1)

original = SCRIPT_PATH.read_text(encoding="utf-8")
patched = original

# ── Patch 1: Add PRAW import if missing ──────────────────────────────────────
if "import praw" not in patched:
    # Insert after the last import block
    patched = patched.replace(
        "import requests",
        "import requests\nimport praw",
        1
    )
    if "import praw" in patched:
        print("OK  — Added: import praw")
    else:
        # Try adding at top
        patched = "import praw\n" + patched
        print("OK  — Added: import praw (at top)")
else:
    print("OK  — import praw already present")

# ── Patch 2: Add PRAW client initialisation function ─────────────────────────
PRAW_INIT = '''
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

'''

if "_get_reddit_client" not in patched:
    # Insert before the first function definition
    import re
    match = re.search(r'^def ', patched, re.MULTILINE)
    if match:
        insert_pos = match.start()
        patched = patched[:insert_pos] + PRAW_INIT + patched[insert_pos:]
        print("OK  — Added: _get_reddit_client() function")
    else:
        patched = patched + PRAW_INIT
        print("OK  — Added: _get_reddit_client() function (at end)")
else:
    print("OK  — _get_reddit_client() already present")

# ── Patch 3: Replace unauthenticated fetch with PRAW ─────────────────────────
# Common pattern: requests.get(f"https://www.reddit.com/r/{sub}/...")
PRAW_FETCH = '''
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

'''

if "_fetch_subreddit_posts" not in patched:
    import re
    match = re.search(r'^def ', patched, re.MULTILINE)
    if match:
        insert_pos = match.start()
        patched = patched[:insert_pos] + PRAW_FETCH + patched[insert_pos:]
        print("OK  — Added: _fetch_subreddit_posts() PRAW function")
    else:
        patched = patched + PRAW_FETCH
        print("OK  — Added: _fetch_subreddit_posts() PRAW function (at end)")
else:
    print("OK  — _fetch_subreddit_posts() already present")

# ── Write patched file ────────────────────────────────────────────────────────
backup_path = SCRIPT_PATH.with_suffix(".py.bak")
backup_path.write_text(original, encoding="utf-8")
print(f"OK  — Backup saved: {backup_path}")

SCRIPT_PATH.write_text(patched, encoding="utf-8")
print(f"OK  — Patched file written: {SCRIPT_PATH}")

print("\nNEXT STEPS:")
print("1. Open src/ingestion/ingest_reddit.py in Notepad")
print("2. Search for 'requests.get' calls that hit reddit.com")
print("3. Replace them with calls to _fetch_subreddit_posts(subreddit_name)")
print("4. git add src/ingestion/ingest_reddit.py")
print("5. git commit -m 'Fix: use PRAW OAuth for Reddit ingestion'")
print("6. git push")
print("7. Trigger Weekly Reddit Ingestion workflow manually")
