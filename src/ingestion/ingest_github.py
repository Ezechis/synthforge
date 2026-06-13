"""
SynthForge — GitHub Ingestion Script
Layer 1: Fetches top prompt engineering repositories and extracts
READMEs, issues, and Jupyter notebooks into structured JSON.

Usage: Usage: py src/ingestion/ingest_github.py
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from github import Github, GithubException
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    GITHUB_TOKEN,
    GITHUB_SEARCH_QUERY,
    GITHUB_TARGET_REPOS,
    GITHUB_RATE_LIMIT,
    DATA_RAW,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingest_github.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = DATA_RAW / "github"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RATE_LIMIT_PAUSE: float = 0.72  # seconds between requests — stays under 5000/hr
MAX_ISSUES_PER_REPO: int = 50
MAX_FILE_SIZE_BYTES: int = 1_000_000  # skip files over 1MB


def fetch_readme(repo) -> str | None:
    """Fetch the decoded README content for a repository.

    Args:
        repo: PyGitHub Repository object.

    Returns:
        Decoded README text or None if not found.
    """
    try:
        readme = repo.get_readme()
        return readme.decoded_content.decode("utf-8", errors="replace")
    except GithubException:
        logger.warning("No README found for %s", repo.full_name)
        return None


def fetch_issues(repo) -> list[dict]:
    """Fetch top issues from a repository.

    Args:
        repo: PyGitHub Repository object.

    Returns:
        List of issue dicts with title, body, comments, and metadata.
    """
    issues = []
    try:
        for i, issue in enumerate(repo.get_issues(state="open", sort="comments")):
            if i >= MAX_ISSUES_PER_REPO:
                break
            issues.append({
                "title": issue.title,
                "body": issue.body or "",
                "comments": issue.comments,
                "created_at": issue.created_at.isoformat(),
                "url": issue.html_url,
            })
            time.sleep(RATE_LIMIT_PAUSE)
    except GithubException as exc:
        logger.warning("Failed to fetch issues for %s: %s", repo.full_name, exc)
    return issues


def fetch_notebooks(repo) -> list[dict]:
    """Fetch Jupyter notebook contents from a repository.

    Args:
        repo: PyGitHub Repository object.

    Returns:
        List of notebook dicts with path and raw content.
    """
    notebooks = []
    try:
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            elif file_content.name.endswith(".ipynb"):
                if file_content.size > MAX_FILE_SIZE_BYTES:
                    logger.info("Skipping large notebook: %s", file_content.path)
                    continue
                try:
                    raw = file_content.decoded_content.decode("utf-8", errors="replace")
                    notebooks.append({
                        "path": file_content.path,
                        "content": raw,
                    })
                    time.sleep(RATE_LIMIT_PAUSE)
                except GithubException as exc:
                    logger.warning("Failed to decode notebook %s: %s", file_content.path, exc)
    except GithubException as exc:
        logger.warning("Failed to fetch notebooks for %s: %s", repo.full_name, exc)
    return notebooks


def process_repository(repo) -> dict:
    """Extract all relevant content from a single repository.

    Args:
        repo: PyGitHub Repository object.

    Returns:
        Structured dict with repo metadata and all extracted content.
    """
    logger.info("Processing: %s (stars: %d)", repo.full_name, repo.stargazers_count)
    return {
        "source": "github",
        "repo_full_name": repo.full_name,
        "repo_url": repo.html_url,
        "stars": repo.stargazers_count,
        "description": repo.description or "",
        "topics": repo.get_topics(),
        "last_updated": repo.updated_at.isoformat(),
        "ingested_at": datetime.utcnow().isoformat(),
        "readme": fetch_readme(repo),
        "issues": fetch_issues(repo),
        "notebooks": fetch_notebooks(repo),
    }


def run_ingestion() -> None:
    """Main ingestion function — searches GitHub and processes top repos.

    Raises:
        ValueError: If GITHUB_TOKEN is not set in environment.
        GithubException: If GitHub API returns an unrecoverable error.
    """
    if not GITHUB_TOKEN:
        raise ValueError(
            "GITHUB_TOKEN is not set. Add it to your .env file before running ingestion."
        )

    client = Github(GITHUB_TOKEN)
    logger.info("Authenticated as: %s", client.get_user().login)
    logger.info("Searching for top %d repos: '%s'", GITHUB_TARGET_REPOS, GITHUB_SEARCH_QUERY)

    repos = client.search_repositories(
        query=f"{GITHUB_SEARCH_QUERY} stars:>100",
        sort="stars",
        order="desc",
    )

    processed = 0
    for repo in repos:
        if processed >= GITHUB_TARGET_REPOS:
            break
        try:
            data = process_repository(repo)
            output_path = OUTPUT_DIR / f"{repo.full_name.replace('/', '_')}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Saved: %s", output_path.name)
            processed += 1
            time.sleep(RATE_LIMIT_PAUSE)
        except GithubException as exc:
            logger.error("Skipping %s due to API error: %s", repo.full_name, exc)
        except OSError as exc:
            logger.error("Failed to write file for %s: %s", repo.full_name, exc)

    logger.info("Ingestion complete. Repositories processed: %d", processed)


if __name__ == "__main__":
    run_ingestion()