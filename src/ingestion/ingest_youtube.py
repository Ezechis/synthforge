"""
ingest_youtube.py -- PromptForge Layer 1: YouTube Transcript Source
===================================================================
Fetches transcripts from curated LLM and prompt engineering playlists
using yt-dlp (video ID extraction) and youtube-transcript-api (captions).
No API key required.

Quality gates:
    Gate 1 -- Minimum transcript word count (MIN_WORDS)
    Gate 2 -- English language transcripts only
    Gate 3 -- Curated playlist whitelist (no random channels)
    Gate 4 -- Metadata tagging with source credibility tier

Run from project root:
    python src/ingestion/ingest_youtube.py
"""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_PATH: str = "data/raw/docs/youtube_transcripts.json"
MIN_WORDS: int = 800
MAX_VIDEOS_PER_PLAYLIST: int = 30
REQUEST_DELAY: float = 1.0

# Curated playlists — Tier 1: academic and high-signal technical content
PLAYLISTS: list[dict[str, Any]] = [
    {
        "url": "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ",
        "channel": "Andrej Karpathy",
        "description": "Neural Networks: Zero to Hero",
        "credibility_tier": "primary",
    },
    {
        "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
        "channel": "3Blue1Brown",
        "description": "Neural Networks series",
        "credibility_tier": "primary",
    },
    {
        "url": "https://www.youtube.com/@YannicKilcher/videos",
        "channel": "Yannic Kilcher",
        "description": "AI paper deep dives",
        "credibility_tier": "primary",
    },
    {
        "url": "https://www.youtube.com/@karpathy/videos",
        "channel": "Andrej Karpathy",
        "description": "Andrej Karpathy channel",
        "credibility_tier": "primary",
    },
    {
        "url": "https://www.youtube.com/@DeepLearningAI/videos",
        "channel": "DeepLearning.AI",
        "description": "Prompt engineering and LLM courses",
        "credibility_tier": "primary",
    },
    {
        "url": "https://www.youtube.com/@HuggingFace/videos",
        "channel": "Hugging Face",
        "description": "Open-source LLM workshops",
        "credibility_tier": "implementation",
    },
    {
        "url": "https://www.youtube.com/@LangChain/videos",
        "channel": "LangChain",
        "description": "LangChain agents and RAG",
        "credibility_tier": "implementation",
    },
    {
        "url": "https://www.youtube.com/@matthew_berman/videos",
        "channel": "Matthew Berman",
        "description": "Practical prompt engineering",
        "credibility_tier": "community",
    },
    {
        "url": "https://www.youtube.com/@aiexplained-official/videos",
        "channel": "AI Explained",
        "description": "AI advancement analysis",
        "credibility_tier": "community",
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

def get_video_ids(playlist_url: str, max_videos: int) -> list[str]:
    """Extract video IDs from a YouTube playlist or channel using yt-dlp.

    Args:
        playlist_url: Full YouTube playlist or channel URL.
        max_videos: Maximum number of video IDs to return.

    Returns:
        List of YouTube video ID strings.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--playlist-end", str(max_videos),
        "--print", "id",
        "--no-warnings",
        "--quiet",
        playlist_url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return ids[:max_videos]
    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timed out for: %s", playlist_url)
        return []
    except Exception as exc:
        logger.error("yt-dlp failed for %s: %s", playlist_url, exc)
        return []


def get_transcript(video_id: str) -> str:
    """Fetch the English transcript for a YouTube video.

    Args:
        video_id: YouTube video ID (11 characters).

    Returns:
        Full transcript as a single string, or empty string on failure.
    """
    try:
        segments = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-GB"], cookies="cookies.txt"
        )
        return " ".join(seg["text"] for seg in segments).strip()
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        return ""
    except Exception as exc:
        logger.warning("Transcript error for %s: %s", video_id, exc)
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch transcripts from all curated playlists and save to disk."""
    Path("data/raw/docs").mkdir(parents=True, exist_ok=True)

    collected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for playlist in PLAYLISTS:
        channel = playlist["channel"]
        url = playlist["url"]
        tier = playlist["credibility_tier"]
        description = playlist["description"]

        logger.info("Processing: %s — %s", channel, description)
        video_ids = get_video_ids(url, MAX_VIDEOS_PER_PLAYLIST)
        logger.info("Found %d video IDs from %s", len(video_ids), channel)

        for video_id in video_ids:
            if video_id in seen_ids:
                continue
            seen_ids.add(video_id)

            transcript = get_transcript(video_id)
            if not transcript:
                continue

            # Gate 1 — minimum word count
            word_count = len(transcript.split())
            if word_count < MIN_WORDS:
                continue

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            collected.append({
                "source": "docs",
                "source_type": "docs",
                "credibility_tier": tier,
                "content_type": "video_transcript",
                "title": f"{channel} — {description} [{video_id}]",
                "author": channel,
                "url": video_url,
                "text": transcript,
                "word_count": word_count,
                "ingested_at": datetime.utcnow().isoformat(),
            })
            logger.info(
                "Saved transcript: %s (%d words)", video_id, word_count
            )
            time.sleep(REQUEST_DELAY)

        logger.info(
            "Playlist complete: %s — running total: %d", channel, len(collected)
        )

    output_path = Path(OUTPUT_PATH)
    output_path.write_text(
        json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d transcripts to %s", len(collected), OUTPUT_PATH)


if __name__ == "__main__":
    main()