"""
src/ingestion/ingest_youtube_groq.py
=====================================
YouTube channel transcription using Groq Whisper API.
Replaces local CPU Whisper (100-135 hour job) with cloud API (~hours).

Processes 18 target channels. Resume-safe — skips already-transcribed
videos on every run. Designed for GitHub Actions: processes BATCH_SIZE_VIDEOS
per run, stops cleanly before the 6-hour runner timeout.

Groq Whisper limits (free tier as of 2026):
  - 7,200 seconds (2 hours) of audio per day
  - 25 MB max file size per request
  - Files > 25 MB are split with ffmpeg before sending

Environment variables:
    GROQ_API_KEY         — Groq API key
    BATCH_SIZE_VIDEOS    — max videos per run (default 30)
    TRANSCRIPTS_DIR      — where .txt transcripts are saved (default data/transcripts)
    RAW_AUDIO_DIR        — temp dir for downloaded audio (default data/audio_tmp)

Author: Ezechinyere Nnabugwu / DeepForge
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
BATCH_SIZE_VIDEOS: int = int(os.environ.get("BATCH_SIZE_VIDEOS", "30"))
TRANSCRIPTS_DIR: Path = Path(os.environ.get("TRANSCRIPTS_DIR", "data/transcripts"))
RAW_AUDIO_DIR: Path = Path(os.environ.get("RAW_AUDIO_DIR", "data/audio_tmp"))
PROGRESS_FILE: Path = Path("data/youtube_progress.json")

WHISPER_MODEL: str = "whisper-large-v3-turbo"
MAX_FILE_SIZE_BYTES: int = 24 * 1024 * 1024   # 24 MB — stays under Groq's 25 MB limit
AUDIO_FORMAT: str = "mp3"
GROQ_RETRY_WAIT_SECONDS: int = 65             # wait after hitting rate limit
MAX_RETRIES: int = 3

# 18 target channels — prompt engineering focus
TARGET_CHANNELS: list[dict] = [
    {"name": "AndrejKarpathy",       "url": "https://www.youtube.com/@AndrejKarpathy"},
    {"name": "AIExplained",          "url": "https://www.youtube.com/@AIExplained-official"},
    {"name": "YannicKilcher",        "url": "https://www.youtube.com/@YannicKilcher"},
    {"name": "SamWitteveen",         "url": "https://www.youtube.com/@samwitteveenai"},
    {"name": "MatthewBerman",        "url": "https://www.youtube.com/@matthew_berman"},
    {"name": "AbhishekThakur",       "url": "https://www.youtube.com/@AbhishekThakurAbhi"},
    {"name": "AIJasonBeck",          "url": "https://www.youtube.com/@AIJasonBeck"},
    {"name": "PromptEngineeringYT",  "url": "https://www.youtube.com/@engineerprompt"},
    {"name": "FahdMirza",            "url": "https://www.youtube.com/@fahdmirza"},
    {"name": "IBMTechnology",        "url": "https://www.youtube.com/@IBMTechnology"},
    {"name": "GoogleDeepMind",       "url": "https://www.youtube.com/@Google_DeepMind"},
    {"name": "ColeMedin",            "url": "https://www.youtube.com/@ColeMedin"},
    {"name": "DataIndependent",      "url": "https://www.youtube.com/@DataIndependent"},
    {"name": "TwoMinutePapers",      "url": "https://www.youtube.com/@TwoMinutePapers"},
    {"name": "LangChainAI",          "url": "https://www.youtube.com/@LangChain"},
    {"name": "AssemblyAI",           "url": "https://www.youtube.com/@AssemblyAI"},
    {"name": "ArizeAI",              "url": "https://www.youtube.com/@ArizeAI"},
    {"name": "NeuralNine",           "url": "https://www.youtube.com/@NeuralNine"},
]

# Max videos per channel to fetch (keeps corpus focused)
MAX_VIDEOS_PER_CHANNEL: int = 40


# ---------------------------------------------------------------------------
# Progress tracking — resume safety
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    """Load the set of already-transcribed video IDs."""
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            done = set(data.get("done", []))
            failed = set(data.get("failed", []))
            logger.info("Progress loaded: %d done, %d failed", len(done), len(failed))
            return {"done": done, "failed": failed}
        except Exception as exc:
            logger.warning("Could not load progress file: %s", exc)
    return {"done": set(), "failed": set()}


def save_progress(progress: dict) -> None:
    """Save progress to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "done":   list(progress["done"]),
        "failed": list(progress["failed"]),
    }
    PROGRESS_FILE.write_text(
        json.dumps(serialisable, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Video list fetching
# ---------------------------------------------------------------------------

def fetch_channel_videos(channel_url: str, channel_name: str) -> list[dict]:
    """
    Fetch video metadata for a YouTube channel using yt-dlp.

    Args:
        channel_url: Full YouTube channel URL.
        channel_name: Human-readable name for logging.

    Returns:
        List of dicts with id, title, duration_seconds.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--playlist-end", str(MAX_VIDEOS_PER_CHANNEL),
        "--print", "%(id)s\t%(title)s\t%(duration)s",
        "--no-warnings",
        channel_url,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        videos = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                video_id   = parts[0].strip()
                title      = parts[1].strip()
                duration   = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                videos.append({
                    "id": video_id,
                    "title": title,
                    "duration_seconds": duration,
                    "channel": channel_name,
                })
        logger.info("Channel %s: %d videos found", channel_name, len(videos))
        return videos
    except subprocess.TimeoutExpired:
        logger.error("Timeout fetching video list for %s", channel_name)
        return []
    except Exception as exc:
        logger.error("Error fetching videos for %s: %s", channel_name, exc)
        return []


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def download_audio(video_id: str, title: str) -> Path | None:
    """
    Download audio for a YouTube video using yt-dlp.

    Args:
        video_id: YouTube video ID.
        title:    Video title (for logging only).

    Returns:
        Path to downloaded audio file, or None on failure.
    """
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_template = str(RAW_AUDIO_DIR / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",                             # extract audio
        "--audio-format", AUDIO_FORMAT,
        "--audio-quality", "5",           # medium quality — sufficient for speech
        "--output", output_template,
        "--no-playlist",
        "--no-warnings",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)
        audio_path = RAW_AUDIO_DIR / f"{video_id}.{AUDIO_FORMAT}"
        if audio_path.exists():
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            logger.info("Downloaded: %s (%.1f MB)", title[:60], size_mb)
            return audio_path
        logger.error("Audio file not found after download: %s", audio_path)
        return None
    except subprocess.CalledProcessError as exc:
        logger.error("yt-dlp failed for %s: %s", video_id, exc.stderr[:200] if exc.stderr else "")
        return None
    except subprocess.TimeoutExpired:
        logger.error("Download timeout for video: %s", video_id)
        return None


# ---------------------------------------------------------------------------
# Audio splitting (for files > MAX_FILE_SIZE_BYTES)
# ---------------------------------------------------------------------------

def split_audio(audio_path: Path) -> list[Path]:
    """
    Split an audio file into chunks under MAX_FILE_SIZE_BYTES using ffmpeg.

    Args:
        audio_path: Path to the audio file to split.

    Returns:
        List of chunk file paths (single-element list if no split needed).
    """
    file_size = audio_path.stat().st_size
    if file_size <= MAX_FILE_SIZE_BYTES:
        return [audio_path]

    # Calculate chunk duration — 20 min chunks as safe default
    chunk_duration_seconds = 1200
    chunk_pattern = str(audio_path.parent / f"{audio_path.stem}_chunk%03d{audio_path.suffix}")

    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_duration_seconds),
        "-c", "copy",
        "-reset_timestamps", "1",
        chunk_pattern,
        "-y", "-loglevel", "error",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        chunks = sorted(audio_path.parent.glob(f"{audio_path.stem}_chunk*.{audio_path.suffix}"))
        logger.info("Split %s into %d chunks", audio_path.name, len(chunks))
        return chunks
    except Exception as exc:
        logger.error("Audio split failed for %s: %s", audio_path.name, exc)
        return [audio_path]  # Return original and let Groq API reject if too large


# ---------------------------------------------------------------------------
# Groq Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_with_groq(audio_path: Path) -> str | None:
    """
    Transcribe an audio file using the Groq Whisper API.

    Args:
        audio_path: Path to the audio file (must be < 25 MB).

    Returns:
        Transcript text string, or None on failure.
    """
    try:
        from groq import Groq  # type: ignore[import]
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq")
        return None

    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(audio_path.name, audio_file.read()),
                    model=WHISPER_MODEL,
                    response_format="text",
                    language="en",
                )
            return str(transcription).strip()

        except Exception as exc:
            error_str = str(exc).lower()
            if "rate_limit" in error_str or "429" in error_str:
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Waiting %ds...",
                    attempt, MAX_RETRIES, GROQ_RETRY_WAIT_SECONDS
                )
                time.sleep(GROQ_RETRY_WAIT_SECONDS)
            else:
                logger.error("Groq API error on attempt %d: %s", attempt, exc)
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(5)

    return None


# ---------------------------------------------------------------------------
# Full transcription pipeline for one video
# ---------------------------------------------------------------------------

def process_video(video: dict) -> bool:
    """
    Download, transcribe, and save transcript for one video.

    Args:
        video: Dict with id, title, channel keys.

    Returns:
        True if transcription succeeded, False otherwise.
    """
    video_id = video["id"]
    title    = video["title"]
    channel  = video["channel"]

    transcript_path = TRANSCRIPTS_DIR / channel / f"{video_id}.txt"
    if transcript_path.exists():
        logger.info("Already transcribed: %s — skipping", video_id)
        return True

    logger.info("Processing: [%s] %s", channel, title[:70])

    # Step 1: Download audio
    audio_path = download_audio(video_id, title)
    if not audio_path:
        return False

    # Step 2: Split if needed
    chunks = split_audio(audio_path)

    # Step 3: Transcribe each chunk
    full_transcript_parts: list[str] = []
    success = True

    for chunk in chunks:
        text = transcribe_with_groq(chunk)
        if text:
            full_transcript_parts.append(text)
        else:
            logger.error("Transcription failed for chunk: %s", chunk.name)
            success = False
            break

        # Clean up chunk if it was a split (not the original)
        if chunk != audio_path:
            try:
                chunk.unlink()
            except OSError:
                pass

    # Step 4: Save transcript
    if success and full_transcript_parts:
        full_text = "\n\n".join(full_transcript_parts)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            f"TITLE: {title}\nCHANNEL: {channel}\nVIDEO_ID: {video_id}\n\n{full_text}",
            encoding="utf-8",
        )
        logger.info("Saved transcript: %s (%d chars)", transcript_path.name, len(full_text))

    # Step 5: Clean up original audio
    try:
        audio_path.unlink()
    except OSError:
        pass

    return success


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch video lists, process up to BATCH_SIZE_VIDEOS, save progress."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY environment variable not set.")
        sys.exit(1)

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    # Collect all pending videos across all channels
    logger.info("Fetching video lists for %d channels...", len(TARGET_CHANNELS))
    all_videos: list[dict] = []
    for channel in TARGET_CHANNELS:
        videos = fetch_channel_videos(channel["url"], channel["name"])
        all_videos.extend(videos)

    pending = [
        v for v in all_videos
        if v["id"] not in progress["done"]
        and v["id"] not in progress["failed"]
        and v["duration_seconds"] > 60    # skip shorts < 1 minute
        and v["duration_seconds"] < 7200  # skip anything > 2 hours
    ]

    logger.info(
        "Total videos: %d | Already done: %d | Pending: %d | This batch: %d",
        len(all_videos),
        len(progress["done"]),
        len(pending),
        min(len(pending), BATCH_SIZE_VIDEOS),
    )

    if not pending:
        logger.info("All videos already transcribed. Nothing to do.")
        return

    # Process up to BATCH_SIZE_VIDEOS
    processed = 0
    for video in pending[:BATCH_SIZE_VIDEOS]:
        success = process_video(video)
        if success:
            progress["done"].add(video["id"])
        else:
            progress["failed"].add(video["id"])

        save_progress(progress)
        processed += 1

        # Brief pause between videos to be respectful to the API
        time.sleep(2)

    logger.info(
        "Batch complete. Processed: %d | Total done: %d | Failed: %d",
        processed,
        len(progress["done"]),
        len(progress["failed"]),
    )

    remaining = len(pending) - processed
    if remaining > 0:
        logger.info("%d videos still pending. Run again to continue.", remaining)


if __name__ == "__main__":
    main()
