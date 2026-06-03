"""
ingest_youtube_actions.py
─────────────────────────
GitHub Actions-compatible YouTube transcription script for SynthForge.

Design contract:
  - Uses Groq Whisper API (not local Whisper) — zero disk space for model weights
  - Outputs one JSONL file per video: yt_transcripts_batch/<video_id>.jsonl
  - Each line in the JSONL is one chunk: {chunk_id, text, metadata}
  - DOES NOT touch ChromaDB — embedding is handled locally by chunk_and_embed.py
  - Resume-safe: reads yt_progress.json, skips completed video IDs
  - Batch-safe: stops after --batch-size videos regardless of time

Usage:
  python src/ingestion/ingest_youtube_actions.py \
    --batch-size 30 \
    --progress-file yt_progress.json \
    --output-dir yt_transcripts_batch

Author: DeepForge / Claude (Sonnet 4.6)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 18 curated channels — same list as the original ingest_youtube_whisper.py
CHANNELS: list[dict[str, str]] = [
    {"handle": "@AndrejKarpathy", "name": "Andrej Karpathy"},
    {"handle": "@lexfridman", "name": "Lex Fridman"},
    {"handle": "@samwitteveenai", "name": "Sam Witteveen"},
    {"handle": "@1littlecoder", "name": "1littlecoder"},
    {"handle": "@AIExplained-official", "name": "AI Explained"},
    {"handle": "@YannicKilcher", "name": "Yannic Kilcher"},
    {"handle": "@bycloud", "name": "bycloud"},
    {"handle": "@aiDotEngineer", "name": "AI Engineer"},
    {"handle": "@Matthew_Berman", "name": "Matthew Berman"},
    {"handle": "@datasciencecastnet", "name": "Data Science Castnet"},
    {"handle": "@TwoMinutePapers", "name": "Two Minute Papers"},
    {"handle": "@MervinPraison", "name": "Mervin Praison"},
    {"handle": "@AssemblyAI", "name": "AssemblyAI"},
    {"handle": "@thecoder5175", "name": "The Coder"},
    {"handle": "@AIJasonBeck", "name": "AI Jason"},
    {"handle": "@AbhishekThakur", "name": "Abhishek Thakur"},
    {"handle": "@GoogleDeepMind", "name": "Google DeepMind"},
    {"handle": "@OpenAI", "name": "OpenAI"},
]

# Keyword filter — only transcribe videos likely relevant to prompt engineering
RELEVANCE_KEYWORDS: list[str] = [
    "prompt", "llm", "gpt", "claude", "gemini", "language model",
    "rag", "retrieval", "embedding", "fine-tun", "instruct",
    "chain of thought", "few-shot", "zero-shot", "agent", "dspy",
    "transformer", "attention", "alignment", "rlhf", "reasoning",
    "diffusion", "multimodal", "context", "inference",
]

# Chunk parameters matching the rest of the SynthForge pipeline
CHUNK_WORD_TARGET: int = 384
CHUNK_WORD_OVERLAP: int = 37

# Max audio duration to attempt transcription (seconds) — skip 3hr conference talks
MAX_DURATION_SECONDS: int = 7200  # 2 hours

# Groq Whisper model — whisper-large-v3 gives best accuracy at free tier cost
GROQ_WHISPER_MODEL: str = "whisper-large-v3"

# Source metadata tag
SOURCE_TAG: str = "youtube"
CREDIBILITY_TIER: str = "practitioner"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

def load_progress(progress_file: Path) -> dict[str, Any]:
    """Load the resumable progress tracker.

    Args:
        progress_file: Path to the JSON progress file.

    Returns:
        Progress dict with keys: completed_ids, completed_count, failed_ids.
    """
    if progress_file.exists():
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            logger.info(
                "Progress loaded: %d completed, %d failed",
                data.get("completed_count", 0),
                len(data.get("failed_ids", [])),
            )
            return data
        except json.JSONDecodeError as exc:
            logger.warning("Progress file corrupt, starting fresh: %s", exc)
    return {"completed_ids": [], "completed_count": 0, "failed_ids": []}


def save_progress(progress: dict[str, Any], progress_file: Path) -> None:
    """Persist progress to disk.

    Args:
        progress: Current progress state.
        progress_file: Path to write the JSON file.
    """
    progress_file.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Video catalogue
# ---------------------------------------------------------------------------

def fetch_channel_videos(channel_handle: str) -> list[dict[str, Any]]:
    """Fetch video metadata for a channel using yt-dlp flat playlist.

    Args:
        channel_handle: YouTube channel handle e.g. '@AndrejKarpathy'.

    Returns:
        List of video metadata dicts. Empty list on failure.
    """
    url = f"https://www.youtube.com/{channel_handle}/videos"
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(duration)s\t%(upload_date)s",
        "--no-warnings",
        "--quiet",
        url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        videos: list[dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            video_id = parts[0].strip()
            title = parts[1].strip() if len(parts) > 1 else ""
            # Duration: yt-dlp returns 'NA' for some — default to 300 (5 min)
            try:
                duration = int(parts[2]) if len(parts) > 2 and parts[2] not in ("NA", "") else 300
            except (ValueError, IndexError):
                duration = 300
            upload_date = parts[3].strip() if len(parts) > 3 else "unknown"

            videos.append({
                "id": video_id,
                "title": title,
                "duration": duration,
                "upload_date": upload_date,
                "channel": channel_handle,
            })
        return videos
    except subprocess.TimeoutExpired:
        logger.warning("Timeout fetching channel: %s", channel_handle)
        return []
    except Exception as exc:
        logger.warning("Error fetching channel %s: %s", channel_handle, exc)
        return []


def is_relevant(title: str) -> bool:
    """Check if a video title contains at least one relevance keyword.

    Args:
        title: Video title string.

    Returns:
        True if relevant.
    """
    title_lower = title.lower()
    return any(kw in title_lower for kw in RELEVANCE_KEYWORDS)


def build_pending_list(
    channels: list[dict[str, str]],
    progress: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the list of videos pending transcription.

    Args:
        channels: List of channel config dicts.
        progress: Current progress tracker.

    Returns:
        Filtered list of pending video dicts, oldest-first.
    """
    completed_ids: set[str] = set(progress.get("completed_ids", []))
    failed_ids: set[str] = set(progress.get("failed_ids", []))
    pending: list[dict[str, Any]] = []

    for channel in channels:
        logger.info("Fetching video list: %s", channel["handle"])
        videos = fetch_channel_videos(channel["handle"])
        logger.info("  Found %d videos in channel", len(videos))

        for video in videos:
            vid_id = video["id"]
            duration = video["duration"]
            title = video["title"]

            if vid_id in completed_ids:
                continue
            if vid_id in failed_ids:
                continue
            # Duration filter: skip NA-default (300) as boundary, skip huge videos
            # Accept: duration == 300 (safe default) OR (60 < duration < MAX)
            if not (duration == 300 or (60 < duration < MAX_DURATION_SECONDS)):
                continue
            if not is_relevant(title):
                continue

            video["channel_name"] = channel["name"]
            pending.append(video)

        time.sleep(1)  # polite crawl rate between channels

    logger.info("Total pending videos: %d", len(pending))
    return pending


# ---------------------------------------------------------------------------
# Transcription via Groq Whisper API
# ---------------------------------------------------------------------------

def download_audio(video_id: str, tmpdir: str) -> Path | None:
    """Download audio for a YouTube video into a temp directory.

    Args:
        video_id: YouTube video ID.
        tmpdir: Temporary directory path.

    Returns:
        Path to the downloaded audio file, or None on failure.
    """
    audio_path = Path(tmpdir) / f"{video_id}.mp3"
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "5",  # 128kbps — sufficient for speech
        "-o", str(audio_path),
        "--no-warnings",
        "--quiet",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning("yt-dlp download failed: %s", result.stderr[:200])
            return None
        return audio_path if audio_path.exists() else None
    except subprocess.TimeoutExpired:
        logger.warning("Audio download timeout: %s", video_id)
        return None
    except Exception as exc:
        logger.warning("Audio download error: %s — %s", video_id, exc)
        return None


def transcribe_with_groq(audio_path: Path, groq_api_key: str) -> str | None:
    """Transcribe an audio file using Groq Whisper API.

    Args:
        audio_path: Path to the audio file.
        groq_api_key: Groq API key string.

    Returns:
        Transcription text, or None on failure.
    """
    # File size check — Groq Whisper limit is 25MB
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 24:
        logger.warning(
            "Audio file too large for Groq API: %.1fMB (limit 25MB) — skipping %s",
            file_size_mb,
            audio_path.name,
        )
        return None

    try:
        from groq import Groq

        client = Groq(api_key=groq_api_key)
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=GROQ_WHISPER_MODEL,
                file=audio_file,
                response_format="text",
            )
        if isinstance(response, str):
            return response
        return getattr(response, "text", str(response))
    except Exception as exc:
        logger.warning("Groq transcription failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_transcript(
    text: str,
    video_meta: dict[str, Any],
    chunk_word_target: int = CHUNK_WORD_TARGET,
    overlap_words: int = CHUNK_WORD_OVERLAP,
) -> list[dict[str, Any]]:
    """Split a transcript into overlapping word-boundary chunks.

    Matches the chunking parameters used by chunk_and_embed.py so BM25
    index entries are consistent with other corpus sources.

    Args:
        text: Full transcript text.
        video_meta: Video metadata dict for tagging each chunk.
        chunk_word_target: Target words per chunk.
        overlap_words: Overlap words between consecutive chunks.

    Returns:
        List of chunk dicts ready for JSONL serialisation.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    step = chunk_word_target - overlap_words
    i = 0
    chunk_index = 0

    while i < len(words):
        chunk_words = words[i : i + chunk_word_target]
        chunk_text = " ".join(chunk_words)

        # SHA-256 chunk ID — deterministic and resume-safe
        chunk_id = hashlib.sha256(
            f"youtube_{video_meta['id']}_{chunk_index}".encode()
        ).hexdigest()[:32]

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "source": SOURCE_TAG,
                "credibility_tier": CREDIBILITY_TIER,
                "video_id": video_meta["id"],
                "title": video_meta["title"],
                "channel": video_meta["channel_name"],
                "upload_date": video_meta.get("upload_date", "unknown"),
                "chunk_index": chunk_index,
                "word_count": len(chunk_words),
                "content_type": "video_transcript",
            },
        })
        i += step
        chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def run_batch(
    batch_size: int,
    progress_file: Path,
    output_dir: Path,
) -> None:
    """Run one transcription batch.

    Args:
        batch_size: Maximum number of videos to process in this run.
        progress_file: Path to the JSON progress tracker.
        output_dir: Directory to write transcript JSONL files.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set. Aborting.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    progress = load_progress(progress_file)
    pending = build_pending_list(CHANNELS, progress)

    if not pending:
        logger.info("No pending videos. All channels up to date.")
        return

    logger.info("Processing batch of %d videos (max %d)", min(len(pending), batch_size), batch_size)
    processed = 0

    for video in pending[:batch_size]:
        video_id = video["id"]
        title = video["title"]
        logger.info(
            "[%d/%d] Processing: %s — %s",
            processed + 1,
            min(len(pending), batch_size),
            video_id,
            title[:60],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Download audio
            audio_path = download_audio(video_id, tmpdir)
            if audio_path is None:
                logger.warning("  Skipping (download failed): %s", video_id)
                progress["failed_ids"].append(video_id)
                save_progress(progress, progress_file)
                processed += 1
                continue

            # Step 2: Transcribe via Groq
            transcript_text = transcribe_with_groq(audio_path, groq_api_key)
            if not transcript_text or len(transcript_text.strip()) < 100:
                logger.warning("  Skipping (transcription empty or too short): %s", video_id)
                progress["failed_ids"].append(video_id)
                save_progress(progress, progress_file)
                processed += 1
                continue

            # Step 3: Chunk and write JSONL
            chunks = chunk_transcript(transcript_text, video)
            if not chunks:
                logger.warning("  No chunks produced for: %s", video_id)
                progress["failed_ids"].append(video_id)
                save_progress(progress, progress_file)
                processed += 1
                continue

            output_path = output_dir / f"{video_id}.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            logger.info("  ✓ %d chunks → %s", len(chunks), output_path.name)

            # Update progress
            progress["completed_ids"].append(video_id)
            progress["completed_count"] = len(progress["completed_ids"])
            save_progress(progress, progress_file)

        processed += 1
        time.sleep(0.5)  # brief pause between videos

    logger.info(
        "Batch complete. Processed: %d | Total completed: %d / 947",
        processed,
        progress["completed_count"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the transcription batch."""
    parser = argparse.ArgumentParser(
        description="SynthForge YouTube transcription — GitHub Actions variant",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Number of videos to process in this run (default: 30)",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("yt_progress.json"),
        help="Path to the JSON progress tracker (default: yt_progress.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("yt_transcripts_batch"),
        help="Directory to write transcript JSONL files (default: yt_transcripts_batch)",
    )
    args = parser.parse_args()
    run_batch(
        batch_size=args.batch_size,
        progress_file=args.progress_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
