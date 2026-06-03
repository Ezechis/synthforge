"""
ingest_youtube_actions.py
─────────────────────────
GitHub Actions-compatible YouTube transcription script for SynthForge.

Strategy (two-tier, in order of preference):
  Tier 1 — youtube-transcript-api: fetches auto-generated or manual captions
            directly from YouTube. Zero audio download, zero bot detection,
            instant. Covers ~70-80% of videos on the channel list.
  Tier 2 — Groq Whisper API: downloads audio via yt-dlp and transcribes.
            Only attempted when Tier 1 returns no captions.

Design contract:
  - Outputs one JSONL file per video: yt_transcripts_batch/<video_id>.jsonl
  - Each line in the JSONL is one chunk: {chunk_id, text, metadata}
  - DOES NOT touch ChromaDB — embedding handled locally
  - Resume-safe: reads yt_progress.json, skips completed/failed/no-transcript IDs
  - Batch-safe: stops after --batch-size videos

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

RELEVANCE_KEYWORDS: list[str] = [
    "prompt", "llm", "gpt", "claude", "gemini", "language model",
    "rag", "retrieval", "embedding", "fine-tun", "instruct",
    "chain of thought", "few-shot", "zero-shot", "agent", "dspy",
    "transformer", "attention", "alignment", "rlhf", "reasoning",
    "diffusion", "multimodal", "context", "inference", "neural",
    "machine learning", "deep learning", "ai ", "artificial intelligence",
    "llama", "mistral", "gemma", "openai", "anthropic", "hugging",
]

CHUNK_WORD_TARGET: int = 384
CHUNK_WORD_OVERLAP: int = 37
MAX_DURATION_SECONDS: int = 7200
GROQ_WHISPER_MODEL: str = "whisper-large-v3"
MIN_TRANSCRIPT_CHARS: int = 200
SOURCE_TAG: str = "youtube"
CREDIBILITY_TIER: str = "practitioner"
CAPTION_LANGUAGES: list[str] = ["en", "en-US", "en-GB", "en-AU"]

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
    """Load the resumable progress tracker."""
    if progress_file.exists():
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            logger.info(
                "Progress loaded: %d completed, %d failed, %d no-transcript",
                data.get("completed_count", 0),
                len(data.get("failed_ids", [])),
                len(data.get("no_transcript_ids", [])),
            )
            return data
        except json.JSONDecodeError as exc:
            logger.warning("Progress file corrupt, starting fresh: %s", exc)
    return {"completed_ids": [], "completed_count": 0, "failed_ids": [], "no_transcript_ids": []}


def save_progress(progress: dict[str, Any], progress_file: Path) -> None:
    """Persist progress to disk."""
    progress_file.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Video catalogue
# ---------------------------------------------------------------------------

def fetch_channel_videos(channel_handle: str) -> list[dict[str, Any]]:
    """Fetch video metadata for a channel using yt-dlp flat playlist."""
    url = f"https://www.youtube.com/{channel_handle}/videos"
    cmd = [
        "yt-dlp", "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(duration)s\t%(upload_date)s",
        "--no-warnings", "--quiet", url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        videos: list[dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            video_id = parts[0].strip()
            title = parts[1].strip() if len(parts) > 1 else ""
            try:
                duration = int(parts[2]) if len(parts) > 2 and parts[2] not in ("NA", "") else 300
            except (ValueError, IndexError):
                duration = 300
            upload_date = parts[3].strip() if len(parts) > 3 else "unknown"
            videos.append({"id": video_id, "title": title, "duration": duration,
                           "upload_date": upload_date, "channel": channel_handle})
        return videos
    except subprocess.TimeoutExpired:
        logger.warning("Timeout fetching channel: %s", channel_handle)
        return []
    except Exception as exc:
        logger.warning("Error fetching channel %s: %s", channel_handle, exc)
        return []


def is_relevant(title: str) -> bool:
    """Check if a video title contains at least one relevance keyword."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in RELEVANCE_KEYWORDS)


def build_pending_list(
    channels: list[dict[str, str]],
    progress: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the list of videos pending transcription."""
    completed_ids: set[str] = set(progress.get("completed_ids", []))
    failed_ids: set[str] = set(progress.get("failed_ids", []))
    no_transcript_ids: set[str] = set(progress.get("no_transcript_ids", []))
    skip_ids = completed_ids | failed_ids | no_transcript_ids
    pending: list[dict[str, Any]] = []

    for channel in channels:
        logger.info("Fetching video list: %s", channel["handle"])
        videos = fetch_channel_videos(channel["handle"])
        logger.info("  Found %d videos in channel", len(videos))
        for video in videos:
            vid_id = video["id"]
            duration = video["duration"]
            if vid_id in skip_ids:
                continue
            if not (duration == 300 or (60 < duration < MAX_DURATION_SECONDS)):
                continue
            if not is_relevant(video["title"]):
                continue
            video["channel_name"] = channel["name"]
            pending.append(video)
        time.sleep(1)

    logger.info("Total pending videos: %d", len(pending))
    return pending


# ---------------------------------------------------------------------------
# Tier 1 — YouTube Transcript API (captions, no download, no bot detection)
# ---------------------------------------------------------------------------

def fetch_transcript_api(video_id: str) -> str | None:
    """Fetch captions using youtube-transcript-api.

    Tries manual captions first, then auto-generated, then translates any
    available language to English. Returns None if no captions exist.

    Args:
        video_id: YouTube video ID.

    Returns:
        Full transcript text as a single string, or None if unavailable.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None

            # Try manual captions in preferred languages
            for lang in CAPTION_LANGUAGES:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    break
                except NoTranscriptFound:
                    continue

            # Try auto-generated captions in preferred languages
            if transcript is None:
                for lang in CAPTION_LANGUAGES:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang])
                        break
                    except NoTranscriptFound:
                        continue

            # Last resort: any language, translate to English
            if transcript is None:
                available = list(transcript_list)
                if available:
                    try:
                        transcript = available[0].translate("en")
                    except Exception:
                        return None

            if transcript is None:
                return None

            segments = transcript.fetch()
            full_text = " ".join(
                seg.get("text", "").replace("\n", " ").strip()
                for seg in segments
                if seg.get("text", "").strip()
            )
            return full_text if len(full_text) >= MIN_TRANSCRIPT_CHARS else None

        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as exc:
            logger.debug("No transcript for %s: %s", video_id, exc)
            return None

    except ImportError:
        logger.error("youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
        return None
    except Exception as exc:
        logger.debug("Transcript API error for %s: %s", video_id, exc)
        return None


# ---------------------------------------------------------------------------
# Tier 2 — Groq Whisper fallback (audio download + transcription)
# ---------------------------------------------------------------------------

def download_audio(video_id: str, tmpdir: str) -> Path | None:
    """Download audio for a YouTube video."""
    audio_path = Path(tmpdir) / f"{video_id}.mp3"
    cmd = [
        "yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "5",
        "-o", str(audio_path), "--no-warnings", "--quiet",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return None
        return audio_path if audio_path.exists() else None
    except Exception:
        return None


def transcribe_with_groq(audio_path: Path, groq_api_key: str) -> str | None:
    """Transcribe audio using Groq Whisper API."""
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 24:
        logger.warning("Audio %.1fMB exceeds Groq 25MB limit — skipping", file_size_mb)
        return None
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=GROQ_WHISPER_MODEL, file=audio_file, response_format="text",
            )
        text = response if isinstance(response, str) else getattr(response, "text", str(response))
        return text if len(text) >= MIN_TRANSCRIPT_CHARS else None
    except Exception as exc:
        logger.warning("Groq transcription failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_transcript(
    text: str,
    video_meta: dict[str, Any],
    source_tier: str,
    chunk_word_target: int = CHUNK_WORD_TARGET,
    overlap_words: int = CHUNK_WORD_OVERLAP,
) -> list[dict[str, Any]]:
    """Split a transcript into overlapping word-boundary chunks."""
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
                "transcript_source": source_tier,
            },
        })
        i += step
        chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def process_video(
    video: dict[str, Any],
    output_dir: Path,
    groq_api_key: str,
) -> str:
    """Process one video through Tier 1 then Tier 2 if needed.

    Returns:
        Status: 'success_captions', 'success_whisper', 'no_transcript', or 'failed'
    """
    video_id = video["id"]

    # Tier 1: caption API (no download, no bot detection)
    logger.info("  Tier 1: caption API...")
    transcript_text = fetch_transcript_api(video_id)

    if transcript_text:
        logger.info("  Tier 1 SUCCESS — %d chars", len(transcript_text))
        chunks = chunk_transcript(transcript_text, video, source_tier="captions")
        if chunks:
            output_path = output_dir / f"{video_id}.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            logger.info("  ✓ %d chunks → %s", len(chunks), output_path.name)
            return "success_captions"

    # Tier 2: Groq Whisper (audio download fallback)
    logger.info("  Tier 2: Groq Whisper fallback...")
    if not groq_api_key:
        logger.warning("  GROQ_API_KEY not set — skipping Tier 2")
        return "no_transcript"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = download_audio(video_id, tmpdir)
        if audio_path is None:
            logger.warning("  Tier 2: audio download failed — marking no_transcript")
            return "no_transcript"

        transcript_text = transcribe_with_groq(audio_path, groq_api_key)
        if not transcript_text:
            return "failed"

        logger.info("  Tier 2 SUCCESS — %d chars", len(transcript_text))
        chunks = chunk_transcript(transcript_text, video, source_tier="whisper")
        if not chunks:
            return "failed"

        output_path = output_dir / f"{video_id}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info("  ✓ %d chunks → %s", len(chunks), output_path.name)
        return "success_whisper"


def run_batch(batch_size: int, progress_file: Path, output_dir: Path) -> None:
    """Run one transcription batch."""
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = load_progress(progress_file)
    pending = build_pending_list(CHANNELS, progress)

    if not pending:
        logger.info("No pending videos. All channels up to date.")
        return

    logger.info("Processing batch of %d videos (max %d)", min(len(pending), batch_size), batch_size)
    stats: dict[str, int] = {"success_captions": 0, "success_whisper": 0,
                              "no_transcript": 0, "failed": 0}

    for i, video in enumerate(pending[:batch_size], 1):
        video_id = video["id"]
        logger.info("[%d/%d] %s — %s", i, min(len(pending), batch_size),
                    video_id, video["title"][:60])

        status = process_video(video, output_dir, groq_api_key)
        stats[status] = stats.get(status, 0) + 1

        if status.startswith("success"):
            progress["completed_ids"].append(video_id)
            progress["completed_count"] = len(progress["completed_ids"])
        elif status == "no_transcript":
            progress["no_transcript_ids"].append(video_id)
        else:
            progress["failed_ids"].append(video_id)

        save_progress(progress, progress_file)
        time.sleep(0.3)

    logger.info("=== BATCH SUMMARY ===")
    logger.info("Captions (Tier 1) : %d", stats["success_captions"])
    logger.info("Whisper  (Tier 2) : %d", stats["success_whisper"])
    logger.info("No transcript      : %d", stats["no_transcript"])
    logger.info("Failed             : %d", stats["failed"])
    logger.info("Total completed    : %d / 947", progress["completed_count"])


def main() -> None:
    """Parse CLI arguments and run the transcription batch."""
    parser = argparse.ArgumentParser(description="SynthForge YouTube transcription")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--progress-file", type=Path, default=Path("yt_progress.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("yt_transcripts_batch"))
    args = parser.parse_args()
    run_batch(batch_size=args.batch_size, progress_file=args.progress_file,
              output_dir=args.output_dir)


if __name__ == "__main__":
    main()
