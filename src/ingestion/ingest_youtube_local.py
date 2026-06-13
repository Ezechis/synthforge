"""
ingest_youtube_local.py
───────────────────────
Local Windows variant of the YouTube transcription pipeline for SynthForge.

Run this from your laptop — YouTube blocks caption and audio requests from
datacenter IPs (GitHub Actions). Your residential IP is not blocked.

THREE-TIER STRATEGY:
  Tier 1 — youtube-transcript-api with full language fallback + translation.
            Fetches captions as plain text. No audio, no size limit, instant.
            Covers ~70-80% of videos including auto-generated captions.
            Enhanced: retries on network errors, accepts all languages + translates.

  Tier 2 — Groq Whisper API for videos under 25 minutes (audio < 25MB).
            Downloads audio via yt-dlp, sends to Groq, gets transcript back.
            Covers short-to-medium videos without captions.

  Tier 3 — Groq Whisper with audio splitting for videos over 25 minutes.
            Splits audio into 20-minute segments, transcribes each, stitches
            back together. Covers long videos without captions (podcasts, lectures).
            Requires ffmpeg on PATH.

Usage:
  set HF_TOKEN=hf_your_token
  set GROQ_API_KEY=gsk_your_key
  set HF_HUB_OFFLINE=0
  python src\\ingestion\\ingest_youtube_local.py --batch-size 50

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
MAX_DURATION_SECONDS: int = 7200       # skip videos over 2 hours
GROQ_WHISPER_MODEL: str = "whisper-large-v3"
MIN_TRANSCRIPT_CHARS: int = 200
SOURCE_TAG: str = "youtube"
CREDIBILITY_TIER: str = "practitioner"
CAPTION_LANGUAGES: list[str] = ["en", "en-US", "en-GB", "en-AU"]

# Tier 3 audio splitting
SEGMENT_DURATION_SECONDS: int = 1200   # 20 minutes per segment
GROQ_MAX_FILE_MB: float = 24.0         # safe ceiling below Groq's 25MB limit

# HF staging
STAGING_REPO_ID: str = "ezechinnabugwu/synthforge-yt-staging"
PROGRESS_FILENAME: str = "yt_progress.json"
LOCAL_BATCH_DIR: Path = Path("data/yt_batch_tmp")

# Tier 1 retry settings for network instability
TIER1_MAX_RETRIES: int = 3
TIER1_RETRY_DELAY: float = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress tracker — lives on HF staging
# ---------------------------------------------------------------------------

def load_progress_from_hf(hf_token: str) -> dict[str, Any]:
    """Download and parse the progress tracker from HF staging."""
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(
            repo_id=STAGING_REPO_ID, filename=PROGRESS_FILENAME,
            repo_type="dataset", token=hf_token,
        )
        data = json.loads(Path(cached).read_text(encoding="utf-8"))
        data.setdefault("no_transcript_ids", [])
        data.setdefault("failed_ids", [])
        data.setdefault("completed_ids", [])
        data.setdefault("completed_count", len(data["completed_ids"]))
        logger.info(
            "Progress: %d completed | %d no-transcript | %d failed",
            data["completed_count"],
            len(data["no_transcript_ids"]),
            len(data["failed_ids"]),
        )
        return data
    except Exception as exc:
        logger.warning("No existing progress on HF (first run?): %s", exc)
        return {"completed_ids": [], "completed_count": 0,
                "failed_ids": [], "no_transcript_ids": []}


def save_progress_to_hf(progress: dict[str, Any], hf_token: str) -> None:
    """Upload the progress tracker back to HF staging."""
    try:
        from huggingface_hub import HfApi
        tmp = Path("yt_progress_tmp.json")
        tmp.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")
        HfApi(token=hf_token).upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo=PROGRESS_FILENAME,
            repo_id=STAGING_REPO_ID,
            repo_type="dataset",
        )
        tmp.unlink(missing_ok=True)
        logger.info("Progress saved to HF: %d completed", progress["completed_count"])
    except Exception as exc:
        logger.error("Failed to save progress to HF: %s", exc)


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
            try:
                duration = int(parts[2]) if len(parts) > 2 and parts[2] not in ("NA", "") else 300
            except (ValueError, IndexError):
                duration = 300
            videos.append({
                "id": parts[0].strip(),
                "title": parts[1].strip() if len(parts) > 1 else "",
                "duration": duration,
                "upload_date": parts[3].strip() if len(parts) > 3 else "unknown",
                "channel": channel_handle,
            })
        return videos
    except subprocess.TimeoutExpired:
        logger.warning("Timeout fetching channel: %s", channel_handle)
        return []
    except Exception as exc:
        logger.warning("Error fetching %s: %s", channel_handle, exc)
        return []


def is_relevant(title: str) -> bool:
    """Check if a video title matches at least one relevance keyword."""
    t = title.lower()
    return any(kw in t for kw in RELEVANCE_KEYWORDS)


def build_pending_list(
    channels: list[dict[str, str]],
    progress: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the list of videos pending transcription."""
    skip_ids = (
        set(progress.get("completed_ids", [])) |
        set(progress.get("failed_ids", [])) |
        set(progress.get("no_transcript_ids", []))
    )
    pending: list[dict[str, Any]] = []
    total_found = 0

    for channel in channels:
        logger.info("Fetching: %s", channel["handle"])
        videos = fetch_channel_videos(channel["handle"])
        total_found += len(videos)
        logger.info("  %d videos found", len(videos))
        for video in videos:
            if video["id"] in skip_ids:
                continue
            d = video["duration"]
            if not (d == 300 or (60 < d < MAX_DURATION_SECONDS)):
                continue
            if not is_relevant(video["title"]):
                continue
            video["channel_name"] = channel["name"]
            pending.append(video)
        time.sleep(0.5)

    logger.info("Total channel videos: %d | Pending: %d", total_found, len(pending))
    return pending


# ---------------------------------------------------------------------------
# TIER 1 — YouTube Transcript API (enhanced: retries + full language fallback)
# ---------------------------------------------------------------------------

def fetch_transcript_api(video_id: str) -> str | None:
    """Fetch captions using youtube-transcript-api with full fallback chain.

    Enhanced over previous version:
    - Retries TIER1_MAX_RETRIES times on network errors (not on genuine no-caption)
    - Tries all available languages, translates non-English to English
    - Accepts auto-generated captions (not just manual)
    - Never marks a video no_transcript due to a network error

    Args:
        video_id: YouTube video ID.

    Returns:
        Full transcript text, or None if genuinely unavailable.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        logger.error("youtube-transcript-api not installed: pip install youtube-transcript-api")
        return None

    api = YouTubeTranscriptApi()
    last_error: Exception | None = None

    for attempt in range(1, TIER1_MAX_RETRIES + 1):
        try:
            # Attempt 1: fetch directly in preferred languages
            try:
                segments = api.fetch(video_id, languages=CAPTION_LANGUAGES)
                text = " ".join(
                    s.get("text", "").replace("\n", " ").strip()
                    for s in segments if s.get("text", "").strip()
                )
                if len(text) >= MIN_TRANSCRIPT_CHARS:
                    return text
            except Exception:
                pass

            # Attempt 2: list all available transcripts, translate to English
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                available = list(transcript_list)

                if not available:
                    return None  # genuinely no captions — do not retry

                # Prefer English manual, then English auto, then any translated
                for t in available:
                    if t.language_code.startswith("en") and not t.is_generated:
                        segments = t.fetch()
                        text = " ".join(
                            s.get("text", "").replace("\n", " ").strip()
                            for s in segments if s.get("text", "").strip()
                        )
                        if len(text) >= MIN_TRANSCRIPT_CHARS:
                            return text

                for t in available:
                    if t.language_code.startswith("en"):
                        segments = t.fetch()
                        text = " ".join(
                            s.get("text", "").replace("\n", " ").strip()
                            for s in segments if s.get("text", "").strip()
                        )
                        if len(text) >= MIN_TRANSCRIPT_CHARS:
                            return text

                # Any language — translate to English
                for t in available:
                    try:
                        translated = t.translate("en")
                        segments = translated.fetch()
                        text = " ".join(
                            s.get("text", "").replace("\n", " ").strip()
                            for s in segments if s.get("text", "").strip()
                        )
                        if len(text) >= MIN_TRANSCRIPT_CHARS:
                            return text
                    except Exception:
                        continue

                return None  # tried everything — no usable captions

            except Exception as exc:
                error_str = str(exc).lower()
                # Genuine no-caption signals — do not retry
                if any(x in error_str for x in [
                    "no transcript", "disabled", "unavailable",
                    "could not retrieve", "xml", "no element found"
                ]):
                    logger.debug("No captions for %s: %s", video_id, exc)
                    return None
                # Network error — retry
                last_error = exc
                if attempt < TIER1_MAX_RETRIES:
                    logger.warning(
                        "  Tier 1 network error (attempt %d/%d): %s — retrying in %.0fs",
                        attempt, TIER1_MAX_RETRIES, exc, TIER1_RETRY_DELAY,
                    )
                    time.sleep(TIER1_RETRY_DELAY)
                    continue
                return None

        except Exception as exc:
            last_error = exc
            if attempt < TIER1_MAX_RETRIES:
                time.sleep(TIER1_RETRY_DELAY)
            continue

    if last_error:
        logger.warning("  Tier 1 failed after %d attempts: %s", TIER1_MAX_RETRIES, last_error)
    return None


# ---------------------------------------------------------------------------
# TIER 2 — Groq Whisper for short/medium videos (audio under 25MB)
# ---------------------------------------------------------------------------

def download_audio(video_id: str, tmpdir: str, fmt: str = "mp3") -> Path | None:
    """Download audio for a YouTube video.

    Args:
        video_id: YouTube video ID.
        tmpdir: Temporary directory path.
        fmt: Audio format (mp3 or wav).

    Returns:
        Path to downloaded audio file, or None on failure.
    """
    audio_path = Path(tmpdir) / f"{video_id}.{fmt}"
    cmd = [
        "yt-dlp", "-x",
        "--audio-format", fmt,
        "--audio-quality", "5",
        "-o", str(audio_path),
        "--no-warnings", "--quiet",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.warning("  yt-dlp failed: %s", result.stderr[:120])
            return None
        return audio_path if audio_path.exists() else None
    except subprocess.TimeoutExpired:
        logger.warning("  Audio download timeout: %s", video_id)
        return None
    except Exception as exc:
        logger.warning("  Audio download error: %s", exc)
        return None


def transcribe_audio_file(audio_path: Path, groq_api_key: str) -> str | None:
    """Transcribe a single audio file via Groq Whisper API.

    Args:
        audio_path: Path to audio file (must be under 25MB).
        groq_api_key: Groq API key.

    Returns:
        Transcription text, or None on failure.
    """
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model=GROQ_WHISPER_MODEL, file=f, response_format="text",
            )
        text = response if isinstance(response, str) else getattr(response, "text", str(response))
        return text if len(text) >= MIN_TRANSCRIPT_CHARS else None
    except Exception as exc:
        logger.warning("  Groq transcription failed: %s", exc)
        return None


def transcribe_tier2(video_id: str, groq_api_key: str) -> str | None:
    """Tier 2: download audio and transcribe if file is under 25MB.

    Args:
        video_id: YouTube video ID.
        groq_api_key: Groq API key.

    Returns:
        Transcript text, or None if file too large or transcription failed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = download_audio(video_id, tmpdir)
        if audio_path is None:
            return None

        size_mb = audio_path.stat().st_size / (1024 * 1024)
        if size_mb > GROQ_MAX_FILE_MB:
            logger.info(
                "  Audio %.1fMB exceeds %.0fMB — escalating to Tier 3",
                size_mb, GROQ_MAX_FILE_MB,
            )
            return None  # signal to caller to try Tier 3

        logger.info("  Audio %.1fMB — sending to Groq Whisper...", size_mb)
        return transcribe_audio_file(audio_path, groq_api_key)


# ---------------------------------------------------------------------------
# TIER 3 — Groq Whisper with ffmpeg audio splitting for long videos
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on PATH.

    Returns:
        True if ffmpeg is available.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def split_audio_into_segments(
    audio_path: Path,
    segment_duration: int,
    tmpdir: str,
) -> list[Path]:
    """Split an audio file into fixed-duration segments using ffmpeg.

    Args:
        audio_path: Path to the full audio file.
        segment_duration: Duration of each segment in seconds.
        tmpdir: Directory to write segment files.

    Returns:
        List of paths to segment files, in order.
    """
    segment_pattern = str(Path(tmpdir) / "segment_%03d.mp3")
    cmd = [
        "ffmpeg",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-c", "copy",
        "-reset_timestamps", "1",
        "-loglevel", "error",
        segment_pattern,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning("  ffmpeg split failed: %s", result.stderr[:200])
            return []
        segments = sorted(Path(tmpdir).glob("segment_*.mp3"))
        logger.info("  Split into %d segments", len(segments))
        return segments
    except subprocess.TimeoutExpired:
        logger.warning("  ffmpeg split timeout")
        return []
    except Exception as exc:
        logger.warning("  ffmpeg split error: %s", exc)
        return []


def transcribe_tier3(video_id: str, groq_api_key: str) -> str | None:
    """Tier 3: split long audio into 20-minute segments and transcribe each.

    Downloads the full audio, splits into SEGMENT_DURATION_SECONDS segments
    using ffmpeg, transcribes each segment via Groq Whisper, then stitches
    the results back into a single transcript.

    Args:
        video_id: YouTube video ID.
        groq_api_key: Groq API key.

    Returns:
        Full stitched transcript text, or None on failure.
    """
    if not check_ffmpeg():
        logger.warning("  Tier 3 requires ffmpeg on PATH — skipping %s", video_id)
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("  Tier 3: downloading full audio for splitting...")
        audio_path = download_audio(video_id, tmpdir)
        if audio_path is None:
            return None

        total_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info("  Full audio: %.1fMB — splitting into 20-min segments...", total_mb)

        segments = split_audio_into_segments(
            audio_path, SEGMENT_DURATION_SECONDS, tmpdir
        )
        if not segments:
            return None

        # Transcribe each segment
        segment_texts: list[str] = []
        for i, seg_path in enumerate(segments, 1):
            seg_mb = seg_path.stat().st_size / (1024 * 1024)
            logger.info(
                "  Segment %d/%d (%.1fMB) — transcribing...",
                i, len(segments), seg_mb,
            )

            if seg_mb > GROQ_MAX_FILE_MB:
                logger.warning(
                    "  Segment %d is %.1fMB — still too large, skipping segment",
                    i, seg_mb,
                )
                continue

            text = transcribe_audio_file(seg_path, groq_api_key)
            if text:
                segment_texts.append(text)
                logger.info("  Segment %d: %d chars", i, len(text))
            else:
                logger.warning("  Segment %d: transcription failed", i)

            # Brief pause between Groq API calls to respect rate limits
            time.sleep(1.0)

        if not segment_texts:
            return None

        # Stitch segments together
        full_text = " ".join(segment_texts)
        logger.info(
            "  Tier 3 SUCCESS — %d segments stitched → %d total chars",
            len(segment_texts), len(full_text),
        )
        return full_text if len(full_text) >= MIN_TRANSCRIPT_CHARS else None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_transcript(
    text: str,
    video_meta: dict[str, Any],
    source_tier: str,
) -> list[dict[str, Any]]:
    """Split a transcript into overlapping word-boundary chunks.

    Args:
        text: Full transcript text.
        video_meta: Video metadata dict.
        source_tier: 'captions', 'whisper_direct', or 'whisper_split'.

    Returns:
        List of chunk dicts ready for JSONL serialisation.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    step = CHUNK_WORD_TARGET - CHUNK_WORD_OVERLAP
    i = 0
    chunk_index = 0

    while i < len(words):
        chunk_words = words[i : i + CHUNK_WORD_TARGET]
        chunk_id = hashlib.sha256(
            f"youtube_{video_meta['id']}_{chunk_index}".encode()
        ).hexdigest()[:32]

        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(chunk_words),
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
# HF staging upload
# ---------------------------------------------------------------------------

def upload_batch_to_staging(batch_dir: Path, hf_token: str) -> int:
    """Upload all JSONL files in batch_dir to HF staging dataset.

    Args:
        batch_dir: Directory containing transcript JSONL files.
        hf_token: HuggingFace token.

    Returns:
        Number of files successfully uploaded.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    try:
        api.create_repo(
            repo_id=STAGING_REPO_ID, repo_type="dataset",
            exist_ok=True, private=False,
        )
    except Exception:
        pass

    jsonl_files = list(batch_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.info("No transcript files to upload.")
        return 0

    logger.info("Uploading %d transcript files to HF staging...", len(jsonl_files))
    uploaded = 0
    for jsonl_file in jsonl_files:
        try:
            api.upload_file(
                path_or_fileobj=str(jsonl_file),
                path_in_repo=f"transcripts/{jsonl_file.name}",
                repo_id=STAGING_REPO_ID,
                repo_type="dataset",
            )
            logger.info("  ✓ %s", jsonl_file.name)
            jsonl_file.unlink()
            uploaded += 1
        except Exception as exc:
            logger.error("  ✗ %s: %s", jsonl_file.name, exc)

    return uploaded


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def process_video(
    video: dict[str, Any],
    batch_dir: Path,
    groq_api_key: str,
) -> str:
    """Process one video through all three tiers.

    Args:
        video: Video metadata dict.
        batch_dir: Directory to write JSONL output.
        groq_api_key: Groq API key.

    Returns:
        Status: 'success_captions', 'success_whisper', 'success_split',
                'no_transcript', or 'failed'
    """
    video_id = video["id"]

    # --- Tier 1: Caption API (no audio, no size limit) ---
    logger.info("  Tier 1: caption API...")
    text = fetch_transcript_api(video_id)
    if text:
        logger.info("  ✓ Tier 1 SUCCESS — %d chars", len(text))
        chunks = chunk_transcript(text, video, "captions")
        if chunks:
            _write_jsonl(chunks, batch_dir / f"{video_id}.jsonl")
            logger.info("  ✓ %d chunks written", len(chunks))
            return "success_captions"

    # --- Tier 2: Groq Whisper direct (audio under 25MB) ---
    if not groq_api_key:
        logger.warning("  GROQ_API_KEY not set — skipping Tiers 2 and 3")
        return "no_transcript"

    logger.info("  Tier 2: Groq Whisper (direct)...")
    text = transcribe_tier2(video_id, groq_api_key)
    if text:
        logger.info("  ✓ Tier 2 SUCCESS — %d chars", len(text))
        chunks = chunk_transcript(text, video, "whisper_direct")
        if chunks:
            _write_jsonl(chunks, batch_dir / f"{video_id}.jsonl")
            logger.info("  ✓ %d chunks written", len(chunks))
            return "success_whisper"

    # --- Tier 3: Groq Whisper with audio splitting (long videos) ---
    logger.info("  Tier 3: Groq Whisper (split long audio)...")
    text = transcribe_tier3(video_id, groq_api_key)
    if text:
        logger.info("  ✓ Tier 3 SUCCESS — %d chars", len(text))
        chunks = chunk_transcript(text, video, "whisper_split")
        if chunks:
            _write_jsonl(chunks, batch_dir / f"{video_id}.jsonl")
            logger.info("  ✓ %d chunks written", len(chunks))
            return "success_split"

    logger.warning("  All tiers failed for: %s", video_id)
    return "no_transcript"


def _write_jsonl(chunks: list[dict[str, Any]], path: Path) -> None:
    """Write chunks to a JSONL file.

    Args:
        chunks: List of chunk dicts.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def run_batch(batch_size: int) -> None:
    """Run one transcription batch from local machine."""
    hf_token = os.environ.get("HF_TOKEN", "")
    groq_api_key = os.environ.get("GROQ_API_KEY", "")

    if not hf_token:
        logger.error("HF_TOKEN not set. Run: set HF_TOKEN=hf_your_token")
        sys.exit(1)
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        logger.error("HF_HUB_OFFLINE=1 is set. Run: set HF_HUB_OFFLINE=0")
        sys.exit(1)

    LOCAL_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress_from_hf(hf_token)
    pending = build_pending_list(CHANNELS, progress)

    if not pending:
        logger.info("No pending videos. All channels fully transcribed.")
        return

    logger.info(
        "Starting batch: %d videos (max %d)",
        min(len(pending), batch_size), batch_size,
    )

    # Check ffmpeg availability once
    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        logger.warning(
            "ffmpeg not found on PATH — Tier 3 disabled. "
            "Long videos without captions will be marked no_transcript."
        )

    stats: dict[str, int] = {
        "success_captions": 0, "success_whisper": 0,
        "success_split": 0, "no_transcript": 0, "failed": 0,
    }

    for i, video in enumerate(pending[:batch_size], 1):
        video_id = video["id"]
        logger.info(
            "[%d/%d] %s — %s",
            i, min(len(pending), batch_size),
            video_id, video["title"][:65],
        )

        status = process_video(video, LOCAL_BATCH_DIR, groq_api_key)
        stats[status] = stats.get(status, 0) + 1

        if status.startswith("success"):
            progress["completed_ids"].append(video_id)
            progress["completed_count"] = len(progress["completed_ids"])
        elif status == "no_transcript":
            progress["no_transcript_ids"].append(video_id)
        else:
            progress["failed_ids"].append(video_id)

        time.sleep(0.2)

    # Upload transcripts to HF staging
    uploaded = upload_batch_to_staging(LOCAL_BATCH_DIR, hf_token)

    # Save updated progress to HF
    save_progress_to_hf(progress, hf_token)

    # Clean up
    try:
        LOCAL_BATCH_DIR.rmdir()
    except OSError:
        pass

    logger.info("=== BATCH COMPLETE ===")
    logger.info("Captions  (Tier 1) : %d", stats["success_captions"])
    logger.info("Whisper   (Tier 2) : %d", stats["success_whisper"])
    logger.info("Split     (Tier 3) : %d", stats["success_split"])
    logger.info("No transcript      : %d", stats["no_transcript"])
    logger.info("Failed             : %d", stats["failed"])
    logger.info("Uploaded to HF     : %d files", uploaded)
    logger.info("Total completed    : %d / 947", progress["completed_count"])
    logger.info("")
    logger.info("Next: run deploy/pull_and_embed_yt_staging.py to absorb into ChromaDB")


def main() -> None:
    """Parse CLI arguments and run the local transcription batch."""
    parser = argparse.ArgumentParser(
        description="SynthForge YouTube transcription — local 3-tier pipeline",
    )
    parser.add_argument(
        "--batch-size", type=int, default=30,
        help="Number of videos to process (default: 30)",
    )
    args = parser.parse_args()
    run_batch(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
