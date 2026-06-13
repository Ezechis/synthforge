"""
ingest_youtube_whisper.py — SynthForge / DeepForge Layer 1: YouTube Corpus
============================================================================
Channel-level ingestion strategy:
  1. Fetch full video list from each target channel (titles only, instant)
  2. Filter by relevance keywords against title + description
  3. Download audio only for relevant videos (yt-dlp, mp3 64kbps)
  4. Transcribe locally with OpenAI Whisper (no API, no cost, CPU-safe)
  5. Save structured JSON for chunk_and_embed.py

Resume-safe: a manifest file tracks every video processed. Restarting
the script skips completed videos automatically. Safe to stop and restart
across multiple days.

Compute estimate (Whisper small, CPU):
  ~2x realtime — 30-min video = ~15 min transcription
  400 videos × 15 min = ~100 hours (run over multiple days in background)

Requirements (local machine only — NOT needed on HF Space):
  pip install yt-dlp openai-whisper
  ffmpeg on PATH (confirmed installed)

Usage:
  python src/ingestion/ingest_youtube_whisper.py

  Optional flags (edit constants below):
    WHISPER_MODEL   — tiny/base/small/medium (default: small)
    MAX_VIDEOS_PER_CHANNEL — cap per channel (default: 150)
    DRY_RUN         — True = fetch and filter only, no download/transcribe

Output:
  data/raw/docs/youtube_transcripts.json
  data/raw/docs/youtube_manifest.json
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH: Path = Path("data/raw/docs/youtube_transcripts.json")
MANIFEST_PATH: Path = Path("data/raw/docs/youtube_manifest.json")
AUDIO_DIR: Path = Path("data/raw/audio")

WHISPER_MODEL: str = "small"       # tiny=fastest, small=best CPU balance, medium=slower
MAX_VIDEOS_PER_CHANNEL: int = 150  # Cap per channel — prevents runaway ingestion
MIN_TRANSCRIPT_WORDS: int = 150    # Discard transcripts shorter than this
KEEP_AUDIO: bool = False           # Delete audio after transcription (saves disk)
DRY_RUN: bool = False              # True = filter only, no download/transcribe
REQUEST_DELAY: float = 2.0         # Seconds between channel fetches (polite crawl)

# ---------------------------------------------------------------------------
# Relevance filter — video must match at least ONE keyword in title or
# description to be downloaded. Case-insensitive substring match.
# ---------------------------------------------------------------------------

RELEVANCE_KEYWORDS: tuple[str, ...] = (
    "prompt engineer",
    "prompting",
    "chain of thought",
    "chain-of-thought",
    "few-shot",
    "few shot",
    "zero-shot",
    "zero shot",
    "in-context learning",
    "in context learning",
    "large language model",
    "language model",
    "llm",
    "gpt",
    "claude",
    "gemini",
    "instruction tuning",
    "instruction follow",
    "rag",
    "retrieval augmented",
    "retrieval-augmented",
    "agent",
    "reasoning",
    "self-consistency",
    "self consistency",
    "tree of thought",
    "tree-of-thought",
    "react prompting",
    "reflexion",
    "tool use",
    "function calling",
    "fine-tuning",
    "finetuning",
    "rlhf",
    "reinforcement learning from human",
    "dspy",
    "jailbreak",
    "hallucination",
    "context window",
    "transformer",
    "attention mechanism",
    "tokeniz",
    "embedding",
    "vector",
    "semantic search",
    "knowledge base",
    "system prompt",
    "structured output",
    "json mode",
    "function call",
    "openai",
    "anthropic",
    "hugging face",
    "huggingface",
    "langchain",
    "llamaindex",
    "mistral",
    "llama",
    "mixtral",
    "deepseek",
    "multimodal",
    "vision language",
    "text to image",
    "stable diffusion",
    "diffusion model",
    "ai agent",
    "autonomous agent",
    "benchmark",
    "evaluation",
    "mmlu",
    "hellaswag",
    "ai safety",
    "alignment",
    "constitutional ai",
)

# ---------------------------------------------------------------------------
# Target channels — 18 channels covering the full PE ecosystem
# Format: {"name": display name, "url": channel URL or @handle}
# ---------------------------------------------------------------------------

TARGET_CHANNELS: list[dict[str, str]] = [
    # ── Foundational educators ─────────────────────────────────────────────
    {
        "name": "Andrej Karpathy",
        "url": "https://www.youtube.com/@AndrejKarpathy",
        "cap": 60,    # All videos are relevant — no cap needed but set generous
    },
    {
        "name": "Yannic Kilcher",
        "url": "https://www.youtube.com/@YannicKilcher",
        "cap": 120,   # Filter heavily — many non-LLM papers
    },
    # ── Applied prompting and LangChain ───────────────────────────────────
    {
        "name": "Sam Witteveen (Red Dragon AI)",
        "url": "https://www.youtube.com/@samwitteveenai",
        "cap": 150,
    },
    {
        "name": "AI Jason",
        "url": "https://www.youtube.com/@AIJasonZ",
        "cap": 100,
    },
    # ── Analysis and comparison ────────────────────────────────────────────
    {
        "name": "AI Explained",
        "url": "https://www.youtube.com/@aiexplained-official",
        "cap": 100,
    },
    {
        "name": "Matthew Berman",
        "url": "https://www.youtube.com/@matthew_berman",
        "cap": 150,   # Large channel — filter needed
    },
    # ── Technical tutorials ────────────────────────────────────────────────
    {
        "name": "AssemblyAI",
        "url": "https://www.youtube.com/@AssemblyAI",
        "cap": 100,
    },
    {
        "name": "Sentdex",
        "url": "https://www.youtube.com/@sentdex",
        "cap": 80,    # Filter to NLP/LLM content
    },
    # ── Official channels ─────────────────────────────────────────────────
    {
        "name": "LangChain",
        "url": "https://www.youtube.com/@LangChain",
        "cap": 150,
    },
    {
        "name": "Hugging Face",
        "url": "https://www.youtube.com/@HuggingFace",
        "cap": 100,
    },
    {
        "name": "DeepLearning.AI",
        "url": "https://www.youtube.com/@Deeplearningai",
        "cap": 80,
    },
    # ── Research and papers ────────────────────────────────────────────────
    {
        "name": "Two Minute Papers",
        "url": "https://www.youtube.com/@TwoMinutePapers",
        "cap": 100,   # Filter to LLM/prompting papers
    },
    {
        "name": "Weights & Biases",
        "url": "https://www.youtube.com/@WeightsBiases",
        "cap": 80,
    },
    # ── Concise explainers ─────────────────────────────────────────────────
    {
        "name": "Fireship",
        "url": "https://www.youtube.com/@Fireship",
        "cap": 60,    # Filter heavily — most videos not PE-related
    },
    {
        "name": "IBM Technology",
        "url": "https://www.youtube.com/@IBMTechnology",
        "cap": 80,
    },
    # ── Frontier model coverage ────────────────────────────────────────────
    {
        "name": "Google DeepMind",
        "url": "https://www.youtube.com/@Google_DeepMind",
        "cap": 60,
    },
    {
        "name": "Databricks",
        "url": "https://www.youtube.com/@Databricks",
        "cap": 80,
    },
    {
        "name": "Connor Shorten (Weaviate)",
        "url": "https://www.youtube.com/@connorshorten6311",
        "cap": 80,
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest() -> dict[str, Any]:
    """Load transcript manifest for resume-safety.

    Returns:
        Dict mapping video ID to status metadata.
    """
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict[str, Any]) -> None:
    """Persist manifest to disk after every video.

    Args:
        manifest: Dict mapping video ID to status metadata.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_existing_transcripts() -> list[dict[str, Any]]:
    """Load previously saved transcripts for append-only updates.

    Returns:
        List of existing transcript dicts, or empty list.
    """
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_transcripts(transcripts: list[dict[str, Any]]) -> None:
    """Persist transcript list to disk.

    Args:
        transcripts: Full list of transcript dicts.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

def is_relevant(title: str, description: str = "") -> bool:
    """Return True if any relevance keyword appears in title or description.

    Args:
        title: Video title.
        description: Video description (optional).

    Returns:
        True if the video is relevant to prompt engineering.
    """
    haystack = (title + " " + description).lower()
    return any(kw in haystack for kw in RELEVANCE_KEYWORDS)


# ---------------------------------------------------------------------------
# Channel video list fetch
# ---------------------------------------------------------------------------

def fetch_channel_videos(
    channel_url: str,
    channel_name: str,
    cap: int,
) -> list[dict[str, Any]]:
    """Fetch video metadata list from a YouTube channel using yt-dlp.

    Does NOT download any audio — only retrieves titles, IDs, URLs.
    Then filters by relevance keywords before returning.

    Args:
        channel_url: YouTube channel URL or @handle URL.
        channel_name: Display name for logging.
        cap: Maximum videos to consider from this channel.

    Returns:
        List of relevant video metadata dicts.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed.")

    logger.info("Fetching video list from: %s", channel_name)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,      # Fetch metadata only — no download
        "playlist_items": f"1-{cap}",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url + "/videos", download=False)
            entries = info.get("entries", []) if info else []
    except Exception as exc:
        logger.error("Failed to fetch channel %s: %s", channel_name, exc)
        return []

    relevant: list[dict[str, Any]] = []
    for entry in entries:
        if not entry:
            continue
        title = entry.get("title", "")
        description = entry.get("description", "") or ""
        video_id = entry.get("id", "")
        url = f"https://www.youtube.com/watch?v={video_id}"

        if is_relevant(title, description):
            relevant.append({
                "id": video_id,
                "url": url,
                "title": title,
                "channel": channel_name,
                "duration_seconds": entry.get("duration", 0),
            })

    logger.info(
        "%s — %d/%d videos passed relevance filter",
        channel_name, len(relevant), len(entries),
    )
    return relevant


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def download_audio(video_id: str, url: str) -> Path | None:
    """Download audio from a YouTube video as MP3.

    Args:
        video_id: YouTube video ID (used for filename).
        url: Full YouTube video URL.

    Returns:
        Path to downloaded audio file, or None on failure.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed.")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_template = str(AUDIO_DIR / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "64",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # Find the downloaded file
        for ext in ["mp3", "m4a", "opus", "webm"]:
            candidate = AUDIO_DIR / f"{video_id}.{ext}"
            if candidate.exists():
                return candidate
    except Exception as exc:
        logger.error("Download failed for %s: %s", url, exc)
    return None


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

_whisper_model_cache = None  # Load once, reuse across all videos


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio using Whisper. Model loaded once and cached.

    Args:
        audio_path: Path to audio file.

    Returns:
        Transcription text, or empty string on failure.
    """
    global _whisper_model_cache

    try:
        import whisper
    except ImportError:
        raise RuntimeError("openai-whisper not installed.")

    if _whisper_model_cache is None:
        logger.info("Loading Whisper model '%s' (once)...", WHISPER_MODEL)
        _whisper_model_cache = whisper.load_model(WHISPER_MODEL)

    t0 = time.time()
    try:
        result = _whisper_model_cache.transcribe(
            str(audio_path),
            language="en",
            verbose=False,
            fp16=False,
        )
        elapsed = time.time() - t0
        text = result.get("text", "").strip()
        word_count = len(text.split())
        logger.info(
            "Transcribed in %.0f seconds — %d words (%.1fx realtime)",
            elapsed, word_count,
            elapsed / max(result.get("duration", elapsed), 1),
        )
        return text
    except Exception as exc:
        logger.error("Transcription failed for %s: %s", audio_path, exc)
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ingest_youtube() -> None:
    """Fetch, filter, download, transcribe and save YouTube corpus.

    Channel-level ingestion: fetches all videos from each target channel,
    filters by relevance, then downloads and transcribes matching videos.
    Resume-safe via manifest — safe to stop and restart at any time.
    """
    manifest = load_manifest()
    transcripts = load_existing_transcripts()
    total_new = 0
    total_skipped = 0
    total_filtered_out = 0

    logger.info(
        "=== YouTube ingestion start === "
        "%d channels | %d already in manifest | DRY_RUN=%s",
        len(TARGET_CHANNELS), len(manifest), DRY_RUN,
    )

    for channel in TARGET_CHANNELS:
        channel_name = channel["name"]
        channel_url = channel["url"]
        cap = channel.get("cap", MAX_VIDEOS_PER_CHANNEL)

        # Step 1: Fetch full video list and filter
        relevant_videos = fetch_channel_videos(channel_url, channel_name, cap)
        total_filtered_out += cap - len(relevant_videos)
        time.sleep(REQUEST_DELAY)

        if not relevant_videos:
            logger.info("%s — no relevant videos found.", channel_name)
            continue

        if DRY_RUN:
            logger.info(
                "DRY RUN — %s: would process %d videos",
                channel_name, len(relevant_videos),
            )
            for v in relevant_videos:
                logger.info("  → %s", v["title"])
            continue

        # Step 2: Process each relevant video
        for video in relevant_videos:
            vid_id = video["id"]

            if vid_id in manifest:
                total_skipped += 1
                continue

            duration_min = video.get("duration_seconds", 0) / 60
            logger.info(
                "Processing [%.0f min]: %s — %s",
                duration_min, channel_name, video["title"],
            )

            # Download audio
            audio_path = download_audio(vid_id, video["url"])
            if not audio_path:
                manifest[vid_id] = {"status": "failed", "reason": "download_error",
                                    "title": video["title"]}
                save_manifest(manifest)
                continue

            # Transcribe
            transcript = transcribe_audio(audio_path)

            # Clean up audio
            if not KEEP_AUDIO and audio_path.exists():
                audio_path.unlink()

            # Quality gate
            word_count = len(transcript.split()) if transcript else 0
            if word_count < MIN_TRANSCRIPT_WORDS:
                logger.warning(
                    "Transcript too short (%d words) — discarding: %s",
                    word_count, video["title"],
                )
                manifest[vid_id] = {"status": "discarded", "reason": "too_short",
                                    "title": video["title"]}
                save_manifest(manifest)
                continue

            # Build corpus entry
            entry: dict[str, Any] = {
                "source": "docs",
                "source_type": "youtube",
                "credibility_tier": "community",
                "content_type": "youtube_transcript",
                "channel": channel_name,
                "title": video["title"],
                "url": video["url"],
                "text": transcript,
                "word_count": word_count,
                "duration_minutes": round(duration_min, 1),
                "ingested_at": datetime.utcnow().isoformat(),
            }

            transcripts.append(entry)
            total_new += 1

            # Save after every video — crash-safe
            save_transcripts(transcripts)
            manifest[vid_id] = {
                "status": "done",
                "title": video["title"],
                "channel": channel_name,
                "words": word_count,
                "ingested_at": entry["ingested_at"],
            }
            save_manifest(manifest)

            logger.info(
                "Saved (%d total): '%s' — %d words",
                total_new, video["title"], word_count,
            )

    logger.info(
        "=== Ingestion complete === "
        "New: %d | Skipped (already done): %d | "
        "Filtered out (not relevant): %d | "
        "Total in file: %d",
        total_new, total_skipped, total_filtered_out, len(transcripts),
    )


if __name__ == "__main__":
    ingest_youtube()
