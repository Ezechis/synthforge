# Fixes youtube-transcript-api v1.x breaking change.
# get_transcript() was removed; replaced with instance method fetch().

import sys
from pathlib import Path

TARGET = Path(r"C:\Users\Ezeking\SynthForge\src\ingestion\ingest_youtube.py")

old = (
    'def get_transcript(video_id: str) -> str:\n'
    '    """Fetch the English transcript for a YouTube video.\n'
    '\n'
    '    Args:\n'
    '        video_id: YouTube video ID (11 characters).\n'
    '\n'
    '    Returns:\n'
    '        Full transcript as a single string, or empty string on failure.\n'
    '    """\n'
    '    try:\n'
    '        segments = YouTubeTranscriptApi.get_transcript(\n'
    '            video_id, languages=["en", "en-US", "en-GB"]\n'
    '        )\n'
    '        return " ".join(seg["text"] for seg in segments).strip()\n'
    '    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):\n'
    '        return ""\n'
    '    except Exception as exc:\n'
    '        logger.warning("Transcript error for %s: %s", video_id, exc)\n'
    '        return ""'
)

new = (
    'def get_transcript(video_id: str) -> str:\n'
    '    """Fetch the English transcript for a YouTube video.\n'
    '\n'
    '    Args:\n'
    '        video_id: YouTube video ID (11 characters).\n'
    '\n'
    '    Returns:\n'
    '        Full transcript as a single string, or empty string on failure.\n'
    '    """\n'
    '    try:\n'
    '        yta = YouTubeTranscriptApi()\n'
    '        fetched = yta.fetch(video_id, languages=["en", "en-US", "en-GB"])\n'
    '        return " ".join(snippet.text for snippet in fetched).strip()\n'
    '    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):\n'
    '        return ""\n'
    '    except Exception as exc:\n'
    '        logger.warning("Transcript error for %s: %s", video_id, exc)\n'
    '        return ""'
)

content = TARGET.read_text(encoding="utf-8")
if content.count(old) != 1:
    print(f"ERROR: found {content.count(old)} matches"); sys.exit(1)
TARGET.write_text(content.replace(old, new, 1), encoding="utf-8")
print("OK — youtube-transcript-api v1.x fetch() method applied")