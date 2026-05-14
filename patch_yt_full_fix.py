# Rewrites the get_transcript function entirely.
# Handles both the original v0.x API and the v1.x API state.
# Adds cookies support to bypass YouTube IP blocks.

import sys
from pathlib import Path

TARGET = Path(r"C:\Users\Ezeking\PromptForge\src\ingestion\ingest_youtube.py")

content = TARGET.read_text(encoding="utf-8")

# The correct final version of the function
CORRECT_FUNC = '''def get_transcript(video_id: str) -> str:
    """Fetch the English transcript for a YouTube video.

    Args:
        video_id: YouTube video ID (11 characters).

    Returns:
        Full transcript as a single string, or empty string on failure.
    """
    try:
        yta = YouTubeTranscriptApi(cookies="cookies.txt")
        fetched = yta.fetch(video_id, languages=["en", "en-US", "en-GB"])
        return " ".join(snippet.text for snippet in fetched).strip()
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        return ""
    except Exception as exc:
        logger.warning("Transcript error for %s: %s", video_id, exc)
        return ""'''

# Try replacing whichever version currently exists
CANDIDATES = [
    # Original v0.x version
    '''def get_transcript(video_id: str) -> str:
    """Fetch the English transcript for a YouTube video.

    Args:
        video_id: YouTube video ID (11 characters).

    Returns:
        Full transcript as a single string, or empty string on failure.
    """
    try:
        segments = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-GB"]
        )
        return " ".join(seg["text"] for seg in segments).strip()
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        return ""
    except Exception as exc:
        logger.warning("Transcript error for %s: %s", video_id, exc)
        return ""''',
    # v1.x version without cookies
    '''def get_transcript(video_id: str) -> str:
    """Fetch the English transcript for a YouTube video.

    Args:
        video_id: YouTube video ID (11 characters).

    Returns:
        Full transcript as a single string, or empty string on failure.
    """
    try:
        yta = YouTubeTranscriptApi()
        fetched = yta.fetch(video_id, languages=["en", "en-US", "en-GB"])
        return " ".join(snippet.text for snippet in fetched).strip()
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        return ""
    except Exception as exc:
        logger.warning("Transcript error for %s: %s", video_id, exc)
        return ""''',
]

patched = False
for candidate in CANDIDATES:
    if candidate in content:
        content = content.replace(candidate, CORRECT_FUNC, 1)
        patched = True
        print("OK — get_transcript rewritten with cookies support")
        break

if not patched:
    print("ERROR: could not find get_transcript function in either known form")
    print("Opening file for manual inspection...")
    # Print the relevant section for diagnosis
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "get_transcript" in line or "YouTubeTranscriptApi" in line:
            print(f"  Line {i+1}: {line}")
    sys.exit(1)

TARGET.write_text(content, encoding="utf-8")