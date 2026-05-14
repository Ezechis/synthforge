# Passes browser cookies to youtube-transcript-api to bypass IP blocks.

import sys
from pathlib import Path

TARGET = Path(r"C:\Users\Ezeking\PromptForge\src\ingestion\ingest_youtube.py")

old = '        yta = YouTubeTranscriptApi()\n        fetched = yta.fetch(video_id, languages=["en", "en-US", "en-GB"])'
new = '        yta = YouTubeTranscriptApi(cookies="cookies.txt")\n        fetched = yta.fetch(video_id, languages=["en", "en-US", "en-GB"])'

content = TARGET.read_text(encoding="utf-8")
if content.count(old) != 1:
    print(f"ERROR: found {content.count(old)} matches"); sys.exit(1)
TARGET.write_text(content.replace(old, new, 1), encoding="utf-8")
print("OK — cookies.txt path added to YouTubeTranscriptApi")