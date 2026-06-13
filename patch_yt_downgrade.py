import sys
from pathlib import Path

TARGET = Path(r"C:\Users\Ezeking\SynthForge\src\ingestion\ingest_youtube.py")
content = TARGET.read_text(encoding="utf-8")

old = '        yta = YouTubeTranscriptApi(cookies="cookies.txt")\n        fetched = yta.fetch(video_id, languages=["en", "en-US", "en-GB"])\n        return " ".join(snippet.text for snippet in fetched).strip()'
new = '        segments = YouTubeTranscriptApi.get_transcript(\n            video_id, languages=["en", "en-US", "en-GB"], cookies="cookies.txt"\n        )\n        return " ".join(seg["text"] for seg in segments).strip()'

if content.count(old) != 1:
    print(f"ERROR: {content.count(old)} matches"); sys.exit(1)
TARGET.write_text(content.replace(old, new, 1), encoding="utf-8")
print("OK — reverted to v0.6.3 API with cookies")