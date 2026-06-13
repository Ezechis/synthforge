# Fix TEXT_KEYS order: pdf_text must come before abstract
# so arXiv full paper text is used instead of abstract-only.

import sys
from pathlib import Path

TARGET = Path(r"C:\Users\Ezeking\SynthForge\src\processing\chunk_and_embed.py")

old = (
    '"text", "content", "body", "abstract", "pdf_text", '
    '"readme", "selftext", "comment_body"'
)
new = (
    '"text", "content", "body", "pdf_text", "abstract", '
    '"readme", "selftext", "comment_body"'
)

content = TARGET.read_text(encoding="utf-8")
if content.count(old) == 0:
    print("ERROR: string not found"); sys.exit(1)
TARGET.write_text(content.replace(old, new, 1), encoding="utf-8")
print("OK — pdf_text now precedes abstract in TEXT_KEYS")