# Fix author field: join list to comma string instead of Python repr.
# Affects arXiv chunks which store authors as a list.

import sys
from pathlib import Path

TARGET = Path(r"C:\Users\Ezeking\PromptForge\src\processing\chunk_and_embed.py")

old = '        "author": str(metadata_source.get("author", metadata_source.get("authors", ""))),'
new = (
    '        "author": (\n'
    '            ", ".join(metadata_source["authors"])\n'
    '            if isinstance(metadata_source.get("authors"), list)\n'
    '            else str(metadata_source.get("author", metadata_source.get("authors", "")))\n'
    '        ),'
)

content = TARGET.read_text(encoding="utf-8")
if content.count(old) != 1:
    print(f"ERROR: found {content.count(old)} matches"); sys.exit(1)
TARGET.write_text(content.replace(old, new, 1), encoding="utf-8")
print("OK — author list formatting fixed")