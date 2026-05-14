import sys
from pathlib import Path

APP = Path(r"C:\Users\Ezeking\hf_space\app.py")
content = APP.read_text(encoding="utf-8")

old = 'ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"'
new = 'ANTHROPIC_MODEL: str = "llama-3.3-70b-versatile"'

if content.count(old) != 1:
    print(f"ERROR: {content.count(old)} matches"); sys.exit(1)
APP.write_text(content.replace(old, new, 1), encoding="utf-8")
print("OK")