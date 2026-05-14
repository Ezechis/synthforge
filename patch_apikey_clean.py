import sys
from pathlib import Path

APP = Path(r"C:\Users\Ezeking\hf_space\app.py")

old = '    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()'
new = '    anthropic_key = "".join(os.environ.get("ANTHROPIC_API_KEY", "").split())'

content = APP.read_text(encoding="utf-8")
if content.count(old) != 1:
    print(f"ERROR: {content.count(old)} matches"); sys.exit(1)
APP.write_text(content.replace(old, new, 1), encoding="utf-8")
print("OK — API key whitespace stripping hardened")