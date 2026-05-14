"""
fix_security.py — Patch the unicode escape bug in security.py docstring.
The backslash path C:\\Users\\... in the triple-quoted docstring causes Python
to interpret \\U as a Unicode escape sequence, which crashes on import.
Fix: replace Windows backslash paths in the docstring with forward slashes.

Run from C:\\Users\\Ezeking\\PromptForge:
  python fix_security.py
"""
from pathlib import Path

SECURITY_PATH = Path(r"C:\Users\Ezeking\hf_space\security.py")

content = SECURITY_PATH.read_text(encoding="utf-8")

# Replace every Windows-style path in the docstring with forward slashes.
# The \\U in C:\\Users is what Python misreads as a Unicode escape.
REPLACEMENTS = [
    (
        r"Placed at: C:\Users\Ezeking\hf_space\security.py",
        "Placed at: C:/Users/Ezeking/hf_space/security.py",
    ),
    (
        r"Imported in app.py as: from security import detect_prompt_injection, check_rate_limit, validate_uploaded_file",
        "Imported in app.py as: from security import detect_prompt_injection, check_rate_limit, validate_uploaded_file",
    ),
]

changed = 0
for find, replace in REPLACEMENTS:
    if find in content:
        content = content.replace(find, replace)
        changed += 1
        print(f"  Fixed: {find[:60]}...")

if changed == 0:
    print("Nothing to fix — pattern not found. Check file contents.")
else:
    SECURITY_PATH.write_text(content, encoding="utf-8")
    print(f"\nFixed {changed} occurrence(s). Now run:")
    print(r"  cd C:\Users\Ezeking\hf_space")
    print(r"  git add security.py")
    print(r'  git commit -m "Fix security.py docstring unicode escape"')
    print(r"  git push")
