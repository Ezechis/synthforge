"""
patch_ui.py — SynthForge UI Redesign Patch
============================================
Reads new_ui_section.py and splices it into hf_space/app.py,
replacing everything from the UI anchor to end of file.
Also patches generate_suggestions to produce 6 suggestions.

Run from C:\\Users\\Ezeking\\SynthForge:
  python patch_ui.py

Requires new_ui_section.py in the same directory as this script.
Creates app.py.bak2 before touching anything.
"""

import shutil
import sys
from pathlib import Path

APP_PATH    = Path(r"C:\Users\Ezeking\hf_space\app.py")
BACKUP_PATH = APP_PATH.with_suffix(".py.bak2")
NEW_UI_PATH = Path(__file__).parent / "new_ui_section.py"

UI_ANCHOR = (
    "# ---------------------------------------------------------------------------\n"
    "# UI\n"
    "# ---------------------------------------------------------------------------"
)

SUGG_PATCHES = [
    (
        '"Generate exactly 3 prompt-engineering follow-up questions. "\n'
        '                    "Return ONLY a JSON array of 3 strings."\n',
        '"Generate exactly 6 prompt-engineering follow-up questions. "\n'
        '                    "Return ONLY a JSON array of 6 strings."\n',
        "generate_suggestions: 3 -> 6",
    ),
    (
        '"max_tokens": 150,',
        '"max_tokens": 300,',
        "generate_suggestions: max_tokens 150 -> 300",
    ),
]


def patch_once(content: str, find: str, replace: str, label: str) -> str:
    if find not in content:
        print(f"\n  x FAILED [{label}] -- anchor not found:")
        print(f"    {find[:100]!r}")
        sys.exit(1)
    print(f"  ok {label}")
    return content.replace(find, replace, 1)


def main() -> None:
    for path, name in [(APP_PATH, "app.py"), (NEW_UI_PATH, "new_ui_section.py")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    print(f"\nSynthForge UI Redesign Patch")
    print(f"Target  : {APP_PATH}")
    print(f"New UI  : {NEW_UI_PATH}\n")

    content     = APP_PATH.read_text(encoding="utf-8")
    new_ui_text = NEW_UI_PATH.read_text(encoding="utf-8")

    shutil.copy(APP_PATH, BACKUP_PATH)
    print(f"Backup  : {BACKUP_PATH}\n")
    print("Applying patches:")

    for find, replace, label in SUGG_PATCHES:
        content = patch_once(content, find, replace, label)

    anchor_pos = content.find(UI_ANCHOR)
    if anchor_pos == -1:
        print("\n  x FAILED [UI anchor] -- not found in app.py")
        sys.exit(1)

    content = content[:anchor_pos] + new_ui_text
    print("  ok UI section replaced with new layout")

    APP_PATH.write_text(content, encoding="utf-8")

    print(f"\nAll patches applied. {APP_PATH.name} updated.")
    print("\nNow run:")
    print(r"  cd C:\Users\Ezeking\hf_space")
    print(r"  git add app.py")
    print(r'  git commit -m "UI: attach inline, Griot top-right, metrics strip, 6 suggestions, numbered history"')
    print(r"  git push")


if __name__ == "__main__":
    main()
