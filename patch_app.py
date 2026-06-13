"""
patch_app.py — SynthForge Session 9 Security Patch
=====================================================
Makes exactly four edits to hf_space/app.py:
  1. Adds security import line
  2. Adds file upload validation gate (size + PDF page cap)
  3. Adds prompt injection + rate limit gates in search handler
  4. Renames ForgeAI → DeepForge in sidebar caption

Run from C:\\Users\\Ezeking\\SynthForge:
  C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe patch_app.py

Creates app.py.bak before touching anything.
Aborts loudly if any anchor string is not found — nothing is written on failure.
"""

import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_PATH: Path = Path(r"C:\Users\Ezeking\hf_space\app.py")
BACKUP_PATH: Path = APP_PATH.with_suffix(".py.bak")


# ---------------------------------------------------------------------------
# Patch engine
# ---------------------------------------------------------------------------

def apply_patch(content: str, label: str, find: str, replace: str) -> str:
    """
    Replace the first occurrence of `find` with `replace` in `content`.

    Args:
        content: Full file text.
        label:   Human-readable name for this patch (printed on success).
        find:    Exact string to locate. Must appear exactly once.
        replace: String to substitute in.

    Returns:
        Updated content string.

    Raises:
        SystemExit: If `find` is not found, prints the anchor and exits
                    without writing anything.
    """
    if find not in content:
        print(f"\n  ✗ PATCH FAILED: [{label}]")
        print(f"    Anchor string not found in app.py.")
        print(f"    Looking for:\n    {find[:120]!r}")
        print("\n  Nothing has been written. app.py is unchanged.")
        sys.exit(1)

    count = content.count(find)
    if count > 1:
        print(f"\n  ✗ PATCH FAILED: [{label}]")
        print(f"    Anchor matches {count} locations — ambiguous. Aborting.")
        sys.exit(1)

    new_content = content.replace(find, replace, 1)
    print(f"  ✓ {label}")
    return new_content


# ---------------------------------------------------------------------------
# Patch definitions
# ---------------------------------------------------------------------------

# --- Edit 1: Import ---
IMPORT_FIND = (
    "from sentence_transformers import CrossEncoder, SentenceTransformer"
)
IMPORT_REPLACE = (
    "from sentence_transformers import CrossEncoder, SentenceTransformer\n"
    "from security import detect_prompt_injection, check_rate_limit, validate_uploaded_file"
)

# --- Edit 2: File upload validation gate ---
# Anchor: the closing paren of st.file_uploader() → blank line → st.columns line
UPLOAD_FIND = (
    '"SynthForge analyses your document alongside retrieved chunks.",\n'
    ')\n'
    '\n'
    'c1, c2, _ = st.columns([1, 1, 6])'
)
UPLOAD_REPLACE = (
    '"SynthForge analyses your document alongside retrieved chunks.",\n'
    ')\n'
    '\n'
    'if uploaded_file is not None:\n'
    '    _is_valid, _validation_reason = validate_uploaded_file(uploaded_file)\n'
    '    if not _is_valid:\n'
    '        st.error(f"\u26a0\ufe0f {_validation_reason}")\n'
    '        uploaded_file = None\n'
    '\n'
    'c1, c2, _ = st.columns([1, 1, 6])'
)

# --- Edit 3: Inject security gates at the top of the search handler ---
# Anchor: the two lines that open the handler block
SEARCH_FIND = (
    'if search_clicked and query.strip():\n'
    '    st.session_state["suggestions"] = []\n'
    '    t0 = time.time()'
)
SEARCH_REPLACE = (
    'if search_clicked and query.strip():\n'
    '    # Security Gate 1 — prompt injection / jailbreak detection\n'
    '    _flagged, _injection_reason = detect_prompt_injection(query.strip())\n'
    '    if _flagged:\n'
    '        st.error(f"\u26a0\ufe0f {_injection_reason}")\n'
    '        st.stop()\n'
    '    # Security Gate 2 — per-session rate limit (20 queries / 10 min)\n'
    '    _allowed, _rate_reason = check_rate_limit()\n'
    '    if not _allowed:\n'
    '        st.warning(_rate_reason)\n'
    '        st.stop()\n'
    '    st.session_state["suggestions"] = []\n'
    '    t0 = time.time()'
)

# --- Edit 4: Branding ---
BRAND_FIND    = "SynthForge \u00b7 ForgeAI Platform \u00b7 by Ezechinyere Nnabugwu"
BRAND_REPLACE = "SynthForge \u00b7 DeepForge Platform \u00b7 by Ezechinyere Nnabugwu"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Read app.py, apply all patches, write back. Backup first."""

    if not APP_PATH.exists():
        print(f"ERROR: app.py not found at {APP_PATH}")
        sys.exit(1)

    print(f"\nSynthForge Session 9 — Security Patch")
    print(f"Target : {APP_PATH}")

    content: str = APP_PATH.read_text(encoding="utf-8")

    # Backup before any modification
    shutil.copy(APP_PATH, BACKUP_PATH)
    print(f"Backup : {BACKUP_PATH}\n")
    print("Applying patches:")

    content = apply_patch(content, "Security import added",           IMPORT_FIND,  IMPORT_REPLACE)
    content = apply_patch(content, "File upload validation gate",      UPLOAD_FIND,  UPLOAD_REPLACE)
    content = apply_patch(content, "Injection + rate-limit in search", SEARCH_FIND,  SEARCH_REPLACE)
    content = apply_patch(content, "ForgeAI → DeepForge branding",     BRAND_FIND,   BRAND_REPLACE)

    APP_PATH.write_text(content, encoding="utf-8")

    print(f"\nAll 4 patches applied. {APP_PATH.name} updated.")
    print("\nNext steps (copy and run in order):")
    print("  cd C:\\Users\\Ezeking\\hf_space")
    print("  git add app.py security.py")
    print('  git commit -m "Security: injection filter, rate limiter, file validation. Branding: DeepForge."')
    print("  git push")


if __name__ == "__main__":
    main()
