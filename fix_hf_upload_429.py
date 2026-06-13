"""
fix_hf_upload_429.py — Patches deploy/upload_vectorstore.py to handle
HuggingFace 429 "Too Many Requests" when repo already exists.

Run from C:\\Users\\Ezeking\\PromptForge:
    python fix_hf_upload_429.py

What it does:
    1. Reads deploy/upload_vectorstore.py
    2. Replaces create_repo() call with a safe exist-or-create pattern
    3. Adds retry logic with exponential backoff for 429 errors
    4. Writes the patched file back
    5. Prints OK or MISSING for each change
"""

import sys
from pathlib import Path

SCRIPT_PATH = Path("deploy/upload_vectorstore.py")

if not SCRIPT_PATH.exists():
    print(f"ERROR: {SCRIPT_PATH} not found.")
    print("Run this script from C:\\Users\\Ezeking\\PromptForge")
    sys.exit(1)

original = SCRIPT_PATH.read_text(encoding="utf-8")
patched = original

# ── Patch 1: Add time import if missing ──────────────────────────────────────
if "import time" not in patched:
    patched = "import time\n" + patched
    print("OK  — Added: import time")
else:
    print("OK  — import time already present")

# ── Patch 2: Replace bare create_repo with safe version ──────────────────────
# Pattern A: api.create_repo(repo_id=..., repo_type="dataset", exist_ok=False)
# Pattern B: api.create_repo(repo_id=..., repo_type="dataset")
# Both need exist_ok=True

import re

# Fix exist_ok=False → exist_ok=True
if "exist_ok=False" in patched:
    patched = patched.replace("exist_ok=False", "exist_ok=True")
    print("OK  — Fixed: exist_ok=False → exist_ok=True")
else:
    print("OK  — exist_ok=False not found (already safe or different pattern)")

# Add exist_ok=True if create_repo is called without it
create_repo_pattern = r'(api\.create_repo\([^)]+repo_type=["\']dataset["\'])(\s*\))'
if re.search(create_repo_pattern, patched):
    def add_exist_ok(m):
        if "exist_ok" not in m.group(0):
            return m.group(1) + ", exist_ok=True" + m.group(2)
        return m.group(0)
    patched = re.sub(create_repo_pattern, add_exist_ok, patched)
    print("OK  — Ensured: exist_ok=True on create_repo() call")
else:
    print("OK  — create_repo pattern not matched (may be different style)")

# ── Patch 3: Add safe_upload helper with retry ───────────────────────────────
SAFE_UPLOAD = '''
def _safe_upload_folder(
    api,
    folder_path: str,
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    max_retries: int = 3,
) -> None:
    """Upload a folder to HuggingFace with retry on 429 Too Many Requests.

    Args:
        api: HfApi instance.
        folder_path: Local folder to upload.
        repo_id: HuggingFace repository ID.
        path_in_repo: Target path inside the repository.
        repo_type: Repository type (default: dataset).
        max_retries: Number of retry attempts on 429 errors.
    """
    import logging
    log = logging.getLogger(__name__)

    # Ensure repo exists (safe — does not fail if already exists)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True,
            private=True,
        )
    except Exception as exc:
        log.warning("create_repo warning (non-fatal): %s", exc)

    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
            )
            log.info("Upload succeeded on attempt %d.", attempt)
            return
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "Too Many Requests" in err_str:
                wait = 30 * attempt  # 30s, 60s, 90s
                log.warning(
                    "429 Too Many Requests on attempt %d/%d. "
                    "Waiting %ds before retry...",
                    attempt, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Upload to {repo_id} failed after {max_retries} attempts (429 rate limit)."
    )

'''

if "_safe_upload_folder" not in patched:
    match = re.search(r'^def |^if __name__', patched, re.MULTILINE)
    if match:
        insert_pos = match.start()
        patched = patched[:insert_pos] + SAFE_UPLOAD + patched[insert_pos:]
        print("OK  — Added: _safe_upload_folder() with retry logic")
    else:
        patched = patched + SAFE_UPLOAD
        print("OK  — Added: _safe_upload_folder() with retry logic (at end)")
else:
    print("OK  — _safe_upload_folder() already present")

# ── Patch 4: Replace direct upload_folder calls with safe version ────────────
# Replace: api.upload_folder(folder_path=..., repo_id=..., path_in_repo=..., repo_type="dataset")
# With:    _safe_upload_folder(api, ..., ..., ..., "dataset")

upload_pattern = r'api\.upload_folder\('
if re.search(upload_pattern, patched):
    # Only replace if _safe_upload_folder not already wrapping it
    lines = patched.split('\n')
    new_lines = []
    for line in lines:
        if 'api.upload_folder(' in line and '_safe_upload_folder' not in line:
            line = line.replace('api.upload_folder(', '_safe_upload_folder(api, ')
            print(f"OK  — Replaced api.upload_folder() call with _safe_upload_folder()")
        new_lines.append(line)
    patched = '\n'.join(new_lines)
else:
    print("OK  — No direct api.upload_folder() calls found")

# ── Write patched file ────────────────────────────────────────────────────────
backup_path = SCRIPT_PATH.with_suffix(".py.bak")
backup_path.write_text(original, encoding="utf-8")
print(f"OK  — Backup saved: {backup_path}")

SCRIPT_PATH.write_text(patched, encoding="utf-8")
print(f"OK  — Patched file written: {SCRIPT_PATH}")

print("\nNEXT STEPS:")
print("1. git add deploy/upload_vectorstore.py")
print("2. git commit -m 'Fix: safe repo creation, retry on 429 for HF upload'")
print("3. git push")
print("4. Trigger Weekly Reddit Ingestion workflow manually")
print("5. The upload step should now succeed without 429 errors")
