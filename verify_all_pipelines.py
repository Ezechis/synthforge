"""
verify_all_pipelines.py — Complete SynthForge Pipeline Verification

Tests all three ingestion pipelines and MIPROv2 readiness.
Run from C:\\Users\\Ezeking\\PromptForge:

    python verify_all_pipelines.py

Outputs a clear PASS/FAIL for each component with exact fix instructions.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# ── Colour helpers (works in Windows CMD) ────────────────────────────────────
def ok(msg):   print(f"  [PASS] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def head(msg): print(f"\n{'='*60}\n  {msg}\n{'='*60}")

ERRORS = []

def record_fail(section, message, fix):
    ERRORS.append({"section": section, "message": message, "fix": fix})
    fail(message)

# ─────────────────────────────────────────────────────────────────────────────
head("1. ENVIRONMENT CHECKS")
# ─────────────────────────────────────────────────────────────────────────────

# Check working directory
cwd = Path.cwd()
if (cwd / "src").exists() and (cwd / "deploy").exists():
    ok(f"Working directory: {cwd}")
else:
    record_fail(
        "Environment",
        f"Wrong directory: {cwd}",
        "Run: cd C:\\Users\\Ezeking\\PromptForge"
    )

# Check Python packages
required_packages = [
    ("chromadb", "chromadb==1.5.8"),
    ("sentence_transformers", "sentence-transformers"),
    ("praw", "praw"),
    ("dspy", "dspy-ai"),
    ("groq", "groq"),
    ("huggingface_hub", "huggingface_hub"),
    ("rank_bm25", "rank-bm25"),
]
for pkg, install_name in required_packages:
    try:
        __import__(pkg)
        ok(f"Package installed: {pkg}")
    except ImportError:
        record_fail(
            "Environment",
            f"Package missing: {pkg}",
            f"pip install {install_name}"
        )

# Check ChromaDB version
try:
    import chromadb
    version = chromadb.__version__
    if version == "1.5.8":
        ok(f"ChromaDB version: {version} (correct)")
    else:
        record_fail(
            "Environment",
            f"ChromaDB version wrong: {version} (need 1.5.8)",
            "pip install chromadb==1.5.8 --force-reinstall"
        )
except Exception as e:
    record_fail("Environment", f"ChromaDB import failed: {e}", "pip install chromadb==1.5.8")

# ─────────────────────────────────────────────────────────────────────────────
head("2. CHROMADB CORPUS CHECK")
# ─────────────────────────────────────────────────────────────────────────────

VECTOR_STORE_PATH = Path("data/vector_store")
COLLECTION_NAME = "synthforge"

try:
    import chromadb
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
    collections = [c.name for c in client.list_collections()]
    ok(f"Collections found: {collections}")

    if collections != ["synthforge"]:
        record_fail(
            "ChromaDB",
            f"Wrong collections: {collections} (expected: ['synthforge'])",
            "The ghost collection is back. Run: deploy/build_bm25_cache.py to verify"
        )
    else:
        ok("Collection integrity: PASS (only 'synthforge' present)")

    col = client.get_collection(COLLECTION_NAME)
    count = col.count()
    ok(f"Chunk count: {count:,}")
    if count < 21000:
        warn(f"Chunk count lower than expected (21,527). Current: {count}")
    elif count > 21527:
        ok(f"Corpus has GROWN beyond 21,527 — ingestion is working!")

except Exception as e:
    record_fail("ChromaDB", f"ChromaDB error: {e}", "Check data/vector_store exists and is not corrupted")

# ─────────────────────────────────────────────────────────────────────────────
head("3. GROQ API KEY CHECK")
# ─────────────────────────────────────────────────────────────────────────────

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_KEY or len(GROQ_KEY) < 20:
    record_fail("Groq", "GROQ_API_KEY not set", "set GROQ_API_KEY=gsk_...")
else:
    ok(f"GROQ_API_KEY present: {GROQ_KEY[:8]}...{GROQ_KEY[-4:]} ({len(GROQ_KEY)} chars)")

    # Test live connection
    try:
        import groq as groq_sdk
        client_groq = groq_sdk.Groq(api_key=GROQ_KEY)
        response = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        ok(f"Groq API live: responded '{response.choices[0].message.content.strip()}'")
    except Exception as e:
        err_str = str(e)
        if "invalid_api_key" in err_str or "Invalid API Key" in err_str:
            record_fail(
                "Groq",
                "Groq API key is INVALID or REVOKED",
                "1. Go to console.groq.com\n"
                "     2. Generate a new API key\n"
                "     3. Update GROQ_API_KEY in run_miprov2_final.py line 14\n"
                "     4. Update GROQ_API_KEY in GitHub Secrets"
            )
        elif "429" in err_str or "rate_limit" in err_str.lower():
            warn("Groq daily limit hit — wait until 1 AM WAT (midnight UTC)")
        else:
            warn(f"Groq connection issue: {err_str[:100]}")

# ─────────────────────────────────────────────────────────────────────────────
head("4. REDDIT PIPELINE CHECK")
# ─────────────────────────────────────────────────────────────────────────────

reddit_vars = {
    "REDDIT_CLIENT_ID": os.environ.get("REDDIT_CLIENT_ID", ""),
    "REDDIT_CLIENT_SECRET": os.environ.get("REDDIT_CLIENT_SECRET", ""),
    "REDDIT_USERNAME": os.environ.get("REDDIT_USERNAME", ""),
    "REDDIT_PASSWORD": os.environ.get("REDDIT_PASSWORD", ""),
}

missing_reddit = [k for k, v in reddit_vars.items() if not v]
if missing_reddit:
    warn(f"Reddit env vars not set locally: {missing_reddit}")
    warn("This is OK — they are set in GitHub Secrets for Actions runs")
    warn("For local testing, set them manually with 'set REDDIT_CLIENT_ID=...' etc.")
else:
    ok("All Reddit env vars present")

# Test PRAW connection if credentials available
if not missing_reddit:
    try:
        import praw
        reddit = praw.Reddit(
            client_id=reddit_vars["REDDIT_CLIENT_ID"],
            client_secret=reddit_vars["REDDIT_CLIENT_SECRET"],
            username=reddit_vars["REDDIT_USERNAME"],
            password=reddit_vars["REDDIT_PASSWORD"],
            user_agent="SynthForge-Ingest/1.0",
        )
        me = reddit.user.me()
        ok(f"PRAW OAuth: authenticated as u/{me}")

        # Test subreddit access
        sub = reddit.subreddit("PromptEngineering")
        posts = list(sub.top(time_filter="month", limit=3))
        ok(f"Reddit r/PromptEngineering: fetched {len(posts)} posts")
        for p in posts[:2]:
            ok(f"  Sample: [{p.score} upvotes] {p.title[:60]}")

    except Exception as e:
        record_fail(
            "Reddit",
            f"PRAW OAuth failed: {e}",
            "Check REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, USERNAME, PASSWORD are correct"
        )
else:
    # Check if ingest_reddit.py uses PRAW
    reddit_script = Path("src/ingestion/ingest_reddit.py")
    if reddit_script.exists():
        content = reddit_script.read_text(encoding="utf-8")
        if "import praw" in content or "praw.Reddit" in content:
            ok("ingest_reddit.py: uses PRAW OAuth ✅")
        elif "requests.get" in content and "reddit.com" in content:
            record_fail(
                "Reddit",
                "ingest_reddit.py uses unauthenticated requests — this causes 403",
                "Apply fix_reddit_praw.py: python fix_reddit_praw.py"
            )
        else:
            warn("ingest_reddit.py: cannot determine auth method")
    else:
        record_fail("Reddit", "ingest_reddit.py not found", "Check src/ingestion/ directory")

# ─────────────────────────────────────────────────────────────────────────────
head("5. YOUTUBE PIPELINE CHECK")
# ─────────────────────────────────────────────────────────────────────────────

# Check local YouTube script
yt_script = Path("src/ingestion/ingest_youtube_local.py")
if yt_script.exists():
    ok("ingest_youtube_local.py: present")
else:
    record_fail("YouTube", "ingest_youtube_local.py not found", "Check src/ingestion/ directory")

# Check pull_and_embed script
embed_script = Path("deploy/pull_and_embed_yt_staging.py")
if embed_script.exists():
    ok("pull_and_embed_yt_staging.py: present")
else:
    record_fail("YouTube", "pull_and_embed_yt_staging.py not found", "Check deploy/ directory")

# Check HF staging progress
try:
    import urllib.request
    url = "https://huggingface.co/datasets/ezechinnabugwu/synthforge-yt-staging/raw/main/yt_progress.json"
    data = json.loads(urllib.request.urlopen(url, timeout=10).read())
    completed = data.get("completed_count", 0)
    failed = len(data.get("failed_ids", []))
    no_transcript = len(data.get("no_transcript_ids", []))
    ok(f"YouTube staging progress: {completed}/947 completed")
    ok(f"  Failed (need Groq Whisper): {failed}")
    ok(f"  No transcript available: {no_transcript}")
    ok(f"  Remaining: {947 - completed - no_transcript}")
    if completed <= 3:
        warn("Only 3 videos completed — YouTube ingestion needs attention")
        warn("Trigger: GitHub Actions → YouTube Batch Transcription → Run workflow")
except Exception as e:
    warn(f"Cannot reach HF staging (may need HF_HUB_OFFLINE=0): {e}")

# Check youtube-transcript-api
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    ok("youtube-transcript-api: installed")
except ImportError:
    record_fail("YouTube", "youtube-transcript-api not installed", "pip install youtube-transcript-api")

# ─────────────────────────────────────────────────────────────────────────────
head("6. HF UPLOAD PIPELINE CHECK")
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
upload_script = Path("deploy/upload_vectorstore.py")

if upload_script.exists():
    content = upload_script.read_text(encoding="utf-8")
    if "exist_ok=True" in content:
        ok("upload_vectorstore.py: exist_ok=True present")
    elif "exist_ok=False" in content:
        record_fail(
            "HF Upload",
            "upload_vectorstore.py has exist_ok=False — causes 429 on existing repos",
            "Apply fix_hf_upload_429.py: python fix_hf_upload_429.py"
        )
    else:
        warn("upload_vectorstore.py: exist_ok not found — may cause 429")
else:
    record_fail("HF Upload", "upload_vectorstore.py not found", "Check deploy/ directory")

# ─────────────────────────────────────────────────────────────────────────────
head("7. MIPROV2 READINESS CHECK")
# ─────────────────────────────────────────────────────────────────────────────

# Check run_miprov2_final.py
final_script = Path("run_miprov2_final.py")
if final_script.exists():
    ok("run_miprov2_final.py: present")
else:
    record_fail(
        "MIPROv2",
        "run_miprov2_final.py not found",
        "copy /Y C:\\Users\\Ezeking\\Downloads\\run_miprov2_final.py C:\\Users\\Ezeking\\PromptForge\\run_miprov2_final.py"
    )

# Check eval file
eval_path = Path("data/evals/PromptForge_Golden_Eval_Stratified_45k_clean.jsonl")
if eval_path.exists():
    line_count = sum(1 for _ in eval_path.open(encoding="utf-8") if _.strip())
    ok(f"Golden eval corpus: {line_count:,} entries at {eval_path}")
else:
    record_fail("MIPROv2", f"Eval file not found: {eval_path}", "Check data/evals/ directory")

# Check for 300k eval
evals_dir = Path("data/evals")
all_evals = list(evals_dir.glob("*.jsonl")) if evals_dir.exists() else []
ok(f"Eval files found: {[f.name for f in all_evals]}")

# Check optimized prompt
opt_path = Path("data/optimization/optimized_prompt_latest.txt")
if opt_path.exists():
    ok(f"Optimized prompt EXISTS: {opt_path}")
    ok(f"  Content preview: {opt_path.read_text(encoding='utf-8')[:100]}...")
else:
    warn("Optimized prompt not yet generated — MIPROv2 has not completed successfully")

# Check dspy
try:
    import dspy
    ok(f"DSPy version: {dspy.__version__}")
    if not dspy.__version__.startswith("3.2"):
        warn(f"DSPy version {dspy.__version__} — expected 3.2.x")
except Exception as e:
    record_fail("MIPROv2", f"DSPy import failed: {e}", "pip install dspy-ai")

# Check optuna
try:
    import optuna
    ok(f"Optuna: installed ({optuna.__version__})")
except ImportError:
    record_fail("MIPROv2", "Optuna not installed", "pip install optuna")

# ─────────────────────────────────────────────────────────────────────────────
head("8. GITHUB ACTIONS STATUS CHECK")
# ─────────────────────────────────────────────────────────────────────────────

workflows_dir = Path(".github/workflows")
if workflows_dir.exists():
    for wf in workflows_dir.glob("*.yml"):
        content = wf.read_text(encoding="utf-8")
        issues = []
        if "github.sha" in content:
            issues.append("cache key uses github.sha (causes re-download every run)")
        if "ubuntu-latest" in content and wf.name == "youtube_batch.yml":
            issues.append("YouTube uses ubuntu-latest instead of self-hosted runner")
        if issues:
            for issue in issues:
                record_fail("GitHub Actions", f"{wf.name}: {issue}",
                           f"Fix {wf.name} — see Session 20 continuity doc")
        else:
            ok(f"{wf.name}: no issues detected")
else:
    warn(".github/workflows not found — check repository structure")

# ─────────────────────────────────────────────────────────────────────────────
head("SUMMARY")
# ─────────────────────────────────────────────────────────────────────────────

if not ERRORS:
    print("\n  ALL CHECKS PASSED — Pipeline is healthy\n")
else:
    print(f"\n  {len(ERRORS)} ISSUE(S) FOUND:\n")
    for i, err in enumerate(ERRORS, 1):
        print(f"  [{i}] {err['section']}: {err['message']}")
        print(f"       FIX: {err['fix']}\n")

print("\nDone. Address failures in order shown above.")
print("Re-run this script after each fix to confirm resolution.")
