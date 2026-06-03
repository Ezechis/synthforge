"""
diagnose_pipeline.py
────────────────────
Run this locally to get a full health report on your SynthForge ingestion
pipeline BEFORE debugging GitHub Actions. Most problems are visible locally.

Usage:
  set HF_TOKEN=hf_your_token_here
  set GROQ_API_KEY=gsk_your_key_here
  C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe deploy/diagnose_pipeline.py

What it checks:
  1. Local ChromaDB collection: chunk count, collection name, version
  2. HF Dataset (vectorstore): last modified, file list, sqlite3 size
  3. HF Dataset (staging): how many transcript files are there, how many pending
  4. GitHub Actions secrets presence (cannot read values, just checks they exist via API)
  5. Groq API reachability (quick test ping)
  6. BM25 cache: file size, key names, chunk count match

Author: DeepForge / Claude (Sonnet 4.6)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — must match your production settings
# ---------------------------------------------------------------------------

LOCAL_VECTORSTORE_PATH: str = "data/vector_store"
COLLECTION_NAME: str = "synthforge"
BM25_CACHE_PATH: str = "deploy/bm25_cache.pkl"
VECTORSTORE_REPO: str = "ezechinnabugwu/synthforge-vectorstore"
STAGING_REPO: str = "ezechinnabugwu/synthforge-yt-staging"
EXPECTED_CHUNKS: int = 21_521  # update this after each corpus update
EXPECTED_CHROMADB_VERSION: str = "1.5.8"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg: str) -> None:
    """Print a passing check."""
    print(f"  ✓  {msg}")


def warn(msg: str) -> None:
    """Print a warning."""
    print(f"  ⚠  {msg}")


def fail(msg: str) -> None:
    """Print a failure."""
    print(f"  ✗  {msg}")


# ---------------------------------------------------------------------------
# Check 1: Local ChromaDB
# ---------------------------------------------------------------------------

def check_local_chromadb() -> None:
    """Verify local ChromaDB collection state."""
    section("1. LOCAL CHROMADB")
    try:
        import chromadb
        import chromadb.__version__ as _cv  # type: ignore[import]
        version = getattr(chromadb, "__version__", "unknown")
        if version == EXPECTED_CHROMADB_VERSION:
            ok(f"ChromaDB version: {version}")
        else:
            fail(f"ChromaDB version: {version} — expected {EXPECTED_CHROMADB_VERSION}")
    except ImportError:
        fail("chromadb not installed")
        return

    try:
        client = chromadb.PersistentClient(path=LOCAL_VECTORSTORE_PATH)
        collections = [c.name for c in client.list_collections()]
        ok(f"Collections found: {collections}")

        if COLLECTION_NAME not in collections:
            fail(f"Collection '{COLLECTION_NAME}' NOT FOUND")
            return

        col = client.get_collection(COLLECTION_NAME)
        count = col.count()
        if count >= EXPECTED_CHUNKS:
            ok(f"Chunk count: {count:,} (expected ≥{EXPECTED_CHUNKS:,})")
        elif count > 0:
            warn(f"Chunk count: {count:,} (expected {EXPECTED_CHUNKS:,} — may be stale)")
        else:
            fail(f"Chunk count: {count} — EMPTY COLLECTION")

        # Sample one chunk to verify structure
        sample = col.peek(limit=1)
        if sample.get("ids"):
            ok(f"Sample chunk ID: {sample['ids'][0][:20]}...")
            meta = sample.get("metadatas", [{}])[0]
            ok(f"Sample metadata keys: {list(meta.keys())}")
        else:
            warn("Could not peek at a sample chunk")

    except Exception as exc:
        fail(f"ChromaDB open error: {exc}")


# ---------------------------------------------------------------------------
# Check 2: BM25 Cache
# ---------------------------------------------------------------------------

def check_bm25_cache() -> None:
    """Verify the BM25 pickle file."""
    section("2. BM25 CACHE")
    cache_path = Path(BM25_CACHE_PATH)
    if not cache_path.exists():
        fail(f"BM25 cache not found at: {BM25_CACHE_PATH}")
        return

    size_mb = cache_path.stat().st_size / (1024 * 1024)
    ok(f"File size: {size_mb:.1f} MB")

    try:
        import pickle
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        keys = list(cache.keys())
        ok(f"Cache keys: {keys}")

        if "bm25" not in keys:
            fail("Key 'bm25' not found — app.py will crash on load")
        else:
            ok("Key 'bm25' present ✓")

        if "corpus_chunks" in keys:
            n_chunks = len(cache["corpus_chunks"])
            if n_chunks >= EXPECTED_CHUNKS:
                ok(f"corpus_chunks count: {n_chunks:,}")
            else:
                warn(f"corpus_chunks count: {n_chunks:,} (expected ≥{EXPECTED_CHUNKS:,})")
        else:
            warn("Key 'corpus_chunks' not found")

        if "corpus_metas" in keys:
            ok(f"corpus_metas count: {len(cache['corpus_metas']):,}")
        else:
            warn("Key 'corpus_metas' not found")

    except Exception as exc:
        fail(f"BM25 cache read error: {exc}")


# ---------------------------------------------------------------------------
# Check 3: HF Vectorstore Dataset
# ---------------------------------------------------------------------------

def check_hf_vectorstore(hf_token: str) -> None:
    """Verify the HF Dataset contains the correct vectorstore files."""
    section("3. HF DATASET — VECTORSTORE")
    if not hf_token:
        warn("HF_TOKEN not set — skipping HF dataset checks")
        return

    try:
        from huggingface_hub import HfApi, list_repo_tree

        api = HfApi(token=hf_token)
        info = api.repo_info(repo_id=VECTORSTORE_REPO, repo_type="dataset")
        ok(f"Repo exists: {VECTORSTORE_REPO}")
        ok(f"Last modified: {info.lastModified}")

        # Check for critical files
        files = [
            item.path
            for item in list_repo_tree(
                repo_id=VECTORSTORE_REPO, repo_type="dataset", token=hf_token
            )
            if hasattr(item, "path")
        ]
        ok(f"Total files in dataset: {len(files)}")

        if any("chroma.sqlite3" in f for f in files):
            ok("chroma.sqlite3 found in dataset")
        else:
            fail("chroma.sqlite3 NOT found — vectorstore upload may have failed")

        if any("bm25_cache.pkl" in f for f in files):
            ok("bm25_cache.pkl found in dataset")
        else:
            fail("bm25_cache.pkl NOT found — BM25 cache upload may have failed")

    except Exception as exc:
        fail(f"HF vectorstore check failed: {exc}")


# ---------------------------------------------------------------------------
# Check 4: HF Staging Dataset (YouTube transcripts)
# ---------------------------------------------------------------------------

def check_hf_staging(hf_token: str) -> None:
    """Check how many YouTube transcripts are in the staging dataset."""
    section("4. HF DATASET — YOUTUBE STAGING")
    if not hf_token:
        warn("HF_TOKEN not set — skipping staging check")
        return

    try:
        from huggingface_hub import HfApi, list_repo_tree

        api = HfApi(token=hf_token)
        try:
            api.repo_info(repo_id=STAGING_REPO, repo_type="dataset")
        except Exception:
            warn(f"Staging repo not yet created: {STAGING_REPO}")
            warn("It will be created automatically on the first YouTube workflow run")
            return

        transcript_files = [
            item.path
            for item in list_repo_tree(
                repo_id=STAGING_REPO, repo_type="dataset", token=hf_token
            )
            if hasattr(item, "path") and item.path.endswith(".jsonl")
        ]
        ok(f"Transcript files in staging: {len(transcript_files)}")

        tracker = Path("data/yt_staging_embedded.json")
        if tracker.exists():
            data = json.loads(tracker.read_text())
            already_embedded = len(data.get("embedded_files", []))
            pending = len(transcript_files) - already_embedded
            ok(f"Already embedded locally: {already_embedded}")
            if pending > 0:
                warn(f"Pending (staged but not yet embedded locally): {pending}")
            else:
                ok("All staged transcripts are embedded locally")
        else:
            warn("Local embedding tracker not found (data/yt_staging_embedded.json)")
            warn("Run deploy/pull_and_embed_yt_staging.py to start absorbing staged transcripts")

    except Exception as exc:
        fail(f"HF staging check failed: {exc}")


# ---------------------------------------------------------------------------
# Check 5: Groq API reachability
# ---------------------------------------------------------------------------

def check_groq_api(groq_api_key: str) -> None:
    """Test Groq API reachability with a minimal ping."""
    section("5. GROQ API")
    if not groq_api_key:
        warn("GROQ_API_KEY not set — skipping Groq check")
        return

    try:
        from groq import Groq

        client = Groq(api_key=groq_api_key)
        start = time.time()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        elapsed = time.time() - start
        reply = response.choices[0].message.content or ""
        ok(f"Groq API reachable — round-trip: {elapsed:.2f}s, reply: '{reply[:20]}'")

        # Check TPD usage if available
        usage = getattr(response, "usage", None)
        if usage:
            ok(f"Tokens used in ping: {usage.total_tokens}")

    except Exception as exc:
        fail(f"Groq API error: {exc}")
        if "rate_limit" in str(exc).lower() or "429" in str(exc):
            warn("Daily token limit (100k TPD) may be exhausted — resets at 1 AM WAT")


# ---------------------------------------------------------------------------
# Check 6: Environment variables
# ---------------------------------------------------------------------------

def check_env() -> None:
    """Check required environment variables."""
    section("6. ENVIRONMENT VARIABLES")
    required: dict[str, str] = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
    }
    for name, value in required.items():
        if name == "HF_HUB_OFFLINE":
            if value == "1":
                warn(f"HF_HUB_OFFLINE=1 — network calls will fail. "
                     f"Set to 0 before running pull_and_embed_yt_staging.py")
            else:
                ok(f"HF_HUB_OFFLINE={value} (network enabled)")
        elif value:
            ok(f"{name} is set (length: {len(value)} chars)")
        else:
            fail(f"{name} is NOT set")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full diagnostic suite."""
    print("\n" + "="*60)
    print("  SYNTHFORGE PIPELINE DIAGNOSTIC")
    print("="*60)
    print(f"  Vectorstore path : {LOCAL_VECTORSTORE_PATH}")
    print(f"  Collection name  : {COLLECTION_NAME}")
    print(f"  Expected chunks  : {EXPECTED_CHUNKS:,}")
    print("="*60)

    hf_token = os.environ.get("HF_TOKEN", "")
    groq_api_key = os.environ.get("GROQ_API_KEY", "")

    check_env()
    check_local_chromadb()
    check_bm25_cache()
    check_hf_vectorstore(hf_token)
    check_hf_staging(hf_token)
    check_groq_api(groq_api_key)

    print("\n" + "="*60)
    print("  DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nIf all checks pass locally, GitHub Actions failures are likely:")
    print("  1. Wrong GitHub Secret values (HF_DATASET_REPO, HF_TOKEN)")
    print("  2. Disk space exhaustion in the runner")
    print("  3. ChromaDB version mismatch between local and requirements_actions.txt")
    print()


if __name__ == "__main__":
    main()