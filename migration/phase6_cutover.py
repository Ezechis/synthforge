"""Phase 6 -- CUTOVER to the HF dataset (HUMAN-RUN, irreversible).

This script is PREPARED for a human to run; the assistant must not execute the
push. It is the only step that writes to production. Safeguards:

  * Offline DRY-RUN by default. It performs local preflight only and prints the
    plan. Nothing touches the network or the token unless you pass --confirm-push.
  * Token from env only. The HF write token is read from HF_TOKEN; it is never
    hardcoded, logged, or persisted.
  * Rollback first. Before overwriting, it records the current production commit
    SHA and snapshot-downloads the live dataset to a timestamped rollback dir.
  * Orphan-safe upload. The clean store has a NEW segment UUID; the old segment
    is removed from the repo via delete_patterns so the Space can't load stale
    files. bm25_cache.pkl is placed at repo ROOT (not under vector_store/).

Usage:
    # 1) safe offline preview (no network, no token):
    py -m migration.phase6_cutover

    # 2) real cutover (you run this, with your write token in the env):
    export HF_TOKEN=<synthforge-write-june2026>
    py -m migration.phase6_cutover --confirm-push
    #   optionally also: --restart-space --space-id <owner/space>
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")     # local preflight stays offline
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

REPO = Path(__file__).resolve().parents[1]
CLEAN_STORE = REPO / "data" / "vector_store_clean"
BM25 = CLEAN_STORE / "bm25_cache.pkl"
ROLLBACK_ROOT = Path(__file__).resolve().parent / "rollback"

REPO_ID = "ezechinnabugwu/synthforge-vectorstore"
REPO_TYPE = "dataset"
COLLECTION = "synthforge"
EXPECTED = 2247


def preflight() -> None:
    """Local, offline integrity checks on the clean store (no network/token)."""
    if not (CLEAN_STORE / "chroma.sqlite3").exists():
        sys.exit(f"FATAL: clean store missing: {CLEAN_STORE/'chroma.sqlite3'}")
    if not BM25.exists():
        sys.exit(f"FATAL: BM25 cache missing: {BM25} (run migration.build_bm25)")

    import chromadb

    client = chromadb.PersistentClient(path=str(CLEAN_STORE))
    names = [c.name for c in client.list_collections()]
    if names != [COLLECTION]:
        sys.exit(f"FATAL: integrity -- expected ['{COLLECTION}'], found {names}")
    coll = client.get_collection(COLLECTION, embedding_function=None)
    n = coll.count()
    if n != EXPECTED:
        sys.exit(f"FATAL: clean store has {n} records, expected {EXPECTED}")
    segs = [p.name for p in CLEAN_STORE.iterdir() if p.is_dir()]
    print("Preflight OK:")
    print(f"  clean store : {CLEAN_STORE}")
    print(f"  collection  : {COLLECTION}  count={n}")
    print(f"  segment dir : {segs}")
    print(f"  bm25 cache  : {BM25} ({BM25.stat().st_size/1e6:.1f} MB)")


def print_plan(args) -> None:
    print("\nCUTOVER PLAN (target = PRODUCTION):")
    print(f"  repo            : {REPO_ID} ({REPO_TYPE}, private)")
    print(f"  upload folder   : {CLEAN_STORE}  ->  repo path 'vector_store/'")
    print(f"                    ignore bm25_cache.pkl; delete_patterns=['**'] "
          f"(removes the old segment {('a0c0596a-...')})")
    print(f"  upload file     : {BM25.name}  ->  repo path 'bm25_cache.pkl' (root)")
    print(f"  rollback dir    : {ROLLBACK_ROOT}/<timestamp>/ (snapshot + SHA)")
    print(f"  restart space   : {'YES -> ' + args.space_id if args.restart_space else 'no (manual)'}")
    if not args.confirm_push:
        print("\nDRY-RUN ONLY. Re-run with --confirm-push (and HF_TOKEN set) to "
              "perform the irreversible cutover.")


def do_cutover(args) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("FATAL: --confirm-push requires HF_TOKEN (the "
                 "synthforge-write-june2026 write token) in the environment.")
    # Network operations require online mode.
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)

    # --- 1. Rollback artifact (record SHA + snapshot the live dataset) ---------
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rollback_dir = ROLLBACK_ROOT / stamp
    rollback_dir.mkdir(parents=True, exist_ok=True)
    info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    (rollback_dir / "PRODUCTION_SHA.txt").write_text(
        f"{REPO_ID}\nsha={info.sha}\nsnapshotted_at={stamp}\n"
    )
    print(f"[rollback] current production sha = {info.sha}")
    print(f"[rollback] downloading live dataset -> {rollback_dir/'snapshot'} ...")
    snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        local_dir=str(rollback_dir / "snapshot"), token=token,
    )
    print("[rollback] snapshot complete. Keep this dir for the rollback window.")

    # --- 2. Upload the clean store (orphan-safe) ------------------------------
    print("[push] uploading vector_store/ (chroma + new segment; deleting orphans)...")
    api.upload_folder(
        folder_path=str(CLEAN_STORE),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo="vector_store",
        ignore_patterns=["bm25_cache.pkl"],   # bm25 belongs at repo root
        delete_patterns=["**"],                # remove the old segment/files
        commit_message="Cohort-B migration: replace corpus with clean "
                        "2,247-record DEEPCORE v0.2.0 store",
    )
    print("[push] uploading bm25_cache.pkl to repo root...")
    api.upload_file(
        path_or_fileobj=str(BM25),
        path_in_repo="bm25_cache.pkl",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Cohort-B migration: rebuilt BM25 cache (2,247 docs)",
    )
    print("[push] upload complete.")

    # --- 3. Post-push verification -------------------------------------------
    files = set(api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE))
    new_seg = [p.name for p in CLEAN_STORE.iterdir() if p.is_dir()][0]
    checks = {
        "vector_store/chroma.sqlite3": "vector_store/chroma.sqlite3" in files,
        f"vector_store/{new_seg}/ present": any(
            f.startswith(f"vector_store/{new_seg}/") for f in files),
        "old segment a0c0596a gone": not any("a0c0596a" in f for f in files),
        "bm25_cache.pkl at root": "bm25_cache.pkl" in files,
    }
    print("[verify]")
    for k, v in checks.items():
        print(f"   {'OK ' if v else 'FAIL'} {k}")
    if not all(checks.values()):
        sys.exit("FATAL: post-push verification failed -- consider rollback.")

    # --- 4. Optional Space restart -------------------------------------------
    if args.restart_space:
        if not args.space_id:
            sys.exit("--restart-space requires --space-id <owner/space>")
        print(f"[space] restarting {args.space_id} ...")
        api.restart_space(repo_id=args.space_id)
        print("[space] restart requested.")
    else:
        print("[space] restart skipped -- restart the Space in the HF UI "
              "(or re-run with --restart-space --space-id <owner/space>).")

    print("\nCUTOVER COMPLETE. Live spot-check now:")
    print("  - open the Space, run a few known queries (e.g. 'chain of thought',")
    print("    'RLHF', 'prompt engineering') and confirm sane results.")
    print(f"  - keep {rollback_dir} for the agreed rollback window.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-push", action="store_true",
                    help="ACTUALLY push to production (requires HF_TOKEN).")
    ap.add_argument("--restart-space", action="store_true")
    ap.add_argument("--space-id", default="", help="owner/space for restart")
    args = ap.parse_args()

    preflight()
    print_plan(args)
    if args.confirm_push:
        print("\n--confirm-push set: performing the irreversible cutover.\n")
        do_cutover(args)


if __name__ == "__main__":
    main()
