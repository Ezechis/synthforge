"""
SynthForge — Book Ingestion Entrypoint
=======================================
This is the script you run from CMD to ingest books into SynthForge.

It imports everything from deepforge.core (the shared ForgeCore engine)
and runs the BookIngestor against the curated BOOK_CATALOGUE.

Output: data/book_all_chunks.jsonl
        data/book_chunks/<source_id>.jsonl  (per-source, for debugging)
        data/book_progress.json             (resume state)

After this script completes, run the standard corpus update sequence:
    python src/processing/chunk_and_embed.py   ← reads book_all_chunks.jsonl
    python deploy/compress_vectorstore.py
    python deploy/build_bm25_cache.py
    python deploy/upload_vectorstore.py

Usage:
    python src/ingestion/ingest_books.py
    python src/ingestion/ingest_books.py --dry-run
    python src/ingestion/ingest_books.py --list
    python src/ingestion/ingest_books.py --id hf_nlp_course
    python src/ingestion/ingest_books.py --min-relevance 4

Author: DeepForge Engineering
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Make deepforge importable when running from project root ──────────────────
# If deepforge/ lives at C:\Users\Ezeking\PromptForge\deepforge\, and you run
# this script from C:\Users\Ezeking\PromptForge\, this sys.path insert is
# not needed (Python finds it). But if needed, uncomment:
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deepforge.core.catalogue.book_catalogue import (
    BOOK_CATALOGUE,
    BLOCKED_NC_BOOKS,
    get_all_safe,
    get_by_relevance,
    print_catalogue_summary,
)
from deepforge.core.ingestion.book_ingestor import BookIngestor
from deepforge.core.schemas import SYNTHFORGE_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_books")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "books_cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SynthForge Book Ingestion — ingests open-licensed books into corpus."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log all actions but download and write nothing.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print the full catalogue summary and exit.",
    )
    p.add_argument(
        "--id", type=str, default=None,
        help="Process only one source by its source_id (e.g. --id hf_nlp_course).",
    )
    p.add_argument(
        "--min-relevance", type=int, default=1,
        help="Only ingest entries with relevance_score >= this value (1-5). Default: 1.",
    )
    p.add_argument(
        "--safe-only", action="store_true",
        help="Only ingest entries where licence_verified=True (strictest mode).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── --list mode ───────────────────────────────────────────────
    if args.list:
        print_catalogue_summary()
        print(f"\n⛔ BLOCKED (NC / All Rights Reserved) — {len(BLOCKED_NC_BOOKS)} books:")
        for b in BLOCKED_NC_BOOKS:
            lic = b.get("licence_type")
            lic_str = lic.value if hasattr(lic, "value") else str(lic)
            print(f"   {b['source_id']:<50} {lic_str}")
        print("\nRun with --dry-run to simulate ingestion, or without flags to ingest.\n")
        return

    # ── Build the working catalogue ───────────────────────────────
    catalogue = BOOK_CATALOGUE

    if args.min_relevance > 1:
        catalogue = [b for b in catalogue if b.get("relevance_score", 0) >= args.min_relevance]
        logger.info("Filtered to relevance >= %d: %d entries", args.min_relevance, len(catalogue))

    if args.safe_only:
        catalogue = get_all_safe()
        logger.info("Safe-only mode: %d verified-safe entries", len(catalogue))

    if not catalogue:
        logger.error("No catalogue entries match the current filters. Exiting.")
        sys.exit(1)

    # ── Initialise ingestor ───────────────────────────────────────
    ingestor = BookIngestor(
        config=SYNTHFORGE_CONFIG,
        output_dir=DATA_DIR,
        cache_dir=CACHE_DIR,
    )

    # ── Run ───────────────────────────────────────────────────────
    progress = ingestor.run(
        source_specs=catalogue,
        dry_run=args.dry_run,
        single_id=args.id,
    )

    # ── Post-run summary ──────────────────────────────────────────
    if not args.dry_run:
        logger.info("=" * 60)
        logger.info("BOOK INGESTION COMPLETE")
        logger.info("  Completed:       %d", len(progress.completed))
        logger.info("  Failed:          %d", len(progress.failed))
        logger.info("  Licence-blocked: %d", len(progress.skipped_licence))
        logger.info("  Total chunks:    %d", progress.total_chunks_written)
        logger.info("")
        logger.info("NEXT: Run the standard corpus update sequence:")
        logger.info("  set HF_HUB_OFFLINE=1")
        logger.info("  python src/processing/chunk_and_embed.py")
        logger.info("  python deploy/compress_vectorstore.py")
        logger.info("  python deploy/build_bm25_cache.py")
        logger.info("  set HF_HUB_OFFLINE=0")
        logger.info("  python deploy/upload_vectorstore.py")

        if progress.failed:
            logger.warning("Failed sources (will retry on next run):")
            for fid in progress.failed:
                logger.warning("  - %s", fid)


if __name__ == "__main__":
    main()
