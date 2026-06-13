"""
SynthForge - arXiv Ingestion Script
Layer 1: Fetches foundational prompt engineering papers,
extracts metadata and PDF text into structured JSON.

Usage: py src/ingestion/ingest_arxiv.py
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import arxiv
import fitz  # pymupdf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    ARXIV_SEARCH_TERMS,
    ARXIV_SEED_COUNT,
    DATA_RAW,
    LOG_DIR,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingest_arxiv.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = DATA_RAW / "arxiv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR: Path = OUTPUT_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)
RATE_LIMIT_PAUSE: float = 5.0   # arXiv requests courtesy pause (seconds)
RATE_LIMIT_429_WAIT: float = 60.0  # wait time when arXiv returns 429
MAX_PDF_PAGES: int = 30          # skip extraction beyond 30 pages


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF file using pymupdf.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text string, or empty string on failure.
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages_to_read = min(len(doc), MAX_PDF_PAGES)
        text = ""
        for page_num in range(pages_to_read):
            text += doc[page_num].get_text()
        doc.close()
        return text.strip()
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", pdf_path.name, exc)
        return ""


def download_pdf(paper: arxiv.Result) -> Path | None:
    """Download a paper's PDF to the local PDF directory.

    Args:
        paper: arxiv.Result object with download_pdf method.

    Returns:
        Path to downloaded PDF, or None on failure.
    """
    safe_id = paper.entry_id.split("/")[-1].replace("/", "_")
    pdf_path = PDF_DIR / f"{safe_id}.pdf"

    if pdf_path.exists():
        logger.info("PDF already exists, skipping download: %s", pdf_path.name)
        return pdf_path

    try:
        paper.download_pdf(dirpath=str(PDF_DIR), filename=pdf_path.name)
        logger.info("Downloaded PDF: %s", pdf_path.name)
        return pdf_path
    except Exception as exc:
        logger.warning("PDF download failed for %s: %s", paper.entry_id, exc)
        return None


def process_paper(paper: arxiv.Result) -> dict:
    """Extract all relevant content from a single arXiv paper.

    Args:
        paper: arxiv.Result object from arXiv API.

    Returns:
        Structured dict with paper metadata and extracted text.
    """
    safe_id = paper.entry_id.split("/")[-1].replace("/", "_")
    pdf_path = download_pdf(paper)
    pdf_text = extract_pdf_text(pdf_path) if pdf_path else ""

    return {
        "source": "arxiv",
        "paper_id": paper.entry_id,
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "abstract": paper.summary,
        "published": paper.published.isoformat(),
        "updated": paper.updated.isoformat(),
        "categories": paper.categories,
        "pdf_url": paper.pdf_url,
        "pdf_text": pdf_text,
        "ingested_at": datetime.utcnow().isoformat(),
    }


def run_ingestion() -> None:
    """Main ingestion function — searches arXiv across all seed terms.

    Deduplicates papers by entry_id. Resumes from existing saved files.
    Handles 429 and 503 rate limiting with automatic retry.
    """
    client = arxiv.Client()

    # Resume: load already-saved paper IDs from disk
    existing_files = set(f.stem for f in OUTPUT_DIR.glob("*.json"))
    seen_ids: set[str] = set(existing_files)
    processed = len(existing_files)
    target = ARXIV_SEED_COUNT
    logger.info("Resuming from %d already-saved papers.", processed)

    for search_term in ARXIV_SEARCH_TERMS:
        if processed >= target:
            break

        logger.info("Searching arXiv for: '%s'", search_term)
        search = arxiv.Search(
            query=search_term,
            max_results=50,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        retry_count = 0
        max_retries = 3

        while retry_count <= max_retries:
            try:
                for paper in client.results(search):
                    if processed >= target:
                        break

                    paper_id = paper.entry_id
                    if paper_id in seen_ids:
                        logger.info("Skipping duplicate: %s", paper.title[:60])
                        continue

                    seen_ids.add(paper_id)

                    try:
                        data = process_paper(paper)
                        safe_id = paper_id.split("/")[-1].replace("/", "_")
                        output_path = OUTPUT_DIR / f"{safe_id}.json"

                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                        logger.info(
                            "[%d/%d] Saved: %s",
                            processed + 1, target, paper.title[:60]
                        )
                        processed += 1
                        time.sleep(5.0)

                    except OSError as exc:
                        logger.error("Failed to write %s: %s", paper_id, exc)

                break  # successful iteration — exit retry loop

            except Exception as exc:
                retry_count += 1
                wait = 60 * retry_count
                logger.warning(
                    "arXiv error for '%s' (attempt %d/%d): %s. Waiting %ds.",
                    search_term, retry_count, max_retries, exc, wait
                )
                time.sleep(wait)

    logger.info("arXiv ingestion complete. Papers processed: %d", processed)


if __name__ == "__main__":
    run_ingestion()