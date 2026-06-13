"""
deepforge.core.ingestion.fetchers
===================================
Source-aware fetchers for all ForgeCore book/document ingestion.

Each fetcher handles one acquisition method and returns raw text
as a list of (text_block, page_or_section_number) tuples — the
universal input format for the shared Chunker.

Supported fetch methods:
  - PDF (pdfplumber): local file or remote URL
  - HTML (requests + BeautifulSoup): single page
  - Project Gutenberg API: by Gutenberg ID
  - Standard Ebooks (standardebooks.org): CC0, by slug
  - Hugging Face Datasets: Institutional Books 1.0, by subset
  - Harvard Library Public Domain Corpus: via HF Hub

Author: DeepForge Engineering
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_RETRIES: int = 3
RETRY_DELAY: float = 5.0
REQUEST_TIMEOUT: int = 60
UA_HEADER: dict[str, str] = {
    "User-Agent": "DeepForge-BookIngestor/1.0 (open-source corpus builder; contact: deepforge@proton.me)"
}
MIN_BLOCK_WORDS: int = 40   # discard blocks shorter than this


# ── Raw text type alias ────────────────────────────────────────────────────────
# List of (text_block, section_index) — section_index is page number for PDFs,
# heading sequence number for HTML, paragraph index for API sources.
RawBlocks = list[tuple[str, int]]


# ── Shared HTTP helper ─────────────────────────────────────────────────────────

def _get_with_retry(url: str, source_id: str, stream: bool = False) -> Optional[requests.Response]:
    """
    GET request with exponential-backoff retries.

    Args:
        url:       Target URL.
        source_id: Used for log context only.
        stream:    If True, return streaming response (for large PDFs).

    Returns:
        requests.Response on success, None if all retries exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, headers=UA_HEADER, timeout=REQUEST_TIMEOUT, stream=stream
            )
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            logger.warning("[%s] HTTP %s on attempt %d/%d", source_id, exc, attempt, MAX_RETRIES)
        except requests.RequestException as exc:
            logger.warning("[%s] Network error attempt %d/%d: %s", source_id, exc, attempt, MAX_RETRIES)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)
    logger.error("[%s] All %d attempts failed for %s", source_id, MAX_RETRIES, url)
    return None


# ── PDF Fetcher ────────────────────────────────────────────────────────────────

def fetch_pdf(url: str, cache_path: Path, source_id: str) -> RawBlocks:
    """
    Download a PDF (if not cached) and extract text page-by-page.

    Args:
        url:        Direct PDF download URL.
        cache_path: Local path to save/load the PDF.
        source_id:  Used for logging.

    Returns:
        RawBlocks: list of (page_text, page_number).
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber required: pip install pdfplumber --break-system-packages") from exc

    # Download if not cached
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        response = _get_with_retry(url, source_id, stream=True)
        if not response:
            return []
        with open(cache_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=65_536):
                fh.write(chunk)
        logger.info("[%s] PDF cached at %s (%.1f MB)",
                    source_id, cache_path.name, cache_path.stat().st_size / 1_048_576)
    else:
        logger.info("[%s] PDF cache hit: %s", source_id, cache_path.name)

    # Extract text
    blocks: RawBlocks = []
    try:
        with pdfplumber.open(cache_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text.split()) >= MIN_BLOCK_WORDS:
                    blocks.append((text, page_num))
    except Exception as exc:
        logger.error("[%s] PDF extraction failed: %s", source_id, exc)
    logger.info("[%s] Extracted %d text blocks from PDF", source_id, len(blocks))
    return blocks


# ── HTML Fetcher ───────────────────────────────────────────────────────────────

def fetch_html(url: str, source_id: str) -> RawBlocks:
    """
    Fetch a single HTML page and split it into heading-bounded sections.

    Removes navigation, scripts, and footers. Splits at h1/h2/h3 headings.

    Args:
        url:       URL to fetch.
        source_id: Used for logging.

    Returns:
        RawBlocks: list of (section_text_with_heading, section_index).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("beautifulsoup4 required: pip install beautifulsoup4 --break-system-packages") from exc

    response = _get_with_retry(url, source_id)
    if not response:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    blocks: RawBlocks = []
    section_idx = 0
    current_heading = "Introduction"
    current_paragraphs: list[str] = []

    def flush() -> None:
        nonlocal section_idx, current_paragraphs
        joined = "\n".join(current_paragraphs).strip()
        if len(joined.split()) >= MIN_BLOCK_WORDS:
            blocks.append((f"{current_heading}\n\n{joined}", section_idx))
            section_idx += 1
        current_paragraphs = []

    for element in soup.find_all(["h1", "h2", "h3", "p", "li", "pre", "blockquote"]):
        if element.name in ("h1", "h2", "h3"):
            flush()
            current_heading = element.get_text(strip=True)
        else:
            text = element.get_text(strip=True)
            if text:
                current_paragraphs.append(text)

    flush()
    logger.info("[%s] Extracted %d sections from HTML at %s", source_id, len(blocks), url)
    return blocks


# ── Project Gutenberg Fetcher ──────────────────────────────────────────────────

GUTENBERG_TEXT_URL = "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
GUTENBERG_ALT_URL  = "https://www.gutenberg.org/files/{gid}/{gid}-0.txt"

def fetch_gutenberg(gutenberg_id: int, cache_dir: Path, source_id: str) -> RawBlocks:
    """
    Fetch a Project Gutenberg plain-text book by Gutenberg ID.

    All Project Gutenberg texts are public domain. The plain-text
    format contains a standard header/footer that is stripped.

    Args:
        gutenberg_id: Gutenberg numerical ID (e.g. 1342 for Pride and Prejudice).
        cache_dir:    Directory to cache the raw .txt file.
        source_id:    Used for logging.

    Returns:
        RawBlocks: list of (paragraph_block, block_index).
    """
    cache_path = cache_dir / f"gutenberg_{gutenberg_id}.txt"

    if not cache_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Try primary URL, fall back to alternate
        for url_template in (GUTENBERG_TEXT_URL, GUTENBERG_ALT_URL):
            url = url_template.format(gid=gutenberg_id)
            response = _get_with_retry(url, source_id)
            if response:
                cache_path.write_text(response.text, encoding="utf-8", errors="replace")
                logger.info("[%s] Gutenberg text cached: %s", source_id, cache_path.name)
                break
        else:
            logger.error("[%s] Failed to fetch Gutenberg ID %d", source_id, gutenberg_id)
            return []

    raw_text = cache_path.read_text(encoding="utf-8", errors="replace")

    # Strip standard Gutenberg header and footer
    start_markers = ["*** START OF", "***START OF", "* START OF"]
    end_markers   = ["*** END OF",   "***END OF",   "* END OF"]
    for marker in start_markers:
        pos = raw_text.find(marker)
        if pos != -1:
            raw_text = raw_text[raw_text.find("\n", pos) + 1:]
            break
    for marker in end_markers:
        pos = raw_text.find(marker)
        if pos != -1:
            raw_text = raw_text[:pos]
            break

    # Split into paragraph blocks (blank-line separated)
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    blocks: RawBlocks = []
    idx = 0
    # Batch paragraphs into ~300-word blocks to match chunker expectations
    batch: list[str] = []
    batch_words = 0
    BATCH_WORD_TARGET = 300

    for para in paragraphs:
        words = len(para.split())
        if batch_words + words > BATCH_WORD_TARGET and batch:
            joined = "\n\n".join(batch)
            if len(joined.split()) >= MIN_BLOCK_WORDS:
                blocks.append((joined, idx))
                idx += 1
            batch = []
            batch_words = 0
        batch.append(para)
        batch_words += words

    if batch:
        joined = "\n\n".join(batch)
        if len(joined.split()) >= MIN_BLOCK_WORDS:
            blocks.append((joined, idx))

    logger.info("[%s] Gutenberg %d → %d text blocks", source_id, gutenberg_id, len(blocks))
    return blocks


# ── Standard Ebooks Fetcher ────────────────────────────────────────────────────
# standardebooks.org — CC0, beautifully formatted public domain books.
# URL pattern: https://standardebooks.org/ebooks/{author}/{title}/text/single-page

STANDARD_EBOOKS_BASE = "https://standardebooks.org/ebooks/{slug}/text/single-page"

def fetch_standard_ebook(slug: str, source_id: str) -> RawBlocks:
    """
    Fetch a Standard Ebooks book as a single HTML page.

    Standard Ebooks are CC0 (public domain + CC0 typography/markup).
    The single-page view returns the entire book as one HTML document.

    Args:
        slug:      Standard Ebooks slug, e.g. "alan-turing/computing-machinery-and-intelligence"
        source_id: Used for logging.

    Returns:
        RawBlocks via fetch_html().
    """
    url = STANDARD_EBOOKS_BASE.format(slug=slug)
    logger.info("[%s] Fetching Standard Ebooks: %s", source_id, url)
    return fetch_html(url, source_id)


# ── Harvard / HF Institutional Books Fetcher ──────────────────────────────────
# Harvard Library Public Domain Corpus is available via HuggingFace Datasets:
# Dataset: "harvard-library/harvard-bibliographic-metadata"
# And the full text corpus: "harvard-library/harvard-library-public-domain-corpus"
# This is explicitly released for AI training and research.

def fetch_hf_dataset_book(
    dataset_repo: str,
    record_id: str,
    text_field: str,
    source_id: str,
    split: str = "train",
) -> RawBlocks:
    """
    Fetch a single book record from a Hugging Face Dataset.

    Used for: Harvard Library Public Domain Corpus, OpenStax via HF,
    and any other text dataset published on the Hub with permissive licences.

    Args:
        dataset_repo: HF dataset repo, e.g. "harvard-library/harvard-library-public-domain-corpus"
        record_id:    The record identifier to filter on.
        text_field:   The dataset column containing the book text.
        source_id:    Used for logging.
        split:        Dataset split to load (default: "train").

    Returns:
        RawBlocks: list of (text_block, block_index).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets required: pip install datasets --break-system-packages"
        ) from exc

    logger.info("[%s] Loading HF dataset %s | record %s", source_id, dataset_repo, record_id)
    try:
        ds = load_dataset(dataset_repo, split=split, trust_remote_code=True)
        # Filter to our record — assumes an 'id' or 'identifier' column
        records = [r for r in ds if str(r.get("id", r.get("identifier", ""))) == record_id]
        if not records:
            logger.warning("[%s] Record '%s' not found in dataset", source_id, record_id)
            return []

        raw_text = records[0].get(text_field, "")
        if not raw_text:
            logger.warning("[%s] Field '%s' is empty", source_id, text_field)
            return []

        # Split into blocks the same way as Gutenberg
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        blocks: RawBlocks = []
        batch: list[str] = []
        batch_words = 0
        BATCH_WORD_TARGET = 300
        idx = 0
        for para in paragraphs:
            words = len(para.split())
            if batch_words + words > BATCH_WORD_TARGET and batch:
                joined = "\n\n".join(batch)
                if len(joined.split()) >= MIN_BLOCK_WORDS:
                    blocks.append((joined, idx))
                    idx += 1
                batch = []
                batch_words = 0
            batch.append(para)
            batch_words += words
        if batch:
            joined = "\n\n".join(batch)
            if len(joined.split()) >= MIN_BLOCK_WORDS:
                blocks.append((joined, idx))

        logger.info("[%s] HF dataset → %d blocks", source_id, len(blocks))
        return blocks

    except Exception as exc:
        logger.error("[%s] HF dataset fetch failed: %s", source_id, exc)
        return []
