"""
deepforge.core.ingestion.book_ingestor
========================================
Concrete BookIngestor — the ForgeCore ingestor for books, textbooks,
technical documentation, and Gutenberg / Standard Ebooks sources.

Inherits all progress tracking, licence gating, JSONL output, and
orchestration from BaseIngestor. Adds book-specific:
  - Source-aware dispatch (PDF, HTML, Gutenberg, Standard Ebook, HF Dataset)
  - BookMeta construction from catalogue entries
  - CorpusChunk assembly via shared Chunker

Usage (from SynthForge entrypoint):
    from deepforge.core.ingestion.book_ingestor import BookIngestor
    from deepforge.core.catalogue.book_catalogue import BOOK_CATALOGUE
    from deepforge.core.schemas import SYNTHFORGE_CONFIG

    ingestor = BookIngestor(config=SYNTHFORGE_CONFIG)
    ingestor.run(BOOK_CATALOGUE)

Author: DeepForge Engineering
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

from deepforge.core.ingestion.base_ingestor import BaseIngestor
from deepforge.core.ingestion.chunker import chunk_raw_blocks
from deepforge.core.ingestion.fetchers import (
    RawBlocks,
    fetch_gutenberg,
    fetch_html,
    fetch_pdf,
    fetch_standard_ebook,
    fetch_hf_dataset_book,
)
from deepforge.core.schemas import (
    BookMeta,
    ContentType,
    CorpusChunk,
    CredibilityTier,
    DownloadType,
    ForgeConfig,
    LicenseType,
    SourceType,
    make_chunk_id,
    to_chroma_meta,
)

logger = logging.getLogger(__name__)


class BookIngestor(BaseIngestor):
    """
    ForgeCore ingestor for books and long-form technical documents.

    Handles five fetch strategies based on the entry's download_type:
      DownloadType.PDF          → fetch_pdf()
      DownloadType.HTML         → fetch_html()
      DownloadType.TEXT         → fetch_gutenberg()   (requires gutenberg_id)
      DownloadType.MARKDOWN     → fetch_standard_ebook() (requires slug)
      DownloadType.JSONL        → fetch_hf_dataset_book() (requires dataset_repo)

    All other logic (licence gating, progress, JSONL write) is in BaseIngestor.
    """

    INGESTOR_NAME: str = "book"
    DEFAULT_DELAY: float = 2.0   # polite delay between book downloads

    def __init__(
        self,
        config: ForgeConfig,
        output_dir: Path = Path("data"),
        cache_dir: Path = Path("data/books_cache"),
        delay_seconds: float | None = None,
    ) -> None:
        """
        Args:
            config:        ForgeConfig for the target Forge product.
            output_dir:    Root data dir (data/book_chunks/, data/book_progress.json).
            cache_dir:     Where raw PDFs and text files are cached.
            delay_seconds: Overrides DEFAULT_DELAY if provided.
        """
        super().__init__(config=config, output_dir=output_dir, delay_seconds=delay_seconds)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── BaseIngestor abstract interface ───────────────────────────────────────

    def get_source_id(self, source_spec: dict[str, Any]) -> str:
        return str(source_spec.get("source_id", "unknown"))

    def get_licence(self, source_spec: dict[str, Any]) -> LicenseType:
        raw = source_spec.get("licence_type", LicenseType.UNKNOWN)
        if isinstance(raw, LicenseType):
            return raw
        try:
            return LicenseType(str(raw))
        except ValueError:
            return LicenseType.UNKNOWN

    def ingest_one(self, source_spec: dict[str, Any]) -> list[CorpusChunk]:
        """
        Download, extract, chunk, and assemble CorpusChunks for one book entry.

        Args:
            source_spec: Dict from BOOK_CATALOGUE (must contain source_id,
                         download_type, and relevant URL / ID fields).

        Returns:
            List of CorpusChunk objects. Empty list on any failure.
        """
        source_id = self.get_source_id(source_spec)

        # ── Extra guard: unverified licence ───────────────────────
        if not source_spec.get("licence_verified", False):
            logger.warning(
                "[%s] licence_verified=False — open %s and confirm the licence "
                "before ingesting. Skipping.",
                source_id, source_spec.get("source_url", "the source page"),
            )
            return []

        # ── Build BookMeta ─────────────────────────────────────────
        book = self._build_meta(source_spec)

        # ── Fetch raw text blocks ──────────────────────────────────
        raw_blocks = self._fetch(source_spec, source_id)
        if not raw_blocks:
            logger.error("[%s] No raw text extracted.", source_id)
            return []

        # ── Chunk ──────────────────────────────────────────────────
        positioned_chunks = chunk_raw_blocks(raw_blocks)
        if not positioned_chunks:
            logger.error("[%s] Chunking produced 0 chunks.", source_id)
            return []

        # ── Assemble CorpusChunks ──────────────────────────────────
        corpus_chunks: list[CorpusChunk] = []
        ingested_at = datetime.datetime.utcnow().isoformat()

        for chunk_text, position in positioned_chunks:
            chunk_id = make_chunk_id(
                source_id,
                position.chapter,
                position.section,
                chunk_text[:200],
            )

            # Clone meta with per-chunk fields
            meta = to_chroma_meta({
                # ── ChunkMeta base fields ──────────────────────────
                "chunk_id": chunk_id,
                "forge_id": self.config.forge_id,
                "source": book.source,
                "source_id": source_id,
                "year": book.year,
                "ingested_at": ingested_at,
                "credibility_tier": book.credibility_tier,
                "license_type": book.licence_type,
                "domain_tags": book.domain_tags,
                "relevance_score": book.relevance_score,
                # ── BookMeta fields ────────────────────────────────
                "title": book.title,
                "authors": book.authors,
                "publisher": book.publisher,
                "edition": book.edition,
                "isbn": book.isbn,
                "doi": book.doi,
                "content_type": book.content_type,
                "source_url": book.source_url,
                "licence_verified": book.licence_verified,
                # ── Chunk-level position ───────────────────────────
                "chapter": position.chapter,
                "section": position.section,
                "page_start": position.page_or_section_num,
            })

            corpus_chunks.append(CorpusChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                meta=meta,
            ))

        logger.info(
            "[%s] ✅ %d chunks assembled (%d raw blocks → %d positioned → %d chunks)",
            source_id,
            len(corpus_chunks),
            len(raw_blocks),
            len(positioned_chunks),
            len(corpus_chunks),
        )
        return corpus_chunks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_meta(self, spec: dict[str, Any]) -> "BookMetaView":
        """
        Extract all BookMeta-relevant fields from a catalogue entry dict
        into a lightweight named view. Avoids mutating the catalogue.
        """
        return BookMetaView(
            source_id=str(spec.get("source_id", "")),
            title=str(spec.get("title", "")),
            authors=str(spec.get("authors", "")),
            year=int(spec.get("year", 0)),
            publisher=str(spec.get("publisher", "")),
            edition=str(spec.get("edition", "")),
            isbn=str(spec.get("isbn", "")),
            doi=str(spec.get("doi", "")),
            licence_type=self.get_licence(spec),
            licence_verified=bool(spec.get("licence_verified", False)),
            source_url=str(spec.get("source_url", "")),
            content_type=spec.get("content_type", ContentType.TEXTBOOK),
            domain_tags=str(spec.get("domain_tags", "")),
            relevance_score=int(spec.get("relevance_score", 3)),
            credibility_tier=spec.get("credibility_tier", CredibilityTier.TIER_2),
            source=spec.get("source", SourceType.BOOK),
        )

    def _fetch(self, spec: dict[str, Any], source_id: str) -> RawBlocks:
        """
        Dispatch to the correct fetcher based on download_type.

        Args:
            spec:      Catalogue entry dict.
            source_id: Used for cache paths and logging.

        Returns:
            RawBlocks from the appropriate fetcher.
        """
        dl_type = spec.get("download_type", DownloadType.HTML)
        if isinstance(dl_type, str):
            try:
                dl_type = DownloadType(dl_type)
            except ValueError:
                dl_type = DownloadType.HTML

        if dl_type == DownloadType.PDF:
            cache_path = self.cache_dir / f"{source_id}.pdf"
            return fetch_pdf(
                url=str(spec.get("download_url", "")),
                cache_path=cache_path,
                source_id=source_id,
            )

        elif dl_type == DownloadType.HTML or dl_type == DownloadType.MARKDOWN:
            # Standard Ebooks has its own fetcher
            if "slug" in spec:
                return fetch_standard_ebook(
                    slug=str(spec["slug"]),
                    source_id=source_id,
                )
            return fetch_html(
                url=str(spec.get("download_url", spec.get("source_url", ""))),
                source_id=source_id,
            )

        elif dl_type == DownloadType.TEXT:
            # Project Gutenberg
            gutenberg_id = spec.get("gutenberg_id")
            if gutenberg_id is None:
                logger.error("[%s] download_type=TEXT but no gutenberg_id provided", source_id)
                return []
            return fetch_gutenberg(
                gutenberg_id=int(gutenberg_id),
                cache_dir=self.cache_dir,
                source_id=source_id,
            )

        elif dl_type == DownloadType.JSONL:
            # HF Datasets
            return fetch_hf_dataset_book(
                dataset_repo=str(spec.get("dataset_repo", "")),
                record_id=str(spec.get("record_id", "")),
                text_field=str(spec.get("text_field", "text")),
                source_id=source_id,
            )

        else:
            logger.error("[%s] Unsupported download_type: %s", source_id, dl_type)
            return []


# ── Lightweight meta view (avoids dataclass mutation) ─────────────────────────

class BookMetaView:
    """Simple container holding BookMeta-relevant fields from a catalogue entry."""
    __slots__ = (
        "source_id", "title", "authors", "year", "publisher",
        "edition", "isbn", "doi", "licence_type", "licence_verified",
        "source_url", "content_type", "domain_tags", "relevance_score",
        "credibility_tier", "source",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
