"""
deepforge.core.ingestion.base_ingestor
=======================================
Abstract base class that every ForgeCore source ingestor must implement.

Design contract:
  - One subclass per source type: BookIngestor, ArxivIngestor, RedditIngestor, ...
  - Each subclass calls super().__init__(config) and implements ingest().
  - ingest() always returns List[CorpusChunk] — the universal output type.
  - The caller (ingest_books.py, ingest_arxiv.py, ...) handles ChromaDB writes.
  - Progress tracking (resume-safety) is handled here at the base level.
  - Commercial licence gating is enforced here — subclasses cannot bypass it.

Adding a new ForgeCore ingestor:
    from deepforge.core.ingestion.base_ingestor import BaseIngestor
    from deepforge.core.schemas import ForgeConfig, CorpusChunk

    class MyIngestor(BaseIngestor):
        def ingest(self, source_spec: dict) -> list[CorpusChunk]:
            ...

Author: DeepForge Engineering
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from deepforge.core.schemas import (
    CorpusChunk,
    ForgeConfig,
    LicenseType,
    validate_commercial_safety,
)

logger = logging.getLogger(__name__)

# ── Progress File Schema ───────────────────────────────────────────────────────

@dataclass
class IngestProgress:
    """
    Resume-safe progress record written after each source is processed.

    The progress file lives at: data/{ingestor_name}_progress.json
    On restart, completed source_ids are skipped automatically.
    """
    ingestor_name: str
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped_licence: list[str] = field(default_factory=list)
    total_chunks_written: int = 0

    def mark_complete(self, source_id: str, chunk_count: int) -> None:
        """Record a successful ingest."""
        if source_id not in self.completed:
            self.completed.append(source_id)
        self.total_chunks_written += chunk_count
        if source_id in self.failed:
            self.failed.remove(source_id)

    def mark_failed(self, source_id: str) -> None:
        """Record a failed ingest (will be retried on next run)."""
        if source_id not in self.failed:
            self.failed.append(source_id)

    def mark_licence_blocked(self, source_id: str) -> None:
        """Record a source blocked due to non-commercial licence."""
        if source_id not in self.skipped_licence:
            self.skipped_licence.append(source_id)

    def is_done(self, source_id: str) -> bool:
        """Return True if this source was already successfully ingested."""
        return source_id in self.completed

    def save(self, path: Path) -> None:
        """Persist progress to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "ingestor_name": self.ingestor_name,
                    "completed": self.completed,
                    "failed": self.failed,
                    "skipped_licence": self.skipped_licence,
                    "total_chunks_written": self.total_chunks_written,
                },
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path, ingestor_name: str) -> "IngestProgress":
        """Load progress from disk, or return a fresh record if file absent."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(**data)
        return cls(ingestor_name=ingestor_name)


# ── Base Ingestor ──────────────────────────────────────────────────────────────

class BaseIngestor(ABC):
    """
    Abstract base class for all ForgeCore source ingestors.

    Subclasses implement ingest_one(source_spec) — the per-source logic.
    This base class handles:
      - Progress tracking and resume-safety
      - Commercial licence enforcement
      - JSONL output writing
      - Polite rate limiting between sources
      - Consistent logging

    The output JSONL file is consumed by chunk_and_embed.py (with JSONL
    loader enabled) or read directly by the ChromaDB writer.

    Attributes:
        config:        ForgeConfig for the target Forge product.
        output_path:   Where the merged JSONL output is written.
        delay_seconds: Pause between successive source downloads (politeness).
    """

    # Subclasses set this to a short identifier: "book" | "arxiv" | "reddit"
    INGESTOR_NAME: str = "base"
    # Seconds to wait between requests (polite crawling)
    DEFAULT_DELAY: float = 1.0

    def __init__(
        self,
        config: ForgeConfig,
        output_dir: Path = Path("data"),
        delay_seconds: float | None = None,
    ) -> None:
        """
        Initialise the ingestor.

        Args:
            config:        ForgeConfig for the Forge product this ingestor serves.
            output_dir:    Root data directory (default: data/).
            delay_seconds: Override the per-request politeness delay.
        """
        self.config = config
        self.output_dir = output_dir
        self.delay_seconds = delay_seconds if delay_seconds is not None else self.DEFAULT_DELAY

        # Paths derived from ingestor name
        self.chunks_dir = output_dir / f"{self.INGESTOR_NAME}_chunks"
        self.progress_path = output_dir / f"{self.INGESTOR_NAME}_progress.json"
        self.output_path = output_dir / f"{self.INGESTOR_NAME}_all_chunks.jsonl"

        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.progress = IngestProgress.load(self.progress_path, self.INGESTOR_NAME)

        self._log = logging.getLogger(f"deepforge.{self.INGESTOR_NAME}")

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def ingest_one(self, source_spec: dict[str, Any]) -> list[CorpusChunk]:
        """
        Ingest a single source and return its chunks.

        Args:
            source_spec: Dict describing the source (varies by subclass).
                         For books: a dict matching BookMeta fields.
                         For arXiv: {"arxiv_id": "2201.11903", ...}

        Returns:
            List of CorpusChunk objects, ready for embedding.
            Returns empty list on failure — do not raise from here.
        """
        ...

    @abstractmethod
    def get_source_id(self, source_spec: dict[str, Any]) -> str:
        """Extract the unique source identifier from a source spec dict."""
        ...

    @abstractmethod
    def get_licence(self, source_spec: dict[str, Any]) -> LicenseType:
        """Extract the declared LicenseType from a source spec dict."""
        ...

    # ── Licence gate (shared, non-overridable) ────────────────────────────────

    def is_safe_to_ingest(self, source_spec: dict[str, Any]) -> bool:
        """
        Commercial licence gate. Called before every ingest_one() call.

        Returns False and logs a warning for any NC/unknown licence.
        This method is intentionally NOT overridable — safety is enforced
        at the base level regardless of subclass behaviour.
        """
        source_id = self.get_source_id(source_spec)
        licence = self.get_licence(source_spec)
        safe = validate_commercial_safety(licence, source_id)
        if not safe:
            self._log.warning(
                "[%s] LICENCE BLOCKED: '%s' is not commercially safe. "
                "SynthForge charges users — this source cannot be ingested.",
                source_id,
                licence.value,
            )
            self.progress.mark_licence_blocked(source_id)
        return safe

    # ── Chunk persistence (shared) ─────────────────────────────────────────────

    def save_chunks_for_source(
        self, source_id: str, chunks: list[CorpusChunk]
    ) -> Path:
        """
        Write chunks for one source to a per-source JSONL file.

        Per-source files allow re-running a single source without touching others.

        Args:
            source_id: Used as the filename stem.
            chunks:    CorpusChunk list to serialise.

        Returns:
            Path to the written JSONL file.
        """
        out_path = self.chunks_dir / f"{source_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.to_jsonl_record(), ensure_ascii=False) + "\n")
        self._log.info("[%s] Saved %d chunks → %s", source_id, len(chunks), out_path.name)
        return out_path

    def merge_all_chunks(self) -> int:
        """
        Merge all per-source JSONL files into one master output JSONL.

        The master file at self.output_path is what chunk_and_embed.py reads.
        Existing master is overwritten — always reflects current state.

        Returns:
            Total number of lines written.
        """
        total = 0
        with open(self.output_path, "w", encoding="utf-8") as out_fh:
            for jsonl_file in sorted(self.chunks_dir.glob("*.jsonl")):
                with open(jsonl_file, "r", encoding="utf-8") as in_fh:
                    for line in in_fh:
                        if line.strip():
                            out_fh.write(line)
                            total += 1
        self._log.info(
            "Merged %d total chunks → %s", total, self.output_path
        )
        return total

    # ── Batch run (shared orchestration) ──────────────────────────────────────

    def run(
        self,
        source_specs: list[dict[str, Any]],
        dry_run: bool = False,
        single_id: str | None = None,
    ) -> IngestProgress:
        """
        Orchestrate ingestion of a list of source specs.

        Handles resume-safety, licence gating, delay, progress save.
        Calls ingest_one() per source spec.

        Args:
            source_specs: List of source dicts (e.g. from book_catalogue).
            dry_run:      If True, log actions but write no files.
            single_id:    If set, process only the source with this ID.

        Returns:
            The final IngestProgress record.
        """
        if single_id:
            source_specs = [
                s for s in source_specs if self.get_source_id(s) == single_id
            ]
            if not source_specs:
                self._log.error("No source found with id '%s'", single_id)
                return self.progress

        total = len(source_specs)
        self._log.info(
            "Starting %s ingest run: %d sources | dry_run=%s",
            self.INGESTOR_NAME, total, dry_run,
        )

        for i, spec in enumerate(source_specs, start=1):
            source_id = self.get_source_id(spec)
            self._log.info("─" * 60)
            self._log.info("[%d/%d] %s", i, total, source_id)

            # ── Resume check ──────────────────────────────────────
            if self.progress.is_done(source_id):
                self._log.info("[%s] Already complete — skipping.", source_id)
                continue

            # ── Licence gate ──────────────────────────────────────
            if not self.is_safe_to_ingest(spec):
                self.progress.save(self.progress_path)
                continue

            # ── Dry run bypass ────────────────────────────────────
            if dry_run:
                self._log.info("[%s] DRY RUN — no fetch, no write.", source_id)
                continue

            # ── Ingest ────────────────────────────────────────────
            try:
                chunks = self.ingest_one(spec)
            except Exception as exc:
                self._log.error(
                    "[%s] Unexpected error in ingest_one: %s", source_id, exc, exc_info=True
                )
                self.progress.mark_failed(source_id)
                self.progress.save(self.progress_path)
                time.sleep(self.delay_seconds)
                continue

            if not chunks:
                self._log.warning("[%s] Produced 0 chunks — marking failed.", source_id)
                self.progress.mark_failed(source_id)
                self.progress.save(self.progress_path)
                time.sleep(self.delay_seconds)
                continue

            # ── Persist ───────────────────────────────────────────
            self.save_chunks_for_source(source_id, chunks)
            self.progress.mark_complete(source_id, len(chunks))
            self.progress.save(self.progress_path)

            time.sleep(self.delay_seconds)

        # ── Merge all per-source files into master JSONL ──────────
        if not dry_run:
            merged = self.merge_all_chunks()
            self._log.info("=" * 60)
            self._log.info("Run complete.")
            self._log.info(
                "  Completed: %d | Failed: %d | Licence-blocked: %d | Total chunks: %d",
                len(self.progress.completed),
                len(self.progress.failed),
                len(self.progress.skipped_licence),
                merged,
            )
            self._log.info(
                "NEXT STEP: python src/processing/chunk_and_embed.py "
                "(reads %s)", self.output_path
            )

        return self.progress

    # ── Utility: iterate chunks from existing JSONL ────────────────────────────

    def iter_existing_chunks(self) -> Iterator[dict[str, Any]]:
        """
        Yield chunk records from the master JSONL file if it exists.
        Useful for inspection or downstream consumers.
        """
        if not self.output_path.exists():
            return
        with open(self.output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)
