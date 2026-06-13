"""
deepforge.core.ingestion.chunker
==================================
Shared text chunker used by ALL ForgeCore ingestion modules.

Parameters are intentionally matched to the existing SynthForge
chunk_and_embed.py pipeline:
  - 384-word target chunk size
  - 37-word overlap between adjacent chunks
  - SHA-256 deterministic chunk IDs
  - Minimum 80 words per chunk (discard stubs)

This module is the single source of truth for chunking behaviour
across all Forge products. Changing a parameter here propagates to
every ingestor automatically.

Author: DeepForge Engineering
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Chunking constants — MUST MATCH chunk_and_embed.py ────────────────────────
# These match the existing SynthForge pipeline. Do not change without also
# updating src/processing/chunk_and_embed.py.

CHUNK_WORD_TARGET: int = 384    # words per chunk (matches existing pipeline)
CHUNK_WORD_OVERLAP: int = 37    # overlap between adjacent chunks (matches pipeline)
MIN_CHUNK_WORDS: int = 80       # discard chunks shorter than this


@dataclass
class ChunkPosition:
    """Location metadata detected from a raw text block."""
    chapter: str = "Unknown"
    section: str = "General"
    page_or_section_num: int = 0


def detect_position(text: str, fallback_num: int = 0) -> ChunkPosition:
    """
    Heuristically detect chapter and section headings from the first
    few lines of a raw text block.

    Args:
        text:         The raw text block (may include a heading on line 1).
        fallback_num: Used as page_or_section_num if nothing is detected.

    Returns:
        ChunkPosition with chapter, section, and page_or_section_num.
    """
    lines = text.strip().split("\n")
    chapter = ""
    section = ""

    chapter_pattern = re.compile(r"^(chapter\s+[\dIVXivx]+|part\s+[\dIVXivx]+)", re.IGNORECASE)

    for line in lines[:6]:
        stripped = line.strip()
        if not stripped:
            continue
        if chapter_pattern.match(stripped):
            chapter = stripped
        elif (
            len(stripped) < 120               # short → likely a heading
            and not stripped.endswith(".")    # headings don't end with period
            and not stripped.startswith("[")  # not a footnote/citation
            and len(stripped.split()) > 1     # at least two words
        ):
            if not section:
                section = stripped

    return ChunkPosition(
        chapter=chapter or "Unknown",
        section=section or "General",
        page_or_section_num=fallback_num,
    )


def chunk_text(
    text: str,
    target_words: int = CHUNK_WORD_TARGET,
    overlap_words: int = CHUNK_WORD_OVERLAP,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[str]:
    """
    Split a text into overlapping word-count windows.

    This is the canonical chunking function shared across all ForgeCore
    ingestion modules. It produces the same chunk boundaries regardless
    of which ingestor calls it.

    Args:
        text:          Input text string.
        target_words:  Target words per chunk.
        overlap_words: Words shared between adjacent chunks.
        min_words:     Minimum words — chunks below this are discarded.

    Returns:
        List of text chunks. May be empty if text is too short.
    """
    words = text.split()
    total = len(words)

    if total < min_words:
        return []
    if total <= target_words:
        return [text]

    chunks: list[str] = []
    stride = target_words - overlap_words
    start = 0

    while start < total:
        end = min(start + target_words, total)
        window = words[start:end]
        if len(window) >= min_words:
            chunks.append(" ".join(window))
        start += stride
        if end == total:
            break

    return chunks


def chunk_raw_blocks(
    blocks: list[tuple[str, int]],
    target_words: int = CHUNK_WORD_TARGET,
    overlap_words: int = CHUNK_WORD_OVERLAP,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[tuple[str, ChunkPosition]]:
    """
    Convert raw text blocks (from any fetcher) into chunks with position metadata.

    Each block is split by chunk_text(). The ChunkPosition is detected once
    per block and applied to all child chunks from that block.

    Args:
        blocks:        Output of any fetcher: list of (text, section_num).
        target_words:  Passed to chunk_text().
        overlap_words: Passed to chunk_text().
        min_words:     Passed to chunk_text().

    Returns:
        List of (chunk_text, ChunkPosition) ready for CorpusChunk.create().
    """
    result: list[tuple[str, ChunkPosition]] = []

    for text_block, section_num in blocks:
        position = detect_position(text_block, fallback_num=section_num)
        raw_chunks = chunk_text(text_block, target_words, overlap_words, min_words)
        for chunk in raw_chunks:
            result.append((chunk, position))

    return result
