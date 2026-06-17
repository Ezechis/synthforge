"""Legacy Cohort-B record -> canonical DEEPCORE ``Chunk`` mapping.

Implements the field mapping from the Cohort-B migration brief. Records flow
through the sanctioned ingestion path (``RawDocument.create`` ->
``Chunk.from_document``) so document_id / chunk_id are computed by the frozen
contract, not by hand.

chunk_index is derived (Cohort B has no explicit index): within each source_id
group, dense-rank by (page_start, chapter, section, text). Identical chunks
(same key incl. text) get the same index -> identical chunk_id -> they collapse
on upsert. Distinct chunks get distinct indices.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from deepforge.core.schemas import (
    Chunk,
    ContentType,
    CredibilityTier,
    ForgeDomain,
    License,
    RawDocument,
    SourceType,
    SynthForgeMeta,
)

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
LEGACY_TAG_DELIMITER = "|"

# Discriminator is the legacy ``source`` field: book / arxiv / doc.
SOURCE_TO_SOURCE_TYPE = {
    "book": SourceType.BOOK,
    "arxiv": SourceType.ARXIV,
    "doc": SourceType.DOCUMENTATION,
}
SOURCE_TO_CONTENT_TYPE = {
    "book": ContentType.BOOK_EXCERPT,
    "arxiv": ContentType.PAPER,
    "doc": ContentType.DOCUMENTATION,
}
SOURCE_TO_CREDIBILITY = {
    "book": CredibilityTier.TIER_2_IMPLEMENTATION,
    "arxiv": CredibilityTier.TIER_1_PRIMARY,
    "doc": CredibilityTier.TIER_2_IMPLEMENTATION,
}
LICENSE_MAP = {
    "cc_by_sa": License.CC_BY_SA,
    "cc_by": License.CC_BY,
    "public_domain": License.PUBLIC_DOMAIN,
    "apache_2": License.APACHE_2_0,
    "mit": License.MIT,
}

# Legacy placeholders that mean "no value".
_EMPTY_SENTINELS = {"", "unknown", "none", "n/a", "null"}


class MappingError(Exception):
    """Raised when a legacy record cannot be mapped (-> DROP list)."""


def _clean(value: Any) -> Optional[str]:
    """Normalise a legacy string field; placeholders -> None."""
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in _EMPTY_SENTINELS:
        return None
    return s


def _clean_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def _sort_key(rec: dict) -> tuple:
    """Deterministic intra-source ordering: page_start, chapter, section, text."""
    page = _clean_int(rec.get("page_start"))
    return (
        page if page is not None else 1 << 30,  # missing pages sort last
        str(rec.get("chapter") or ""),
        str(rec.get("section") or ""),
        str(rec.get("_text") or ""),
    )


def derive_chunk_indices(records: list[dict]) -> dict[str, int]:
    """Map each record _id -> derived chunk_index (dense rank within source_id).

    Records sharing a source_id and an identical sort key (incl. text) receive
    the SAME index, so they will collapse to one chunk_id on write.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["source_id"]].append(r)

    index_by_id: dict[str, int] = {}
    for _sid, recs in groups.items():
        # Distinct sort keys, ordered -> dense rank gives the chunk_index.
        distinct_keys = sorted({_sort_key(r) for r in recs})
        rank = {key: i for i, key in enumerate(distinct_keys)}
        for r in recs:
            index_by_id[r["_id"]] = rank[_sort_key(r)]
    return index_by_id


def _technique_tags(rec: dict) -> list[str]:
    raw = rec.get("domain_tags")
    if not raw:
        return []
    return [t.strip() for t in str(raw).split(LEGACY_TAG_DELIMITER) if t.strip()]


def _published_at(rec: dict) -> Optional[datetime]:
    year = _clean_int(rec.get("year"))
    if year is None:
        return None
    return datetime(year, 1, 1, tzinfo=timezone.utc)


def build_chunk(rec: dict, *, chunk_index: int, token_count: int) -> Chunk:
    """Construct a canonical Chunk from a legacy Cohort-B record.

    Raises MappingError for records that cannot be mapped (caller -> DROP).
    """
    source = _clean(rec.get("source"))
    if source not in SOURCE_TO_SOURCE_TYPE:
        raise MappingError(f"unknown source sub-type {source!r}")

    license_type = _clean(rec.get("license_type"))
    license_ = LICENSE_MAP.get((license_type or "").lower())
    if license_ is None:
        raise MappingError(f"unmappable license_type {license_type!r}")

    source_url = _clean(rec.get("source_url"))
    if not source_url:
        raise MappingError("missing source_url")

    text = rec.get("_text")
    if not text:
        raise MappingError("missing chunk text")

    document = RawDocument.create(
        source_type=SOURCE_TO_SOURCE_TYPE[source],
        content_type=SOURCE_TO_CONTENT_TYPE[source],
        source_url=source_url,
        raw_content=text,
        license=license_,
        credibility_tier=SOURCE_TO_CREDIBILITY[source],
        domain=ForgeDomain.PROMPT_ENGINEERING,
        title=_clean(rec.get("title")) or "",
        author=_clean(rec.get("authors")),
        published_at=_published_at(rec),
    )

    meta = SynthForgeMeta(
        technique_tags=_technique_tags(rec),
        isbn=_clean(rec.get("isbn")),
        publisher=_clean(rec.get("publisher")),
        edition=_clean(rec.get("edition")),
        doi=_clean(rec.get("doi")),
        page_start=_clean_int(rec.get("page_start")),
        chapter=_clean(rec.get("chapter")),
    )

    return Chunk.from_document(
        document,
        text=text,
        chunk_index=chunk_index,
        token_count=token_count,
        domain_metadata=meta,
        topic_tags=[],
        quality_score=_clean_int(rec.get("relevance_score")),
        embedding_model=EMBEDDING_MODEL,
    )
