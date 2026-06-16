"""DEEPCORE BaseSchema -- Layers 0-2 (primitives, ingestion, processing).

This module is the domain-agnostic data contract shared by every Forge on the
DeepForge platform (SynthForge, CodeForge, MathForge, ...). It defines the
"skeleton" that DEEPCORE moves through its retrieval-augmented-generation
pipeline, while leaving each Forge free to attach strictly-typed,
domain-specific metadata via a registry of ``BaseDomainMeta`` subclasses.

Scope of this file (Layers 0-2):
    L0  Primitives   shared enumerations (source, content, credibility,
                     license, domain) that tag every object.
    L1  Ingestion    ``RawDocument`` -- the validated output of any ingestor.
    L2  Processing   ``Chunk`` -- the workhorse stored in / retrieved from the
                     shared ChromaDB collection, carrying typed
                     ``domain_metadata``, a deterministic SHA-256 id, and
                     ChromaDB-safe (de)serialisation.

Layers 3-5 (retrieval ``ScoredChunk``, generation ``SynthesizedAnswer``,
operations ``IngestionRun``) are intentionally deferred. They compose over the
objects defined here -- ``ScoredChunk`` is expected to wrap a ``Chunk`` and
hold scores itself; ``Citation`` references a ``Chunk`` by id -- so they will
not require edits to this file when they land.

Design decisions (settled):
    * Closed base, open extension. Adding Forge #N never edits ``Chunk``; it
      registers a ``BaseDomainMeta`` subclass via ``@register_domain_meta``.
    * Per-Forge metadata is strongly typed and validated, not a loose dict. The
      ``DomainMetaRegistry`` keeps ``Chunk`` a single homogeneous type, so a
      mixed-domain ``list[Chunk]`` (cross-domain retrieval) just works.
    * Permissive now, tighten later. Domain metadata fields start optional and
      ``extra='allow'`` while each domain settles; tighten an individual
      subclass to ``extra='forbid'`` once its shape stabilises.
    * Fail closed on licensing. A non-commercial, proprietary, or unknown
      license raises ``LicenseViolationError`` at construction -- DeepForge
      charges users, so such content must never enter the corpus.
    * Resume-safe ids. Chunk ids are a deterministic SHA-256 over
      (document_id, chunk_index, text), so re-ingestion after a power loss is
      idempotent.

Requires Pydantic v2.
"""

from __future__ import annotations

import hashlib
import types
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Optional, Union, get_args, get_origin

try:
    import pydantic
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "DEEPCORE schemas require the 'pydantic' package (v2). "
        "Install it with: pip install 'pydantic>=2'"
    ) from exc

if int(pydantic.VERSION.split(".", 1)[0]) < 2:  # pragma: no cover
    raise ImportError(
        f"DEEPCORE schemas require Pydantic v2, found {pydantic.VERSION}. "
        "Pin a v2 release compatible with your installed DSPy version."
    )

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --------------------------------------------------------------------------- #
# Module metadata & named constants                                           #
# --------------------------------------------------------------------------- #

SCHEMA_VERSION: str = "0.1.0"
"""Bump on any breaking change to a stored shape; used to gate migrations."""

# Chunking defaults -- sentence-aware splitting target (project spec: 512/50).
DEFAULT_CHUNK_SIZE_TOKENS: int = 512
DEFAULT_CHUNK_OVERLAP_TOKENS: int = 50

# Deterministic id hashing.
_HASH_ALGORITHM: str = "sha256"
_HASH_ENCODING: str = "utf-8"

# ChromaDB metadata values must be str | int | float | bool (no lists/nesting).
# List-valued fields are flattened to a single delimited string for storage.
CHROMA_LIST_DELIMITER: str = " | "
# Domain-specific metadata is stored under this key prefix so it can be
# round-tripped back into the correct ``BaseDomainMeta`` subclass.
CHROMA_META_PREFIX: str = "meta_"

# Quality scoring (e.g. the Reddit 4-gate filter): chunks below the retained
# threshold are discarded upstream; the field is kept here for provenance.
MIN_RETAINED_QUALITY_SCORE: int = 3
MAX_QUALITY_SCORE: int = 5


__all__ = [
    "SCHEMA_VERSION",
    "SchemaError",
    "LicenseViolationError",
    "ChromaMetadataError",
    "UnknownForgeDomainError",
    "ChunkIntegrityError",
    "SourceType",
    "ContentType",
    "CredibilityTier",
    "License",
    "UncertaintyLevel",
    "ForgeDomain",
    "COMMERCIAL_USE_PROHIBITED_LICENSES",
    "RawDocument",
    "BaseDomainMeta",
    "DomainMetaRegistry",
    "register_domain_meta",
    "SynthForgeMeta",
    "Chunk",
]


# --------------------------------------------------------------------------- #
# Exceptions                                                                   #
# --------------------------------------------------------------------------- #

class SchemaError(Exception):
    """Base class for all DEEPCORE schema errors."""


class LicenseViolationError(SchemaError):
    """Raised when content carries a license that forbids commercial use.

    DeepForge charges users, so any non-commercial (CC *-NC-*), proprietary, or
    unknown license is rejected at construction time (fail closed).
    """


class ChromaMetadataError(SchemaError):
    """Raised when (de)serialising to/from ChromaDB-safe primitive metadata."""


class UnknownForgeDomainError(SchemaError):
    """Raised when a ``ForgeDomain`` has no registered metadata model."""


class ChunkIntegrityError(SchemaError):
    """Raised when a chunk's declared domain and metadata disagree."""


# --------------------------------------------------------------------------- #
# Layer 0 -- Primitive enumerations                                           #
# --------------------------------------------------------------------------- #

class SourceType(str, Enum):
    """Origin of a document. Extend as new ingestion sources are added."""

    YOUTUBE = "youtube"
    ARXIV = "arxiv"
    REDDIT = "reddit"
    GITHUB = "github"
    BOOK = "book"
    RSS = "rss"
    HACKERNEWS = "hackernews"
    STACKOVERFLOW = "stackoverflow"
    LESSWRONG = "lesswrong"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class ContentType(str, Enum):
    """The form of the content, independent of its source."""

    PAPER = "paper"
    CODE = "code"
    VIDEO_TRANSCRIPT = "video_transcript"
    FORUM_POST = "forum_post"
    DOCUMENTATION = "documentation"
    BOOK_EXCERPT = "book_excerpt"
    ARTICLE = "article"
    OTHER = "other"


class CredibilityTier(str, Enum):
    """Source hierarchy the generation layer reads to weight evidence.

    Tier 1 outranks Tier 2 outranks Tier 3 when sources conflict.
    """

    TIER_1_PRIMARY = "tier_1_primary"                # peer-reviewed / primary
    TIER_2_IMPLEMENTATION = "tier_2_implementation"  # official docs / code
    TIER_3_COMMUNITY = "tier_3_community"            # practitioner / forum


class License(str, Enum):
    """Content license. Commercial usability is derived, not stored.

    ``UNKNOWN`` is treated as commercially prohibited (fail closed): if a
    license cannot be proven to permit commercial use, the content must not be
    monetised.
    """

    PUBLIC_DOMAIN = "public_domain"
    CC0 = "cc0"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    CC_BY_ND = "cc_by_nd"
    CC_BY_NC = "cc_by_nc"
    CC_BY_NC_SA = "cc_by_nc_sa"
    CC_BY_NC_ND = "cc_by_nc_nd"
    MIT = "mit"
    APACHE_2_0 = "apache_2_0"
    BSD_3_CLAUSE = "bsd_3_clause"
    GPL_3_0 = "gpl_3_0"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"

    @property
    def allows_commercial_use(self) -> bool:
        """Whether this license permits the commercial use DeepForge requires."""
        return self not in COMMERCIAL_USE_PROHIBITED_LICENSES


# Licenses incompatible with a paid product. NC = non-commercial clause;
# PROPRIETARY and UNKNOWN are rejected by default (fail closed). ND (no
# derivatives) is *not* gated here -- chunking/synthesis raises a separate
# derivative-rights question to revisit per source.
COMMERCIAL_USE_PROHIBITED_LICENSES: "frozenset[License]" = frozenset(
    {
        License.CC_BY_NC,
        License.CC_BY_NC_SA,
        License.CC_BY_NC_ND,
        License.PROPRIETARY,
        License.UNKNOWN,
    }
)


class UncertaintyLevel(str, Enum):
    """Confidence tier for a synthesised claim. Reserved for the L4 contract."""

    WELL_ESTABLISHED = "well_established"  # replicated, broad consensus
    EMERGING = "emerging"                  # solid but context-dependent
    SPECULATIVE = "speculative"            # community claim, no strong evidence


class ForgeDomain(str, Enum):
    """Which Forge owns a record. Grows as domains are added to DeepForge."""

    PROMPT_ENGINEERING = "prompt_engineering"  # SynthForge (ground-zero Forge)
    CODE = "code"                              # CodeForge (planned)
    MATH = "math"                              # MathForge (planned)
    MACHINE_LEARNING = "machine_learning"      # MLForge (planned)


# --------------------------------------------------------------------------- #
# Layer 1 -- Ingestion contract                                               #
# --------------------------------------------------------------------------- #

class RawDocument(BaseModel):
    """A single source document, validated as it enters the pipeline.

    This is the output every ingestor must produce before chunking. The
    commercial-license gate is enforced here so prohibited content is rejected
    at the earliest possible point; ingestors should construct ``RawDocument``
    inside a ``try/except LicenseViolationError`` to skip-and-log gracefully.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(..., min_length=1)
    source_type: SourceType
    content_type: ContentType
    source_url: str = Field(..., min_length=1)
    title: str = ""
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    license: License
    credibility_tier: CredibilityTier
    domain: ForgeDomain
    language: str = "en"
    raw_content: str = Field(..., min_length=1)
    # Free-form source provenance (ChromaDB-safe primitives only). Permissive
    # by design -- kept loose at the document level.
    source_metadata: dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _enforce_commercial_license(self) -> "RawDocument":
        """Reject content whose license forbids commercial use (fail closed)."""
        if not self.license.allows_commercial_use:
            raise LicenseViolationError(
                f"License '{self.license.value}' forbids commercial use; "
                f"document '{self.document_id}' rejected from a paid corpus."
            )
        return self

    @staticmethod
    def compute_document_id(source_url: str) -> str:
        """Deterministic id from the source URL (idempotent re-ingestion)."""
        digest = hashlib.new(_HASH_ALGORITHM)
        digest.update(source_url.encode(_HASH_ENCODING))
        return digest.hexdigest()

    @classmethod
    def create(
        cls,
        *,
        source_type: SourceType,
        content_type: ContentType,
        source_url: str,
        raw_content: str,
        license: License,
        credibility_tier: CredibilityTier,
        domain: ForgeDomain,
        title: str = "",
        author: Optional[str] = None,
        published_at: Optional[datetime] = None,
        language: str = "en",
        source_metadata: Optional[dict[str, Union[str, int, float, bool]]] = None,
    ) -> "RawDocument":
        """Construct a document with a deterministic id derived from its URL."""
        return cls(
            document_id=cls.compute_document_id(source_url),
            source_type=source_type,
            content_type=content_type,
            source_url=source_url,
            raw_content=raw_content,
            license=license,
            credibility_tier=credibility_tier,
            domain=domain,
            title=title,
            author=author,
            published_at=published_at,
            language=language,
            source_metadata=source_metadata or {},
        )


# --------------------------------------------------------------------------- #
# Layer 2 -- Domain metadata (closed base, open extension via registry)       #
# --------------------------------------------------------------------------- #

class BaseDomainMeta(BaseModel):
    """Base class for per-Forge, strongly-typed chunk metadata.

    Each Forge subclasses this and registers it with ``@register_domain_meta``.
    ``extra='allow'`` keeps the shape permissive while a domain is still
    settling; tighten an individual subclass to ``extra='forbid'`` once its
    fields stabilise.
    """

    model_config = ConfigDict(extra="allow")

    forge: ForgeDomain


class DomainMetaRegistry:
    """Maps each ``ForgeDomain`` to its ``BaseDomainMeta`` subclass.

    This registry is the extension point that keeps ``Chunk`` a closed, single
    type: adding a Forge registers a metadata class here instead of editing
    ``Chunk`` or a hard-coded union.
    """

    _registry: ClassVar[dict[ForgeDomain, type[BaseDomainMeta]]] = {}

    @classmethod
    def register(cls, domain: ForgeDomain, meta_cls: type[BaseDomainMeta]) -> None:
        """Register ``meta_cls`` as the metadata model for ``domain``."""
        existing = cls._registry.get(domain)
        if existing is not None and existing is not meta_cls:
            raise SchemaError(
                f"ForgeDomain '{domain.value}' is already registered to "
                f"{existing.__name__}; refusing to rebind to {meta_cls.__name__}."
            )
        cls._registry[domain] = meta_cls

    @classmethod
    def get(cls, domain: ForgeDomain) -> type[BaseDomainMeta]:
        """Return the metadata model registered for ``domain``."""
        try:
            return cls._registry[domain]
        except KeyError as exc:
            raise UnknownForgeDomainError(
                f"No metadata model registered for ForgeDomain '{domain.value}'. "
                f"Register one with @register_domain_meta."
            ) from exc

    @classmethod
    def is_registered(cls, domain: ForgeDomain) -> bool:
        """Whether ``domain`` has a registered metadata model."""
        return domain in cls._registry


def register_domain_meta(domain: ForgeDomain):
    """Class decorator registering a ``BaseDomainMeta`` subclass for ``domain``."""

    def _decorator(meta_cls: type[BaseDomainMeta]) -> type[BaseDomainMeta]:
        if not issubclass(meta_cls, BaseDomainMeta):
            raise SchemaError(
                f"{meta_cls.__name__} must subclass BaseDomainMeta to be "
                f"registered as domain metadata."
            )
        DomainMetaRegistry.register(domain, meta_cls)
        return meta_cls

    return _decorator


@register_domain_meta(ForgeDomain.PROMPT_ENGINEERING)
class SynthForgeMeta(BaseDomainMeta):
    """Prompt-engineering metadata (SynthForge -- the ground-zero Forge).

    Permissive phase: every domain-specific field is optional so chunks can be
    populated partially while the prompt-engineering schema settles. Tighten to
    ``extra='forbid'`` and required fields as the domain stabilises.
    """

    forge: ForgeDomain = ForgeDomain.PROMPT_ENGINEERING
    technique_tags: list[str] = Field(default_factory=list)
    prompt_pattern: Optional[str] = None
    target_model_family: Optional[str] = None
    task_category: Optional[str] = None


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _is_list_field(annotation: Any) -> bool:
    """Whether a Pydantic field annotation is a (possibly optional) list type."""
    origin = get_origin(annotation)
    if origin is list:
        return True
    union_type = getattr(types, "UnionType", None)
    if origin is Union or (union_type is not None and origin is union_type):
        return any(_is_list_field(arg) for arg in get_args(annotation))
    return False


# --------------------------------------------------------------------------- #
# Layer 2 -- Processing contract                                              #
# --------------------------------------------------------------------------- #

class Chunk(BaseModel):
    """A retrievable unit of a document, stored in the shared vector store.

    A ``Chunk`` is immutable: its ``chunk_id`` is a deterministic hash of its
    content, so mutating it would invalidate its identity. Document-level fields
    (source, license, credibility, domain, provenance) are denormalised onto
    every chunk so the vector store can filter on them without a join.

    The embedding vector itself is NOT stored here -- ChromaDB owns it; only the
    embedding model name is recorded for provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    token_count: int = Field(..., ge=0)

    # Denormalised document context (for metadata filtering at retrieval time).
    source_type: SourceType
    content_type: ContentType
    credibility_tier: CredibilityTier
    license: License
    domain: ForgeDomain
    source_url: str = Field(..., min_length=1)
    title: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None

    # Generic, domain-agnostic enrichment.
    topic_tags: list[str] = Field(default_factory=list)
    quality_score: Optional[int] = Field(default=None, ge=0, le=MAX_QUALITY_SCORE)
    embedding_model: Optional[str] = None

    # Strongly-typed, per-Forge metadata (validated against the registry).
    domain_metadata: BaseDomainMeta

    @model_validator(mode="after")
    def _enforce_commercial_license(self) -> "Chunk":
        """Reject chunks whose license forbids commercial use (fail closed).

        Also surfaces prohibited content already in the store when a chunk is
        reconstructed via ``from_chroma_metadata`` -- such content fails loudly
        so it can be purged.
        """
        if not self.license.allows_commercial_use:
            raise LicenseViolationError(
                f"License '{self.license.value}' forbids commercial use; "
                f"chunk '{self.chunk_id}' rejected from a paid corpus."
            )
        return self

    @model_validator(mode="after")
    def _enforce_domain_metadata_integrity(self) -> "Chunk":
        """Ensure declared domain and attached metadata agree with the registry."""
        if self.domain_metadata.forge != self.domain:
            raise ChunkIntegrityError(
                f"Chunk domain '{self.domain.value}' disagrees with metadata "
                f"forge '{self.domain_metadata.forge.value}'."
            )
        expected_cls = DomainMetaRegistry.get(self.domain)
        if not isinstance(self.domain_metadata, expected_cls):
            raise ChunkIntegrityError(
                f"Chunk domain '{self.domain.value}' expects metadata type "
                f"{expected_cls.__name__}, got "
                f"{type(self.domain_metadata).__name__}."
            )
        return self

    @staticmethod
    def compute_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
        """Deterministic id over (document_id, chunk_index, text).

        Identical inputs always yield the same id, making re-ingestion after an
        interruption idempotent (resume-safety).
        """
        digest = hashlib.new(_HASH_ALGORITHM)
        digest.update(
            f"{document_id}:{chunk_index}:{text}".encode(_HASH_ENCODING)
        )
        return digest.hexdigest()

    @classmethod
    def from_document(
        cls,
        document: RawDocument,
        *,
        text: str,
        chunk_index: int,
        token_count: int,
        domain_metadata: BaseDomainMeta,
        topic_tags: Optional[list[str]] = None,
        quality_score: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> "Chunk":
        """Build a chunk, inheriting document context and computing its id."""
        return cls(
            chunk_id=cls.compute_chunk_id(document.document_id, chunk_index, text),
            document_id=document.document_id,
            chunk_index=chunk_index,
            text=text,
            token_count=token_count,
            source_type=document.source_type,
            content_type=document.content_type,
            credibility_tier=document.credibility_tier,
            license=document.license,
            domain=document.domain,
            source_url=document.source_url,
            title=document.title or None,
            author=document.author,
            published_at=document.published_at,
            topic_tags=topic_tags or [],
            quality_score=quality_score,
            embedding_model=embedding_model,
            domain_metadata=domain_metadata,
        )

    def to_chroma_metadata(self) -> dict[str, Union[str, int, float, bool]]:
        """Flatten to a ChromaDB-safe metadata dict (primitives only).

        The chunk ``text`` is NOT included -- store it in ChromaDB's
        ``documents`` and ``chunk_id`` in ``ids``. Lists become delimiter-joined
        strings; enums become their values; datetimes become ISO-8601 strings;
        ``None`` values are omitted (ChromaDB cannot store null).
        """
        metadata: dict[str, Union[str, int, float, bool]] = {
            "schema_version": SCHEMA_VERSION,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "source_type": self.source_type.value,
            "content_type": self.content_type.value,
            "credibility_tier": self.credibility_tier.value,
            "license": self.license.value,
            "domain": self.domain.value,
            "source_url": self.source_url,
        }
        if self.title is not None:
            metadata["title"] = self.title
        if self.author is not None:
            metadata["author"] = self.author
        if self.published_at is not None:
            metadata["published_at"] = self.published_at.isoformat()
        if self.topic_tags:
            metadata["topic_tags"] = CHROMA_LIST_DELIMITER.join(self.topic_tags)
        if self.quality_score is not None:
            metadata["quality_score"] = self.quality_score
        if self.embedding_model is not None:
            metadata["embedding_model"] = self.embedding_model

        metadata.update(self._flatten_domain_metadata())
        return metadata

    def _flatten_domain_metadata(self) -> dict[str, Union[str, int, float, bool]]:
        """Flatten domain metadata to prefixed, ChromaDB-safe primitives."""
        flat: dict[str, Union[str, int, float, bool]] = {}
        for name, value in self.domain_metadata.model_dump().items():
            if value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            key = f"{CHROMA_META_PREFIX}{name}"
            if isinstance(value, list):
                if not all(
                    isinstance(item, (str, int, float, bool)) for item in value
                ):
                    raise ChromaMetadataError(
                        f"Domain metadata field '{name}' contains non-primitive "
                        f"list items; cannot store in ChromaDB."
                    )
                flat[key] = CHROMA_LIST_DELIMITER.join(str(item) for item in value)
            elif isinstance(value, (str, int, float, bool)):
                flat[key] = value
            else:
                raise ChromaMetadataError(
                    f"Domain metadata field '{name}' has non-primitive type "
                    f"{type(value).__name__}; ChromaDB metadata must be flat."
                )
        return flat

    @classmethod
    def from_chroma_metadata(
        cls,
        *,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> "Chunk":
        """Rebuild a ``Chunk`` from its ChromaDB id, document text and metadata.

        Inverse of ``to_chroma_metadata``. Uses the stored ``domain`` to look up
        the correct metadata model in the registry and reconstruct typed
        ``domain_metadata``, splitting delimiter-joined list fields back into
        lists.
        """
        try:
            domain = ForgeDomain(metadata["domain"])
        except KeyError as exc:
            raise ChromaMetadataError(
                "ChromaDB metadata is missing required key 'domain'."
            ) from exc
        except ValueError as exc:
            raise ChromaMetadataError(
                f"Unrecognised ForgeDomain '{metadata.get('domain')}' in metadata."
            ) from exc

        meta_cls = DomainMetaRegistry.get(domain)
        domain_metadata = cls._rebuild_domain_metadata(meta_cls, metadata)

        published_at_raw = metadata.get("published_at")
        published_at = (
            datetime.fromisoformat(published_at_raw)
            if isinstance(published_at_raw, str)
            else None
        )
        topic_tags_raw = metadata.get("topic_tags")
        topic_tags = (
            topic_tags_raw.split(CHROMA_LIST_DELIMITER)
            if isinstance(topic_tags_raw, str) and topic_tags_raw
            else []
        )
        quality_score_raw = metadata.get("quality_score")
        quality_score = (
            int(quality_score_raw) if quality_score_raw is not None else None
        )

        try:
            return cls(
                chunk_id=chunk_id,
                document_id=metadata["document_id"],
                chunk_index=int(metadata["chunk_index"]),
                text=text,
                token_count=int(metadata["token_count"]),
                source_type=SourceType(metadata["source_type"]),
                content_type=ContentType(metadata["content_type"]),
                credibility_tier=CredibilityTier(metadata["credibility_tier"]),
                license=License(metadata["license"]),
                domain=domain,
                source_url=metadata["source_url"],
                title=metadata.get("title"),
                author=metadata.get("author"),
                published_at=published_at,
                topic_tags=topic_tags,
                quality_score=quality_score,
                embedding_model=metadata.get("embedding_model"),
                domain_metadata=domain_metadata,
            )
        except KeyError as exc:
            raise ChromaMetadataError(
                f"ChromaDB metadata is missing required key: {exc}."
            ) from exc

    @staticmethod
    def _rebuild_domain_metadata(
        meta_cls: type[BaseDomainMeta], metadata: dict[str, Any]
    ) -> BaseDomainMeta:
        """Reconstruct typed domain metadata from prefixed primitive keys."""
        fields: dict[str, Any] = {}
        for raw_key, value in metadata.items():
            if not raw_key.startswith(CHROMA_META_PREFIX):
                continue
            name = raw_key[len(CHROMA_META_PREFIX):]
            field = meta_cls.model_fields.get(name)
            if field is not None and _is_list_field(field.annotation):
                fields[name] = (
                    value.split(CHROMA_LIST_DELIMITER)
                    if isinstance(value, str) and value
                    else []
                )
            else:
                fields[name] = value
        try:
            return meta_cls(**fields)
        except (pydantic.ValidationError, ValueError, TypeError) as exc:
            raise ChromaMetadataError(
                f"Failed to rebuild {meta_cls.__name__} from metadata: {exc}."
            ) from exc


# --------------------------------------------------------------------------- #
# Self-test (run: python schemas.py)                                          #
# --------------------------------------------------------------------------- #

def _self_test() -> None:
    """Round-trip self-test: construct -> flatten -> reconstruct -> assert."""
    document = RawDocument.create(
        source_type=SourceType.ARXIV,
        content_type=ContentType.PAPER,
        source_url="https://arxiv.org/abs/2201.11903",
        raw_content="Chain-of-thought prompting elicits reasoning in LLMs.",
        license=License.CC_BY,
        credibility_tier=CredibilityTier.TIER_1_PRIMARY,
        domain=ForgeDomain.PROMPT_ENGINEERING,
        title="Chain-of-Thought Prompting",
        author="Wei et al.",
    )

    chunk = Chunk.from_document(
        document,
        text="Few-shot chain-of-thought outperforms zero-shot on multi-step tasks.",
        chunk_index=0,
        token_count=12,
        domain_metadata=SynthForgeMeta(
            technique_tags=["chain_of_thought", "few_shot"],
            prompt_pattern="reasoning",
            task_category="arithmetic",
        ),
        topic_tags=["reasoning", "prompting"],
        quality_score=5,
        embedding_model="bge-large-en-v1.5",
    )

    metadata = chunk.to_chroma_metadata()
    assert all(
        isinstance(v, (str, int, float, bool)) for v in metadata.values()
    ), "ChromaDB metadata must contain only primitives."

    restored = Chunk.from_chroma_metadata(
        chunk_id=chunk.chunk_id, text=chunk.text, metadata=metadata
    )
    assert restored == chunk, "Round-trip mismatch between chunk and restored."
    assert isinstance(restored.domain_metadata, SynthForgeMeta)
    assert restored.domain_metadata.technique_tags == [
        "chain_of_thought",
        "few_shot",
    ]

    # License gate must reject non-commercial content (fail closed).
    rejected = False
    try:
        RawDocument.create(
            source_type=SourceType.REDDIT,
            content_type=ContentType.FORUM_POST,
            source_url="https://reddit.com/r/example/comments/abc",
            raw_content="A non-commercially licensed post.",
            license=License.CC_BY_NC,
            credibility_tier=CredibilityTier.TIER_3_COMMUNITY,
            domain=ForgeDomain.PROMPT_ENGINEERING,
        )
    except LicenseViolationError:
        rejected = True
    assert rejected, "License gate failed to reject a CC-BY-NC document."

    # Unknown license is treated as commercially prohibited.
    assert not License.UNKNOWN.allows_commercial_use, "UNKNOWN must fail closed."

    # Deterministic ids: identical inputs reproduce the same id.
    assert chunk.chunk_id == Chunk.compute_chunk_id(
        document.document_id, 0, chunk.text
    ), "Chunk id is not deterministic."

    print("schemas.py self-test: PASS")
    print(f"  schema_version       = {SCHEMA_VERSION}")
    print(f"  registered domains   = "
          f"{[d.value for d in ForgeDomain if DomainMetaRegistry.is_registered(d)]}")
    print(f"  chunk_id             = {chunk.chunk_id[:16]}...")
    print(f"  chroma metadata keys = {sorted(metadata)}")


if __name__ == "__main__":
    _self_test()
