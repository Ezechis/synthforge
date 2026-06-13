"""
deepforge/core/schemas.py — DeepForge Base Schema (Rosetta Stone Architecture)

Universal metadata standard for ALL DeepForge corpora.
Every Knowledge Object across all 24 Forges inherits this base schema
plus optional Forge-specific extension profiles.

This is the foundational file that makes cross-Forge synthesis possible.
All ingestion scripts must produce chunks conforming to this schema.

Design decisions (settled):
- Dataclasses over TypedDict for IDE support and validation
- Optional fields default to None — never fail on missing data
- Forge-specific extensions added as separate dataclasses
- SHA-256 chunk_id as primary key across all Forges
- evidence_strength and confidence_label drive retrieval weighting
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class EvidenceStrength(str, Enum):
    """How strongly is this claim supported by evidence?"""
    STRONG = "Strong"
    MODERATE = "Moderate"
    WEAK = "Weak"
    ANECDOTAL = "Anecdotal"
    SPECULATIVE = "Speculative"


class ReproducibilityStatus(str, Enum):
    """Has this finding been independently reproduced?"""
    INDEPENDENTLY_REPRODUCED = "Independently Reproduced"
    PARTIALLY_REPRODUCED = "Partially Reproduced"
    NOT_REPRODUCED = "Not Reproduced"
    CONTRADICTED = "Contradicted"
    UNKNOWN = "Unknown"


class ContradictionStatus(str, Enum):
    """Does this claim conflict with other sources?"""
    CONSISTENT = "Consistent"
    DEBATED = "Debated"
    CONTRADICTED = "Contradicted"
    UNRESOLVED = "Unresolved"


class ConfidenceLabel(str, Enum):
    """Human-readable confidence label shown in SynthForge answers."""
    WELL_ESTABLISHED = "[WELL-ESTABLISHED]"
    PRODUCTION_PROVEN = "[PRODUCTION-PROVEN]"
    EMERGING = "[EMERGING]"
    LIMITED_EVIDENCE = "[LIMITED EVIDENCE]"
    SPECULATIVE = "[SPECULATIVE]"
    DEPRECATED = "[DEPRECATED]"
    CORPUS_GAP = "[CORPUS GAP]"


class KnowledgeHalfLife(str, Enum):
    """How quickly does this knowledge become stale?"""
    SHORT = "SHORT"    # < 6 months (e.g. specific model benchmarks)
    MEDIUM = "MEDIUM"  # 1-2 years (e.g. best practices)
    LONG = "LONG"      # 5+ years (e.g. foundational theory)


class KnowledgeEra(str, Enum):
    """Which era of AI development does this knowledge belong to?"""
    PRE_2024 = "Pre-2024"
    TRANSITION_2024_2026 = "2024-2026"
    CURRENT_2026_PLUS = "2026+"


class SourceTier(int, Enum):
    """Source credibility tier (used by generation system prompt)."""
    PEER_REVIEWED = 1      # arXiv papers, academic journals
    EMPIRICAL = 2           # GitHub implementations, official docs, books
    PRACTITIONER = 3        # Reddit, HN, Stack Overflow, YouTube


class Difficulty(str, Enum):
    """Content difficulty level."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ComputeTier(str, Enum):
    """Compute intensity for reproducing this technique."""
    C0 = "C0"  # CPU only, seconds
    C1 = "C1"  # CPU, minutes
    C2 = "C2"  # Single GPU, minutes
    C3 = "C3"  # Single GPU, hours
    C4 = "C4"  # Multi-GPU, hours
    C5 = "C5"  # Cluster, days


# ── Profile Dataclasses ───────────────────────────────────────────────────────

@dataclass
class EpistemicProfile:
    """How trustworthy and well-supported is this knowledge?

    This is the most important profile — it drives how SynthForge weights
    retrieved content during generation.
    """
    evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE
    reproducibility: ReproducibilityStatus = ReproducibilityStatus.UNKNOWN
    contradiction_status: ContradictionStatus = ContradictionStatus.CONSISTENT
    confidence_label: ConfidenceLabel = ConfidenceLabel.EMERGING
    negative_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to flat dict for ChromaDB metadata storage."""
        return {
            "epistemic_evidence_strength": self.evidence_strength.value,
            "epistemic_reproducibility": self.reproducibility.value,
            "epistemic_contradiction": self.contradiction_status.value,
            "epistemic_confidence": self.confidence_label.value,
        }


@dataclass
class TemporalProfile:
    """How fresh and time-sensitive is this knowledge?"""
    knowledge_half_life: KnowledgeHalfLife = KnowledgeHalfLife.MEDIUM
    era: KnowledgeEra = KnowledgeEra.TRANSITION_2024_2026
    last_verified: Optional[str] = None      # YYYY-MM-DD
    next_review_suggested: Optional[str] = None  # YYYY-MM-DD

    def to_dict(self) -> dict:
        return {
            "temporal_half_life": self.knowledge_half_life.value,
            "temporal_era": self.era.value,
            "temporal_last_verified": self.last_verified or "",
        }


@dataclass
class ResourceProfile:
    """What compute/data resources are needed to apply this knowledge?"""
    compute_tier: ComputeTier = ComputeTier.C1
    low_resource_friendly: bool = True
    energy_sensitivity: str = "Low"
    latency_profile: str = ""

    def to_dict(self) -> dict:
        return {
            "resource_compute_tier": self.compute_tier.value,
            "resource_low_resource_friendly": str(self.low_resource_friendly),
        }


@dataclass
class CrossForgeEdge:
    """A relationship between this chunk and content in another Forge."""
    target_forge: str           # e.g. "MLForge", "MathForge"
    relationship: str           # e.g. "DEPENDS_ON", "IMPLEMENTS", "EXTENDS"
    target_concept: str         # e.g. "Gradient Descent"
    description: str = ""


# ── Master Chunk Schema ───────────────────────────────────────────────────────

@dataclass
class DeepForgeChunk:
    """Universal Knowledge Object for all DeepForge corpora.

    This is the single source of truth for what a chunk looks like
    across all 24 Forges. Every ingestion pipeline must produce chunks
    conforming to this schema.

    The chunk_id is the SHA-256 hash of the text content — this makes
    ingestion resume-safe and deduplication trivial.

    Args:
        text: The actual text content of this chunk.
        source: Data source identifier (e.g. 'arxiv', 'reddit', 'github').
        forge: Which Forge this belongs to (e.g. 'SynthForge', 'MLForge').
        chunk_id: SHA-256 of text (auto-computed if not provided).
    """
    # Required fields
    text: str
    source: str
    forge: str = "SynthForge"

    # Auto-computed
    chunk_id: str = field(default="")

    # Taxonomy
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    category: str = ""
    topic_tags: list[str] = field(default_factory=list)

    # Source metadata
    title: str = ""
    author: str = ""
    url: str = ""
    date: str = ""
    credibility_tier: SourceTier = SourceTier.EMPIRICAL

    # Quality profiles
    epistemic: EpistemicProfile = field(default_factory=EpistemicProfile)
    temporal: TemporalProfile = field(default_factory=TemporalProfile)
    resource: ResourceProfile = field(default_factory=ResourceProfile)

    # Cross-forge relationships
    cross_forge_edges: list[CrossForgeEdge] = field(default_factory=list)

    # Forge-specific extension (JSON string for ChromaDB storage)
    forge_extension: str = ""

    def __post_init__(self) -> None:
        """Auto-compute chunk_id from text if not provided."""
        if not self.chunk_id:
            self.chunk_id = hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest()

    def to_chromadb_metadata(self) -> dict:
        """Serialise to flat dict suitable for ChromaDB metadata storage.

        ChromaDB metadata values must be str, int, float, or bool.
        Nested objects are flattened with prefix notation.

        Returns:
            Flat dict of metadata fields.
        """
        meta = {
            # Core
            "source": self.source,
            "forge": self.forge,
            "difficulty": self.difficulty.value,
            "category": self.category,
            "title": self.title,
            "author": self.author,
            "url": self.url,
            "date": self.date,
            "credibility_tier": int(self.credibility_tier),
            # Profiles
            **self.epistemic.to_dict(),
            **self.temporal.to_dict(),
            **self.resource.to_dict(),
            # Extension
            "forge_extension": self.forge_extension,
        }
        # Remove empty strings to keep metadata lean
        return {k: v for k, v in meta.items() if v != "" and v is not None}

    def to_jsonl_dict(self) -> dict:
        """Serialise to dict suitable for JSONL storage.

        Returns:
            Full dict including nested objects.
        """
        d = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "forge": self.forge,
            "difficulty": self.difficulty.value,
            "category": self.category,
            "topic_tags": self.topic_tags,
            "title": self.title,
            "author": self.author,
            "url": self.url,
            "date": self.date,
            "credibility_tier": int(self.credibility_tier),
            "epistemic": asdict(self.epistemic),
            "temporal": asdict(self.temporal),
            "resource": asdict(self.resource),
        }
        return d


# ── Forge-Specific Extension Profiles ────────────────────────────────────────

@dataclass
class SynthForgeExtension:
    """Extension profile for SynthForge (Prompt Engineering)."""
    token_efficiency: float = 0.0
    injection_risk: str = "Low"           # Low / Medium / High
    model_compatibility: list[str] = field(default_factory=list)
    delimiter_sensitivity: list[str] = field(default_factory=list)
    structured_output_support: bool = False
    prompt_pattern_type: str = ""         # e.g. chain-of-thought, few-shot


@dataclass
class MLForgeExtension:
    """Extension profile for MLForge (Machine Learning)."""
    compute_bracket: str = ""
    benchmark_references: list[str] = field(default_factory=list)
    hardware_mapping: list[str] = field(default_factory=list)
    training_vs_inference: str = "Both"   # Training / Inference / Both
    quantization_compatible: bool = True


@dataclass
class MathForgeExtension:
    """Extension profile for MathForge (Mathematics)."""
    axiomatic_dependencies: list[str] = field(default_factory=list)
    proof_complexity: str = "Medium"       # Low / Medium / High
    formal_verification_status: str = "Unverified"
    computational_complexity: str = ""    # Big-O notation


@dataclass
class CodeForgeExtension:
    """Extension profile for CodeForge (Software Engineering)."""
    language_support: list[str] = field(default_factory=list)
    framework_versions: list[str] = field(default_factory=list)
    test_coverage: float = 0.0
    reproducibility_score: float = 0.0


# ── Factory Functions ─────────────────────────────────────────────────────────

def chunk_from_arxiv(
    text: str,
    title: str,
    authors: str,
    arxiv_id: str,
    date: str,
    forge: str = "SynthForge",
) -> DeepForgeChunk:
    """Create a DeepForgeChunk from an arXiv paper.

    Args:
        text: Chunk text content.
        title: Paper title.
        authors: Comma-separated author names.
        arxiv_id: arXiv ID (e.g. '2305.12345').
        date: Publication date (YYYY-MM-DD).
        forge: Target Forge name.

    Returns:
        DeepForgeChunk with arXiv-appropriate metadata.
    """
    return DeepForgeChunk(
        text=text,
        source="arxiv",
        forge=forge,
        title=title,
        author=authors,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        date=date,
        credibility_tier=SourceTier.PEER_REVIEWED,
        epistemic=EpistemicProfile(
            evidence_strength=EvidenceStrength.STRONG,
            reproducibility=ReproducibilityStatus.UNKNOWN,
            confidence_label=ConfidenceLabel.EMERGING,
        ),
        temporal=TemporalProfile(
            knowledge_half_life=KnowledgeHalfLife.MEDIUM,
            era=_date_to_era(date),
        ),
    )


def chunk_from_reddit(
    text: str,
    subreddit: str,
    post_id: str,
    author: str,
    score: int,
    date: str,
    permalink: str,
    forge: str = "SynthForge",
) -> DeepForgeChunk:
    """Create a DeepForgeChunk from a Reddit post or comment.

    Args:
        text: Post or comment text.
        subreddit: Subreddit name without r/ prefix.
        post_id: Reddit post ID.
        author: Reddit username.
        score: Upvote score.
        date: Post date (YYYY-MM-DD).
        permalink: Full Reddit URL.
        forge: Target Forge name.

    Returns:
        DeepForgeChunk with Reddit-appropriate metadata.
    """
    return DeepForgeChunk(
        text=text,
        source="reddit",
        forge=forge,
        title=f"r/{subreddit} post {post_id}",
        author=author,
        url=permalink,
        date=date,
        credibility_tier=SourceTier.PRACTITIONER,
        epistemic=EpistemicProfile(
            evidence_strength=EvidenceStrength.ANECDOTAL,
            confidence_label=ConfidenceLabel.LIMITED_EVIDENCE,
        ),
        temporal=TemporalProfile(
            knowledge_half_life=KnowledgeHalfLife.SHORT,
            era=_date_to_era(date),
        ),
    )


def chunk_from_github(
    text: str,
    repo_name: str,
    file_path: str,
    url: str,
    date: str,
    forge: str = "SynthForge",
) -> DeepForgeChunk:
    """Create a DeepForgeChunk from a GitHub repository file.

    Args:
        text: File content chunk.
        repo_name: Repository name (e.g. 'owner/repo').
        file_path: Path to file within repository.
        url: Direct URL to file.
        date: Last commit date (YYYY-MM-DD).
        forge: Target Forge name.

    Returns:
        DeepForgeChunk with GitHub-appropriate metadata.
    """
    return DeepForgeChunk(
        text=text,
        source="github",
        forge=forge,
        title=f"{repo_name}: {file_path}",
        url=url,
        date=date,
        credibility_tier=SourceTier.EMPIRICAL,
        epistemic=EpistemicProfile(
            evidence_strength=EvidenceStrength.MODERATE,
            confidence_label=ConfidenceLabel.EMERGING,
        ),
        temporal=TemporalProfile(
            knowledge_half_life=KnowledgeHalfLife.MEDIUM,
            era=_date_to_era(date),
        ),
    )


def chunk_from_youtube(
    text: str,
    video_id: str,
    title: str,
    channel: str,
    date: str,
    forge: str = "SynthForge",
) -> DeepForgeChunk:
    """Create a DeepForgeChunk from a YouTube transcript.

    Args:
        text: Transcript chunk text.
        video_id: YouTube video ID.
        title: Video title.
        channel: Channel name.
        date: Upload date (YYYY-MM-DD).
        forge: Target Forge name.

    Returns:
        DeepForgeChunk with YouTube-appropriate metadata.
    """
    return DeepForgeChunk(
        text=text,
        source="youtube",
        forge=forge,
        title=title,
        author=channel,
        url=f"https://youtube.com/watch?v={video_id}",
        date=date,
        credibility_tier=SourceTier.PRACTITIONER,
        epistemic=EpistemicProfile(
            evidence_strength=EvidenceStrength.ANECDOTAL,
            confidence_label=ConfidenceLabel.LIMITED_EVIDENCE,
        ),
        temporal=TemporalProfile(
            knowledge_half_life=KnowledgeHalfLife.SHORT,
            era=_date_to_era(date),
        ),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _date_to_era(date_str: str) -> KnowledgeEra:
    """Convert a date string to a KnowledgeEra enum value.

    Args:
        date_str: Date in YYYY-MM-DD or YYYY format.

    Returns:
        Appropriate KnowledgeEra value.
    """
    try:
        year = int(date_str[:4])
        if year < 2024:
            return KnowledgeEra.PRE_2024
        elif year <= 2025:
            return KnowledgeEra.TRANSITION_2024_2026
        else:
            return KnowledgeEra.CURRENT_2026_PLUS
    except (ValueError, TypeError):
        return KnowledgeEra.TRANSITION_2024_2026


# ── Schema Validation ─────────────────────────────────────────────────────────

def validate_chunk(chunk: DeepForgeChunk) -> list[str]:
    """Validate a DeepForgeChunk and return list of validation errors.

    Args:
        chunk: Chunk to validate.

    Returns:
        List of error strings. Empty list means chunk is valid.
    """
    errors = []
    if not chunk.text or len(chunk.text.strip()) < 10:
        errors.append("text is empty or too short (< 10 chars)")
    if not chunk.chunk_id or len(chunk.chunk_id) != 64:
        errors.append("chunk_id must be 64-char SHA-256 hex string")
    if not chunk.source:
        errors.append("source is required")
    if not chunk.forge:
        errors.append("forge is required")
    return errors


if __name__ == "__main__":
    # Self-test
    print("Testing DeepForge Base Schema...")

    # Test chunk creation
    chunk = chunk_from_arxiv(
        text="Chain-of-thought prompting enables language models to perform complex reasoning.",
        title="Chain-of-Thought Prompting Elicits Reasoning in LLMs",
        authors="Wei, Jason et al.",
        arxiv_id="2201.11903",
        date="2022-01-28",
    )

    errors = validate_chunk(chunk)
    assert not errors, f"Validation failed: {errors}"
    print(f"  chunk_id: {chunk.chunk_id[:16]}...")
    print(f"  source: {chunk.source}")
    print(f"  credibility_tier: {chunk.credibility_tier}")
    print(f"  confidence: {chunk.epistemic.confidence_label.value}")

    meta = chunk.to_chromadb_metadata()
    print(f"  ChromaDB metadata keys: {list(meta.keys())}")
    assert "epistemic_confidence" in meta
    assert "source" in meta
    assert "credibility_tier" in meta

    # Test Reddit chunk
    reddit_chunk = chunk_from_reddit(
        text="I've been using structured XML tags for system prompts and it works much better.",
        subreddit="PromptEngineering",
        post_id="abc123",
        author="u/practitioner",
        score=150,
        date="2026-01-15",
        permalink="https://reddit.com/r/PromptEngineering/comments/abc123",
    )
    assert reddit_chunk.credibility_tier == SourceTier.PRACTITIONER
    print(f"  Reddit chunk tier: {reddit_chunk.credibility_tier}")

    print("\nAll schema tests PASSED.")
    print(f"\nSchema location: deepforge/core/schemas.py")
    print("Import with: from deepforge.core.schemas import DeepForgeChunk, chunk_from_arxiv")
