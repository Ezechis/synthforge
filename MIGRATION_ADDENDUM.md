# MIGRATION ADDENDUM — corrections & authoritative schema facts

**Amends:** the Migration Strategy document (SynthForge legacy corpus → DEEPCORE
BaseSchema).
**Applies to:** Cowork (fold these into the strategy doc and the ADR) **and**
Claude Code (treat as binding inputs to Step 0 and execution).
**Source of the schema facts below:** the verified `schemas.py` itself.

---

## 1. CORRECTION — license recovery must be per-source, never an ownership claim (MANDATORY)

**Strike** the corpus-level license example in §2.2 of the strategy doc:
> ~~"all SynthForge content is originally authored/curated and licensed for the
> owner's own commercial use"~~

That framing is **invalid and must not be used.** The SynthForge corpus is
*ingested third-party content* — arXiv papers, Reddit posts, YouTube
transcripts, GitHub code, books — none of which the operator authored or owns.
"Curated it" is not "may license it commercially." Asserting a blanket
owner's-commercial-use license over third-party content is exactly the
fabrication the fail-closed gate exists to prevent, and it creates real legal
exposure for a paid product.

**Replace** the corpus-level recovery check with this:

> Corpus-level license recovery is legitimate **only** as documented evidence of
> the *per-source* license that ingestion filtered for. Valid evidence includes:
> the `BaseIngestor`'s allowed-license list, an ingestion config or manifest, or
> a per-source note in the SynthForge tree showing what each source was gated to
> (e.g. "arXiv ingested under CC-BY only," "books restricted to public-domain /
> CC," "GitHub under MIT/Apache/BSD"). The recovered license is then assigned
> **per source**, reflecting what was actually permitted at ingestion — not a
> single value asserted over the whole corpus on the basis of curation.
>
> If a given source has **no such per-source documentation** and its content's
> license cannot otherwise be established, that source's chunks are
> **unsubstantiated** → the gate drops them (per §7). This includes the realistic
> case where the pre-schema pipeline never recorded or enforced licensing at all.

The PASS / FAIL / PARTIAL verdict structure stays exactly as written — only the
*basis* for a PASS changes from "ownership" to "documented per-source filtering."

---

## 2. Authoritative schema facts (resolves the §2.4 deferral)

These come straight from `schemas.py`, so Step 0's field audit can be precise
rather than exploratory.

**`Chunk` required fields with NO default** (every one must be present or the
chunk fails construction):
`chunk_id`, `document_id`, `chunk_index`, `text`, `token_count`,
`source_type`, `content_type`, `credibility_tier`, `license`, `domain`,
`source_url`, `domain_metadata`.

**`Chunk` optional / defaulted fields** (no recovery needed):
`title`, `author`, `published_at`, `topic_tags` (default `[]`),
`quality_score` (default `None`), `embedding_model` (default `None`).

**`SynthForgeMeta`:** `forge` is defaulted; `technique_tags`, `prompt_pattern`,
`target_model_family`, `task_category` are all optional. An empty
`SynthForgeMeta()` validates — so the `meta_*` side is low-risk, as the strategy
doc guessed.

**`schema_version` is NOT a recovery concern.** It is not a `Chunk` constructor
field — `to_chroma_metadata()` stamps it automatically on every write. Remove it
from the "fields to recover" list entirely; it is always satisfied.

**Chunk id composition (confirms the checkpoint is valid):**
`chunk_id = sha256(f"{document_id}:{chunk_index}:{text}")`, where
`document_id = sha256(source_url)`. The id depends on **`source_url`,
`chunk_index`, and `text` only** — not on `license`, `domain`, or
`schema_version`. Therefore re-running the embed loop reproduces identical ids
for the same input, and "skip if id already in destination" is a sound
resume-after-power-cut mechanism.

---

## 3. CORRECTION — Branch B (re-embed) also requires `source_url` and `chunk_index`

Because the new id is computed from `source_url` + `chunk_index` + `text`, the
re-embed branch silently depends on both being recoverable from legacy metadata,
not just on the license and embedding-model findings. **Add to Step 0's §2.1
census:** confirm legacy chunks carry a recoverable `source_url` and a positional
index. If `chunk_index` is absent, derive it from per-document ordering (a
consistent, repeatable ordering) before the id-as-checkpoint logic is relied on.
If `source_url` is absent, that is a hard blocker for id computation and must be
flagged.

---

## 4. CORRECTION — `license` is not the only source-fact field needing recovery

The strategy doc frames license as the sole pivot and expects the other required
fields to be "empty or near-empty." That is slightly optimistic. Of the required
fields, these encode facts about the *source* and must be recovered or derived
(they cannot simply be stamped like `domain`):

- `source_url` — recover from legacy (also needed for the id; see §3).
- `source_type` — recover from a legacy `source` key, or derive.
- `content_type` — derive from source type (e.g. arXiv → `paper`, Reddit →
  `forum_post`, YouTube → `video_transcript`).
- `credibility_tier` — **needs an explicit derivation rule**, not a stamp: a
  source-type → tier mapping (e.g. peer-reviewed/arXiv → `TIER_1_PRIMARY`,
  official docs → `TIER_2_IMPLEMENTATION`, Reddit/forum → `TIER_3_COMMUNITY`).
- `token_count` — recompute from `text` with the tokenizer.

Most are recoverable or derivable, so they are not expected to force drops — but
Step 0's §2.4 audit should treat `source_type`, `content_type`, and
`credibility_tier` as derivation-rule work, not formalities, and the derivation
rules should be written down (they belong in the ADR / schema usage guide).

---

## 5. NOTE — count reconciliation must account for de-duplication

Because ids are deterministic, any legacy *duplicate* (same `source_url` +
`chunk_index` + `text`) collapses to a single record on write. So the final
corpus count may be **lower** than "starting count − |DROP list|" by the number
of duplicates. Reconcile the §A5/§B5 count assertions against the duplicate count
from the §2.5 baseline scan rather than treating any shortfall as data loss.

---

## Boundaries unchanged

Everything in the original briefs still holds: Cowork proposes and documents and
never edits code or the schema; Claude Code executes only under human sign-off,
never modifies `schemas.py`, never touches secrets, never pushes or deploys, and
runs migrations only against a copy. These corrections change *inputs*, not roles.
