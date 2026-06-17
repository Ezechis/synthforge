# Cohort-B → DEEPCORE migration tooling

Scripts that migrate the **2,247 license-verified Cohort-B records** from the
legacy `synthforge` ChromaDB collection into the canonical DEEPCORE `Chunk`
schema (`deepforge/core/schemas.py`, v0.2.0). Cohort A (19,274) and the 6
YouTube records are intentionally dropped — the license/identification gate
working as designed.

## Safety rails (enforced in code)
- **Read-only source.** `_store.connect_ro` opens the legacy sqlite with
  `mode=ro&immutable=1`; it physically cannot mutate the working store. No
  write helper exists in this package for the legacy store.
- **Reuse vectors.** Embeddings are never recomputed; Phase 4 reattaches the
  legacy stored vector by id. Only `token_count` is recomputed (bge tokenizer).
- **Two human gates.** The dry-run report (before any write) and the cutover
  (before anything touches the HF dataset). Phase 4+ does not run unprompted.

## Files
| File | Phase | Writes? | Purpose |
|---|---|---|---|
| `_store.py` | — | no | Read-only sqlite connection helper. |
| `_load.py` | — | no | Pivot the EAV metadata into per-record dicts. |
| `tokenizer.py` | — | no | bge-large-en-v1.5 token counts via cached `tokenizer.json` (avoids the broken sklearn DLL). |
| `inspect_cohortb.py` | 2 | no | Cohort-B identification + field coverage. |
| `check_prereqs.py` | 2 | no | Cardinality / license / range / tokenizer checks. |
| `mapping.py` | 3–5 | no | Legacy record → canonical `Chunk` (via `RawDocument.create`/`Chunk.from_document`). |
| `dry_run.py` | 3 | no | Validate every record by full chroma round-trip; emit UPDATE/DROP/dedup + `reports/dry_run_plan.json`. |
| `phase4_apply.py` | 4 | **fresh store only** | Reuse legacy vectors (from a copy) + upsert migrated chunks into `data/vector_store_clean`. Resume-safe. |
| `build_bm25.py` | 5 | fresh store only | Rebuild the BM25 cache for the 2,247-record corpus (bundled with the clean store). |
| `phase5_validate.py` | 5 | no | Count/round-trip/vector/id-set reconciliation + BM25 & dense retrieval spot-checks. |

## chunk_index derivation
Cohort B has no explicit index. Within each `source_id` group, records are
dense-ranked by `(page_start, chapter, section, text)`. Identical chunks (same
key incl. text) get the same index → identical `chunk_id` → they collapse on
upsert; distinct chunks get distinct indices. (`source_id`↔`source_url` is 1:1,
so this is consistent with `document_id = sha256(source_url)`.)

## Status
Phases 1–5 complete (dry run approved). The clean store lives at
`data/vector_store_clean/` (2,247 chunks + `bm25_cache.pkl`); the legacy store
and the HF dataset are untouched. **Stopped before the Phase-6 cutover** — the
only step that writes to production — which awaits explicit human approval.

Validation (Phase 5): count 2,247; all reconstruct via `from_chroma_metadata`;
all vectors dim 1024 & L2-normalised; id-set matches the plan; reconciliation
2247 − 0 drop − 0 collapse = 2247. The legacy/clean source-copy working dir
`data/vector_store_src_copy/` can be deleted after cutover.
