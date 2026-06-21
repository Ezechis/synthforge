# CLAUDE.md — DeepForge / DEEPCORE schema integration

## Your role
You are the **executor** for establishing `deepforge/core/schemas.py` (the DEEPCORE
BaseSchema, Layers 0–2) across the SynthForge codebase. You read files, run
code, edit the ingestion and retrieval code, and write and run tests. You do
the integration. You do **not** design or alter the contract.

## Authoritative artifact
`deepforge/core/schemas.py` is the **frozen contract**. It has already been
hand-verified: the self-test passes, and the license gate, the chunk integrity
guard, and the deterministic SHA-256 ids are all confirmed working. Everything
else in the repo adapts to it — never the reverse.

## What you DO
1. Place `schemas.py` at `deepforge/core/schemas.py`; make imports resolve repo-wide.
2. Run `py deepforge/core/schemas.py` and confirm it prints `self-test: PASS` in this
   environment before doing anything else.
3. Refactor every ingestor (YouTube, arXiv, Reddit, books, RSS) to:
   - build documents with `RawDocument.create(...)`, wrapped in
     `try/except LicenseViolationError` to skip-and-log non-commercial content;
   - build chunks with `Chunk.from_document(...)`;
   - write with
     `collection.add(ids=[c.chunk_id], documents=[c.text], metadatas=[c.to_chroma_metadata()])`.
4. Wire the retrieval read path to reconstruct chunks via
   `Chunk.from_chroma_metadata(...)`.
5. Write unit tests: round-trip, license gate (CC-BY-NC rejected, UNKNOWN
   rejected), integrity guard (domain ≠ metadata raises), deterministic id.
   Add one integration test against a **throwaway local** ChromaDB collection.
6. Execute the legacy-corpus migration **only per the approved strategy** (from
   the Cowork migration doc / the human's decision) and **only against a copy** —
   never production.
7. Run all tests, report results, and fix any failures **in the ingestors or
   adapters — never in `schemas.py`**.

## What you must NEVER do
- **Never modify `schemas.py`.** It is the contract every Forge and the entire
  corpus depend on. If you believe a change is required, STOP, write a short
  proposal stating what and why, and wait for human approval. Do not edit it
  speculatively, "improve" it, or refactor it.
- **Never add or change** schema fields, enums, the discriminator, or the
  registry mechanism. New `ForgeDomain`s or metadata models are deliberate human
  decisions, not your improvisation.
- **Never change** the ChromaDB version pin (`1.5.8`) or the collection name
  (`synthforge`). These are load-bearing — drift causes silent corpus
  corruption. Never recreate any deleted or ghost collection.
- **Never run a migration or test against the live/production ChromaDB or the
  HF Space vectorstore.** Work on a local copy only. Production migration is a
  human-gated step.
- **Never touch credentials or secrets** — GitHub Actions secrets, HF tokens,
  `.env`, `.key`/`.pem` files, or the HF Space settings.
- **Never push or deploy.** No `git push`, no force-push, no history rewrite, no
  Space deploy. Work on a branch and commit incrementally; the human reviews and
  pushes.
- **Never decide a strategic or architectural fork yourself.** If one arises
  (e.g. which migration approach to take), STOP and surface the options to the
  human.

## Model routing
Use **Sonnet** for the mechanical ingestor refactors. Escalate to **Opus / Fable**
only for reasoning through the migration strategy, and only if asked.

## Done when
All ingestors write via the schema; round-trip + gate + integrity tests pass
against a local ChromaDB; the approved migration has run against a copy;
`schemas.py` is byte-for-byte unchanged; and all work sits on a branch awaiting
human review.
