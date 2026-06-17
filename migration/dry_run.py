"""Phase 3 -- dry run (NO writes).

For every Cohort-B record: compute the proposed Chunk per the mapping, validate
by constructing it and round-tripping through to/from_chroma_metadata. Produces
an UPDATE list, a DROP list (with reasons), a dedup analysis (chunk_id
collisions), count reconciliation, and ~10 sample mappings. Writes a plan
artifact (migration/reports/dry_run_plan.json) for Phase 4 to consume.

Reads the legacy store strictly read-only. Writes nothing to any vector store.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from deepforge.core.schemas import Chunk
from migration._load import load_records
from migration._store import connect_ro
from migration.inspect_cohortb import is_cohort_b
from migration.mapping import build_chunk, derive_chunk_indices
from migration.tokenizer import count_tokens

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _roundtrip_ok(chunk: Chunk) -> None:
    """Raise if the chunk does not survive a chroma metadata round-trip."""
    md = chunk.to_chroma_metadata()
    if not all(isinstance(v, (str, int, float, bool)) for v in md.values()):
        raise ValueError("non-primitive in chroma metadata")
    restored = Chunk.from_chroma_metadata(
        chunk_id=chunk.chunk_id, text=chunk.text, metadata=md
    )
    if restored != chunk:
        raise ValueError("round-trip mismatch")


def main() -> None:
    conn = connect_ro()
    recs = load_records(conn)
    conn.close()
    cohort_b = [r for r in recs.values() if is_cohort_b(r)]
    print(f"Cohort B records: {len(cohort_b)}")

    index_by_id = derive_chunk_indices(cohort_b)

    update: list[dict] = []          # validated proposed chunks
    drop: list[dict] = []            # (id, reason)
    by_chunk_id: dict[str, list[dict]] = defaultdict(list)

    for r in cohort_b:
        rid = r["_id"]
        try:
            idx = index_by_id[rid]
            tc = count_tokens(r["_text"])
            chunk = build_chunk(r, chunk_index=idx, token_count=tc)
            _roundtrip_ok(chunk)
        except Exception as exc:  # noqa: BLE001 - dry run: route ANY failure to DROP
            drop.append({"legacy_id": rid, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        entry = {
            "legacy_id": rid,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "source": r.get("source"),
            "token_count": chunk.token_count,
        }
        update.append(entry)
        by_chunk_id[chunk.chunk_id].append(entry)

    # Dedup: records whose computed chunk_id collides (collapse on write).
    collisions = {cid: es for cid, es in by_chunk_id.items() if len(es) > 1}
    collapsed = sum(len(es) - 1 for es in collisions.values())
    distinct_chunks = len(by_chunk_id)

    print("\n================ DRY-RUN REPORT ================")
    print(f"Cohort B in            : {len(cohort_b)}")
    print(f"UPDATE (validated)     : {len(update)}")
    print(f"DROP (failed)          : {len(drop)}")
    print(f"  by reason            : "
          f"{dict(Counter(d['reason'].split(':')[0] for d in drop))}")
    print(f"chunk_id collisions    : {len(collisions)} groups, "
          f"{collapsed} records collapse")
    print(f"distinct chunks (final): {distinct_chunks}")
    print("\nReconciliation:")
    print(f"  {len(cohort_b)} in  - {len(drop)} drop - {collapsed} collapse "
          f"= {len(cohort_b) - len(drop) - collapsed}")
    print(f"  distinct chunk_ids   = {distinct_chunks}")
    assert len(cohort_b) - len(drop) - collapsed == distinct_chunks, "arithmetic!"

    print("\nUPDATE by source:")
    for s, c in Counter(e["source"] for e in update).most_common():
        print(f"  {s:6s}: {c}")

    if drop:
        print("\nDROP list (first 20):")
        for d in drop[:20]:
            print(f"  {d['legacy_id'][:16]}  {d['reason']}")

    if collisions:
        print("\nCollision examples (first 5 groups):")
        for cid, es in list(collisions.items())[:5]:
            print(f"  chunk_id {cid[:16]} <- {len(es)} records "
                  f"(idx={es[0]['chunk_index']}, src={es[0]['source']})")

    # ~10 full sample mappings across book/arxiv/doc.
    print("\n================ SAMPLE MAPPINGS (~10) ================")
    samples = _pick_samples(cohort_b, index_by_id, per_source=(("book", 4), ("arxiv", 3), ("doc", 3)))
    sample_dump = []
    for r in samples:
        chunk = build_chunk(r, chunk_index=index_by_id[r["_id"]],
                            token_count=count_tokens(r["_text"]))
        md = chunk.to_chroma_metadata()
        sample_dump.append({"legacy_id": r["_id"], "chroma_metadata": md,
                            "text_head": r["_text"][:160]})
        print(f"\n--- {r.get('source')} | legacy {r['_id'][:12]} ---")
        print(f"  source_id   : {r.get('source_id')}")
        print(f"  -> chunk_id  : {chunk.chunk_id}")
        print(f"  -> doc_id    : {chunk.document_id}")
        print(f"  -> idx       : {chunk.chunk_index}  token_count: {chunk.token_count}")
        print(f"  -> src/ctype : {chunk.source_type.value} / {chunk.content_type.value}")
        print(f"  -> cred/lic  : {chunk.credibility_tier.value} / {chunk.license.value}")
        print(f"  -> title     : {chunk.title}")
        print(f"  -> author    : {chunk.author}")
        print(f"  -> published : {chunk.published_at}")
        print(f"  -> quality   : {chunk.quality_score}")
        m = chunk.domain_metadata
        print(f"  -> tags      : {m.technique_tags}")
        print(f"  -> bib       : isbn={m.isbn} pub={m.publisher} ed={m.edition} "
              f"doi={m.doi} page={m.page_start} ch={m.chapter}")

    # Persist plan artifact for Phase 4.
    REPORTS_DIR.mkdir(exist_ok=True)
    plan = {
        "cohort_b_count": len(cohort_b),
        "update_count": len(update),
        "drop_count": len(drop),
        "collapsed": collapsed,
        "distinct_chunks": distinct_chunks,
        "update": update,
        "drop": drop,
        "collisions": {cid: [e["legacy_id"] for e in es]
                       for cid, es in collisions.items()},
    }
    (REPORTS_DIR / "dry_run_plan.json").write_text(json.dumps(plan, indent=2))
    (REPORTS_DIR / "dry_run_samples.json").write_text(json.dumps(sample_dump, indent=2))
    print(f"\nPlan written: {REPORTS_DIR / 'dry_run_plan.json'}")
    print("NO WRITES were made to any vector store. STOP for human approval.")


def _pick_samples(cohort_b, index_by_id, per_source):
    out = []
    for src, n in per_source:
        picks = [r for r in cohort_b if r.get("source") == src][:n]
        out.extend(picks)
    return out


if __name__ == "__main__":
    main()
