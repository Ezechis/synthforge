"""Phase 4 -- build the clean Cohort-B corpus in a FRESH store.

Reads legacy vectors from a COPY of the legacy store (data/vector_store_src_copy,
opened by ChromaDB; the original data/vector_store is never touched) and upserts
the migrated chunks into a brand-new ``synthforge`` collection at
data/vector_store_clean. Embeddings are REUSED verbatim -- no inference.

Resume-safe: chunk_id is deterministic, so re-running skips chunks already in the
destination. Vectors are fetched and written in batches; ChromaDB persists each
batch to sqlite as it is written.

Run:  py -m migration.phase4_apply  [--batch 100] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb

from migration._load import load_records
from migration._store import connect_ro
from migration.inspect_cohortb import is_cohort_b
from migration.mapping import build_chunk

REPO = Path(__file__).resolve().parents[1]
SRC_COPY = REPO / "data" / "vector_store_src_copy"
DEST = REPO / "data" / "vector_store_clean"
PLAN = Path(__file__).resolve().parent / "reports" / "dry_run_plan.json"
COLLECTION = "synthforge"
EXPECTED_DIM = 1024


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    plan = json.loads(PLAN.read_text())
    update = plan["update"]
    if args.limit:
        update = update[: args.limit]
    print(f"plan UPDATE entries: {len(update)}")

    # Legacy records (read-only) for rebuilding chunks deterministically.
    conn = connect_ro()
    recs = {r["_id"]: r for r in load_records(conn).values() if is_cohort_b(r)}
    conn.close()

    # Source vectors: open the COPY (never the original).
    src_client = chromadb.PersistentClient(path=str(SRC_COPY))
    src = src_client.get_collection(COLLECTION, embedding_function=None)

    # Destination: fresh store, cosine space, no embedding function (we supply
    # vectors explicitly so nothing is ever inferred).
    DEST.mkdir(parents=True, exist_ok=True)
    dest_client = chromadb.PersistentClient(path=str(DEST))
    dest = dest_client.get_or_create_collection(
        COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"destination existing count: {dest.count()}")

    written = skipped = 0
    for batch in _chunked(update, args.batch):
        chunk_ids = [e["chunk_id"] for e in batch]
        already = set(dest.get(ids=chunk_ids)["ids"])  # resume checkpoint

        todo = [e for e in batch if e["chunk_id"] not in already]
        skipped += len(batch) - len(todo)
        if not todo:
            continue

        # Fetch legacy vectors by legacy_id; map by returned id (get may reorder).
        legacy_ids = [e["legacy_id"] for e in todo]
        got = src.get(ids=legacy_ids, include=["embeddings"])
        vec_by_id = {i: v for i, v in zip(got["ids"], got["embeddings"])}

        ids, embs, docs, metas = [], [], [], []
        for e in todo:
            rec = recs[e["legacy_id"]]
            vec = vec_by_id.get(e["legacy_id"])
            if vec is None:
                raise RuntimeError(f"no legacy vector for {e['legacy_id']}")
            if len(vec) != EXPECTED_DIM:
                raise RuntimeError(f"bad dim {len(vec)} for {e['legacy_id']}")
            chunk = build_chunk(
                rec, chunk_index=e["chunk_index"], token_count=e["token_count"]
            )
            if chunk.chunk_id != e["chunk_id"]:
                raise RuntimeError(
                    f"nondeterministic chunk_id for {e['legacy_id']}: "
                    f"{chunk.chunk_id} != {e['chunk_id']}"
                )
            ids.append(chunk.chunk_id)
            embs.append([float(x) for x in vec])
            docs.append(chunk.text)
            metas.append(chunk.to_chroma_metadata())

        dest.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        written += len(ids)
        print(f"  upserted {written:5d} / {len(update)}  (skipped {skipped})")

    final = dest.count()
    print(f"\nDONE. written={written} skipped={skipped} dest.count={final}")


if __name__ == "__main__":
    main()
