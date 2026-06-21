"""Phase 5 -- validate the fresh clean store (still NO cutover).

Checks:
  1. exactly one 'synthforge' collection; count == expected.
  2. every record reconstructs via Chunk.from_chroma_metadata (license gate +
     integrity guard re-run on read).
  3. vectors present, dim 1024, L2-normalised.
  4. id set matches the Phase-3 plan; per-source counts reconcile (arithmetic).
  5. BM25 keyword spot-check (real text queries -- sparse path needs no model).
  6. dense spot-check via vector probe (use a known chunk's own vector as the
     query; the bge model can't load locally, so text->vector is validated on
     the Space, not here).
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from pathlib import Path

import chromadb

from deepforge.core.schemas import Chunk

REPO = Path(__file__).resolve().parents[1]
CLEAN_STORE = REPO / "data" / "vector_store_clean"
PLAN = Path(__file__).resolve().parent / "reports" / "dry_run_plan.json"
COLLECTION = "synthforge"
EXPECTED = 2247
DIM = 1024
SRC_COUNTS = {"book": 1814, "arxiv": 384, "doc": 49}
SRC_TYPE_COUNTS = {"book": 1814, "arxiv": 384, "documentation": 49}


def main() -> None:
    plan = json.loads(PLAN.read_text())
    plan_ids = {e["chunk_id"] for e in plan["update"]}

    client = chromadb.PersistentClient(path=str(CLEAN_STORE))
    names = [c.name for c in client.list_collections()]
    assert names == [COLLECTION], f"integrity: found {names}"
    coll = client.get_collection(COLLECTION, embedding_function=None)

    n = coll.count()
    print(f"[1] collection 'synthforge' count = {n}  (expected {EXPECTED})")
    assert n == EXPECTED, "count mismatch"

    data = coll.get(include=["documents", "metadatas", "embeddings"])
    ids, docs, metas, embs = (
        data["ids"], data["documents"], data["metadatas"], data["embeddings"]
    )

    # [2] reconstruct every chunk
    ok = 0
    src_counter = Counter()
    stype_counter = Counter()
    for cid, doc, meta in zip(ids, docs, metas):
        chunk = Chunk.from_chroma_metadata(chunk_id=cid, text=doc, metadata=meta)
        ok += 1
        stype_counter[chunk.source_type.value] += 1
    print(f"[2] reconstructed via from_chroma_metadata: {ok}/{n}")
    assert ok == n

    # [3] vectors
    bad_dim = sum(1 for v in embs if len(v) != DIM)
    norms = [math.sqrt(sum(x * x for x in v)) for v in embs]
    not_norm = sum(1 for nm in norms if abs(nm - 1.0) > 1e-3)
    print(f"[3] vectors: dim!={DIM}: {bad_dim} ; L2!=1.0 (tol 1e-3): {not_norm} ; "
          f"norm range [{min(norms):.4f}, {max(norms):.4f}]")
    assert bad_dim == 0 and not_norm == 0

    # [4] id-set + per-source reconciliation
    id_set = set(ids)
    print(f"[4] id set == plan: {id_set == plan_ids}  "
          f"(dest {len(id_set)} / plan {len(plan_ids)})")
    assert id_set == plan_ids
    print(f"    source_type counts: {dict(stype_counter)}")
    assert dict(stype_counter) == SRC_TYPE_COUNTS, "per-source mismatch"
    print(f"    reconciliation: {plan['cohort_b_count']} cohortB "
          f"- {plan['drop_count']} drop - {plan['collapsed']} collapse "
          f"= {plan['cohort_b_count'] - plan['drop_count'] - plan['collapsed']} "
          f"== dest {n}")

    # [5] BM25 keyword spot-check (real text queries; no model needed)
    cache = pickle.loads((CLEAN_STORE / "bm25_cache.pkl").read_bytes())
    bm25, bm_docs, bm_metas = (
        cache["bm25"], cache["corpus_chunks"], cache["corpus_metas"]
    )
    assert len(bm_docs) == n, "bm25 corpus size mismatch"
    print(f"\n[5] BM25 keyword spot-check ({len(bm_docs)} docs):")
    for q in ["chain of thought reasoning", "attention is all you need",
              "prompt engineering", "reinforcement learning from human feedback"]:
        scores = bm25.get_scores(q.lower().split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        print(f"  q={q!r}")
        for i in top:
            m = bm_metas[i]
            print(f"     score={scores[i]:5.1f} [{m.get('source_type')}] "
                  f"{str(m.get('title'))[:48]!r} :: {bm_docs[i][:60].strip()!r}")

    # [6] dense spot-check via vector probe
    print("\n[6] dense vector-probe spot-check (self + neighbours):")
    id_to_idx = {cid: k for k, cid in enumerate(ids)}
    probes = []
    for src in ("book", "arxiv", "doc"):
        for e in plan["update"]:
            if e["source"] == src:
                probes.append(e["chunk_id"]); break
    for cid in probes:
        k = id_to_idx[cid]
        res = coll.query(query_embeddings=[embs[k]], n_results=3,
                         include=["metadatas", "distances"])
        top_ids = res["ids"][0]
        m0 = res["metadatas"][0][0]
        self_hit = top_ids[0] == cid
        print(f"  probe [{m0.get('source_type')}] {str(m0.get('title'))[:40]!r}: "
              f"self_top={self_hit} dists={[round(d,3) for d in res['distances'][0]]}")
        assert self_hit, "probe did not return itself as top hit"

    print("\nPhase 5 validation: PASS  (no cutover performed)")


if __name__ == "__main__":
    main()
