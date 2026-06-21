"""Phase-2 prerequisite checks (read-only): cardinality, ranges, tokenizer."""

from __future__ import annotations

from collections import Counter, defaultdict

from migration._load import load_records
from migration._store import connect_ro
from migration.inspect_cohortb import is_cohort_b


def main() -> None:
    conn = connect_ro()
    recs = load_records(conn)
    conn.close()
    b = [r for r in recs.values() if is_cohort_b(r)]

    # source_id <-> source_url cardinality (document_id = sha256(source_url),
    # chunk_index grouped by source_id -> they must agree, i.e. be 1:1).
    sid_to_urls = defaultdict(set)
    url_to_sids = defaultdict(set)
    for r in b:
        sid_to_urls[r["source_id"]].add(r["source_url"])
        url_to_sids[r["source_url"]].add(r["source_id"])
    sid_multi = {k: v for k, v in sid_to_urls.items() if len(v) > 1}
    url_multi = {k: v for k, v in url_to_sids.items() if len(v) > 1}
    print(f"distinct source_id : {len(sid_to_urls)}")
    print(f"distinct source_url: {len(url_to_sids)}")
    print(f"source_id -> >1 source_url : {len(sid_multi)}")
    for k, v in list(sid_multi.items())[:10]:
        print(f"    {k}: {sorted(v)}")
    print(f"source_url -> >1 source_id : {len(url_multi)}")
    for k, v in list(url_multi.items())[:10]:
        print(f"    {k}: {sorted(v)}")

    # license mappability
    LICENSE_MAP = {"cc_by_sa", "cc_by", "public_domain", "apache_2", "mit"}
    bad_lic = Counter(r["license_type"] for r in b
                      if r.get("license_type") not in LICENSE_MAP)
    print(f"\nunmappable license_type values: {dict(bad_lic)}")

    # quality / relevance_score range (Chunk.quality_score must be 0..5)
    rs = [r.get("relevance_score") for r in b]
    rs_int = [v for v in rs if isinstance(v, int) and not isinstance(v, bool)]
    print(f"relevance_score: n={len(rs_int)} min={min(rs_int)} max={max(rs_int)} "
          f"dist={dict(Counter(rs_int))}")
    out_of_range = [v for v in rs_int if v < 0 or v > 5]
    print(f"relevance_score out of [0,5]: {len(out_of_range)}")

    # bib-field emptiness (empty string -> None)
    print("\nbib field populated (non-empty, non-'Unknown'):")
    for f in ("isbn", "publisher", "edition", "doi", "page_start", "chapter"):
        pop = sum(1 for r in b
                  if str(r.get(f, "")).strip() not in ("", "Unknown", "None"))
        print(f"  {f:10s}: {pop}/{len(b)}")

    # chapter / section distinctness (tiebreak quality)
    print(f"\ndistinct chapter values: {len(set(str(r.get('chapter')) for r in b))}")
    print(f"distinct section values: {len(set(str(r.get('section')) for r in b))}")

    # bge tokenizer availability
    print("\n=== tokenizer availability ===")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        n = len(tok("hello world chain of thought")["input_ids"])
        print(f"  transformers AutoTokenizer OK (sample token len={n})")
    except Exception as exc:  # noqa: BLE001
        print(f"  transformers tokenizer unavailable: {type(exc).__name__}: {exc}")
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        print("  sentence_transformers importable")
    except Exception as exc:  # noqa: BLE001
        print(f"  sentence_transformers unavailable: {type(exc).__name__}")


if __name__ == "__main__":
    main()
