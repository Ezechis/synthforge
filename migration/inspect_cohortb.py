"""Inspect Cohort B (read-only): discriminator fields, value coverage, samples."""

from __future__ import annotations

from collections import Counter

from migration._load import load_records
from migration._store import connect_ro


def is_cohort_b(rec: dict) -> bool:
    """Per-source license fields present: license_type set & licence_verified True."""
    lt = rec.get("license_type")
    return rec.get("licence_verified") is True and bool(lt)


def main() -> None:
    conn = connect_ro()
    recs = load_records(conn)
    conn.close()

    cohort_b = [r for r in recs.values() if is_cohort_b(r)]
    print(f"total records          : {len(recs)}")
    print(f"cohort B (licence_verified=True & license_type set): {len(cohort_b)}")

    def dist(field: str) -> Counter:
        return Counter(str(r.get(field)) for r in cohort_b)

    for field in ("source", "content_type", "forge_id", "license_type",
                  "credibility_tier", "domain_tags", "year"):
        print(f"\n=== distinct {field} (Cohort B) ===")
        for v, c in dist(field).most_common(20):
            print(f"  {c:5d}  {v!r}")

    # Coverage of fields needed by the mapping.
    print("\n=== coverage / null checks (Cohort B) ===")
    def present(field, pred):
        return sum(1 for r in cohort_b if pred(r.get(field)))
    nonempty = lambda v: v is not None and v != ""
    isint = lambda v: isinstance(v, int) and not isinstance(v, bool)
    print(f"  source_url non-empty   : {present('source_url', nonempty)}")
    print(f"  source_url startswith http: "
          f"{sum(1 for r in cohort_b if str(r.get('source_url','')).startswith('http'))}")
    print(f"  source_id non-empty    : {present('source_id', nonempty)}")
    print(f"  page_start is int      : {present('page_start', isint)}")
    print(f"  page_start non-empty   : {present('page_start', nonempty)}")
    print(f"  chapter non-empty      : {present('chapter', nonempty)}")
    print(f"  section non-empty      : {present('section', nonempty)}")
    print(f"  _text non-empty        : {present('_text', nonempty)}")
    print(f"  title non-empty        : {present('title', nonempty)}")
    print(f"  authors non-empty      : {present('authors', nonempty)}")
    print(f"  relevance_score is int : {present('relevance_score', isint)}")
    print(f"  domain_tags non-empty  : {present('domain_tags', nonempty)}")

    # One full sample per content_type.
    print("\n=== one full sample per content_type ===")
    seen = set()
    for r in cohort_b:
        ct = r.get("content_type")
        if ct in seen:
            continue
        seen.add(ct)
        print(f"\n--- content_type={ct!r} (id={r['_id']}) ---")
        for k, v in sorted(r.items()):
            if k == "_text":
                v = (str(v)[:120] + "...") if v else v
            print(f"   {k}: {v!r}")


if __name__ == "__main__":
    main()
