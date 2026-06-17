"""Load legacy records from the read-only ChromaDB sqlite into plain dicts.

Pivots the EAV ``embedding_metadata`` table into one dict per embedding. The
chunk text is stored under the ``#document`` key by ChromaDB; we expose it as
``_text``. The user-facing record id (``embedding_id``) is exposed as ``_id``
and the internal rowid as ``_rowid`` (needed later to fetch the stored vector).

Read-only only: this module never writes.
"""

from __future__ import annotations

from typing import Any

from migration._store import connect_ro

# Internal/reserved EAV keys we surface under friendlier names. ChromaDB 1.5.x
# stores chunk text under "chroma:document" (older builds used "#document").
_DOCUMENT_KEYS = ("chroma:document", "#document")


def _coalesce(row: Any) -> Any:
    """Return the single populated value for an EAV row (NULL-aware)."""
    if row["string_value"] is not None:
        return row["string_value"]
    if row["int_value"] is not None:
        return row["int_value"]
    if row["float_value"] is not None:
        return row["float_value"]
    if row["bool_value"] is not None:
        return bool(row["bool_value"])
    return None


def load_records(conn) -> dict[str, dict[str, Any]]:
    """Return {embedding_id: {meta..., _id, _rowid, _text}} for all records."""
    cur = conn.cursor()
    # rowid (embeddings.id) -> embedding_id (the user-facing chunk id string)
    records: dict[int, dict[str, Any]] = {}
    rowid_to_eid: dict[int, str] = {}
    for r in cur.execute("SELECT id, embedding_id FROM embeddings"):
        rowid_to_eid[r["id"]] = r["embedding_id"]
        records[r["id"]] = {"_rowid": r["id"], "_id": r["embedding_id"]}

    for r in cur.execute(
        "SELECT id, key, string_value, int_value, float_value, bool_value "
        "FROM embedding_metadata"
    ):
        rec = records.get(r["id"])
        if rec is None:
            continue
        value = _coalesce(r)
        key = r["key"]
        if key in _DOCUMENT_KEYS:
            rec["_text"] = value
        else:
            rec[key] = value

    return {rec["_id"]: rec for rec in records.values()}


if __name__ == "__main__":
    conn = connect_ro()
    recs = load_records(conn)
    conn.close()
    print(f"loaded {len(recs)} records")
    # distinct keys
    keys: dict[str, int] = {}
    for rec in recs.values():
        for k in rec:
            keys[k] = keys.get(k, 0) + 1
    print("\n=== key -> occurrence count (across all records) ===")
    for k, c in sorted(keys.items(), key=lambda kv: -kv[1]):
        print(f"  {c:6d}  {k}")
