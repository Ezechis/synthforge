"""Read-only access helpers for the legacy SynthForge ChromaDB store.

Cohort-B migration tooling. Every connection here is opened with
``mode=ro&immutable=1`` so the working store at
``data/vector_store/chroma.sqlite3`` can NEVER be mutated by this code
(copy-first / reuse-vectors rail). Do not add write helpers to this module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SQLITE = REPO_ROOT / "data" / "vector_store" / "chroma.sqlite3"
LEGACY_VECTOR_DIR = REPO_ROOT / "data" / "vector_store"


def connect_ro(db_path: Path = LEGACY_SQLITE) -> sqlite3.Connection:
    """Open the legacy sqlite strictly read-only (cannot mutate the file)."""
    if not db_path.exists():
        raise FileNotFoundError(f"Legacy store not found: {db_path}")
    uri = f"file:{db_path.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == "__main__":
    conn = connect_ro()
    cur = conn.cursor()
    print("=== tables ===")
    for row in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ):
        print(" ", row["name"])
    print("\n=== collections ===")
    try:
        for row in cur.execute("SELECT * FROM collections"):
            print(" ", dict(row))
    except sqlite3.Error as exc:
        print("  (no collections table?)", exc)
    conn.close()
