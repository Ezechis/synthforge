"""Download chroma.sqlite3 from HF and map each UUID folder to its collection name."""
import sqlite3
from huggingface_hub import hf_hub_download

TOKEN = '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'
REPO = 'ezechinnabugwu/synthforge-vectorstore'

print("[1/2] Downloading chroma.sqlite3 from HF...")
sqlite_path = hf_hub_download(
    repo_id=REPO,
    repo_type='dataset',
    filename='vector_store/chroma.sqlite3',
    token=TOKEN,
)
print(f"   downloaded to: {sqlite_path}")

print("\n[2/2] Reading collection -> UUID mapping...")
conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()

# ChromaDB stores collections in the 'collections' table with id (UUID) and name
cur.execute("SELECT id, name FROM collections")
rows = cur.fetchall()

print("\n=== Collections registered in HF chroma.sqlite3 ===")
for uuid, name in rows:
    print(f"  {name:20s}  ->  {uuid}")

# Also count records per collection via the embeddings table if present
print("\n=== Record counts per collection ===")
for uuid, name in rows:
    try:
        cur.execute("SELECT COUNT(*) FROM embeddings WHERE collection_id = ?", (uuid,))
        count = cur.fetchone()[0]
        print(f"  {name:20s}  {count:>10,} records")
    except sqlite3.OperationalError as e:
        print(f"  {name:20s}  (could not count: {e})")

conn.close()
