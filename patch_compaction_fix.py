# Patches chunk_and_embed.py main() to auto-recover from ChromaDB compaction errors.

import sys
from pathlib import Path

TARGET = Path(r"C:\Users\Ezeking\PromptForge\src\processing\chunk_and_embed.py")


def patch(old, new, label):
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(old)
    if count == 0:
        print(f"ERROR [{label}]: string not found"); sys.exit(1)
    if count > 1:
        print(f"ERROR [{label}]: found {count} times - ambiguous"); sys.exit(1)
    TARGET.write_text(content.replace(old, new, 1), encoding="utf-8")
    print(f"OK    [{label}]")


# Replace the bare process_json_file call with a retry loop that
# reinitialises the ChromaDB client on compaction errors.
patch(
    '            stored, skipped = process_json_file(json_path, model, collection, existing_ids)',
    '            for _attempt in range(5):\n'
    '                try:\n'
    '                    stored, skipped = process_json_file(\n'
    '                        json_path, model, collection, existing_ids\n'
    '                    )\n'
    '                    break\n'
    '                except Exception as exc:\n'
    '                    err = str(exc).lower()\n'
    '                    if "compaction" in err or "segments" in err:\n'
    '                        logger.warning(\n'
    '                            "ChromaDB compaction error on %s (attempt %d) "\n'
    '                            "-- reinitialising client in 10s.",\n'
    '                            json_path.name, _attempt + 1,\n'
    '                        )\n'
    '                        time.sleep(10)\n'
    '                        client = chromadb.PersistentClient(\n'
    '                            path=VECTOR_STORE_PATH,\n'
    '                            settings=Settings(anonymized_telemetry=False),\n'
    '                        )\n'
    '                        collection = client.get_or_create_collection(\n'
    '                            name=COLLECTION_NAME,\n'
    '                            metadata={"hnsw:space": "cosine"},\n'
    '                        )\n'
    '                        existing_ids = get_existing_ids(collection)\n'
    '                        stored, skipped = 0, 0\n'
    '                    else:\n'
    '                        logger.error(\n'
    '                            "Unhandled error on %s: %s", json_path.name, exc\n'
    '                        )\n'
    '                        stored, skipped = 0, 0\n'
    '                        break',
    "compaction auto-recovery in main()"
)

print("\nCompaction fix applied.")