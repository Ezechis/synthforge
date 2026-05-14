# Batch B patcher for chunk_and_embed.py
# No Windows paths in this docstring to avoid unicode escape errors.

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


# ── Patch 1+2: pdf_text in TEXT_KEYS + insert _CREDIBILITY_MAP ───────────────
patch(
    'TEXT_KEYS: tuple[str, ...] = (\n'
    '    "text", "content", "body", "abstract", "readme", "selftext", "comment_body"\n'
    ')',
    'TEXT_KEYS: tuple[str, ...] = (\n'
    '    "text", "content", "body", "abstract", "pdf_text", "readme", "selftext", "comment_body"\n'
    ')\n'
    '\n'
    '# Maps source identifier to credibility tier for the source hierarchy.\n'
    '_CREDIBILITY_MAP: dict[str, str] = {\n'
    '    "arxiv": "primary",\n'
    '    "github": "implementation",\n'
    '    "docs": "implementation",\n'
    '    "reddit": "community",\n'
    '}',
    "Patch 1+2 TEXT_KEYS + CREDIBILITY_MAP"
)

# ── Patch 3: Reddit per-post processing (replaces metadata collapse) ──────────
patch(
    '    if isinstance(raw, list):\n'
    '        # Reddit files: list of post dicts\n'
    '        if not raw:\n'
    '            return 0, 0\n'
    '        metadata_source: dict[str, Any] = raw[0] if isinstance(raw[0], dict) else {}\n'
    '        parts: list[str] = []\n'
    '        for item in raw:\n'
    '            if isinstance(item, dict):\n'
    '                t = extract_text_from_dict(item)\n'
    '                if t:\n'
    '                    parts.append(t)\n'
    '        text_content = "\\n\\n".join(parts)',
    '    if isinstance(raw, list):\n'
    '        # Reddit files: list of post dicts.\n'
    '        # Process each post individually so every chunk carries that\n'
    '        # post\'s own title, author, and URL (not just post #1\'s metadata).\n'
    '        if not raw:\n'
    '            return 0, 0\n'
    '        agg_stored = agg_skipped = 0\n'
    '        for item in raw:\n'
    '            if not isinstance(item, dict):\n'
    '                continue\n'
    '            item_text = extract_text_from_dict(item)\n'
    '            if not item_text:\n'
    '                continue\n'
    '            item_source: str = item.get("source", json_path.stem)\n'
    '            item_chunks = chunk_text(item_text)\n'
    '            if not item_chunks:\n'
    '                continue\n'
    '            item_ids = [\n'
    '                make_chunk_id(item_source, c, i) for i, c in enumerate(item_chunks)\n'
    '            ]\n'
    '            item_new_idx = [i for i, cid in enumerate(item_ids) if cid not in existing_ids]\n'
    '            agg_skipped += len(item_chunks) - len(item_new_idx)\n'
    '            if not item_new_idx:\n'
    '                continue\n'
    '            item_meta: dict[str, Any] = {\n'
    '                "source": item_source,\n'
    '                "source_type": item.get("source_type", item_source),\n'
    '                "credibility_tier": item.get(\n'
    '                    "credibility_tier", _CREDIBILITY_MAP.get(item_source, "community")\n'
    '                ),\n'
    '                "date": str(item.get("created_utc", item.get("date", ""))),\n'
    '                "author": str(item.get("author", "")),\n'
    '                "url": item.get("url", item.get("permalink", "")),\n'
    '                "title": item.get("title", json_path.stem),\n'
    '                "file": json_path.name,\n'
    '            }\n'
    '            item_new_chunks = [item_chunks[i] for i in item_new_idx]\n'
    '            item_new_ids = [item_ids[i] for i in item_new_idx]\n'
    '            item_new_metas = [\n'
    '                {**item_meta, "chunk_position": i, "chunk_total": len(item_chunks)}\n'
    '                for i in item_new_idx\n'
    '            ]\n'
    '            for start in range(0, len(item_new_chunks), BATCH_SIZE):\n'
    '                end = start + BATCH_SIZE\n'
    '                btexts = item_new_chunks[start:end]\n'
    '                bids = item_new_ids[start:end]\n'
    '                bmetas = item_new_metas[start:end]\n'
    '                try:\n'
    '                    embs = model.encode(\n'
    '                        btexts, normalize_embeddings=True, show_progress_bar=False\n'
    '                    ).tolist()\n'
    '                except Exception as exc:\n'
    '                    logger.error(\n'
    '                        "Embedding failed for Reddit batch in %s: %s",\n'
    '                        json_path.name, exc,\n'
    '                    )\n'
    '                    continue\n'
    '                if upsert_with_retry(collection, bids, embs, btexts, bmetas):\n'
    '                    agg_stored += len(btexts)\n'
    '                    time.sleep(0.1)\n'
    '        return agg_stored, agg_skipped',
    "Patch 3 Reddit per-post processing"
)

# ── Patch 4: base_meta credibility_tier for dict docs (arXiv/GitHub/docs) ────
patch(
    '        "source_type": metadata_source.get("source_type", "unknown"),\n'
    '        "credibility_tier": metadata_source.get("credibility_tier", "unknown"),',
    '        "source_type": metadata_source.get("source_type", source),\n'
    '        "credibility_tier": metadata_source.get("credibility_tier", _CREDIBILITY_MAP.get(source, "unknown")),',
    "Patch 4 base_meta credibility_tier"
)

print("\nBatch B patches applied to chunk_and_embed.py.")