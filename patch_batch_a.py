# Batch A patcher for PromptForge hf_space app.py
# Fixes: source_type->source, add author, lift 80-word cap, fix counters

import sys
from pathlib import Path

APP = Path(r"C:\Users\Ezeking\hf_space\app.py")

def patch(old, new, label):
    content = APP.read_text(encoding="utf-8")
    if content.count(old) == 0:
        print(f"ERROR [{label}]: string not found"); sys.exit(1)
    if content.count(old) > 1:
        print(f"ERROR [{label}]: found more than once"); sys.exit(1)
    APP.write_text(content.replace(old, new, 1), encoding="utf-8")
    print(f"OK    [{label}]")

# Fix 1 - source filter where_clause
patch(
    '            where_clause = {"source_type": {"$eq": source_filter[0]}}\n'
    '        else:\n'
    '            where_clause = {"source_type": {"$in": source_filter}}',
    '            where_clause = {"source": {"$eq": source_filter[0]}}\n'
    '        else:\n'
    '            where_clause = {"source": {"$in": source_filter}}',
    "Fix 1 source filter"
)

# Fix 2 - context label sent to LLM (source_type->source, add author, 3->6 chunks, 80->150 words)
patch(
    "    for chunk in retrieved_chunks[:3]:\n"
    "        meta = chunk[\"metadata\"]\n"
    "        source_label = (\n"
    "            f\"[{meta.get('source_type','unknown').upper()} | \"\n"
    "            f\"{meta.get('title','')[:60]} | \"\n"
    "            f\"credibility: {meta.get('credibility_tier','unknown')}]\"\n"
    "        )\n"
    "        block = f\"{source_label}\\n{chr(32).join(chunk['text'].split()[:300])}\"\n"
    "        block_words = len(block.split())\n"
    "        if word_count + block_words > MAX_CONTEXT_WORDS:\n"
    "            break\n"
    "        block=\" \".join(block.split()[:80]);context_parts.append(block)\n"
    "        word_count += block_words",
    "    for chunk in retrieved_chunks[:6]:\n"
    "        meta = chunk[\"metadata\"]\n"
    "        source_label = (\n"
    "            f\"[{meta.get('source','unknown').upper()} | \"\n"
    "            f\"Author: {meta.get('author','') or 'N/A'} | \"\n"
    "            f\"{meta.get('title','')[:60]} | \"\n"
    "            f\"credibility: {meta.get('credibility_tier','unknown')}]\"\n"
    "        )\n"
    "        block = f\"{source_label}\\n{chr(32).join(chunk['text'].split()[:300])}\"\n"
    "        block_words = len(block.split())\n"
    "        if word_count + block_words > MAX_CONTEXT_WORDS:\n"
    "            break\n"
    "        block=\" \".join(block.split()[:150]);context_parts.append(block)\n"
    "        word_count += block_words",
    "Fix 2 context label"
)

# Fix 3 - arXiv counter
patch(
    'arxiv_count = sum(1 for r in results if r["metadata"].get("source_type") == "arxiv")',
    'arxiv_count = sum(1 for r in results if r["metadata"].get("source") == "arxiv")',
    "Fix 3 arXiv counter"
)

# Fix 4 - source badge
patch(
    'source_type = meta.get("source_type", "unknown")',
    'source_type = meta.get("source", "unknown")',
    "Fix 4 source badge"
)

print("\nBatch A complete.")