"""
filter_rss_quality.py -- Remove noise posts from rss_feeds.json
Removes posts under 300 words and keyword-irrelevant content.
Saves filtered version and reports statistics.
"""

import json
from pathlib import Path

INPUT = Path("data/raw/docs/rss_feeds.json")

RELEVANCE_KEYWORDS = {
    "prompt", "llm", "language model", "gpt", "claude", "gemini",
    "chain of thought", "few-shot", "zero-shot", "instruction",
    "fine-tun", "rlhf", "alignment", "agent", "rag", "retrieval",
    "hallucin", "transformer", "embedding", "inference", "reasoning",
    "anthropic", "openai", "mistral", "llama", "chatgpt",
    "in-context", "context window", "benchmark", "neural", "attention",
    "deep learning", "machine learning", "dataset", "evaluation",
    "parameter", "token", "generation", "model", "training",
}

data = json.loads(INPUT.read_text(encoding="utf-8"))
print(f"Original posts: {len(data)}")

kept = []
removed_short = 0
removed_irrelevant = 0

for post in data:
    text = post.get("text", "")
    words = len(text.split())

    # Gate 1 — minimum length
    if words < 300:
        removed_short += 1
        continue

    # Gate 2 — keyword relevance
    haystack = (post.get("title", "") + " " + text[:1000]).lower()
    if not any(kw in haystack for kw in RELEVANCE_KEYWORDS):
        removed_irrelevant += 1
        continue

    kept.append(post)

print(f"Removed (too short):    {removed_short}")
print(f"Removed (irrelevant):   {removed_irrelevant}")
print(f"Kept:                   {len(kept)}")
print(f"Reduction:              {len(data)-len(kept)} posts ({(len(data)-len(kept))/len(data)*100:.1f}%)")

INPUT.write_text(
    json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Saved filtered RSS to {INPUT}")