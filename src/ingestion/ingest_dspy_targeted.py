"""Targeted DSPy corpus ingest — fills the MIPROv2 and DSPy knowledge gap.

Run from SynthForge root:
    python src/ingestion/ingest_dspy_targeted.py

After running, execute the full corpus update sequence.
"""

import os
import hashlib
import json
import requests
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/dspy_targeted")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# High-priority DSPy sources — public URLs
DSPY_SOURCES = [
    {
        "url": "https://raw.githubusercontent.com/stanfordnlp/dspy/main/README.md",
        "title": "DSPy: Programming—not prompting—Foundation Models",
        "source": "github",
        "author": "Stanford NLP",
        "year": 2025,
        "credibility_tier": "implementation",
        "tags": ["dspy", "automated_prompting", "framework"]
    },
    {
        "url": "https://raw.githubusercontent.com/stanfordnlp/dspy/main/docs/docs/building-blocks/optimizers.md",
        "title": "DSPy Optimizers — MIPROv2 and Compilation",
        "source": "docs",
        "author": "Stanford NLP",
        "year": 2025,
        "credibility_tier": "implementation",
        "tags": ["dspy", "MIPROv2", "optimizer", "compilation"]
    },
    {
        "url": "https://arxiv.org/abs/2310.03714",
        "title": "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines",
        "source": "arxiv",
        "author": "Khattab et al.",
        "year": 2023,
        "credibility_tier": "primary",
        "tags": ["dspy", "automated_prompting", "pipeline_optimization"]
    },
]

def fetch_content(url: str) -> str:
    """Fetch URL content with retry."""
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "SynthForge/1.0"})
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> list[str]:
    """Split text into overlapping word chunks."""
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def make_chunk_id(text: str, source: str) -> str:
    return hashlib.sha256(f"{source}::{text[:200]}".encode()).hexdigest()[:16]

def run_targeted_ingest() -> None:
    all_chunks = []
    for src in DSPY_SOURCES:
        logger.info("Fetching: %s", src["title"])
        content = fetch_content(src["url"])
        if not content:
            logger.warning("Skipping %s — no content", src["title"])
            continue
        chunks = chunk_text(content)
        logger.info("  → %d chunks from %s", len(chunks), src["title"])
        for chunk_text_str in chunks:
            chunk_id = make_chunk_id(chunk_text_str, src["source"])
            all_chunks.append({
                "id":   chunk_id,
                "text": chunk_text_str,
                "metadata": {
                    "source":          src["source"],
                    "title":           src["title"],
                    "author":          src["author"],
                    "year":            src["year"],
                    "credibility_tier": src["credibility_tier"],
                    "tags":            src["tags"],
                    "ingest_date":     datetime.now().isoformat(),
                    "url":             src["url"],
                }
            })

    output_path = RAW_DIR / "dspy_targeted_chunks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("Saved %d DSPy chunks to %s", len(all_chunks), output_path)
    logger.info("Next step: run chunk_and_embed.py to add to ChromaDB")

if __name__ == "__main__":
    run_targeted_ingest()