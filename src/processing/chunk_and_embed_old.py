"""
SynthForge - Chunking and Embedding Pipeline
Layer 2: Reads raw JSON from all sources, splits into 512-token chunks
with 50-token overlap, preserves code blocks, tags metadata, embeds
with bge-large-en-v1.5, and stores in ChromaDB.

Usage: py src/processing/chunk_and_embed.py
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Generator

import chromadb
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DATA_RAW,
    DATA_PROCESSED,
    VECTOR_STORE_DIR,
    LOG_DIR,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "chunk_and_embed.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BATCH_SIZE: int = 32          # chunks per embedding batch
MIN_CHUNK_CHARS: int = 100    # discard chunks shorter than this
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def estimate_tokens(text: str) -> int:
    """Estimate token count using word-based approximation.

    Args:
        text: Input text string.

    Returns:
        Estimated token count (1 token ~ 0.75 words).
    """
    return int(len(text.split()) / 0.75)


def extract_code_blocks(text: str) -> tuple[str, list[str]]:
    """Extract code blocks from text, replacing with placeholders.

    Code blocks are never split mid-block per architecture decision.

    Args:
        text: Raw text possibly containing markdown code blocks.

    Returns:
        Tuple of (text with placeholders, list of extracted code blocks).
    """
    code_blocks: list[str] = []

    def replace_block(match: re.Match) -> str:
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    cleaned_text = CODE_BLOCK_PATTERN.sub(replace_block, text)
    return cleaned_text, code_blocks


def restore_code_blocks(text: str, code_blocks: list[str]) -> str:
    """Restore extracted code blocks from placeholders.

    Args:
        text: Text with code block placeholders.
        code_blocks: Original code block strings.

    Returns:
        Text with code blocks restored.
    """
    for i, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)
    return text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> Generator[str, None, None]:
    """Split text into overlapping token-aware chunks.

    Preserves code blocks intact. Uses sentence boundaries where possible.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.

    Yields:
        Individual text chunks.
    """
    if not text or len(text.strip()) < MIN_CHUNK_CHARS:
        return

    # Extract code blocks before splitting
    text_with_placeholders, code_blocks = extract_code_blocks(text)

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text_with_placeholders)

    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            chunk_text_raw = " ".join(current_chunk)
            chunk_restored = restore_code_blocks(chunk_text_raw, code_blocks)
            if len(chunk_restored.strip()) >= MIN_CHUNK_CHARS:
                yield chunk_restored.strip()

            # Overlap: keep last N tokens worth of sentences
            overlap_chunk: list[str] = []
            overlap_tokens = 0
            for sent in reversed(current_chunk):
                sent_tokens = estimate_tokens(sent)
                if overlap_tokens + sent_tokens <= overlap:
                    overlap_chunk.insert(0, sent)
                    overlap_tokens += sent_tokens
                else:
                    break
            current_chunk = overlap_chunk
            current_tokens = overlap_tokens

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Yield final chunk
    if current_chunk:
        chunk_text_raw = " ".join(current_chunk)
        chunk_restored = restore_code_blocks(chunk_text_raw, code_blocks)
        if len(chunk_restored.strip()) >= MIN_CHUNK_CHARS:
            yield chunk_restored.strip()


def process_github_file(data: dict) -> list[dict]:
    """Extract chunks from a GitHub repository JSON file.

    Args:
        data: Parsed JSON dict from ingest_github.py output.

    Returns:
        List of chunk dicts with text and metadata.
    """
    chunks = []
    base_meta = {
        "source": "github",
        "repo": data.get("repo_full_name", ""),
        "url": data.get("repo_url", ""),
        "stars": str(data.get("stars", 0)),
        "credibility_tier": "implementation",
        "content_type": "readme",
        "ingested_at": data.get("ingested_at", ""),
    }

    # README chunks
    if readme := data.get("readme"):
        for chunk in chunk_text(readme):
            chunks.append({
                "text": chunk,
                "metadata": {**base_meta, "content_type": "readme"},
            })

    # Issue chunks
    for issue in data.get("issues", []):
        issue_text = f"{issue.get('title', '')}\n{issue.get('body', '')}"
        for chunk in chunk_text(issue_text):
            chunks.append({
                "text": chunk,
                "metadata": {**base_meta, "content_type": "issue",
                             "url": issue.get("url", base_meta["url"])},
            })

    # Notebook chunks
    for notebook in data.get("notebooks", []):
        for chunk in chunk_text(notebook.get("content", "")):
            chunks.append({
                "text": chunk,
                "metadata": {**base_meta, "content_type": "notebook",
                             "path": notebook.get("path", "")},
            })

    return chunks


def process_arxiv_file(data: dict) -> list[dict]:
    """Extract chunks from an arXiv paper JSON file.

    Args:
        data: Parsed JSON dict from ingest_arxiv.py output.

    Returns:
        List of chunk dicts with text and metadata.
    """
    chunks = []
    base_meta = {
        "source": "arxiv",
        "paper_id": data.get("paper_id", ""),
        "title": data.get("title", ""),
        "authors": ", ".join(data.get("authors", [])[:3]),
        "published": data.get("published", ""),
        "url": data.get("pdf_url", ""),
        "credibility_tier": "primary",
        "content_type": "paper",
        "ingested_at": data.get("ingested_at", ""),
    }

    # Abstract — always include
    if abstract := data.get("abstract"):
        for chunk in chunk_text(abstract):
            chunks.append({
                "text": chunk,
                "metadata": {**base_meta, "content_type": "abstract"},
            })

    # Full PDF text
    if pdf_text := data.get("pdf_text"):
        for chunk in chunk_text(pdf_text):
            chunks.append({
                "text": chunk,
                "metadata": {**base_meta, "content_type": "paper_body"},
            })

    return chunks


def embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: SentenceTransformer,
    existing_ids: set[str],
) -> int:
    """Embed chunks and store in ChromaDB collection.

    Args:
        chunks: List of chunk dicts with text and metadata.
        collection: ChromaDB collection to store into.
        model: Loaded SentenceTransformer embedding model.
        existing_ids: Set of chunk IDs already in the collection.

    Returns:
        Number of chunks successfully stored.
    """
    stored = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # Generate deterministic IDs from source + content hash
        ids = [
            f"{c['metadata']['source']}_{abs(hash(c['text']))}"
            for c in batch
        ]

        # Skip already-stored chunks
        new_indices = [j for j, id_ in enumerate(ids) if id_ not in existing_ids]
        if not new_indices:
            continue

        new_texts = [texts[j] for j in new_indices]
        new_metadatas = [metadatas[j] for j in new_indices]
        new_ids = [ids[j] for j in new_indices]

        try:
            embeddings = model.encode(
                new_texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            collection.add(
                documents=new_texts,
                embeddings=embeddings,
                metadatas=new_metadatas,
                ids=new_ids,
            )

            existing_ids.update(new_ids)
            stored += len(new_ids)

        except Exception as exc:
            logger.error("Embedding batch failed: %s", exc)

    return stored


def run_pipeline() -> None:
    """Main pipeline — chunks all raw sources and embeds into ChromaDB.

    Processes GitHub and arXiv sources. Resume-safe: skips
    already-embedded chunks using deterministic ID generation.
    """
    logger.info("Initialising embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model loaded successfully.")

    # Initialise ChromaDB
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_or_create_collection(
        name="synthforge",
        metadata={"hnsw:space": "cosine"},
    )

    # Load existing IDs for resume capability
    existing_ids: set[str] = set(collection.get()["ids"])
    logger.info("Existing chunks in vector store: %d", len(existing_ids))

    total_stored = 0

    # ── Process GitHub ────────────────────────────────────────────────────────
    github_files = list((DATA_RAW / "github").glob("*.json"))
    logger.info("Processing %d GitHub files...", len(github_files))

    for file_path in github_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            chunks = process_github_file(data)
            stored = embed_and_store(chunks, collection, model, existing_ids)
            total_stored += stored
            logger.info("GitHub %s: %d chunks stored", file_path.stem[:50], stored)
        except Exception as exc:
            logger.error("Failed to process %s: %s", file_path.name, exc)

    # ── Process arXiv ─────────────────────────────────────────────────────────
    arxiv_files = list((DATA_RAW / "arxiv").glob("*.json"))
    logger.info("Processing %d arXiv files...", len(arxiv_files))

    for file_path in arxiv_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            chunks = process_arxiv_file(data)
            stored = embed_and_store(chunks, collection, model, existing_ids)
            total_stored += stored
            logger.info("arXiv %s: %d chunks stored", file_path.stem[:50], stored)
        except Exception as exc:
            logger.error("Failed to process %s: %s", file_path.name, exc)

    final_count = collection.count()
    logger.info("Pipeline complete. New chunks stored: %d", total_stored)
    logger.info("Total chunks in vector store: %d", final_count)


if __name__ == "__main__":
    run_pipeline()