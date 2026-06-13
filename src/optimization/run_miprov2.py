#!/usr/bin/env python3
"""
SynthForge DSPy MIPROv2 Optimizer
===================================
Optimizes the generation instruction using DSPy's MIPROv2 algorithm.
Pre-retrieves context for each training example from local ChromaDB,
then optimizes only the generation instruction — retrieval is fixed.

Output: data/optimization/optimized_prompt_latest.txt

Usage:
    python src/optimization/run_miprov2.py
    python src/optimization/run_miprov2.py --train-size 30 --dev-size 15 --max-steps 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import dspy

# ── Constants ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GENERATION_MODEL: str = "groq/llama-3.1-8b-instant"
EVAL_PATH: Path = Path("data/evals/PromptForge_Golden_Eval_Stratified_45k_clean.jsonl")
VECTOR_STORE_PATH: Path = Path("data/vector_store")
COLLECTION_NAME: str = "synthforge"
OUTPUT_DIR: Path = Path("data/optimization")
DEFAULT_TRAIN_SIZE: int = 496
DEFAULT_DEV_SIZE: int = 200
DEFAULT_MAX_STEPS: int = 10
TOP_K_CHUNKS: int = 8
RANDOM_SEED: int = 42

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Module-level retrieval singletons (loaded once, reused for all examples) ──
_embedding_model: Any = None
_chroma_collection: Any = None


def _load_retrieval_resources() -> tuple[Any, Any]:
    """Lazy-load SentenceTransformer and ChromaDB collection once per process.

    Returns:
        Tuple of (embedding_model, chroma_collection).

    Raises:
        RuntimeError: If ChromaDB collection cannot be loaded.
    """
    global _embedding_model, _chroma_collection

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model BAAI/bge-large-en-v1.5 ...")
        _embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        log.info("Embedding model loaded.")

    if _chroma_collection is None:
        import chromadb
        if not VECTOR_STORE_PATH.exists():
            raise RuntimeError(
                f"Vector store not found at {VECTOR_STORE_PATH}. "
                "Run chunk_and_embed.py first."
            )
        client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
        try:
            _chroma_collection = client.get_collection(COLLECTION_NAME)
            count = _chroma_collection.count()
            log.info("ChromaDB collection '%s' loaded — %d chunks.", COLLECTION_NAME, count)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ChromaDB collection '{COLLECTION_NAME}': {exc}"
            ) from exc

    return _embedding_model, _chroma_collection


def retrieve_context(query: str, top_k: int = TOP_K_CHUNKS) -> str:
    """Retrieve top-k chunks from local ChromaDB for a given query.

    Args:
        query: The search query string.
        top_k: Number of chunks to retrieve.

    Returns:
        Concatenated chunk texts separated by dividers, or empty string on failure.
    """
    try:
        model, collection = _load_retrieval_resources()
        embedding = model.encode(query, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        chunks = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        parts: list[str] = []
        for chunk, meta in zip(chunks, metas):
            source = meta.get("source", "unknown")
            parts.append(f"[Source: {source}]\n{chunk}")

        return "\n\n---\n\n".join(parts)

    except Exception as exc:
        log.warning("Retrieval failed for query '%s...': %s", query[:60], exc)
        return ""


# ── DSPy Signature ─────────────────────────────────────────────────────────────
class GenerateAnswer(dspy.Signature):
    """You are SynthForge, a synthesis engine over a curated corpus of prompt engineering
    research spanning arXiv papers, GitHub implementations, and practitioner sources.
    Generate a technically precise, component-dense answer. Your answer MUST contain:
    named technique definitions, theoretical basis, empirical evidence with named studies,
    concrete implementation details, and known failure modes. Prioritize 2026 sources.
    Vague answers that describe without specifying named components will be rejected."""

    context: str = dspy.InputField(
        desc="Retrieved chunks from the SynthForge corpus, ranked by relevance"
    )
    query: str = dspy.InputField(
        desc="The user's prompt engineering question"
    )
    answer: str = dspy.OutputField(
        desc=(
            "Detailed synthesis answer (400+ words) containing: "
            "technique name and definition, theoretical mechanism, "
            "empirical evidence with named papers or benchmarks, "
            "implementation example or pseudocode, and known failure modes."
        )
    )


# ── DSPy Module ────────────────────────────────────────────────────────────────
class SynthForgeRAG(dspy.Module):
    """Minimal RAG module for MIPROv2 optimization of the generation instruction.

    Retrieval is pre-computed and passed as a fixed input field.
    MIPROv2 optimizes only the generation instruction and few-shot examples.
    """

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, query: str, context: str) -> dspy.Prediction:
        """Generate an answer given query and pre-retrieved context.

        Args:
            query: The user's question.
            context: Pre-retrieved corpus chunks as a formatted string.

        Returns:
            DSPy Prediction containing the answer field.
        """
        return self.generate(context=context, query=query)


# ── Metric ─────────────────────────────────────────────────────────────────────
def component_coverage_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Optional[Any] = None,
) -> float:
    """Score an answer by coverage of expected technical components.

    Uses keyword matching — fast, zero extra API calls.
    Falls back to word-count heuristic when no expected_components defined.
    Normalized to [0.0, 1.0].

    Args:
        example: DSPy example containing expected_components list.
        prediction: DSPy prediction containing answer string.
        trace: Optional trace object from DSPy (unused).

    Returns:
        Float coverage score between 0.0 and 1.0.
    """
    answer = getattr(prediction, "answer", "") or ""
    answer_lower = answer.lower()
    expected: list[str] = example.get("expected_components", [])

    if not expected:
        # Fallback: reward longer, denser answers up to 400 words
        word_count = len(answer_lower.split())
        return min(1.0, word_count / 400.0)

    hits = sum(1 for component in expected if component.lower() in answer_lower)
    return hits / len(expected)


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_eval_records(path: Path) -> list[dict]:
    """Load golden eval set from JSONL file.

    Args:
        path: Path to the JSONL eval file.

    Returns:
        List of eval record dicts.

    Raises:
        FileNotFoundError: If eval file does not exist.
        ValueError: If file contains no valid records.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Eval file not found: {path}\n"
            "Expected: data/evals/golden_eval_set.jsonl"
        )

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", line_num, exc)

    if not records:
        raise ValueError(f"No valid records found in {path}")

    log.info("Loaded %d eval records.", len(records))
    return records


def build_dspy_examples(records: list[dict]) -> list[dspy.Example]:
    """Convert eval records to DSPy Examples with pre-retrieved context.

    Loads retrieval resources once, then queries for each record sequentially.
    Progress logged every 10 records.

    Args:
        records: List of eval record dicts from golden_eval_set.jsonl.

    Returns:
        List of DSPy Example objects with query, context, expected_components.
    """
    log.info("Pre-retrieving context for %d examples...", len(records))
    examples: list[dspy.Example] = []

    for idx, record in enumerate(records):
        query: str = record.get("query", "").strip()
        if not query:
            log.warning("Skipping record %d — empty query.", idx)
            continue

        context = retrieve_context(query)

        example = dspy.Example(
            query=query,
            context=context,
            expected_components=record.get("expected_components", []),
            difficulty=record.get("difficulty", "intermediate"),
        ).with_inputs("query", "context")

        examples.append(example)

        if (idx + 1) % 10 == 0:
            log.info("  Pre-retrieved %d / %d", idx + 1, len(records))

    log.info("Built %d DSPy examples.", len(examples))
    return examples


# ── Optimization ───────────────────────────────────────────────────────────────
def run_miprov2(train_size: int, dev_size: int, max_steps: int) -> None:
    """Execute full MIPROv2 optimization run and persist the optimized instruction.

    Args:
        train_size: Number of training examples to use.
        dev_size: Number of dev/validation examples to use.
        max_steps: Number of MIPROv2 optimization trials.

    Raises:
        ValueError: If eval set does not have enough records.
        RuntimeError: If GROQ_API_KEY is not set.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Run: set GROQ_API_KEY=your_key"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configure DSPy language model
    lm = dspy.LM(
        model=GENERATION_MODEL,
        api_key=GROQ_API_KEY,
        max_tokens=1024,
        num_retries=10,
        temperature=0.0,
    )
    dspy.configure(lm=lm)
    log.info("DSPy configured: %s", GENERATION_MODEL)

    # Load and split eval records
    records = load_eval_records(EVAL_PATH)
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    total_needed = train_size + dev_size
    if len(records) < total_needed:
        raise ValueError(
            f"Requested {total_needed} records (train={train_size}, dev={dev_size}) "
            f"but eval set only has {len(records)}. Reduce --train-size or --dev-size."
        )

    train_records = records[:train_size]
    dev_records = records[train_size: train_size + dev_size]
    log.info("Split — train: %d | dev: %d", len(train_records), len(dev_records))

    # Pre-retrieve context (loads embedding model + ChromaDB once)
    log.info("=== Building training set ===")
    trainset = build_dspy_examples(train_records)
    log.info("=== Building dev set ===")
    devset = build_dspy_examples(dev_records)

    # Initialise the program
    program = SynthForgeRAG()

    # Run MIPROv2
    from dspy.teleprompt import MIPROv2

    teleprompter = MIPROv2(
        metric=component_coverage_metric,
        auto=None,        
        num_candidates=8,
        init_temperature=1.4,
        verbose=True,
        track_stats=True,
           
    )

    log.info("=== Starting MIPROv2 — %d trials ===", max_steps)
    optimized_program = teleprompter.compile(
        student=program,
        trainset=trainset,
        num_trials=max_steps,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        minibatch=False,
        
        
    )

    # Extract optimized instruction text
    try:
        optimized_instruction: str = optimized_program.generate.signature.instructions
    except AttributeError:
        optimized_instruction = str(optimized_program.generate.signature)

    # Persist output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = OUTPUT_DIR / "optimized_prompt_latest.txt"
    backup_path = OUTPUT_DIR / f"optimized_prompt_{timestamp}.txt"

    header = (
        f"# SynthForge MIPROv2 Optimized Prompt\n"
        f"# Generated : {datetime.now().isoformat()}\n"
        f"# Model     : {GENERATION_MODEL}\n"
        f"# Train size: {train_size} | Dev size: {dev_size} | Steps: {max_steps}\n"
        f"# ─────────────────────────────────────────────────────────────────────\n\n"
    )

    full_output = header + optimized_instruction + "\n"
    latest_path.write_text(full_output, encoding="utf-8")
    backup_path.write_text(full_output, encoding="utf-8")

    log.info("✅ Optimized prompt → %s", latest_path)
    log.info("✅ Backup           → %s", backup_path)
    log.info("\n%s\nOPTIMIZED INSTRUCTION:\n%s\n%s", "=" * 60, optimized_instruction, "=" * 60)


# ── Entry Point ────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SynthForge MIPROv2 Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-size", type=int, default=DEFAULT_TRAIN_SIZE,
        help="Number of training examples."
    )
    parser.add_argument(
        "--dev-size", type=int, default=DEFAULT_DEV_SIZE,
        help="Number of dev/validation examples."
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help="Number of MIPROv2 optimization trials."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_miprov2(
        train_size=args.train_size,
        dev_size=args.dev_size,
        max_steps=args.max_steps,
    )