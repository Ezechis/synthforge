"""
run_miprov2_final.py — Definitive MIPROv2 launcher.

Sets GROQ_API_KEY in ALL layers that litellm checks:
  1. os.environ (standard)
  2. litellm module globals
  3. dspy.LM api_key parameter

Usage:
    cd C:\\Users\\Ezeking\\PromptForge
    set HF_HUB_OFFLINE=1
    C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe run_miprov2_final.py
"""

import os
import sys
import random
from pathlib import Path
import json
import logging
import argparse
import datetime

# ── Step 1: Set in os.environ FIRST before any imports ───────────────────────
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = GROQ_KEY
os.environ["GROQ_API_KEY_HEADER"] = f"Bearer {GROQ_KEY}"

print(f"Step 1 OK — os.environ set: {GROQ_KEY[:8]}...{GROQ_KEY[-4:]}")

# ── Step 2: Now import litellm and patch its globals ──────────────────────────
try:
    import litellm
    litellm.api_key = GROQ_KEY
    litellm.groq_key = GROQ_KEY
    # litellm reads from os.environ["GROQ_API_KEY"] internally
    # but also caches — force-clear the cache
    if hasattr(litellm, "_lazy_module_dir"):
        pass  # not needed
    print(f"Step 2 OK — litellm.groq_key set")
except ImportError:
    print("Step 2 SKIP — litellm not yet imported (normal)")

# ── Step 3: Import dspy and patch LM configuration ───────────────────────────
try:
    import dspy
    print(f"Step 3 OK — dspy imported")
except ImportError as e:
    print(f"Step 3 FAIL — dspy import error: {e}")
    sys.exit(1)

# ── Step 4: Verify key is readable ───────────────────────────────────────────
assert os.environ.get("GROQ_API_KEY") == GROQ_KEY, "ENV VAR LOST"
print(f"Step 4 OK — key verified in environment")

# ── Step 5: Configure dspy LM with explicit key ───────────────────────────────
GENERATION_MODEL = "groq/llama-3.1-8b-instant"
lm = dspy.LM(
    model=GENERATION_MODEL,
    api_key=GROQ_KEY,
    api_base="https://api.groq.com/openai/v1",
    max_tokens=1024,
    num_retries=10,
    temperature=0.0,
)
dspy.configure(lm=lm)
print(f"Step 5 OK — dspy LM configured: {GENERATION_MODEL}")

# ── Step 6: Test the connection before running MIPROv2 ────────────────────────
print("Step 6 — Testing Groq connection...")
try:
    test_response = lm("Say OK in one word.")
    print(f"Step 6 OK — Groq responded: {str(test_response)[:50]}")
except Exception as e:
    print(f"Step 6 FAIL — Groq connection test failed: {e}")
    print("\nThe API key is being rejected by Groq.")
    print("Possible causes:")
    print("  1. Key has been revoked — generate a new key at console.groq.com")
    print("  2. Daily token limit hit — wait until 1 AM WAT (midnight UTC)")
    print("  3. Network issue — check internet connection")
    sys.exit(1)

# ── Step 7: Run MIPROv2 with all resources already configured ─────────────────
print("\nAll checks passed. Starting MIPROv2...\n")

from pathlib import Path
import json
import logging
import argparse
import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Constants
EVAL_PATH = Path("data/evals/golden_eval_set.jsonl")
VECTOR_STORE_PATH = Path("data/vector_store")
COLLECTION_NAME = "synthforge"
OUTPUT_DIR = Path("data/optimization")
DEFAULT_TRAIN_SIZE = 100
DEFAULT_DEV_SIZE = 50
DEFAULT_MAX_STEPS = 10
TOP_K_CHUNKS = 3
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load retrieval resources
log.info("Loading retrieval resources...")
try:
    import chromadb
    from sentence_transformers import SentenceTransformer

    _embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    _client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
    _chroma_collection = _client.get_collection(COLLECTION_NAME)
    count = _chroma_collection.count()
    log.info("ChromaDB loaded: %d chunks in '%s'", count, COLLECTION_NAME)
except Exception as e:
    log.error("Failed to load retrieval resources: %s", e)
    sys.exit(1)


def retrieve_context(query: str, top_k: int = TOP_K_CHUNKS) -> str:
    """Retrieve top-k chunks from ChromaDB for a query."""
    embedding = _embedding_model.encode(query, normalize_embeddings=True).tolist()
    results = _chroma_collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    chunks = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    parts = []
    for doc, meta in zip(chunks, metas):
        source = meta.get("source", "unknown")
        parts.append(f"[{source}] {doc}")
    return "\n\n".join(parts)


# DSPy Signature
class GenerateAnswer(dspy.Signature):
    """SynthForge RAG signature."""
    context: str = dspy.InputField(desc="Retrieved chunks from the SynthForge corpus")
    query: str = dspy.InputField(desc="The user's prompt engineering question")
    answer: str = dspy.OutputField(desc=(
        "A comprehensive, component-dense answer. Must contain: "
        "named technique, theoretical mechanism, empirical evidence, "
        "implementation example, and known failure modes."
    ))


class SynthForgeRAG(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, query: str, context: str) -> dspy.Prediction:
        return self.generate(context=context, query=query)


# Metric
def component_coverage_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Score based on presence of required components."""
    answer = getattr(prediction, "answer", "") or ""
    answer_lower = answer.lower()
    expected: list[str] = example.get("expected_components", [])
    if not expected:
        return 0.0
    hits = sum(1 for c in expected if c.lower() in answer_lower)
    return hits / len(expected)


# Load eval records
def load_eval_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def build_dspy_examples(records: list[dict]) -> list[dspy.Example]:
    examples = []
    for idx, record in enumerate(records):
        query: str = record.get("query", "").strip()
        context = retrieve_context(query)
        example = dspy.Example(
            query=query,
            context=context,
            expected_components=record.get("expected_components", []),
            difficulty=record.get("difficulty", "intermediate"),
        ).with_inputs("query", "context")
        if (idx + 1) % 10 == 0:
            log.info("Built %d DSPy examples.", idx + 1)
        examples.append(example)
    return examples


# Parse args
parser = argparse.ArgumentParser(description="SynthForge MIPROv2 Optimizer")
parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
parser.add_argument("--dev-size", type=int, default=DEFAULT_DEV_SIZE)
parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
args = parser.parse_args()

train_size = args.train_size
dev_size = args.dev_size
max_steps = args.max_steps

log.info("Loading eval records from %s", EVAL_PATH)
if not EVAL_PATH.exists():
    log.error("Eval file not found: %s", EVAL_PATH)
    sys.exit(1)

records = load_eval_records(EVAL_PATH)
if not records:
    log.error("No records found in eval file: %s", EVAL_PATH)
    sys.exit(1)

total_needed = train_size + dev_size
if len(records) < total_needed:
    log.error(
        "Not enough records in eval file. Needed: %d, available: %d",
        total_needed,
        len(records),
    )
    sys.exit(1)

log.info("Loaded %d records. Using %d train + %d dev.", len(records), train_size, dev_size)

# Shuffle with fixed seed for reproducibility
random.seed(RANDOM_SEED)
random.shuffle(records)

train_records = records[:train_size]
dev_records = records[train_size:train_size + dev_size]

log.info("=== Building training set ===")
trainset = build_dspy_examples(train_records)
log.info("=== Building dev set ===")
devset = build_dspy_examples(dev_records)

program = SynthForgeRAG()

from dspy.teleprompt import MIPROv2

teleprompter = MIPROv2(
    metric=component_coverage_metric,
    auto=None,
    num_candidates=4,
    init_temperature=1.4,
    verbose=True,
    track_stats=True,
)

log.info("=== Starting MIPROv2 with %d trials ===", max_steps)
optimized_program = teleprompter.compile(
    student=program,
    trainset=trainset,
    num_trials=max_steps,
    max_bootstrapped_demos=1,
    max_labeled_demos=2,
    minibatch=False,
)

# Extract and save optimized instruction
try:
    optimized_instruction = optimized_program.generate.signature.instructions
except AttributeError:
    optimized_instruction = str(optimized_program.generate.signature)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
latest_path = OUTPUT_DIR / "optimized_prompt_latest.txt"
backup_path = OUTPUT_DIR / f"optimized_prompt_{timestamp}.txt"

header = (
    f"# SynthForge MIPROv2 Optimized Instruction\n"
    f"# Generated: {timestamp}\n"
    f"# Train size: {train_size} | Dev size: {dev_size} | Trials: {max_steps}\n"
    f"{'=' * 60}\n\n"
)
full_output = header + optimized_instruction + "\n"
latest_path.write_text(full_output, encoding="utf-8")
backup_path.write_text(full_output, encoding="utf-8")

log.info("\n%s\nOPTIMIZED INSTRUCTION:\n%s\n%s",
         "=" * 60, optimized_instruction, "=" * 60)
log.info("Saved to %s", latest_path)
optimized_program.save(str(OUTPUT_DIR / "optimized_program.json"))
log.info("Full DSPy program saved to optimized_program.json")
print("\nMIPROv2 COMPLETE. Optimized prompt saved. Please run 'python verify_all_pipelines.py' to verify the pipelines.")
