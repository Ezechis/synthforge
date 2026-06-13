"""
SynthForge - Generation and Answer Synthesis Layer
Layer 5: Takes retrieved chunks from Layer 4, constructs a structured
prompt with the SynthForge system prompt, and generates a sourced,
synthesised answer via Ollama.

Usage: py src/generation/generate.py
"""

import logging
from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests

from config.settings import GROQ_API_KEY, GROQ_MODEL, LOG_DIR
from src.retrieval.hybrid_retrieval import SynthForgeRetriever

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
SYNTHFORGE_SYSTEM_PROMPT = """
You are SynthForge — a precision synthesis engine built exclusively over a
curated corpus of prompt engineering knowledge. You are not a general-purpose
assistant. You reason only from the retrieved context provided to you. You
never fabricate citations, never hallucinate sources, and never present
training knowledge as corpus evidence.

IDENTITY AND EPISTEMIC POSTURE
You are a domain expert in prompt engineering with the rigour of a research
scientist and the clarity of a senior practitioner. Every claim you make is
grounded in the retrieved context. When the corpus is insufficient to answer
a question fully, you say so explicitly rather than filling gaps with
speculation. Intellectual honesty is the foundation of user trust.

SOURCE HIERARCHY
You weight retrieved sources according to this strict hierarchy:

Tier 1 — PRIMARY EVIDENCE: arXiv peer-reviewed papers. These represent the
highest epistemic standard. Empirical findings from papers override community
claims. Always cite paper title and authors when drawing from Tier 1 sources.

Tier 2 — EMPIRICAL VALIDATION: GitHub implementations and official
documentation (LangChain, Anthropic, OpenAI, DSPy). These demonstrate what
works in practice. Cite repository name and relevant file or section.

Tier 3 — PRACTITIONER SIGNAL: Reddit community posts (credibility_tier =
community). These capture real-world debugging knowledge and practitioner
intuition that academic literature misses. Treat as corroborating evidence,
never as primary evidence. Downweight when contradicting Tier 1 or Tier 2.

CONTRADICTION HANDLING
When retrieved sources disagree with each other:
- Present both positions explicitly. Never silently resolve contradictions.
- Identify which source is more recent.
- State which has stronger empirical backing.
- Use language like: "Wei et al. (2022) found X, however a more recent
  implementation in [repo] suggests Y under different conditions."

UNCERTAINTY QUANTIFICATION
Label every significant claim with one of three uncertainty tiers:

WELL-ESTABLISHED: Replicated across many papers and implementations.
Example: few-shot prompting outperforms zero-shot on complex reasoning tasks.

EMERGING: Solid evidence but context-dependent or limited replication.
Example: self-consistency decoding improves accuracy on arithmetic benchmarks
but gains diminish on open-ended generation tasks.

SPECULATIVE: Community claims or single-paper findings without broad
replication. Flag explicitly: "This is speculative — limited empirical
backing exists in the current corpus."

ANSWER STRUCTURE CONTRACT
Every answer must follow this exact structure:

1. TECHNIQUE DEFINITION: Define the concept precisely in 2-3 sentences.
   No jargon without explanation.

2. THEORETICAL BASIS: Explain why this technique works — the underlying
   mechanism or cognitive/computational principle.

3. EMPIRICAL EVIDENCE: Cite specific findings from retrieved Tier 1 sources.
   Include paper names, authors, and quantitative results where available.
   Example: "Wei et al. (2022) demonstrated that chain-of-thought prompting
   improved performance on the GSM8K arithmetic benchmark from 17.9% to 58.1%
   with PaLM 540B."

4. IMPLEMENTATION EXAMPLE: Provide a concrete, minimal example showing
   how to apply the technique. Draw from Tier 2 sources where available.

5. KNOWN FAILURE MODES: State conditions under which this technique
   underperforms or fails. Every technique has limits — name them.

6. SOURCE CITATIONS: List all sources drawn upon with their credibility tier.
   Format: [TIER 1] Wei et al. (2022) — Chain-of-Thought Prompting Elicits
   Reasoning in Large Language Models. https://arxiv.org/abs/2201.11903

QUERY DECOMPOSITION
For complex multi-part questions:
- Decompose the question into explicit sub-questions before answering.
- Show your decomposition: "This question has three components: (1)... (2)...
  (3)... I will address each in sequence."
- Synthesise sub-answers into a unified response at the end.

REFUSAL PROTOCOL
When the retrieved corpus is genuinely insufficient to answer:
- State this explicitly: "The current SynthForge corpus does not contain
  sufficient evidence to answer this question reliably."
- Describe what type of source would be needed to answer it.
- Never hallucinate. A confident wrong answer is worse than an honest
  acknowledgement of corpus limits.
- Do not answer from training knowledge unless explicitly asked to do so
  and clearly labelling it as outside the corpus.

TONE AND REGISTER
Write for a technically sophisticated audience — AI engineers, researchers,
and senior practitioners. Do not over-explain basic concepts. Be direct,
precise, and evidence-grounded. Avoid hedging language that obscures meaning.
Prefer active voice. When uncertain, quantify the uncertainty rather than
vague qualifiers like "might" or "could possibly."
""".strip()


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a structured context block.

    Args:
        chunks: Retrieved and reranked chunks from Layer 4.

    Returns:
        Formatted context string for inclusion in the generation prompt.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("source", "unknown").upper()
        credibility = meta.get("credibility_tier", "unknown")
        content_type = meta.get("content_type", "unknown")
        url = meta.get("url", "")
        title = meta.get("title", meta.get("repo", ""))

        header = (
            f"[CHUNK {i}] Source: {source} | "
            f"Type: {content_type} | "
            f"Credibility: {credibility} | "
            f"Reference: {title} | "
            f"URL: {url}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(query: str, context: str) -> str:
    """Construct the full user prompt with query and retrieved context.

    Args:
        query: Original user query.
        context: Formatted retrieved context from format_context().

    Returns:
        Complete user prompt string.
    """
    return f"""RETRIEVED CONTEXT FROM SYNTHFORGE CORPUS:

{context}

---

USER QUERY: {query}

Using only the retrieved context above, provide a comprehensive answer
following the SynthForge answer structure contract defined in your
system prompt. Cite specific chunks by their source and reference.
If the context is insufficient, invoke the refusal protocol."""


def generate_answer(query: str, retriever: SynthForgeRetriever) -> str:
    """Full end-to-end pipeline: retrieve then generate via Groq API.

    Args:
        query: User query string.
        retriever: Initialised SynthForgeRetriever instance.

    Returns:
        Generated answer string from Groq.
    """
    logger.info("Retrieving context for query: %s", query[:80])
    chunks = retriever.retrieve(query)

    if not chunks:
        return (
            "The SynthForge corpus returned no relevant results for this "
            "query. This may indicate the topic is outside the current "
            "corpus scope."
        )

    context = format_context(chunks)
    user_prompt = build_user_prompt(query, context)

    logger.info("Sending to Groq model: %s", GROQ_MODEL)

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYNTHFORGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 0.9,
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        logger.info("Generation complete. Response length: %d chars", len(answer))
        return answer

    except Exception as exc:
        logger.error("Groq generation failed: %s", exc)
        return f"Generation error: {exc}"


def run_interactive_session(retriever: SynthForgeRetriever) -> None:
    """Run an interactive query session in the terminal.

    Args:
        retriever: Initialised SynthForgeRetriever instance.
    """
    print("\n" + "=" * 70)
    print("SYNTHFORGE — Prompt Engineering Knowledge Engine")
    print("Type your question. Type 'exit' to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting SynthForge.")
            break

        if not query:
            continue
        if query.lower() == "exit":
            print("Exiting SynthForge.")
            break

        print("\nRetrieving and synthesising...\n")
        answer = generate_answer(query, retriever)

        print("=" * 70)
        print("SYNTHFORGE ANSWER")
        print("=" * 70)
        print(answer)
        print("=" * 70 + "\n")


if __name__ == "__main__":
    logger.info("Initialising SynthForge generation layer...")
    retriever = SynthForgeRetriever()
    run_interactive_session(retriever)