"""
ingest_arxiv_expand.py -- PromptForge arXiv Corpus Expansion
============================================================
Expands the arXiv corpus from ~200 to 500+ papers by searching
topic-by-topic across the core prompt engineering literature.
Skips papers already downloaded. Downloads PDFs and extracts text.

Run from project root:
    python src/ingestion/ingest_arxiv_expand.py
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import arxiv
import pymupdf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR: str = "data/raw/arxiv"
MAX_PER_QUERY: int = 50
REQUEST_DELAY: float = 3.0

# Targeted topic queries covering the full prompt engineering landscape
SEARCH_QUERIES: list[str] = [
    "chain of thought prompting large language models",
    "few-shot prompting in-context learning",
    "self-consistency prompting reasoning",
    "retrieval augmented generation RAG",
    "ReAct reasoning acting language model agents",
    "Reflexion self-reflection language model",
    "Tree of Thoughts deliberate problem solving",
    "constitutional AI RLHF alignment",
    "prompt injection adversarial attacks LLM",
    "instruction tuning fine-tuning language models",
    "hallucination mitigation large language models",
    "tool use function calling language model agents",
    "structured output generation JSON language model",
    "DSPy programming language model pipelines",
    "automatic prompt optimization engineering",
    "multi-agent systems LLM collaboration",
    "long context window language model prompting",
    "zero-shot chain of thought reasoning",
    "prompt compression token efficiency",
    "evaluation benchmarks language model prompting",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_existing_ids(output_dir: Path) -> set[str]:
    """Return arXiv IDs already downloaded to avoid re-fetching.

    Args:
        output_dir: Directory containing existing arXiv JSON files.

    Returns:
        Set of arXiv ID strings already on disk.
    """
    existing: set[str] = set()
    for json_file in output_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "arxiv_id" in data:
                existing.add(data["arxiv_id"])
        except Exception:
            pass
    return existing


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract full text from a PDF using pymupdf.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text string, or empty string on failure.
    """
    try:
        doc = pymupdf.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return " ".join(pages).strip()
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", pdf_path.name, exc)
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Search arXiv by topic and download new papers with full text."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = get_existing_ids(output_dir)
    logger.info("Found %d existing papers — will skip these.", len(existing_ids))

    total_saved = 0
    seen_this_run: set[str] = set()

    for query in SEARCH_QUERIES:
        logger.info("Searching: '%s'", query)

        search = arxiv.Search(
            query=query,
            max_results=MAX_PER_QUERY,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        try:
            results = list(search.results())
        except Exception as exc:
            logger.error("arXiv search failed for '%s': %s", query, exc)
            time.sleep(REQUEST_DELAY)
            continue

        for paper in results:
            arxiv_id: str = paper.entry_id.split("/")[-1]
            safe_id = arxiv_id.replace("/", "_")

            if arxiv_id in existing_ids or arxiv_id in seen_this_run:
                continue

            seen_this_run.add(arxiv_id)

            # Download PDF and extract text
            pdf_path = output_dir / f"{safe_id}.pdf"
            pdf_text = ""
            try:
                paper.download_pdf(dirpath=str(output_dir), filename=f"{safe_id}.pdf")
                pdf_text = extract_pdf_text(pdf_path)
                pdf_path.unlink(missing_ok=True)  # Remove PDF after extraction
            except Exception as exc:
                logger.warning("PDF download failed for %s: %s", arxiv_id, exc)

            doc: dict[str, Any] = {
                "source": "arxiv",
                "source_type": "arxiv",
                "credibility_tier": "primary",
                "arxiv_id": arxiv_id,
                "title": paper.title,
                "authors": [str(a) for a in paper.authors],
                "abstract": paper.summary.replace("\n", " "),
                "pdf_text": pdf_text,
                "url": paper.entry_id,
                "published": str(paper.published),
                "categories": paper.categories,
                "ingested_at": datetime.utcnow().isoformat(),
            }

            json_path = output_dir / f"{safe_id}.json"
            json_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total_saved += 1
            logger.info("Saved: %s — %s", arxiv_id, paper.title[:60])
            time.sleep(REQUEST_DELAY)

        logger.info(
            "Query complete — %d new papers saved so far.", total_saved
        )

    logger.info("arXiv expansion complete. Total new papers: %d", total_saved)


if __name__ == "__main__":
    main()