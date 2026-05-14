"""
PromptForge - Official Documentation Ingestion
Layer 1: Scrapes LangChain, Anthropic, OpenAI Cookbook, DSPy,
Gemini, Mistral, and Hugging Face documentation.
No credentials required - all publicly accessible.

Usage: py src/ingestion/ingest_docs.py
"""

import json
import logging
import time
import re
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DATA_RAW, LOG_DIR

# -- Logging setup -------------------------------------------------------------
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingest_docs.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
OUTPUT_DIR: Path = DATA_RAW / "docs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS: dict[str, str] = {
    "User-Agent": "PromptForge/0.1 research tool (read-only)",
}

RATE_LIMIT_PAUSE: float = 2.0
MIN_TEXT_LENGTH: int = 200

# -- Documentation Sources -----------------------------------------------------
DOC_SOURCES: list[dict] = [
    {
        "name": "anthropic_docs",
        "base_url": "https://docs.anthropic.com",
        "urls": [
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview",
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-generator",
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-examples",
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-of-thought",
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/extended-thinking",
            "https://docs.anthropic.com/en/docs/about-claude/models/overview",
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/be-clear-and-direct",
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags",
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/give-claude-a-role",
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response",
        ],
        "credibility_tier": "primary",
        "content_type": "documentation",
    },
    {
        "name": "openai_docs",
        "base_url": "https://platform.openai.com",
        "urls": [
            "https://platform.openai.com/docs/guides/prompt-engineering",
            "https://platform.openai.com/docs/guides/text-generation",
            "https://platform.openai.com/docs/guides/few-shot-learning",
            "https://platform.openai.com/docs/guides/chain-of-thought",
            "https://platform.openai.com/docs/guides/function-calling",
            "https://platform.openai.com/docs/guides/structured-outputs",
        ],
        "credibility_tier": "primary",
        "content_type": "documentation",
    },
    {
        "name": "langchain_docs",
        "base_url": "https://python.langchain.com",
        "urls": [
            "https://python.langchain.com/docs/concepts/prompt_templates/",
            "https://python.langchain.com/docs/concepts/few_shot_prompting/",
            "https://python.langchain.com/docs/concepts/rag/",
            "https://python.langchain.com/docs/concepts/agents/",
            "https://python.langchain.com/docs/concepts/chains/",
            "https://python.langchain.com/docs/concepts/lcel/",
            "https://python.langchain.com/docs/tutorials/rag/",
            "https://python.langchain.com/docs/tutorials/chatbot/",
            "https://python.langchain.com/docs/tutorials/agents/",
        ],
        "credibility_tier": "implementation",
        "content_type": "documentation",
    },
    {
        "name": "huggingface_docs",
        "base_url": "https://huggingface.co",
        "urls": [
            "https://huggingface.co/docs/transformers/main/en/tasks/prompting",
            "https://huggingface.co/docs/transformers/main/en/tasks/text_generation",
            "https://huggingface.co/docs/peft/main/en/index",
            "https://huggingface.co/docs/trl/main/en/index",
            "https://huggingface.co/learn/cookbook/en/index",
            "https://huggingface.co/learn/nlp-course/en/chapter1/1",
        ],
        "credibility_tier": "implementation",
        "content_type": "documentation",
    },
    {
        "name": "mistral_docs",
        "base_url": "https://docs.mistral.ai",
        "urls": [
            "https://docs.mistral.ai/guides/prompting/",
            "https://docs.mistral.ai/guides/rag/",
            "https://docs.mistral.ai/guides/function_calling/",
            "https://docs.mistral.ai/guides/prefix/",
        ],
        "credibility_tier": "implementation",
        "content_type": "documentation",
    },
    {
        "name": "google_ai_docs",
        "base_url": "https://ai.google.dev",
        "urls": [
            "https://ai.google.dev/gemini-api/docs/prompting-intro",
            "https://ai.google.dev/gemini-api/docs/prompting-strategies",
            "https://ai.google.dev/gemini-api/docs/system-instructions",
            "https://ai.google.dev/gemini-api/docs/few-shot-examples",
        ],
        "credibility_tier": "implementation",
        "content_type": "documentation",
    },
    {
        "name": "lilianweng_blog",
        "base_url": "https://lilianweng.github.io",
        "urls": [
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
            "https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/",
        ],
        "credibility_tier": "primary",
        "content_type": "blog",
    },
    {
        "name": "simonwillison_blog",
        "base_url": "https://simonwillison.net",
        "urls": [
            "https://simonwillison.net/2023/May/2/prompt-injection-explained/",
            "https://simonwillison.net/2024/Mar/5/prompt-injection-attacks/",
            "https://simonwillison.net/2023/Apr/25/dual-llm-pattern/",
        ],
        "credibility_tier": "implementation",
        "content_type": "blog",
    },
]


def extract_text_from_url(url: str) -> str | None:
    """Fetch and extract clean text content from a URL.

    Args:
        url: Target URL to scrape.

    Returns:
        Cleaned text content or None on failure.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove navigation, headers, footers, scripts
        for tag in soup.find_all([
            "nav", "header", "footer", "script",
            "style", "aside", "advertisement"
        ]):
            tag.decompose()

        # Extract main content — try common content containers
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(class_=re.compile(r"content|main|article|docs"))
            or soup.find("body")
        )

        if not main_content:
            return None

        # Extract text with spacing
        text = main_content.get_text(separator="\n", strip=True)

        # Clean excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip() if len(text) >= MIN_TEXT_LENGTH else None

    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def process_source(source: dict) -> list[dict]:
    """Scrape all URLs for a documentation source.

    Args:
        source: Source config dict with name, urls, and metadata.

    Returns:
        List of page dicts with text and metadata.
    """
    pages = []
    source_name = source["name"]
    logger.info("Processing source: %s (%d URLs)", source_name, len(source["urls"]))

    for url in source["urls"]:
        text = extract_text_from_url(url)
        time.sleep(RATE_LIMIT_PAUSE)

        if not text:
            logger.warning("No content extracted from: %s", url)
            continue

        page = {
            "source": "docs",
            "source_name": source_name,
            "url": url,
            "text": text,
            "credibility_tier": source["credibility_tier"],
            "content_type": source["content_type"],
            "ingested_at": datetime.utcnow().isoformat(),
        }
        pages.append(page)
        logger.info("Extracted %d chars from: %s", len(text), url)

    return pages


def run_ingestion() -> None:
    """Main ingestion function — scrapes all documentation sources."""
    total_pages = 0

    for source in DOC_SOURCES:
        pages = process_source(source)

        if not pages:
            logger.warning("No pages extracted from: %s", source["name"])
            continue

        output_path = OUTPUT_DIR / f"{source['name']}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)

        logger.info(
            "Saved %d pages from %s", len(pages), source["name"]
        )
        total_pages += len(pages)

    logger.info("Docs ingestion complete. Total pages saved: %d", total_pages)


if __name__ == "__main__":
    run_ingestion()