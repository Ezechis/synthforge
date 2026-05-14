"""
app.py — PromptForge Streamlit Frontend (Hugging Face Spaces Edition)
=====================================================================
Features:
  - Hybrid BM25 + dense retrieval with cross-encoder reranking
  - File upload: PDF, Word, TXT, Markdown — supplements retrieval
  - Query suggestions, feedback, source confidence bar, copy button
  - Search history, live HN news, Griot Protocol articles panel
  - BM25 loaded from pre-built pickle for fast cold starts

Environment variables required in HF Space secrets:
    GROQ_API_KEY       — Groq API key
    HF_TOKEN           — HF read token
    HF_DATASET_REPO    — e.g. "ezechinnabugwu/promptforge-vectorstore"
"""

import json
import logging
import os
import pickle
import time
from io import BytesIO
from pathlib import Path

import chromadb
import requests
import streamlit as st
import streamlit.components.v1 as components
from chromadb.config import Settings
from huggingface_hub import snapshot_download
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME: str = "promptforge"
VECTOR_STORE_PATH: str = "/tmp/promptforge_vectorstore"
EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str = "llama-3.3-70b-versatile"
MAX_CONTEXT_TOKENS: int = 1500
WORDS_PER_TOKEN: float = 0.75
MAX_CONTEXT_WORDS: int = int(MAX_CONTEXT_TOKENS * WORDS_PER_TOKEN)
MIN_RETRIEVAL_SCORE: float = -6.5
HN_ALGOLIA_URL: str = "https://hn.algolia.com/api/v1/search"
FILE_CONTEXT_MAX_WORDS: int = 600

# ---------------------------------------------------------------------------
# The Griot Protocol — Series 1 by Ezechinyere Nnabugwu
# Set url="#" to show SOON badge. Replace with live URL when published.
# ---------------------------------------------------------------------------

EZE_ARTICLES: list[dict] = [
    {
        "title": "Structural Prompt Architecture",
        "url": "#",
        "description": "Markdown, XML tags, and JSON schema — mechanistic explanations of why each structure works.",
    },
    {
        "title": "Four Eras of Prompt Engineering",
        "url": "#",
        "description": "A periodization framework tracing the field from incantation through optimization.",
    },
    {
        "title": "DSPy Applied to Nigerian Education and Fintech",
        "url": "#",
        "description": "Practical DSPy application anchored in African use cases.",
    },
    {
        "title": "DSPy vs. TextGrad vs. GEPA vs. Promptomatix",
        "url": "#",
        "description": "Diagnostic decision framework across four variables. DeepSeek V3.2 as cost-efficient alternative.",
    },
    {
        "title": "The Formation of the Modern Prompt Engineer",
        "url": "#",
        "description": "Traced through Adaeze Okonkwo: five mental moves, four specification principles, six formation stages.",
    },
]

EXAMPLE_QUERIES: list[str] = [
    "What is chain-of-thought prompting?",
    "How does self-consistency decoding work?",
    "When does few-shot prompting fail?",
    "ReAct vs Reflexion — what is the difference?",
    "How do I reduce hallucination in LLM outputs?",
    "What is DSPy and how does it differ from manual prompting?",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector store download
# ---------------------------------------------------------------------------

def download_vectorstore() -> None:
    """Download ChromaDB + BM25 cache from HF Dataset on cold start."""
    vs_path = Path(VECTOR_STORE_PATH)
    if (vs_path / "chroma.sqlite3").exists():
        return

    dataset_repo = os.environ.get("HF_DATASET_REPO", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not dataset_repo:
        st.error("HF_DATASET_REPO secret not set.")
        st.stop()

    vs_path.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            local_dir=VECTOR_STORE_PATH,
            token=hf_token or None,
            ignore_patterns=["*.md", ".gitattributes"],
        )
    except Exception as exc:
        st.error(f"Failed to download vector store: {exc}")
        st.stop()


# ---------------------------------------------------------------------------
# Resource loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading PromptForge knowledge base...")
def load_all_resources():
    """Download and load all models and corpus. Cached for session lifetime."""
    download_vectorstore()

    chroma_path = str(Path(VECTOR_STORE_PATH) / "vector_store")
    if not Path(chroma_path).exists():
        chroma_path = VECTOR_STORE_PATH

    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    bm25_cache_path = Path(VECTOR_STORE_PATH) / "bm25_cache.pkl"
    if bm25_cache_path.exists():
        try:
            with open(bm25_cache_path, "rb") as fh:
                cache = pickle.load(fh)
            bm25 = cache["bm25"]
            corpus_chunks = cache["corpus_chunks"]
            corpus_metas = cache["corpus_metas"]
            logger.info("BM25 cache loaded: %d chunks.", len(corpus_chunks))
        except Exception as exc:
            logger.warning("BM25 cache failed (%s) — rebuilding...", exc)
            all_docs = collection.get(include=["documents", "metadatas"])
            corpus_chunks, corpus_metas = all_docs["documents"], all_docs["metadatas"]
            bm25 = BM25Okapi([d.lower().split() for d in corpus_chunks])
    else:
        all_docs = collection.get(include=["documents", "metadatas"])
        corpus_chunks, corpus_metas = all_docs["documents"], all_docs["metadatas"]
        bm25 = BM25Okapi([d.lower().split() for d in corpus_chunks])

    return collection, embed_model, reranker, bm25, corpus_chunks, corpus_metas


# ---------------------------------------------------------------------------
# File extraction
# ---------------------------------------------------------------------------

def extract_file_content(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF, Word, TXT, or Markdown file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Extracted and truncated text string, or error message string.
    """
    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    try:
        if filename.endswith((".txt", ".md")):
            text = raw_bytes.decode("utf-8", errors="ignore")

        elif filename.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(BytesIO(raw_bytes))
                text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                return "[PDF support unavailable — pypdf not installed on this Space]"

        elif filename.endswith(".docx"):
            try:
                from docx import Document
                doc = Document(BytesIO(raw_bytes))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                return "[Word support unavailable — python-docx not installed on this Space]"

        else:
            return ""

        words = text.split()
        if le