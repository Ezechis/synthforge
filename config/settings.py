"""
SynthForge — Central Configuration
All constants, paths, and credentials live here.
Every pipeline script imports from this module only.
Never hardcode values in individual scripts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

#  Paths 
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_RAW: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED: Path = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR: Path = BASE_DIR / "data" / "vector_store"
LOG_DIR: Path = BASE_DIR / "logs"

#  Credentials (loaded from .env — never hardcoded) 
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "SynthForge/0.1")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "***REMOVED***")

#  GitHub Ingestion 



GITHUB_RATE_LIMIT: int = 5_000           # requests/hr authenticated
GITHUB_TARGET_REPOS: int = 50            # top N repos by stars
GITHUB_SEARCH_QUERY: str = "prompt engineering"

#  arXiv Ingestion 
ARXIV_SEED_COUNT: int = 200
ARXIV_SEARCH_TERMS: list[str] = [
    "prompt engineering",
    "chain of thought prompting",
    "few-shot learning language model",
    "instruction tuning",
    "retrieval augmented generation",
]

#  Reddit Quality Gates 
REDDIT_MIN_POST_UPVOTES: int = 50        # Gate 1 — minimum post upvotes
REDDIT_MIN_COMMENT_UPVOTES: int = 20     # Gate 1 — minimum comment upvotes
REDDIT_MIN_QUALITY_SCORE: float = 3.0   # Gate 3 — LLM scorer threshold (1-5)
REDDIT_TARGET_SUBREDDITS: list[str] = [
    "PromptEngineering",
    "LocalLLaMA",
    "MachineLearning",
    "LanguageModelSafety",
    "ChatGPTPro",
]

#  Chunking 
CHUNK_SIZE_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 50

#  Embedding 
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"  # MUST match corpus + Space query model
#  Generation 
OLLAMA_MODEL: str = "llama3:8b"
OLLAMA_BASE_URL: str = "http://localhost:11434"
GROQ_MODEL: str = "llama-3.3-70b-versatile"
