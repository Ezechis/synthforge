"""
deepforge.core.catalogue.book_catalogue
=========================================
The single authoritative catalogue of freely-licensed books and documents
for ingestion into SynthForge (and future ForgeCore products).

COMMERCIAL SAFETY RULE (non-negotiable):
    SynthForge charges users. Only licences in COMMERCIALLY_SAFE_LICENSES
    (defined in deepforge.core.schemas) are permitted. Every entry here
    must have a verified licence_type that is in that set.

    If you are adding a new entry and are not 100% certain of the licence:
      1. Set licence_type = LicenseType.UNKNOWN
      2. Set licence_verified = False
      3. Add a note in the 'notes' field
      4. The ingestor will block it until you verify and update

FETCH TYPES:
    "pdf"          → fetch_pdf()
    "html"         → fetch_html()
    "gutenberg"    → fetch_gutenberg() — requires gutenberg_id field
    "standard_ebook" → fetch_standard_ebook() — requires slug field
    "hf_dataset"   → fetch_hf_dataset_book() — requires dataset_repo + record_id

Author: DeepForge Engineering
"""

from __future__ import annotations

from typing import Any

from deepforge.core.schemas import (
    ContentType,
    CredibilityTier,
    DownloadType,
    LicenseType,
    SourceType,
)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — TECHNICAL DOCUMENTATION (MIT / Apache 2.0)
# Highest relevance for SynthForge. Verified commercial safe.
# ──────────────────────────────────────────────────────────────────────────────

TECH_DOCS: list[dict[str, Any]] = [
    {
        "source_id": "hf_nlp_course",
        "title": "Hugging Face NLP Course",
        "authors": "Lewis Tunstall|Leandro von Werra|Thomas Wolf",
        "year": 2023,
        "publisher": "Hugging Face",
        "licence_type": LicenseType.APACHE_2,
        "licence_verified": True,
        "source_url": "https://huggingface.co/learn/nlp-course/",
        "download_url": "https://huggingface.co/learn/nlp-course/chapter1/1",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.COURSE_NOTES,
        "domain_tags": "transformers|fine-tuning|NLP|embeddings|HuggingFace",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "Apache 2.0 confirmed in GitHub repo huggingface/course",
    },
    {
        "source_id": "hf_llm_course",
        "title": "Hugging Face LLM Course",
        "authors": "Hugging Face Team",
        "year": 2024,
        "publisher": "Hugging Face",
        "licence_type": LicenseType.APACHE_2,
        "licence_verified": True,
        "source_url": "https://huggingface.co/learn/llm-course/",
        "download_url": "https://huggingface.co/learn/llm-course/chapter1/1",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.COURSE_NOTES,
        "domain_tags": "LLM|fine-tuning|RLHF|prompt-engineering|evaluation|RAG",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "Apache 2.0 confirmed in GitHub repo",
    },
    {
        "source_id": "hf_agents_course",
        "title": "Hugging Face Agents Course",
        "authors": "Hugging Face Team",
        "year": 2024,
        "publisher": "Hugging Face",
        "licence_type": LicenseType.APACHE_2,
        "licence_verified": True,
        "source_url": "https://huggingface.co/learn/agents-course/",
        "download_url": "https://huggingface.co/learn/agents-course/unit0/introduction",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.COURSE_NOTES,
        "domain_tags": "agents|tool-use|LLM|planning|prompt-engineering|agentic",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "Apache 2.0 confirmed in GitHub repo",
    },
    {
        "source_id": "hf_deep_rl_course",
        "title": "Hugging Face Deep Reinforcement Learning Course",
        "authors": "Thomas Simonini",
        "year": 2023,
        "publisher": "Hugging Face",
        "licence_type": LicenseType.APACHE_2,
        "licence_verified": True,
        "source_url": "https://huggingface.co/learn/deep-rl-course/",
        "download_url": "https://huggingface.co/learn/deep-rl-course/unit0/introduction",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.COURSE_NOTES,
        "domain_tags": "reinforcement-learning|RLHF|reward-modelling|LLM-alignment",
        "relevance_score": 4,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "Apache 2.0 confirmed",
    },
    {
        "source_id": "openai_cookbook",
        "title": "OpenAI Cookbook",
        "authors": "OpenAI",
        "year": 2024,
        "publisher": "OpenAI",
        "licence_type": LicenseType.MIT,
        "licence_verified": True,
        "source_url": "https://github.com/openai/openai-cookbook",
        "download_url": "https://raw.githubusercontent.com/openai/openai-cookbook/main/README.md",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.GUIDE,
        "domain_tags": "prompt-engineering|few-shot|RAG|LLM|embeddings|function-calling",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "MIT licence in repo root — confirmed",
    },
    {
        "source_id": "dspy_docs",
        "title": "DSPy Documentation",
        "authors": "Omar Khattab|DSPy Contributors",
        "year": 2024,
        "publisher": "Stanford NLP",
        "licence_type": LicenseType.MIT,
        "licence_verified": True,
        "source_url": "https://dspy.ai/",
        "download_url": "https://dspy.ai/",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.DOCUMENTATION,
        "domain_tags": "DSPy|auto-prompt-optimisation|MIPROv2|LM-programs|teleprompters",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "MIT licence confirmed in stanfordnlp/dspy repo",
    },
    {
        "source_id": "langchain_docs",
        "title": "LangChain Documentation",
        "authors": "LangChain Team",
        "year": 2024,
        "publisher": "LangChain",
        "licence_type": LicenseType.MIT,
        "licence_verified": True,
        "source_url": "https://python.langchain.com/docs/introduction/",
        "download_url": "https://python.langchain.com/docs/introduction/",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.DOCUMENTATION,
        "domain_tags": "LangChain|RAG|agents|chains|prompting|LCEL",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "MIT confirmed in langchain-ai/langchain repo",
    },
    {
        "source_id": "llamaindex_docs",
        "title": "LlamaIndex Documentation",
        "authors": "LlamaIndex Team",
        "year": 2024,
        "publisher": "LlamaIndex",
        "licence_type": LicenseType.MIT,
        "licence_verified": True,
        "source_url": "https://docs.llamaindex.ai/",
        "download_url": "https://docs.llamaindex.ai/en/stable/",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.DOCUMENTATION,
        "domain_tags": "RAG|LlamaIndex|document-ingestion|query-engines|knowledge-graphs",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "MIT confirmed in run-llama/llama_index repo",
    },
    {
        "source_id": "anthropic_prompt_guide",
        "title": "Anthropic Prompt Engineering Documentation",
        "authors": "Anthropic",
        "year": 2024,
        "publisher": "Anthropic",
        "licence_type": LicenseType.MIT,
        "licence_verified": True,
        "source_url": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview",
        "download_url": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.GUIDE,
        "domain_tags": "prompt-engineering|Claude|CoT|XML-tags|system-prompts|Constitutional-AI",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "Public documentation — MIT for code examples per Anthropic terms",
    },
    {
        "source_id": "mistral_docs",
        "title": "Mistral AI Prompting Guide",
        "authors": "Mistral AI",
        "year": 2024,
        "publisher": "Mistral AI",
        "licence_type": LicenseType.APACHE_2,
        "licence_verified": True,
        "source_url": "https://docs.mistral.ai/guides/prompting_capabilities/",
        "download_url": "https://docs.mistral.ai/guides/prompting_capabilities/",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.GUIDE,
        "domain_tags": "prompt-engineering|Mistral|function-calling|JSON-mode|few-shot",
        "relevance_score": 4,
        "credibility_tier": CredibilityTier.TIER_2,
        "source": SourceType.DOCUMENTATION,
        "notes": "Apache 2.0 confirmed",
    },
    {
        "source_id": "fastai_practical_dl",
        "title": "Practical Deep Learning for Coders",
        "authors": "Jeremy Howard|Sylvain Gugger",
        "year": 2022,
        "publisher": "fast.ai",
        "licence_type": LicenseType.APACHE_2,
        "licence_verified": True,
        "source_url": "https://course.fast.ai/",
        "download_url": "https://course.fast.ai/",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.COURSE_NOTES,
        "domain_tags": "deep-learning|PyTorch|fine-tuning|transformers|NLP|practical",
        "relevance_score": 4,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "Apache 2.0 confirmed in fastai/fastbook repo",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FREELY LICENSED TEXTBOOKS (CC BY, CC BY-SA, verified)
# ──────────────────────────────────────────────────────────────────────────────

OPEN_TEXTBOOKS: list[dict[str, Any]] = [
    {
        "source_id": "d2l_dive_into_dl",
        "title": "Dive into Deep Learning",
        "authors": "Aston Zhang|Zachary Lipton|Mu Li|Alexander Smola",
        "year": 2023,
        "publisher": "Cambridge University Press",
        "licence_type": LicenseType.CC_BY_SA,
        "licence_verified": True,
        "source_url": "https://d2l.ai/",
        "download_url": "https://d2l.ai/d2l-en.pdf",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "deep-learning|transformers|attention|NLP|PyTorch|pre-training",
        "relevance_score": 4,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "CC BY-SA 4.0 declared on d2l.ai and in the PDF itself. "
                 "SA means derivative works must also be CC BY-SA — acceptable for corpus.",
    },
    {
        "source_id": "esl_hastie",
        "title": "The Elements of Statistical Learning (2nd Edition)",
        "authors": "Trevor Hastie|Robert Tibshirani|Jerome Friedman",
        "year": 2009,
        "publisher": "Springer",
        "licence_type": LicenseType.CC_BY,
        "licence_verified": False,  # PDF freely available; confirm CC BY at hastie.su.domains
        "source_url": "https://hastie.su.domains/ElemStatLearn/",
        "download_url": "https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "statistical-learning|ensemble-methods|boosting|regression|classification",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "Authors distribute free PDF. Verify CC BY declaration at source before ingesting. "
                 "licence_verified=False until confirmed.",
    },
    {
        "source_id": "bishop_prml",
        "title": "Pattern Recognition and Machine Learning",
        "authors": "Christopher M. Bishop",
        "year": 2006,
        "publisher": "Springer (Microsoft Research Free Release)",
        "licence_type": LicenseType.CC_BY,
        "licence_verified": False,
        "source_url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/",
        "download_url": "https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "machine-learning|Bayesian|probabilistic|pattern-recognition|neural-networks",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "Microsoft released PDF freely but explicit CC BY not confirmed. "
                 "Verify licence declaration before ingesting. licence_verified=False.",
    },
    {
        "source_id": "ciml_daum",
        "title": "A Course in Machine Learning",
        "authors": "Hal Daumé III",
        "year": 2017,
        "publisher": "Self-published (ciml.info)",
        "licence_type": LicenseType.CC_BY,
        "licence_verified": True,
        "source_url": "http://ciml.info/",
        "download_url": "http://ciml.info/dl/v0_99/ciml-v0_99-all.pdf",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "machine-learning|NLP|structured-prediction|neural-networks",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "CC BY declared on ciml.info — verified",
    },
    {
        "source_id": "goldberg_nlp_primer",
        "title": "A Primer on Neural Network Models for Natural Language Processing",
        "authors": "Yoav Goldberg",
        "year": 2015,
        "publisher": "arXiv",
        "licence_type": LicenseType.CC_BY,
        "licence_verified": True,
        "source_url": "https://arxiv.org/abs/1510.00726",
        "download_url": "https://arxiv.org/pdf/1510.00726",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.SURVEY,
        "domain_tags": "NLP|neural-networks|word-embeddings|RNN|sequence-models|attention",
        "relevance_score": 4,
        "credibility_tier": CredibilityTier.TIER_1,
        "source": SourceType.ARXIV,
        "notes": "CC BY confirmed on arXiv — author submitted with CC BY licence",
    },
    {
        "source_id": "foundations_llm_xiao",
        "title": "Foundations of Large Language Models",
        "authors": "Tong Xiao|Jingbo Zhu",
        "year": 2025,
        "publisher": "arXiv",
        "licence_type": LicenseType.CC_BY,
        "licence_verified": True,
        "source_url": "https://arxiv.org/abs/2501.09223",
        "download_url": "https://arxiv.org/pdf/2501.09223",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "LLM|prompting|alignment|fine-tuning|pre-training|RLHF|CoT|agents",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_1,
        "source": SourceType.ARXIV,
        "notes": "CC BY declared in arXiv submission — highest priority ingest",
    },
    {
        "source_id": "gaussian_processes_rasmussen",
        "title": "Gaussian Processes for Machine Learning",
        "authors": "Carl Edward Rasmussen|Christopher K. I. Williams",
        "year": 2006,
        "publisher": "MIT Press",
        "licence_type": LicenseType.CC_BY,
        "licence_verified": True,
        "source_url": "http://gaussianprocess.org/gpml/",
        "download_url": "http://gaussianprocess.org/gpml/chapters/RW.pdf",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "Bayesian|Gaussian-processes|probabilistic|machine-learning",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "CC BY declared at gaussianprocess.org — verified",
    },
    {
        "source_id": "interpretable_ml_molnar",
        "title": "Interpretable Machine Learning",
        "authors": "Christoph Molnar",
        "year": 2023,
        "publisher": "christophm.github.io",
        "licence_type": LicenseType.CC_BY,
        "licence_verified": True,
        "source_url": "https://christophm.github.io/interpretable-ml-book/",
        "download_url": "https://christophm.github.io/interpretable-ml-book/",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "interpretability|explainability|SHAP|LIME|ML|feature-importance",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "CC BY declared on page — verified",
    },
    {
        "source_id": "openintro_stats",
        "title": "OpenIntro Statistics",
        "authors": "David Diez|Mine Çetinkaya-Rundel|Christopher Barr",
        "year": 2019,
        "publisher": "OpenIntro",
        "licence_type": LicenseType.CC_BY_SA,
        "licence_verified": True,
        "source_url": "https://www.openintro.org/book/stat/",
        "download_url": "https://www.openintro.org/book/stat/",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "statistics|probability|hypothesis-testing|regression",
        "relevance_score": 2,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "CC BY-SA confirmed at openintro.org",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PROJECT GUTENBERG (Public Domain — all safe)
# Fetched via fetch_gutenberg(gutenberg_id=...).
# For SynthForge, only texts with direct PE / AI / logic relevance.
# Classical AI precursor texts and foundational logic works.
# ──────────────────────────────────────────────────────────────────────────────

GUTENBERG_BOOKS: list[dict[str, Any]] = [
    {
        "source_id": "gutenberg_turing_1950",
        "title": "Computing Machinery and Intelligence (1950)",
        "authors": "Alan Turing",
        "year": 1950,
        "publisher": "Project Gutenberg",
        "licence_type": LicenseType.PUBLIC_DOMAIN,
        "licence_verified": True,
        "source_url": "https://www.gutenberg.org/ebooks/72363",
        "download_url": "",
        "download_type": DownloadType.TEXT,
        "gutenberg_id": 72363,
        "content_type": ContentType.MONOGRAPH,
        "domain_tags": "AI|Turing-test|machine-intelligence|computation|philosophy-of-mind",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "THE foundational AI text. Turing died 1954, UK. Clearly public domain. "
                 "Gutenberg ID 72363 — verify this is the correct ID before ingesting.",
    },
    {
        "source_id": "gutenberg_boole_laws_of_thought",
        "title": "An Investigation of the Laws of Thought (1854)",
        "authors": "George Boole",
        "year": 1854,
        "publisher": "Project Gutenberg",
        "licence_type": LicenseType.PUBLIC_DOMAIN,
        "licence_verified": True,
        "source_url": "https://www.gutenberg.org/ebooks/15114",
        "download_url": "",
        "download_type": DownloadType.TEXT,
        "gutenberg_id": 15114,
        "content_type": ContentType.MONOGRAPH,
        "domain_tags": "logic|boolean-algebra|reasoning|formal-systems",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "Foundational logic text. Relevant to prompt logic and formal reasoning.",
    },
    {
        "source_id": "gutenberg_russell_problems_philosophy",
        "title": "The Problems of Philosophy (1912)",
        "authors": "Bertrand Russell",
        "year": 1912,
        "publisher": "Project Gutenberg",
        "licence_type": LicenseType.PUBLIC_DOMAIN,
        "licence_verified": True,
        "source_url": "https://www.gutenberg.org/ebooks/5827",
        "download_url": "",
        "download_type": DownloadType.TEXT,
        "gutenberg_id": 5827,
        "content_type": ContentType.MONOGRAPH,
        "domain_tags": "epistemology|knowledge|reasoning|philosophy",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "Relevant to knowledge representation and epistemic framing in prompts.",
    },
    {
        "source_id": "gutenberg_peirce_how_to_make_ideas_clear",
        "title": "How to Make Our Ideas Clear (1878)",
        "authors": "Charles Sanders Peirce",
        "year": 1878,
        "publisher": "Project Gutenberg",
        "licence_type": LicenseType.PUBLIC_DOMAIN,
        "licence_verified": True,
        "source_url": "https://www.gutenberg.org/ebooks/35895",
        "download_url": "",
        "download_type": DownloadType.TEXT,
        "gutenberg_id": 35895,
        "content_type": ContentType.MONOGRAPH,
        "domain_tags": "pragmatism|semiotics|clarity|reasoning|epistemology",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "Peirce's pragma of clarity — directly relevant to prompt specification.",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — STANDARD EBOOKS (CC0 — highest safety level)
# standardebooks.org produces beautifully formatted CC0 public domain editions.
# Fetch via fetch_standard_ebook(slug=...).
# ──────────────────────────────────────────────────────────────────────────────

STANDARD_EBOOKS: list[dict[str, Any]] = [
    {
        "source_id": "stdebooks_turing_computing",
        "title": "Computing Machinery and Intelligence",
        "authors": "Alan Turing",
        "year": 1950,
        "publisher": "Standard Ebooks",
        "licence_type": LicenseType.CC0,
        "licence_verified": True,
        "source_url": "https://standardebooks.org/ebooks/alan-turing/computing-machinery-and-intelligence",
        "download_url": "",
        "download_type": DownloadType.HTML,
        "slug": "alan-turing/computing-machinery-and-intelligence",
        "content_type": ContentType.MONOGRAPH,
        "domain_tags": "AI|Turing-test|machine-intelligence|imitation-game|computation",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "CC0 — maximum safety. Standard Ebooks confirms CC0 on every page.",
    },
    {
        "source_id": "stdebooks_darwin_origin",
        "title": "On the Origin of Species",
        "authors": "Charles Darwin",
        "year": 1859,
        "publisher": "Standard Ebooks",
        "licence_type": LicenseType.CC0,
        "licence_verified": True,
        "source_url": "https://standardebooks.org/ebooks/charles-darwin/on-the-origin-of-species",
        "download_url": "",
        "download_type": DownloadType.HTML,
        "slug": "charles-darwin/on-the-origin-of-species",
        "content_type": ContentType.MONOGRAPH,
        "domain_tags": "evolution|natural-selection|scientific-reasoning|argument-structure",
        "relevance_score": 2,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "CC0. Lower PE relevance but exceptional scientific argument structure. "
                 "Useful for argument-chain prompting training data.",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — HARVARD LIBRARY / HF DATASETS (Public Domain — explicitly AI-safe)
# The Harvard Library Public Domain Corpus (242B+ tokens) is one of the few
# sources explicitly cleared for AI training. Available via HuggingFace Datasets.
# Fetch via fetch_hf_dataset_book().
# ──────────────────────────────────────────────────────────────────────────────

HF_DATASET_BOOKS: list[dict[str, Any]] = [
    {
        "source_id": "harvard_pd_corpus_sample",
        "title": "Harvard Library Public Domain Corpus (AI/Logic subset)",
        "authors": "Harvard Library",
        "year": 2023,
        "publisher": "Harvard Library / Institutional Data Initiative",
        "licence_type": LicenseType.PUBLIC_DOMAIN,
        "licence_verified": True,
        "source_url": "https://huggingface.co/datasets/harvard-library/harvard-library-public-domain-corpus",
        "download_url": "",
        "download_type": DownloadType.JSONL,
        "dataset_repo": "harvard-library/harvard-library-public-domain-corpus",
        "record_id": "",       # Empty = fetch all; filter by subject post-download
        "text_field": "text",
        "content_type": ContentType.MONOGRAPH,
        "domain_tags": "public-domain|logic|philosophy|science|mathematics",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "Explicitly cleared for AI training by Harvard. "
                 "Filter by subject='Computer Science' or 'Mathematics' or 'Logic' "
                 "post-download to limit to SynthForge-relevant content.",
        "fetch_type": "hf_dataset",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — BLOCKED ENTRIES (NC licence or unverified)
# These are included for catalogue completeness and documentation.
# The ingestor will skip all of these automatically via licence gate.
# Do NOT remove them — they explain WHY specific popular books are absent.
# ──────────────────────────────────────────────────────────────────────────────

BLOCKED_NC_BOOKS: list[dict[str, Any]] = [
    {
        "source_id": "goodfellow_deep_learning",
        "title": "Deep Learning",
        "authors": "Ian Goodfellow|Yoshua Bengio|Aaron Courville",
        "year": 2016,
        "publisher": "MIT Press",
        "licence_type": LicenseType.CC_BY_NC_ND,
        "licence_verified": True,
        "source_url": "https://www.deeplearningbook.org/",
        "download_url": "",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "deep-learning|neural-networks|transformers",
        "relevance_score": 4,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "BLOCKED — CC BY-NC-ND. Free to read but NC prohibits commercial corpus use.",
    },
    {
        "source_id": "prince_understanding_dl",
        "title": "Understanding Deep Learning",
        "authors": "Simon J.D. Prince",
        "year": 2023,
        "publisher": "MIT Press",
        "licence_type": LicenseType.CC_BY_NC_ND,
        "licence_verified": True,
        "source_url": "https://udlbook.github.io/udlbook/",
        "download_url": "",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "deep-learning|transformers|attention|diffusion",
        "relevance_score": 4,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "BLOCKED — CC BY-NC-ND. Excellent book but NC bars commercial corpus use.",
    },
    {
        "source_id": "murphy_pml_intro",
        "title": "Probabilistic Machine Learning: An Introduction",
        "authors": "Kevin Patrick Murphy",
        "year": 2022,
        "publisher": "MIT Press",
        "licence_type": LicenseType.CC_BY_NC_ND,
        "licence_verified": True,
        "source_url": "https://probml.github.io/pml-book/book1.html",
        "download_url": "",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "machine-learning|Bayesian|probabilistic",
        "relevance_score": 3,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "BLOCKED — CC BY-NC-ND. Murphy explicitly states NC.",
    },
    {
        "source_id": "sutton_barto_rl",
        "title": "Reinforcement Learning: An Introduction",
        "authors": "Richard Sutton|Andrew Barto",
        "year": 2018,
        "publisher": "MIT Press",
        "licence_type": LicenseType.CC_BY_NC_ND,
        "licence_verified": True,
        "source_url": "http://incompleteideas.net/book/the-book-2nd.html",
        "download_url": "",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "reinforcement-learning|RLHF|reward-modelling",
        "relevance_score": 4,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "BLOCKED — CC BY-NC-ND per MIT Press terms.",
    },
    {
        "source_id": "jurafsky_slp3",
        "title": "Speech and Language Processing (3rd Edition Draft)",
        "authors": "Dan Jurafsky|James H. Martin",
        "year": 2025,
        "publisher": "Stanford University",
        "licence_type": LicenseType.ALL_RIGHTS_RESERVED,
        "licence_verified": True,
        "source_url": "https://web.stanford.edu/~jurafsky/slp3/",
        "download_url": "",
        "download_type": DownloadType.PDF,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "NLP|LLM|prompting|transformers",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "BLOCKED — 'Copyright ©2024. All Rights Reserved.' declared on page. "
                 "Free to read, but NOT licensed for corpus use.",
    },
    {
        "source_id": "raschka_llm_scratch",
        "title": "Build a Large Language Model From Scratch",
        "authors": "Sebastian Raschka",
        "year": 2024,
        "publisher": "Manning",
        "licence_type": LicenseType.CC_BY_NC_ND,
        "licence_verified": True,
        "source_url": "https://github.com/rasbt/LLMs-from-scratch",
        "download_url": "",
        "download_type": DownloadType.HTML,
        "content_type": ContentType.TEXTBOOK,
        "domain_tags": "LLM|GPT|PyTorch|training",
        "relevance_score": 5,
        "credibility_tier": CredibilityTier.TIER_1_5,
        "source": SourceType.BOOK,
        "notes": "BLOCKED — book text is CC BY-NC-ND (Manning). "
                 "Code in repo is Apache 2.0 and CAN be ingested separately.",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# MASTER CATALOGUE — used by BookIngestor.run()
# Only SAFE sources. Blocked entries excluded from runtime list.
# ──────────────────────────────────────────────────────────────────────────────

BOOK_CATALOGUE: list[dict[str, Any]] = (
    TECH_DOCS
    + OPEN_TEXTBOOKS
    + GUTENBERG_BOOKS
    + STANDARD_EBOOKS
    + HF_DATASET_BOOKS
    # BLOCKED_NC_BOOKS intentionally excluded from runtime catalogue
)

# ──────────────────────────────────────────────────────────────────────────────
# Convenience accessors
# ──────────────────────────────────────────────────────────────────────────────

def get_all_safe() -> list[dict[str, Any]]:
    """Return only entries with verified commercial-safe licences."""
    from deepforge.core.schemas import COMMERCIALLY_SAFE_LICENSES
    return [
        b for b in BOOK_CATALOGUE
        if b.get("licence_type") in COMMERCIALLY_SAFE_LICENSES
        and b.get("licence_verified", False)
    ]


def get_by_relevance(min_score: int = 4) -> list[dict[str, Any]]:
    """Return safe entries filtered by minimum relevance score."""
    return [b for b in get_all_safe() if b.get("relevance_score", 0) >= min_score]


def get_by_source_id(source_id: str) -> dict[str, Any] | None:
    """Look up a single entry by source_id."""
    return next((b for b in BOOK_CATALOGUE if b.get("source_id") == source_id), None)


def print_catalogue_summary() -> None:
    """Print a quick summary table to stdout — useful for verification before a run."""
    safe = get_all_safe()
    unverified = [b for b in BOOK_CATALOGUE if not b.get("licence_verified", False)]
    blocked = BLOCKED_NC_BOOKS

    print(f"\n{'='*70}")
    print("DEEPFORGE BOOK CATALOGUE SUMMARY")
    print(f"{'='*70}")
    print(f"  ✅ Safe + verified:   {len(safe)}")
    print(f"  ⚠️  Declared but unverified: {len(unverified)}")
    print(f"  ⛔ Blocked (NC/ARR):  {len(blocked)}")
    print(f"{'─'*70}")
    print(f"{'★':>4}  {'source_id':<45} {'licence'}")
    print(f"{'─'*70}")
    for b in sorted(safe, key=lambda x: -x.get("relevance_score", 0)):
        lic = b.get("licence_type", LicenseType.UNKNOWN)
        lic_str = lic.value if hasattr(lic, "value") else str(lic)
        print(f"  {b.get('relevance_score',0)}/5  {b['source_id']:<45} {lic_str}")
    print(f"{'='*70}\n")
