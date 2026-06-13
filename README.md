# SynthForge

**A prompt-engineering knowledge synthesis engine built on a zero-cost RAG stack.**

SynthForge ingests knowledge from arXiv papers, YouTube technical channels, Reddit practitioner communities, books, and RSS/forum sources, then answers questions about prompt engineering with sourced, synthesized responses — not search results. It is the first product of the planned **DeepForge** multi-domain platform, and every component runs on free-tier infrastructure.

**Live demo:** [huggingface.co/spaces/ezechinnabugwu/SynthForge](https://huggingface.co/spaces/ezechinnabugwu/SynthForge)

---

## Architecture

Six layers between a raw source and an answer:

```
Ingestion → Chunking & Embedding → Vector Store → Hybrid Retrieval → Reranking → Generation
```

1. **Ingestion** — source-specific ingestors (arXiv, YouTube transcripts, Reddit, books, RSS, Hacker News, LessWrong, Stack Overflow) with a licence gate enforced at the `BaseIngestor` level: non-commercial-licensed content is rejected before it can enter the corpus.
2. **Chunking & embedding** — sentence-aware chunking with SHA-256 chunk IDs, embedded locally with `bge-large-en-v1.5`. Deterministic IDs make every pipeline resume-safe: a power cut mid-run (a real constraint in Lagos) costs nothing but time.
3. **Vector store** — ChromaDB **1.5.8, pinned everywhere** (local, CI, deployment). Version drift between environments silently corrupts the corpus; the pin and a collection-integrity guard are the defense.
4. **Hybrid retrieval** — dense vector search + BM25 (`rank_bm25`) sparse retrieval, merged.
5. **Reranking** — cross-encoder scores retrieved chunks against the specific query before the LLM sees them.
6. **Generation** — Groq free-tier inference, with the generation prompt under active optimization via DSPy MIPROv2.

Current corpus: **21,500+ chunks** across all sources, growing via automated pipelines.

## Automation

All ingestion runs unattended on GitHub Actions:

- **arXiv** — daily workflow
- **Reddit** — weekly workflow (PRAW)
- **YouTube** — runs on a **self-hosted Oracle Cloud ARM runner**, because GitHub's datacenter IPs are blocked for transcript retrieval. The residential-adjacent runner is the architectural fix, not a workaround.

The vectorstore is versioned as a HuggingFace Dataset and synced to the Space on deploy.

## Evaluation

- 42,845-query golden evaluation corpus
- Four-metric runner: component coverage, ROUGE, BERTScore, and LLM-as-judge composite
- Stratified train/dev sets feeding DSPy MIPROv2 prompt optimization

## Why zero-cost matters

This project is a working reference for builders in markets where a GPU instance costs a month's income. Every constraint documented here — free-tier inference limits, embedding on CPU, outage-resilient checkpointing, self-hosted runners on Always Free cloud tiers — is solved in the open so it doesn't have to be solved twice.

## Stack

ChromaDB 1.5.8 · bge-large-en-v1.5 · rank_bm25 · cross-encoder reranking · DSPy + MIPROv2 · Groq · Streamlit · GitHub Actions · Oracle Cloud ARM · HuggingFace Spaces & Datasets

## Roadmap

- Reddit PRAW pipeline hardening
- MIPROv2-optimized generation prompt
- Frontier-model generation tier
- `ForgeCore`: extraction of the reusable engine powering future domain Forges

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Ezechinyere Nnabugwu
