"""
src/eval/run_live_baseline.py
=============================
Live baseline quality eval for SynthForge — judge-scored, per-cluster.

Pipeline per query:  retrieve  ->  generate (Groq llama-3.3-70b-versatile)
                     ->  judge   (Groq llama-3.1-8b-instant, 1-10 vs rubric)

Design contract (do not drift):
  * Generation is called ONLY through the existing public entrypoint
    `src.generation.generate.generate_answer(query, retriever)`. The prompt
    internals (the SynthForge system prompt / Source Confidence logic) are
    owned and actively edited elsewhere; this harness never re-implements or
    imports them.
  * The corpus is READ-ONLY. We read the live ChromaDB collection `synthforge`
    via SynthForgeRetriever; we never write to the vectorstore or the HF dataset.
  * The golden eval set is a PRIVATE HF dataset and is NEVER committed to the
    repo. It is pulled at runtime via HF_TOKEN into `data/` (which is
    .gitignored) and read from there.

Golden record shape (JSONL):
    {id, query, expected_components, tier1_required, category, difficulty, domain}
  `expected_components` is the scoring rubric (a list of required ideas/terms).

Run modes (token-budget aware):
  * sample (default) — stratified sample, every normalized category represented,
    ~EVAL_SAMPLE_SIZE records total. Use this for the first baseline.
  * full            — larger stratified draw (EVAL_SAMPLE_SIZE still caps total);
                      schedule OFF-PEAK, after the 01:00 WAT Groq quota reset.

Outputs (written under data/evals/results/, .gitignored; uploaded as CI artifact):
  * eval_live_baseline_<ts>.csv     — one row per scored query
  * eval_live_baseline_<ts>.json    — full structured summary
  * eval_live_baseline_<ts>.md      — human-readable per-cluster table
  * latest_live_baseline.{csv,json,md}

SANITY GUARD: a Groq judge that silently falls back to a flat score (the classic
symptom of an invalid/missing GROQ_API_KEY) must NOT be reported as a baseline.
This harness (a) pings the judge before the run and aborts on auth failure, and
(b) never substitutes a default score on judge failure — failed judges are
recorded as None and excluded, and an excessive judge-failure rate or a
degenerate zero-variance distribution aborts the run with a non-zero exit.

Environment variables:
    GROQ_API_KEY        -- required; generation + judge
    HF_TOKEN            -- required; pull the private golden dataset (and vectorstore)
    GOLDEN_DATASET_REPO -- default: ezechinnabugwu/synthforge-golden-evals
    GOLDEN_DATASET_FILE -- default: synthforge_corpus_500k.jsonl
    EVAL_MODE           -- "sample" (default) | "full"
    EVAL_SAMPLE_SIZE    -- target total records (default: 300)
    EVAL_SEED           -- RNG seed for reproducible sampling (default: 42)
    EVAL_JUDGE_QPS_SLEEP-- seconds to sleep between queries (default: 3.0)
    VECTOR_STORE_PATH   -- informational; retriever uses config.settings.VECTOR_STORE_DIR

Author: Ezechinyere Nnabugwu / DeepForge
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

# Repo root on path so `config.*` and `src.*` imports resolve when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live_baseline")

# ---------------------------------------------------------------------------
# Constants  (named, with source comments — never hardcode inline)
# ---------------------------------------------------------------------------

GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
# Judge model: small + high RPM on Groq free tier. Generation model is owned by
# config.settings.GROQ_MODEL (llama-3.3-70b-versatile) and reached via generate_answer().
JUDGE_MODEL: str = "llama-3.1-8b-instant"

GOLDEN_DATASET_REPO: str = os.environ.get(
    "GOLDEN_DATASET_REPO", "ezechinnabugwu/synthforge-golden-evals"
)
GOLDEN_DATASET_FILE: str = os.environ.get(
    "GOLDEN_DATASET_FILE", "synthforge_corpus_500k.jsonl"
)

# Pulled golden data lands under data/ which is .gitignored -> never committed.
GOLDEN_LOCAL_DIR: Path = REPO_ROOT / "data" / "evals" / "_remote"
RESULTS_DIR: Path = REPO_ROOT / "data" / "evals" / "results"

EVAL_MODE: str = os.environ.get("EVAL_MODE", "sample").strip().lower()
EVAL_SAMPLE_SIZE: int = int(os.environ.get("EVAL_SAMPLE_SIZE", "300"))
EVAL_SEED: int = int(os.environ.get("EVAL_SEED", "42"))
# Courtesy pause between queries: each query = 1 x 70b generate + 1 x 8b judge.
JUDGE_QPS_SLEEP: float = float(os.environ.get("EVAL_JUDGE_QPS_SLEEP", "3.0"))

# Sanity-guard thresholds.
MAX_JUDGE_FAILURE_RATE: float = 0.20   # abort if >20% of judge calls fail
SUSPICIOUS_FLAT_SCORE: float = 7.0     # the classic invalid-key fallback value

# Canonical cluster map. Keys are de-punctuated, lowercased category strings
# (so "task_specific" and "taskspecific" collapse to the same key); values are
# the canonical display names. Anything not listed groups by its de-punctuated
# key and is logged, so unseen duplicate pairs still collapse together.
CANONICAL_CLUSTERS: dict[str, str] = {
    "coretechnique": "core_technique",
    "taskspecific": "task_specific",
    "structuredoutput": "structured_output",
    "contextmanagement": "context_management",
    "modelspecific": "model_specific",
    "modelarchitecture": "model_architecture",
    "modelscale": "model_scale",
    "failuremodes": "failure_modes",
    "bestpractices": "best_practices",
    "prompttuning": "prompt_tuning",
    "promptpatterns": "prompt_patterns",
    "multimodalvoice": "multimodal_voice",
    "hitlux": "hitl_ux",
    "edgecomputing": "edge_computing",
    "futureparadigms": "future_paradigms",
    "finetuning": "fine_tuning",
    "costoptimization": "cost_optimization",
}


# ---------------------------------------------------------------------------
# Golden set: pull (private HF) -> load -> validate -> normalize
# ---------------------------------------------------------------------------

def download_golden_set() -> Path:
    """Pull the private golden eval dataset from HF into a .gitignored dir.

    Returns:
        Local path to the downloaded golden JSONL file.

    Raises:
        SystemExit: if HF_TOKEN is missing or the download fails.
    """
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        logger.error("HF_TOKEN missing — cannot pull private golden dataset. Aborting.")
        raise SystemExit(2)

    GOLDEN_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=GOLDEN_DATASET_REPO,
            filename=GOLDEN_DATASET_FILE,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(GOLDEN_LOCAL_DIR),
        )
        logger.info("Golden set pulled: %s/%s", GOLDEN_DATASET_REPO, GOLDEN_DATASET_FILE)
        return Path(local_path)
    except Exception as exc:  # noqa: BLE001 - surface any HF failure clearly
        logger.error("Failed to pull golden dataset %s: %s", GOLDEN_DATASET_REPO, exc)
        raise SystemExit(2) from exc


def normalize_category(raw: Any) -> str:
    """Collapse category-label variants to a canonical cluster name.

    De-punctuates and lowercases so 'task_specific' / 'taskspecific' / 'Task Specific'
    all map to one key, then returns the canonical display name.

    Args:
        raw: Raw category value from a golden record (may be missing/non-string).

    Returns:
        Canonical cluster name, or 'uncategorized' if absent.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "uncategorized"
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    if not key:
        return "uncategorized"
    return CANONICAL_CLUSTERS.get(key, key)


def is_valid_record(rec: dict) -> bool:
    """A record is usable only if it has a query and a non-empty rubric.

    Args:
        rec: Parsed golden record.

    Returns:
        True if the record can be scored.
    """
    query = rec.get("query")
    rubric = rec.get("expected_components")
    if not isinstance(query, str) or not query.strip():
        return False
    if not isinstance(rubric, list) or not any(str(c).strip() for c in rubric):
        return False
    return True


def load_and_clean(golden_path: Path) -> list[dict]:
    """Stream the golden JSONL, keep valid records, normalize categories, dedupe ids.

    Args:
        golden_path: Local path to the golden JSONL file.

    Returns:
        List of cleaned records (category replaced with its canonical name;
        original kept under '_raw_category').
    """
    seen_ids: set[str] = set()
    cleaned: list[dict] = []
    total = malformed = invalid = dupes = 0

    with open(golden_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not is_valid_record(rec):
                invalid += 1
                continue
            rid = str(rec.get("id", f"AUTO-{total}"))
            if rid in seen_ids:
                dupes += 1
                continue
            seen_ids.add(rid)
            rec["_raw_category"] = rec.get("category", "")
            rec["category"] = normalize_category(rec.get("category"))
            rec["id"] = rid
            cleaned.append(rec)

    logger.info(
        "Golden load: %d lines | kept %d | malformed %d | invalid %d | dupes %d",
        total, len(cleaned), malformed, invalid, dupes,
    )
    return cleaned


# ---------------------------------------------------------------------------
# Stratified sampling — every normalized category represented
# ---------------------------------------------------------------------------

def stratified_sample(records: list[dict], target_total: int, seed: int) -> list[dict]:
    """Draw a per-category stratified sample with every category represented.

    Allocation: give each non-empty category a floor of 1, then distribute the
    remaining budget proportionally to category size. Per-category draw is
    capped at availability. Deterministic given `seed`.

    Args:
        records: Cleaned golden records (with canonical 'category').
        target_total: Desired total sample size.
        seed: RNG seed for reproducibility.

    Returns:
        Sampled records (shuffled deterministically).
    """
    import random

    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_cat[rec["category"]].append(rec)

    categories = sorted(by_cat.keys())
    n_cats = len(categories)
    if target_total >= len(records) or n_cats == 0:
        logger.info("Target >= corpus or no categories — using all %d records.", len(records))
        out = list(records)
        rng.shuffle(out)
        return out

    # Floor of 1 per category (bounded by availability).
    alloc: dict[str, int] = {c: min(1, len(by_cat[c])) for c in categories}
    used = sum(alloc.values())
    remaining = max(0, target_total - used)

    # Proportional distribution of the remainder by category size.
    total_size = sum(len(by_cat[c]) for c in categories)
    if remaining > 0 and total_size > 0:
        for c in categories:
            extra = int(remaining * (len(by_cat[c]) / total_size))
            alloc[c] = min(len(by_cat[c]), alloc[c] + extra)

    # Top up any rounding shortfall, largest categories first.
    def shortfall() -> int:
        return target_total - sum(alloc.values())

    for c in sorted(categories, key=lambda x: len(by_cat[x]), reverse=True):
        if shortfall() <= 0:
            break
        room = len(by_cat[c]) - alloc[c]
        if room > 0:
            alloc[c] += min(room, shortfall())

    sample: list[dict] = []
    for c in categories:
        pool = list(by_cat[c])
        rng.shuffle(pool)
        sample.extend(pool[: alloc[c]])

    rng.shuffle(sample)
    logger.info(
        "Stratified sample: %d records across %d categories (target %d).",
        len(sample), n_cats, target_total,
    )
    return sample


# ---------------------------------------------------------------------------
# Judge — Groq llama-3.1-8b-instant, score answer vs rubric on 1-10
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a prompt-engineering RAG system. "
    "You score how well an ANSWER satisfies a RUBRIC of expected components for "
    "a QUERY. Reward coverage of the rubric ideas, factual grounding, correct "
    "citations, and clarity; penalize missing rubric items, hallucinated facts, "
    "and vagueness. Output ONLY JSON."
)

JUDGE_USER_TEMPLATE = (
    "QUERY:\n{query}\n\n"
    "RUBRIC (expected components — each should be addressed):\n{rubric}\n\n"
    "ANSWER TO SCORE:\n{answer}\n\n"
    "Score the answer from 1 (fails the rubric / unusable) to 10 (fully covers the "
    "rubric, well grounded, correctly cited). Reply with ONLY valid JSON:\n"
    '{{"score": <integer 1-10>, "components_covered": ["..."], "rationale": "<one sentence>"}}'
)


def _post_groq(payload: dict, timeout: int = 30) -> Optional[dict]:
    """POST to Groq with light retry on 429. Returns parsed JSON or None on failure."""
    groq_key = "".join(os.environ.get("GROQ_API_KEY", "").split())
    if not groq_key:
        return None
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429 and attempt < 2:
                wait = 15 * (2 ** attempt)
                logger.warning("Judge rate limited (429); waiting %ds (retry %d/3).", wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            logger.error("Judge HTTP error: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Judge call failed: %s", exc)
            return None
    return None


def judge_ping() -> bool:
    """Pre-flight: confirm the Groq judge is reachable and authenticated.

    Returns:
        True if the judge answered with a parseable score; False otherwise.
        A False here means we must NOT trust any downstream scores.
    """
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                query="What is 2+2?",
                rubric="- the number four",
                answer="The answer is four (4).",
            )},
        ],
        "max_tokens": 120,
        "temperature": 0.0,
    }
    data = _post_groq(payload)
    if not data:
        return False
    score, _, _ = _parse_judge(data)
    return score is not None


def _parse_judge(data: dict) -> tuple[Optional[int], list[str], str]:
    """Extract (score, components_covered, rationale) from a Groq judge response."""
    try:
        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Tolerate prose around the JSON object.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(match.group(0) if match else raw)
        score = obj.get("score")
        score = int(round(float(score))) if score is not None else None
        if score is not None:
            score = max(1, min(10, score))
        covered = [str(c).strip() for c in obj.get("components_covered", []) if str(c).strip()]
        rationale = str(obj.get("rationale", "")).strip()
        return score, covered, rationale
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse judge response: %s", exc)
        return None, [], ""


def judge_answer(query: str, answer: str, rubric: list[str]) -> tuple[Optional[int], list[str], str]:
    """Score an answer against its rubric via the Groq judge.

    Returns:
        (score|None, components_covered, rationale). None score = judge failed
        and the row is EXCLUDED from the mean (never defaulted to a fixed value).
    """
    if not answer.strip():
        return None, [], "empty answer"
    rubric_str = "\n".join(f"- {str(c).strip()}" for c in rubric if str(c).strip())
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                query=query,
                rubric=rubric_str,
                answer=" ".join(answer.split()[:600]),
            )},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    data = _post_groq(payload)
    if not data:
        return None, [], "judge call failed"
    return _parse_judge(data)


def has_tier1_source(chunks: list[dict]) -> bool:
    """True if any retrieved chunk is a Tier-1 (arxiv / primary) source."""
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        if meta.get("source", "") == "arxiv" or meta.get("credibility_tier", "") == "primary":
            return True
    return False


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run_eval(records: list[dict]) -> dict:
    """Retrieve -> generate -> judge each record; aggregate per-cluster.

    Args:
        records: Sampled, cleaned golden records.

    Returns:
        Summary dict with overall + per-cluster means/variance and per-query rows.

    Raises:
        SystemExit: on the sanity guards (excessive judge failures / degenerate flat scores).
    """
    # Imported here so a missing-deps / import error surfaces only when we actually run.
    from src.generation.generate import generate_answer
    from src.retrieval.hybrid_retrieval import SynthForgeRetriever

    logger.info("Initialising retriever (read-only) over collection 'synthforge'...")
    retriever = SynthForgeRetriever()
    corpus_size = retriever.collection.count()
    logger.info("Corpus size: %d chunks.", corpus_size)

    rows: list[dict] = []
    judge_failures = 0
    answered = 0

    for i, rec in enumerate(records, 1):
        rid = rec["id"]
        query = rec["query"]
        rubric = rec["expected_components"]
        category = rec["category"]

        logger.info("[%d/%d] %s (%s) — %s", i, len(records), rid, category, query[:60])

        # Retrieve (read-only) for the tier-1 signal; generation re-retrieves internally.
        try:
            chunks = retriever.retrieve(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("Retrieval failed for %s: %s", rid, exc)
            chunks = []
        t1_present = has_tier1_source(chunks) if chunks else False

        # Generate via the existing public entrypoint (llama-3.3-70b-versatile).
        answer = generate_answer(query, retriever)
        is_answer = bool(answer) and not answer.startswith("Generation error:")
        if is_answer:
            answered += 1

        # Judge vs rubric (1-10). None => excluded from means.
        score, covered, rationale = (None, [], "")
        if is_answer:
            score, covered, rationale = judge_answer(query, answer, rubric)
            if score is None:
                judge_failures += 1

        rows.append({
            "id": rid,
            "category": category,
            "raw_category": rec.get("_raw_category", ""),
            "domain": rec.get("domain", ""),
            "difficulty": rec.get("difficulty", ""),
            "tier1_required": bool(rec.get("tier1_required", False)),
            "tier1_present": t1_present,
            "chunks_retrieved": len(chunks),
            "answered": is_answer,
            "score": score,
            "components_total": len([c for c in rubric if str(c).strip()]),
            "components_covered": len(covered),
            "rationale": rationale,
        })
        logger.info("  score=%s | t1=%s | chunks=%d", score, t1_present, len(chunks))
        time.sleep(JUDGE_QPS_SLEEP)

    # ---- Sanity guards -------------------------------------------------
    n_judged = sum(1 for r in rows if r["score"] is not None)
    n_attempted_judge = sum(1 for r in rows if r["answered"])
    if n_attempted_judge:
        fail_rate = judge_failures / n_attempted_judge
        if fail_rate > MAX_JUDGE_FAILURE_RATE:
            logger.error(
                "SANITY ABORT: judge failure rate %.0f%% > %.0f%% — likely invalid GROQ_API_KEY "
                "or judge outage. Not reporting a baseline.",
                fail_rate * 100, MAX_JUDGE_FAILURE_RATE * 100,
            )
            raise SystemExit(3)

    scores = [r["score"] for r in rows if r["score"] is not None]
    if not scores:
        logger.error("SANITY ABORT: zero valid judge scores — nothing to report.")
        raise SystemExit(3)

    overall_mean = statistics.mean(scores)
    overall_var = statistics.pvariance(scores) if len(scores) > 1 else 0.0
    if overall_var == 0.0 and abs(overall_mean - SUSPICIOUS_FLAT_SCORE) < 1e-9:
        logger.error(
            "SANITY ABORT: every judge score is a flat %.1f (zero variance). This is the "
            "signature of a fallback/invalid key, not a real baseline.",
            SUSPICIOUS_FLAT_SCORE,
        )
        raise SystemExit(3)

    # ---- Per-cluster aggregation ---------------------------------------
    per_cat: dict[str, list[int]] = defaultdict(list)
    per_cat_n: dict[str, int] = defaultdict(int)
    for r in rows:
        per_cat_n[r["category"]] += 1
        if r["score"] is not None:
            per_cat[r["category"]].append(r["score"])

    cluster_table = []
    for cat in sorted(per_cat_n.keys()):
        cat_scores = per_cat[cat]
        cluster_table.append({
            "category": cat,
            "n_sampled": per_cat_n[cat],
            "n_scored": len(cat_scores),
            "mean_score": round(statistics.mean(cat_scores), 3) if cat_scores else None,
            "variance": round(statistics.pvariance(cat_scores), 3) if len(cat_scores) > 1 else 0.0,
            "stdev": round(statistics.pstdev(cat_scores), 3) if len(cat_scores) > 1 else 0.0,
        })

    summary = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "judge_model": JUDGE_MODEL,
        "generation_model": _generation_model_name(),
        "eval_mode": EVAL_MODE,
        "corpus_size": corpus_size,
        "queries_sampled": len(rows),
        "queries_answered": answered,
        "answer_rate": round(answered / len(rows), 3) if rows else 0.0,
        "queries_judged": n_judged,
        "judge_failures": judge_failures,
        "overall_mean_score": round(overall_mean, 3),
        "overall_variance": round(overall_var, 3),
        "overall_stdev": round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
        "score_scale": "1-10 (judge vs expected_components rubric)",
        "tier1_coverage_rate": round(
            sum(1 for r in rows if r["tier1_present"]) / len(rows), 3) if rows else 0.0,
        "per_cluster": cluster_table,
        "per_query": rows,
    }
    return summary


def _generation_model_name() -> str:
    """Report the configured generation model without importing prompt internals."""
    try:
        from config.settings import GROQ_MODEL
        return GROQ_MODEL
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_reports(summary: dict) -> dict[str, Path]:
    """Write CSV + JSON + Markdown reports (and 'latest_*' copies).

    Returns:
        Mapping of report kind -> path written.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    csv_path = RESULTS_DIR / f"eval_live_baseline_{ts}.csv"
    json_path = RESULTS_DIR / f"eval_live_baseline_{ts}.json"
    md_path = RESULTS_DIR / f"eval_live_baseline_{ts}.md"

    # Per-query CSV
    fields = [
        "id", "category", "raw_category", "domain", "difficulty",
        "tier1_required", "tier1_present", "chunks_retrieved",
        "answered", "score", "components_total", "components_covered", "rationale",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in summary["per_query"]:
            writer.writerow(row)

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")

    # Stable 'latest' copies for quick CI access.
    (RESULTS_DIR / "latest_live_baseline.csv").write_text(
        csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    (RESULTS_DIR / "latest_live_baseline.json").write_text(
        json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (RESULTS_DIR / "latest_live_baseline.md").write_text(
        md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {"csv": csv_path, "json": json_path, "md": md_path}


def _render_markdown(summary: dict) -> str:
    """Render a human-readable per-cluster baseline table."""
    lines = [
        "# SynthForge — Live Baseline (judge-scored)",
        "",
        f"- Run (UTC): `{summary['run_timestamp']}`",
        f"- Mode: `{summary['eval_mode']}`",
        f"- Generation model: `{summary['generation_model']}`",
        f"- Judge model: `{summary['judge_model']}`  ·  Scale: {summary['score_scale']}",
        f"- Corpus size: {summary['corpus_size']:,} chunks",
        f"- Queries sampled / answered / judged: "
        f"{summary['queries_sampled']} / {summary['queries_answered']} / {summary['queries_judged']}",
        f"- Judge failures (excluded): {summary['judge_failures']}",
        "",
        f"## Overall: mean **{summary['overall_mean_score']}/10** "
        f"(variance {summary['overall_variance']}, stdev {summary['overall_stdev']})",
        f"Answer rate {summary['answer_rate']*100:.1f}% · "
        f"Tier-1 coverage {summary['tier1_coverage_rate']*100:.1f}%",
        "",
        "## Per-cluster",
        "",
        "| Cluster | n sampled | n scored | mean /10 | variance | stdev |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(summary["per_cluster"], key=lambda x: (x["mean_score"] is None, x["mean_score"] or 0)):
        mean = "—" if c["mean_score"] is None else f"{c['mean_score']:.2f}"
        lines.append(
            f"| {c['category']} | {c['n_sampled']} | {c['n_scored']} | "
            f"{mean} | {c['variance']} | {c['stdev']} |"
        )
    lines.append("")
    lines.append(
        "_Clusters are ordered weakest-mean first to surface where the book-heavy "
        "corpus is thin. Low `n scored` with low mean = sparse coverage, not just low quality._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Pull golden -> clean -> stratified-sample -> retrieve/generate/judge -> report."""
    if not os.environ.get("GROQ_API_KEY", "").strip():
        logger.error("GROQ_API_KEY missing — generation and judge both require it. Aborting.")
        raise SystemExit(2)

    # Pre-flight: judge must be reachable, or scores are meaningless.
    logger.info("Pinging judge (%s) to validate GROQ_API_KEY...", JUDGE_MODEL)
    if not judge_ping():
        logger.error(
            "SANITY ABORT: judge pre-flight failed. GROQ_API_KEY is likely invalid or the "
            "judge model is unreachable. Refusing to produce a (fake) baseline."
        )
        raise SystemExit(3)
    logger.info("Judge reachable. Proceeding.")

    golden_path = download_golden_set()
    records = load_and_clean(golden_path)
    if not records:
        logger.error("No valid golden records after cleaning. Aborting.")
        raise SystemExit(2)

    target = EVAL_SAMPLE_SIZE
    if EVAL_MODE == "full":
        # 'full' is still budget-capped; raise the ceiling but never run the raw 500k.
        target = max(EVAL_SAMPLE_SIZE, int(os.environ.get("EVAL_FULL_SIZE", "3000")))
        logger.info("FULL mode — target %d (run OFF-PEAK after 01:00 WAT Groq reset).", target)

    sample = stratified_sample(records, target_total=target, seed=EVAL_SEED)

    # Report per-cluster n of the sample up-front (token-budget transparency).
    cat_n: dict[str, int] = defaultdict(int)
    for r in sample:
        cat_n[r["category"]] += 1
    logger.info("Sample per-cluster n: %s", dict(sorted(cat_n.items())))

    summary = run_eval(sample)
    paths = write_reports(summary)

    logger.info("=" * 60)
    logger.info("LIVE BASELINE COMPLETE")
    logger.info("  Overall mean: %.3f/10 (var %.3f)",
                summary["overall_mean_score"], summary["overall_variance"])
    logger.info("  Reports: %s", {k: str(v) for k, v in paths.items()})
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
