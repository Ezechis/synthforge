"""
patch_quality.py — SynthForge Quality Gap Closure
====================================================
Six targeted edits to app.py that close the quality gap:

  1. Hardens the system prompt — confidence quantification, temporal
     awareness, grounding discipline, harder refusal protocol
  2. Adds verify_answer_grounding() — LLM-as-Judge using llama-3.1-8b-instant
  3. Hooks the judge into the search pipeline after generate_answer()
  4. Adds _auto_search flag to session state init
  5. Suggestion buttons set _auto_search flag
  6. Search trigger fires on _auto_search OR button click

Run from C:\\Users\\Ezeking\\SynthForge:
  C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe patch_quality.py
"""

import shutil
import sys
from pathlib import Path

APP_PATH   = Path(r"C:\Users\Ezeking\hf_space\app.py")
APP_BACKUP = Path(r"C:\Users\Ezeking\hf_space\app.py.bak_quality")


def patch_once(content: str, find: str, replace: str, label: str) -> str:
    if find not in content:
        print(f"\n  x FAILED [{label}]")
        print(f"    Anchor: {find[:80]!r}")
        sys.exit(1)
    print(f"  ok {label}")
    return content.replace(find, replace, 1)


# =============================================================================
# NEW SYSTEM PROMPT — replaces the current one in full
# =============================================================================

OLD_PROMPT_START = 'SYSTEM_PROMPT = """You are SynthForge'
OLD_PROMPT_END   = 'REFUSAL: If genuinely insufficient evidence, say so. Do not hallucinate."""'

NEW_SYSTEM_PROMPT = '''SYSTEM_PROMPT = """You are SynthForge — the most rigorously sourced synthesis engine for prompt engineering knowledge. Your corpus spans arXiv peer-reviewed papers, GitHub repositories, official documentation (Anthropic, OpenAI, LangChain, DSPy), Reddit practitioner discussions, and YouTube transcripts.

CORE DIRECTIVE: Every factual claim must trace directly to retrieved context. Never fabricate citations. Never synthesise beyond what the evidence supports. Uncertainty must be named, never hidden.

SOURCE HIERARCHY (enforce in every answer):
- Tier 1 (Primary): arXiv peer-reviewed papers — highest epistemic weight
- Tier 2 (Empirical): GitHub implementations, official docs — validated practice
- Tier 3 (Signal): Reddit, YouTube — practitioner insight, explicitly discounted
- USER DOCUMENT: Highest priority for document-specific questions only

MANDATORY ANSWER STRUCTURE:
1. TECHNIQUE DEFINITION — precise, one paragraph, no claims beyond retrieved context
2. THEORETICAL BASIS — mechanistic explanation grounded in Tier 1 or Tier 2 sources
3. EMPIRICAL EVIDENCE — specific benchmarks, numbers, years, and paper names as retrieved
4. IMPLEMENTATION EXAMPLE — concrete pseudocode or numbered steps from retrieved sources
5. KNOWN FAILURE MODES — when, why, and under what conditions it breaks, sourced explicitly
6. TEMPORAL NOTE — flag pre-2024 evidence; state whether 2025-2026 practice confirms or updates it
7. SOURCE CITATIONS — every source used, with credibility tier and publication year

CONFIDENCE QUANTIFICATION (mandatory label on every empirical claim):
- [WELL-ESTABLISHED]: Replicated across 5+ independent sources or official documentation consensus
- [EMERGING]: Supported by 2-4 studies; promising but context-dependent
- [SPECULATIVE]: Single study, community observation, or pre-2023 finding not yet replicated

GROUNDING DISCIPLINE:
- If fewer than 2 Tier 1 sources support a claim: flag [LIMITED EVIDENCE — single source]
- If the corpus has no evidence for a query component: state [CORPUS GAP — insufficient evidence]
- Never resolve contradictions silently — present both positions, note recency and strength
- Tag pre-2024 findings with [Pre-2024] — never present outdated practice as current

ABSOLUTE PROHIBITIONS:
- No invented benchmark numbers or fabricated paper titles
- No confident synthesis beyond retrieved context
- No Tier 3 claims without explicit [Community Signal — unverified] label
- No answer without at least one explicit citation per major claim"""'''


# =============================================================================
# JUDGE FUNCTION — inserted before fetch_hn_news()
# =============================================================================

JUDGE_FUNCTION = '''
def verify_answer_grounding(
    query: str,
    answer: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    LLM-as-Judge: verify the generated answer is grounded in retrieved chunks.

    Uses llama-3.1-8b-instant via Groq — fast, cheap, on free tier.
    Appends a grounding warning if unsupported claims are detected.
    Never blocks — original answer returned on any failure.

    Args:
        query: Original user query.
        answer: Generated answer to verify.
        retrieved_chunks: Chunks used during generation.

    Returns:
        Answer string, optionally annotated with a grounding warning.
    """
    import json as _json

    groq_key = "".join(os.environ.get("GROQ_API_KEY", "").split())
    if not groq_key or not answer.strip():
        return answer

    # Top 5 chunks, 250 words each — compact context for the judge
    context_parts: list[str] = []
    for i, chunk in enumerate(retrieved_chunks[:5], 1):
        meta  = chunk.get("metadata", {})
        src   = meta.get("source", "unknown").upper()
        words = chunk.get("text", "").split()[:250]
        context_parts.append(f"[SOURCE {i}: {src}]\\n{' '.join(words)}")
    context = "\\n\\n".join(context_parts)

    judge_prompt = (
        "You are a hallucination detector for a RAG system about prompt engineering.\\n\\n"
        f"RETRIEVED SOURCES:\\n{context}\\n\\n"
        f"ANSWER TO CHECK (first 500 words):\\n{' '.join(answer.split()[:500])}\\n\\n"
        "Find specific factual claims in the answer NOT supported by the sources above. "
        "Focus on: benchmark numbers, paper titles, percentages, named techniques stated as fact.\\n\\n"
        "Reply with ONLY valid JSON, nothing else:\\n"
        '{"grounded": true_or_false, "issues": ["unsupported claim 1", "etc — empty list if none"]}'
    )

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": 300,
                "temperature": 0.0,
            },
            timeout=12,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result  = _json.loads(raw)
        issues  = [str(i).strip() for i in result.get("issues", []) if str(i).strip()]
        grounded = result.get("grounded", True)

        if not grounded and issues:
            issue_text = "; ".join(f"*{iss}*" for iss in issues[:3])
            warning = (
                "\\n\\n---\\n"
                "\u26a0\ufe0f **Grounding Verification:** The following claims could not be "
                "fully verified against retrieved sources and may require independent "
                "confirmation: " + issue_text
            )
            logger.info("Judge flagged %d unsupported claim(s).", len(issues))
            return answer + warning

        logger.debug("Judge: answer fully grounded.")
        return answer

    except Exception as exc:
        logger.debug("Judge skipped (non-blocking): %s", exc)
        return answer

'''


def main() -> None:
    if not APP_PATH.exists():
        print(f"ERROR: {APP_PATH} not found.")
        sys.exit(1)

    print(f"\nSynthForge Quality Gap Closure Patch")
    print(f"Target  : {APP_PATH}\n")

    raw     = APP_PATH.read_bytes()
    content = raw.decode("utf-8").replace("\r\n", "\n")

    shutil.copy(APP_PATH, APP_BACKUP)
    print(f"Backup  : {APP_BACKUP.name}\n")
    print("Applying patches:")

    # ------------------------------------------------------------------ #
    # Patch 1: Replace system prompt                                       #
    # ------------------------------------------------------------------ #
    # Find the start and end of the full SYSTEM_PROMPT block
    start_idx = content.find(OLD_PROMPT_START)
    end_idx   = content.find(OLD_PROMPT_END)

    if start_idx == -1 or end_idx == -1:
        print(f"\n  x FAILED [System prompt] — anchor not found")
        print(f"    Start found: {start_idx != -1} | End found: {end_idx != -1}")
        sys.exit(1)

    end_idx += len(OLD_PROMPT_END)
    content = content[:start_idx] + NEW_SYSTEM_PROMPT + content[end_idx:]
    print("  ok System prompt hardened (confidence, temporal, grounding)")

    # ------------------------------------------------------------------ #
    # Patch 2: Insert verify_answer_grounding() before fetch_hn_news()    #
    # ------------------------------------------------------------------ #
    content = patch_once(
        content,
        "@st.cache_data(ttl=300)\ndef fetch_hn_news()",
        JUDGE_FUNCTION + "@st.cache_data(ttl=300)\ndef fetch_hn_news()",
        "verify_answer_grounding() inserted",
    )

    # ------------------------------------------------------------------ #
    # Patch 3: Hook judge after generate_answer() in search pipeline       #
    # ------------------------------------------------------------------ #
    content = patch_once(
        content,
        "        answer   = generate_answer(query.strip(), results, file_context)\n"
        "\n"
        "    elapsed  = time.time() - t0",
        "        answer   = generate_answer(query.strip(), results, file_context)\n"
        "        answer   = verify_answer_grounding(query.strip(), answer, results)\n"
        "\n"
        "    elapsed  = time.time() - t0",
        "Judge hooked into search pipeline",
    )

    # ------------------------------------------------------------------ #
    # Patch 4: Add _auto_search to session state init                      #
    # ------------------------------------------------------------------ #
    content = patch_once(
        content,
        '    ("last_query", None), ("last_had_file", False),\n]:\n'
        '    if key not in st.session_state:\n'
        '        st.session_state[key] = default',
        '    ("last_query", None), ("last_had_file", False), ("_auto_search", False),\n]:\n'
        '    if key not in st.session_state:\n'
        '        st.session_state[key] = default',
        "_auto_search flag added to session init",
    )

    # ------------------------------------------------------------------ #
    # Patch 5: Suggestion buttons set _auto_search                         #
    # ------------------------------------------------------------------ #
    SUGG_OLD = (
        '                if st.button(f"\u2192 {s}", key=f"sg_{i}", use_container_width=True):\n'
        '                    st.session_state.update({"query_input": s, "suggestions": []})\n'
        '                    st.rerun()'
    )
    SUGG_NEW = (
        '                if st.button(f"\u2192 {s}", key=f"sg_{i}", use_container_width=True):\n'
        '                    st.session_state.update({"query_input": s, "suggestions": [], "_auto_search": True})\n'
        '                    st.rerun()'
    )
    content = patch_once(content, SUGG_OLD, SUGG_NEW, "Suggestion buttons set _auto_search flag")

    # ------------------------------------------------------------------ #
    # Patch 6: Search trigger fires on _auto_search OR button click        #
    # ------------------------------------------------------------------ #
    content = patch_once(
        content,
        "if search_clicked and query.strip():",
        '_auto_search = st.session_state.get("_auto_search", False)\n'
        'if _auto_search:\n'
        '    st.session_state["_auto_search"] = False\n'
        'if (search_clicked or _auto_search) and query.strip():',
        "Search trigger respects _auto_search flag",
    )

    # Restore CRLF if original had it
    if b"\r\n" in raw:
        content = content.replace("\n", "\r\n")

    APP_PATH.write_bytes(content.encode("utf-8"))
    print(f"\nAll 6 patches applied. {APP_PATH.name} updated.")
    print("\nDeploy:")
    print(r"  cd C:\Users\Ezeking\hf_space")
    print(r"  git add app.py")
    print(r'  git commit -m "Quality: LLM-as-Judge, hardened system prompt, auto-submit suggestions"')
    print(r"  git push")


if __name__ == "__main__":
    main()
