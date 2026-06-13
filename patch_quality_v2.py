"""
patch_quality_v2.py — SynthForge Quality Patch (remaining steps)
==================================================================
System prompt already hardened. This script applies the remaining 5 patches:

  1. Adds verify_answer_grounding() — LLM-as-Judge via llama-3.1-8b-instant
  2. Hooks judge into search pipeline after generate_answer()
  3. Adds _auto_search flag to session state init
  4. Suggestion buttons set _auto_search flag
  5. Search trigger fires on _auto_search OR button click

Run from C:\\Users\\Ezeking\\SynthForge:
  C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe patch_quality_v2.py
"""

import shutil
import sys
from pathlib import Path

APP_PATH   = Path(r"C:\Users\Ezeking\hf_space\app.py")
APP_BACKUP = Path(r"C:\Users\Ezeking\hf_space\app.py.bak_quality2")


def patch_once(content: str, find: str, replace: str, label: str) -> str:
    if find not in content:
        print(f"\n  x FAILED [{label}]")
        print(f"    Anchor: {find[:80]!r}")
        sys.exit(1)
    print(f"  ok {label}")
    return content.replace(find, replace, 1)


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

    Uses llama-3.1-8b-instant via Groq — fast, free tier.
    Appends a grounding warning if unsupported claims detected.
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

    # Top 5 chunks, 250 words each
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
        "Find specific factual claims in the answer NOT supported by the sources. "
        "Focus on: benchmark numbers, paper titles, percentages, techniques stated as fact.\\n\\n"
        "Reply with ONLY valid JSON:\\n"
        '{"grounded": true_or_false, "issues": ["unsupported claim 1", "etc"]}'
    )

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {groq_key}",
                     "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": 300,
                "temperature": 0.0,
            },
            timeout=12,
        )
        resp.raise_for_status()
        raw     = resp.json()["choices"][0]["message"]["content"].strip()
        raw     = raw.replace("```json", "").replace("```", "").strip()
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

    print(f"\nSynthForge Quality Patch v2 (remaining 5 patches)")
    print(f"Target  : {APP_PATH}\n")

    raw     = APP_PATH.read_bytes()
    content = raw.decode("utf-8").replace("\r\n", "\n")

    shutil.copy(APP_PATH, APP_BACKUP)
    print(f"Backup  : {APP_BACKUP.name}\n")
    print("System prompt: already hardened — skipping")
    print("\nApplying remaining patches:")

    # ------------------------------------------------------------------
    # Patch 1: Insert verify_answer_grounding() before fetch_hn_news()
    # ------------------------------------------------------------------
    if "def verify_answer_grounding(" in content:
        print("  ok verify_answer_grounding() already present — skipping")
    else:
        content = patch_once(
            content,
            "@st.cache_data(ttl=300)\ndef fetch_hn_news()",
            JUDGE_FUNCTION + "@st.cache_data(ttl=300)\ndef fetch_hn_news()",
            "verify_answer_grounding() inserted before fetch_hn_news()",
        )

    # ------------------------------------------------------------------
    # Patch 2: Hook judge into pipeline after generate_answer()
    # ------------------------------------------------------------------
    if "verify_answer_grounding(query.strip(), answer, results)" in content:
        print("  ok Judge already hooked into pipeline — skipping")
    else:
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

    # ------------------------------------------------------------------
    # Patch 3: Add _auto_search to session state init
    # ------------------------------------------------------------------
    if '"_auto_search"' in content:
        print("  ok _auto_search already in session state — skipping")
    else:
        content = patch_once(
            content,
            '    ("last_query", None), ("last_had_file", False),\n'
            ']:\n'
            '    if key not in st.session_state:\n'
            '        st.session_state[key] = default',
            '    ("last_query", None), ("last_had_file", False), ("_auto_search", False),\n'
            ']:\n'
            '    if key not in st.session_state:\n'
            '        st.session_state[key] = default',
            "_auto_search flag added to session state init",
        )

    # ------------------------------------------------------------------
    # Patch 4: Suggestion buttons set _auto_search flag
    # ------------------------------------------------------------------
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
    if '"_auto_search": True' in content:
        print("  ok Suggestion buttons already set _auto_search — skipping")
    else:
        content = patch_once(content, SUGG_OLD, SUGG_NEW,
                             "Suggestion buttons set _auto_search flag")

    # ------------------------------------------------------------------
    # Patch 5: Search trigger respects _auto_search
    # ------------------------------------------------------------------
    if "_auto_search = st.session_state.get" in content:
        print("  ok Search trigger already respects _auto_search — skipping")
    else:
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
    print(f"\nAll patches applied. {APP_PATH.name} updated.")
    print("\nDeploy:")
    print(r"  cd C:\Users\Ezeking\hf_space")
    print(r"  git add app.py")
    print(r'  git commit -m "Quality: LLM-as-Judge, auto-submit suggestions"')
    print(r"  git push")


if __name__ == "__main__":
    main()
