# Adds persistent search history to PromptForge HF Space UI.
# Three patches to hf_space/app.py:
#   1. Initialize history list in session state
#   2. Append each query+answer to history after generation
#   3. Render collapsible history panel at bottom of page

import sys
from pathlib import Path

APP = Path(r"C:\Users\Ezeking\hf_space\app.py")


def patch(old, new, label):
    content = APP.read_text(encoding="utf-8")
    count = content.count(old)
    if count == 0:
        print(f"ERROR [{label}]: string not found"); sys.exit(1)
    if count > 1:
        print(f"ERROR [{label}]: found {count} times"); sys.exit(1)
    APP.write_text(content.replace(old, new, 1), encoding="utf-8")
    print(f"OK    [{label}]")


# ── Patch 1: initialise history in session state after resources load ─────────
patch(
    '    collection, embed_model, reranker, bm25, corpus_chunks, corpus_metas = load_all_resources()',
    '    collection, embed_model, reranker, bm25, corpus_chunks, corpus_metas = load_all_resources()\n\n'
    'if "search_history" not in st.session_state:\n'
    '    st.session_state["search_history"] = []',
    "Patch 1 history init"
)

# ── Patch 2: append query+answer to history after generation ──────────────────
patch(
    '    st.markdown(answer)\n\n    # Sources',
    '    st.markdown(answer)\n\n'
    '    # Store in session history\n'
    '    st.session_state["search_history"].append({\n'
    '        "query": query.strip(),\n'
    '        "answer": answer,\n'
    '        "elapsed": f"{elapsed:.1f}s",\n'
    '        "chunks": len(results),\n'
    '        "arxiv": arxiv_count,\n'
    '        "timestamp": time.strftime("%H:%M:%S"),\n'
    '    })\n\n'
    '    # Sources',
    "Patch 2 append to history"
)

# ── Patch 3: render history panel at bottom of page ───────────────────────────
patch(
    'elif search_clicked and not query.strip():\n    st.warning("Please enter a query.")',
    'elif search_clicked and not query.strip():\n'
    '    st.warning("Please enter a query.")\n\n'
    '# ---------------------------------------------------------------------------\n'
    '# Search History\n'
    '# ---------------------------------------------------------------------------\n'
    'if st.session_state.get("search_history"):\n'
    '    st.markdown("---")\n'
    '    st.markdown("### \U0001f553 Search History")\n'
    '    for item in reversed(st.session_state["search_history"]):\n'
    '        with st.expander(\n'
    '            f"[{item[\'timestamp\']}]  {item[\'query\'][:80]}",\n'
    '            expanded=False,\n'
    '        ):\n'
    '            st.caption(\n'
    '                f"{item[\'elapsed\']} \u00b7 {item[\'chunks\']} chunks \u00b7 {item[\'arxiv\']} arXiv sources"\n'
    '            )\n'
    '            st.markdown(item["answer"])',
    "Patch 3 history display"
)

print("\nSearch history patches complete.")