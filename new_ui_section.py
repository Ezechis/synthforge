# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="PromptForge", page_icon="🔥", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');
.main-header { font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700;
    color:#FF6B35; letter-spacing:-0.02em; margin-bottom:0.1rem; }
.sub-header { font-family:'Inter',sans-serif; font-size:0.9rem; color:#666; margin-bottom:1rem; }
.source-badge { display:inline-block; padding:2px 8px; border-radius:3px; font-size:0.7rem;
    font-weight:700; font-family:'Space Mono',monospace; margin-right:4px; }
.arxiv-badge  { background:#B31B1B; color:white; }
.github-badge { background:#238636; color:white; }
.reddit-badge { background:#FF4500; color:white; }
.docs-badge   { background:#0066CC; color:white; }
.youtube-badge { background:#CC0000; color:white; }
.confidence-bar-container { margin:6px 0 10px 0; }
.confidence-label { font-size:0.68rem; font-family:'Space Mono',monospace; color:#888; margin-bottom:2px; }
.confidence-bar { height:5px; border-radius:3px; margin-bottom:5px; }
.tier1-bar { background:#B31B1B; }
.tier2-bar { background:#238636; }
.tier3-bar { background:#FF4500; }
.suggestion-header { font-family:'Space Mono',monospace; font-size:0.78rem; color:#888;
    text-transform:uppercase; letter-spacing:0.08em; margin:1rem 0 0.4rem 0; }
.example-label { font-family:'Space Mono',monospace; font-size:0.75rem; color:#555;
    text-transform:uppercase; letter-spacing:0.06em; margin:0.8rem 0 0.4rem 0; }
.article-item { padding:5px 0; border-bottom:1px solid #222; }
.article-title { font-size:0.8rem; font-weight:600; color:#FF6B35; line-height:1.3; }
.coming-soon-badge { display:inline-block; padding:1px 5px; border-radius:3px; font-size:0.6rem;
    font-weight:700; background:#1a1a1a; color:#888; border:1px solid #333; margin-left:4px; }
.hn-item { padding:5px 0; border-bottom:1px solid #1a1a1a; }
.hn-title { font-size:0.78rem; color:#ddd; line-height:1.3; }
.hn-meta { font-size:0.68rem; color:#555; font-family:'Space Mono',monospace; margin-top:2px; }
.file-badge { display:inline-block; padding:3px 10px; border-radius:4px;
    background:#1a2a1a; border:1px solid #238636; font-size:0.75rem; color:#4caf50;
    font-family:'Space Mono',monospace; margin:4px 0; }
.metric-block { margin-bottom:8px; }
.metric-label { font-size:0.68rem; font-family:'Space Mono',monospace; color:#888;
    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1px; }
.metric-value { font-size:0.95rem; font-family:'Space Mono',monospace; color:#ddd; font-weight:700; }
.history-num { color:#FF6B35; font-weight:700; font-family:'Space Mono',monospace;
    font-size:0.78rem; margin-right:5px; }
</style>
""", unsafe_allow_html=True)

# Load resources
with st.spinner("Initialising PromptForge..."):
    collection, embed_model, reranker, bm25, corpus_chunks, corpus_metas = load_all_resources()

# Session state
for key, default in [
    ("search_history", []), ("feedback", {}), ("query_input", ""), ("suggestions", []),
    ("last_answer", None), ("last_metrics", None), ("last_results", None),
    ("last_query", None), ("last_had_file", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# SIDEBAR — Settings, Source filter, Live HN, Quick Queries
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Retrieval Settings")
    n_results = st.slider("Candidate chunks", 10, 50, 20, 5)

    st.markdown("### 🗂️ Source Filter")
    active_sources = [
        k for k, v in {
            "arxiv":   st.checkbox("arXiv papers",       value=True),
            "github":  st.checkbox("GitHub repos",        value=True),
            "reddit":  st.checkbox("Reddit discussions",  value=True),
            "docs":    st.checkbox("Official docs",       value=True),
            "youtube": st.checkbox("YouTube transcripts", value=True),
        }.items() if v
    ] or None

    st.markdown("---")
    st.markdown("### 📡 Live: AI on HN")
    for s in fetch_hn_news():
        st.markdown(
            f'<div class="hn-item"><div class="hn-title">'
            f'<a href="{s["url"]}" target="_blank" style="color:#ddd;text-decoration:none;">'
            f'{s["title"]}</a></div>'
            f'<div class="hn-meta">▲ {s["points"]} · 💬 {s["comments"]}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 💡 Quick Queries")
    for ex in EXAMPLE_QUERIES:
        if st.button(ex, use_container_width=True, key=f"sb_{ex[:15]}"):
            st.session_state["query_input"] = ex
            st.rerun()

    st.markdown("---")
    st.caption("PromptForge · DeepForge Platform · by Ezechinyere Nnabugwu")

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-header">🔥 PromptForge</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Prompt Engineering Knowledge Synthesis Engine — '
    f'{collection.count():,} chunks across arXiv · GitHub · Reddit · Docs · YouTube</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# SEARCH ROW — Query box + Search + Clear + Attach (all on same plane)
# =============================================================================
query = st.text_area(
    "Ask anything about prompt engineering:",
    value=st.session_state.get("query_input", ""),
    height=130,
    placeholder="e.g. What empirical evidence exists for chain-of-thought prompting?",
    key="query_input",
)

# Search | Clear | 📎 Attach — one row
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])

with btn_col1:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

with btn_col2:
    if st.button("✕ Clear", use_container_width=True):
        st.session_state.update({
            "query_input": "", "suggestions": [],
            "last_answer": None, "last_metrics": None,
            "last_results": None, "last_query": None, "last_had_file": False,
        })
        st.rerun()

with btn_col3:
    # Compact file uploader — label hidden, sits flush with buttons
    uploaded_file = st.file_uploader(
        "📎 Attach a document (PDF, Word, TXT, Markdown):",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed",
        help="Max 5 MB · 50 pages. Supplements the knowledge base.",
    )
    if uploaded_file is not None:
        _is_valid, _validation_reason = validate_uploaded_file(uploaded_file)
        if not _is_valid:
            st.error(f"⚠️ {_validation_reason}")
            uploaded_file = None

# Example queries — shown before first search
if not st.session_state["search_history"]:
    st.markdown('<div class="example-label">Try asking:</div>', unsafe_allow_html=True)
    ex_cols = st.columns(3)
    for i, ex in enumerate(EXAMPLE_QUERIES):
        with ex_cols[i % 3]:
            if st.button(ex, key=f"me_{i}", use_container_width=True):
                st.session_state["query_input"] = ex
                st.rerun()

st.markdown("---")

# =============================================================================
# TWO-COLUMN BODY
# Left (75%): Answer, copy, feedback, retrieved sources, 6 suggestions
# Right (25%): Griot Protocol → Metrics strip (top to bottom)
# =============================================================================
col_main, col_right = st.columns([3, 1])

# ── RIGHT PANEL — Griot Protocol at top, Metrics below ───────────────────────
with col_right:

    # Griot Protocol
    st.markdown("**📝 The Griot Protocol**")
    st.caption("Series 1 · Ezechinyere Nnabugwu")
    for art in EZE_ARTICLES:
        live = art["url"] != "#"
        if live:
            st.markdown(
                f'<div class="article-item"><div class="article-title">'
                f'<a href="{art["url"]}" target="_blank" style="color:#FF6B35;text-decoration:none;">'
                f'{art["title"]}</a></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="article-item"><div class="article-title">{art["title"]}'
                f'<span class="coming-soon-badge">SOON</span></div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Metrics strip — visible after any search
    if st.session_state["last_metrics"]:
        m    = st.session_state["last_metrics"]
        conf = m["conf"]

        st.markdown("**📊 Source Confidence**")
        st.markdown(
            f'<div class="confidence-bar-container">'
            f'<div class="confidence-label">T1 · arXiv · {conf["tier1"]} · {conf["tier1_pct"]}%</div>'
            f'<div class="confidence-bar tier1-bar" style="width:{max(conf["tier1_pct"],2)}%;"></div>'
            f'<div class="confidence-label">T2 · Docs/GH · {conf["tier2"]} · {conf["tier2_pct"]}%</div>'
            f'<div class="confidence-bar tier2-bar" style="width:{max(conf["tier2_pct"],2)}%;"></div>'
            f'<div class="confidence-label">T3 · Reddit/YT · {conf["tier3"]} · {conf["tier3_pct"]}%</div>'
            f'<div class="confidence-bar tier3-bar" style="width:{max(conf["tier3_pct"],2)}%;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-block"><div class="metric-label">⏱ Response time</div>'
            f'<div class="metric-value">{m["elapsed"]:.1f}s</div></div>'
            f'<div class="metric-block"><div class="metric-label">🤖 Model</div>'
            f'<div class="metric-value" style="font-size:0.78rem;">Llama 3.3 70B</div></div>'
            f'<div class="metric-block"><div class="metric-label">📦 Chunks</div>'
            f'<div class="metric-value">{m["chunks"]}</div></div>'
            f'<div class="metric-block"><div class="metric-label">📄 arXiv sources</div>'
            f'<div class="metric-value">{m["arxiv_n"]}</div></div>',
            unsafe_allow_html=True,
        )

# ── LEFT / MAIN PANEL — Answer, sources, 6 suggestions ───────────────────────
with col_main:

    if st.session_state["last_answer"]:
        answer      = st.session_state["last_answer"]
        results     = st.session_state["last_results"]
        query_shown = st.session_state["last_query"]

        if st.session_state.get("last_had_file"):
            st.markdown(
                '<div class="file-badge">📎 Document added to context</div>',
                unsafe_allow_html=True,
            )

        st.markdown("### 📋 Answer")
        st.markdown(answer)

        # Copy button
        safe = (
            answer
            .replace("\\", "\\\\")
            .replace("`", "'")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )
        components.html(
            f"""<script>function cp(){{
                navigator.clipboard.writeText("{safe}").then(()=>{{
                    var b=document.getElementById('cb');
                    b.innerText='✓ Copied!'; b.style.background='#238636';
                    setTimeout(()=>{{b.innerText='📋 Copy Answer';b.style.background='#1e1e1e';}},2000);
                }});
            }}</script>
            <button id="cb" onclick="cp()" style="background:#1e1e1e;color:#ccc;border:1px solid #333;
                padding:6px 16px;border-radius:4px;cursor:pointer;font-size:13px;font-family:monospace;">
                📋 Copy Answer</button>""",
            height=44,
        )

        # Feedback
        qk       = str(hash(query_shown))
        existing = st.session_state["feedback"].get(qk)
        st.markdown("**Was this answer useful?**")
        fb1, fb2, _ = st.columns([1, 1, 8])
        with fb1:
            if st.button("👍", key=f"up_{qk}",
                         type="primary" if existing == "up" else "secondary"):
                st.session_state["feedback"][qk] = "up"
                st.toast("Thanks!", icon="✅")
        with fb2:
            if st.button("👎", key=f"dn_{qk}",
                         type="primary" if existing == "down" else "secondary"):
                st.session_state["feedback"][qk] = "down"
                st.toast("Noted.", icon="📝")

        # Retrieved Sources
        with st.expander("📄 Retrieved Sources", expanded=False):
            for r in results:
                meta = r["metadata"]
                src  = meta.get("source", "unknown")
                st.markdown(
                    f'<span class="source-badge {src}-badge">{src.upper()}</span> '
                    f'**{str(meta.get("title", ""))[:80]}** (score: {r["score"]:.3f})',
                    unsafe_allow_html=True,
                )
                if meta.get("url"):
                    st.markdown(f'[{meta["url"]}]({meta["url"]})')
                st.text(r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"])
                st.divider()

    # 6 Follow-up suggestions — 3 columns × 2 rows
    if st.session_state.get("suggestions"):
        st.markdown('<div class="suggestion-header">You might also ask:</div>',
                    unsafe_allow_html=True)
        sg_cols = st.columns(3)
        for i, s in enumerate(st.session_state["suggestions"][:6]):
            with sg_cols[i % 3]:
                if st.button(f"→ {s}", key=f"sg_{i}", use_container_width=True):
                    st.session_state.update({"query_input": s, "suggestions": []})
                    st.rerun()

# =============================================================================
# SEARCH EXECUTION
# =============================================================================
if search_clicked and query.strip():
    # Security Gate 1: injection
    _flagged, _injection_reason = detect_prompt_injection(query.strip())
    if _flagged:
        st.error(f"⚠️ {_injection_reason}")
        st.stop()
    # Security Gate 2: rate limit
    _allowed, _rate_reason = check_rate_limit()
    if not _allowed:
        st.warning(_rate_reason)
        st.stop()

    t0 = time.time()

    file_context = ""
    had_file     = False
    if uploaded_file:
        with st.spinner(f"Extracting from {uploaded_file.name}..."):
            file_context = extract_file_content(uploaded_file)
        if file_context and not file_context.startswith("["):
            had_file = True
        elif file_context.startswith("["):
            st.warning(file_context)
            file_context = ""

    with st.spinner("Retrieving and synthesising..."):
        results = hybrid_retrieve(
            query=query.strip(), collection=collection, embed_model=embed_model,
            reranker=reranker, bm25=bm25, corpus_chunks=corpus_chunks,
            corpus_metas=corpus_metas, n_results=n_results, source_filter=active_sources,
        )
        if not results:
            st.warning("No relevant chunks found. Try enabling more source filters.")
            st.stop()

        filtered = [r for r in results if r["score"] > MIN_RETRIEVAL_SCORE]
        results  = filtered if len(filtered) >= 3 else results[:3]
        answer   = generate_answer(query.strip(), results, file_context)

    elapsed  = time.time() - t0
    arxiv_n  = sum(1 for r in results if r["metadata"].get("source") == "arxiv")
    conf     = compute_source_confidence(results)

    st.session_state.update({
        "last_answer":   answer,
        "last_query":    query.strip(),
        "last_results":  results,
        "last_had_file": had_file,
        "last_metrics": {
            "elapsed": elapsed,
            "chunks":  len(results),
            "arxiv_n": arxiv_n,
            "conf":    conf,
        },
    })

    st.session_state["search_history"].append({
        "query":     query.strip(),
        "answer":    answer,
        "elapsed":   f"{elapsed:.1f}s",
        "chunks":    len(results),
        "arxiv":     arxiv_n,
        "timestamp": time.strftime("%H:%M:%S"),
        "had_file":  had_file,
    })

    with st.spinner("Generating follow-up suggestions..."):
        st.session_state["suggestions"] = generate_suggestions(query.strip(), answer)

    st.rerun()

elif search_clicked:
    st.warning("Please enter a query.")

# =============================================================================
# FEEDBACK SUMMARY
# =============================================================================
if st.session_state["feedback"]:
    pos   = sum(1 for v in st.session_state["feedback"].values() if v == "up")
    total = len(st.session_state["feedback"])
    st.caption(f"Session feedback: {pos}/{total} answers rated useful")

# =============================================================================
# SEARCH HISTORY — numbered, bottom of page
# =============================================================================
if st.session_state["search_history"]:
    st.markdown("---")
    st.markdown("### 🕔 Search History")
    for i, item in enumerate(reversed(st.session_state["search_history"]), 1):
        file_icon = " 📎" if item.get("had_file") else ""
        label     = f"#{i}  [{item['timestamp']}]  {item['query'][:80]}{file_icon}"
        with st.expander(label, expanded=False):
            st.caption(
                f"{item['elapsed']} · {item['chunks']} chunks · {item['arxiv']} arXiv"
            )
            st.markdown(item["answer"])
