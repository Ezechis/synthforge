"""
PromptForge - Streamlit Frontend
Layer 5 Interface: Web UI for querying the PromptForge knowledge engine.
Connects retrieval and generation layers into a clean user interface.

Usage: streamlit run src/app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.hybrid_retrieval import PromptForgeRetriever
from src.generation.generate import generate_answer, PROMPTFORGE_SYSTEM_PROMPT

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PromptForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FF6B35;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #1a1a2e;
        border-left: 4px solid #FF6B35;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .source-chip {
        display: inline-block;
        background-color: #16213e;
        color: #FF6B35;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin: 0.2rem;
        border: 1px solid #FF6B35;
    }
    .metric-box {
        background-color: #16213e;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Retriever Initialisation (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_retriever() -> PromptForgeRetriever:
    """Load and cache the retriever across all sessions.

    Returns:
        Initialised PromptForgeRetriever instance.
    """
    with st.spinner("Initialising PromptForge knowledge engine..."):
        return PromptForgeRetriever()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(retriever: PromptForgeRetriever) -> dict:
    """Render sidebar with corpus stats and settings.

    Args:
        retriever: Initialised retriever for corpus statistics.

    Returns:
        Dict of user-selected settings.
    """
    with st.sidebar:
        st.markdown("### ⚡ PromptForge")
        st.markdown("*Prompt Engineering Knowledge Engine*")
        st.divider()

        st.markdown("**Corpus Statistics**")
        total_chunks = retriever.collection.count()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", f"{total_chunks:,}")
        with col2:
            st.metric("Sources", "2")

        st.divider()

        st.markdown("**Retrieval Settings**")
        top_n = st.slider(
            "Chunks to retrieve",
            min_value=3,
            max_value=12,
            value=8,
            help="More chunks = more comprehensive but slower"
        )

        st.divider()

        st.markdown("**Source Filter**")
        use_arxiv = st.checkbox("arXiv Papers", value=True)
        use_github = st.checkbox("GitHub Repos", value=True)

        st.divider()

        st.markdown("**Example Queries**")
        examples = [
            "What is chain-of-thought prompting?",
            "How does few-shot prompting work?",
            "What is self-consistency decoding?",
            "Explain retrieval augmented generation",
            "What are the failure modes of zero-shot prompting?",
            "How does instruction tuning affect prompt design?",
        ]
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.example_query = example

        st.divider()
        st.markdown(
            "<small>Built by Ezechinyere Nnabugwu</small>",
            unsafe_allow_html=True
        )

    return {"top_n": top_n, "use_arxiv": use_arxiv, "use_github": use_github}


# ── Results Display ───────────────────────────────────────────────────────────
def render_sources(chunks: list[dict]) -> None:
    """Render retrieved source chunks in an expandable section.

    Args:
        chunks: Retrieved and reranked chunks from retrieval layer.
    """
    with st.expander(f"View {len(chunks)} Retrieved Sources", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            source = meta.get("source", "unknown").upper()
            credibility = meta.get("credibility_tier", "unknown")
            content_type = meta.get("content_type", "unknown")
            url = meta.get("url", "")
            title = meta.get("title", meta.get("repo", "Unknown"))
            score = chunk.get("final_score", 0)

            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**[{i}] {title[:60]}**")
                with col2:
                    st.markdown(f"`{source}` · `{content_type}`")
                with col3:
                    st.markdown(f"Score: `{score:.3f}`")

                if url:
                    st.markdown(f"[View Source]({url})")
                st.markdown(f"*{chunk['text'][:300]}...*")
                st.divider()


# ── Main App ──────────────────────────────────────────────────────────────────
def main() -> None:
    """Main Streamlit application entry point."""

    # Header
    st.markdown(
        '<p class="main-header">⚡ PromptForge</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Prompt Engineering Knowledge Engine — '
        'Synthesising arXiv papers, GitHub implementations, '
        'and practitioner knowledge</p>',
        unsafe_allow_html=True
    )

    # Load retriever
    retriever = load_retriever()

    # Render sidebar
    settings = render_sidebar(retriever)

    # Query input
    query_value = st.session_state.get("example_query", "")
    if query_value:
        st.session_state.example_query = ""

    query = st.text_area(
        "Ask anything about prompt engineering:",
        value=query_value,
        height=100,
        placeholder="e.g. What is the empirical evidence for chain-of-thought prompting?",
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_clicked = st.button(
            "Search",
            type="primary",
            use_container_width=True,
        )
    with col2:
        clear_clicked = st.button(
            "Clear",
            use_container_width=True,
        )

    if clear_clicked:
        st.rerun()

    # Process query
    if search_clicked and query.strip():
        with st.spinner("Retrieving and synthesising..."):
            start_time = time.time()

            # Retrieve chunks
            chunks = retriever.retrieve(query.strip(), top_n=settings["top_n"])

            # Filter by source if needed
            if not settings["use_arxiv"]:
                chunks = [c for c in chunks if c["metadata"].get("source") != "arxiv"]
            if not settings["use_github"]:
                chunks = [c for c in chunks if c["metadata"].get("source") != "github"]

            if not chunks:
                st.warning("No relevant results found. Try a different query.")
                return

            # Generate answer
            answer = generate_answer(query.strip(), retriever)
            elapsed = time.time() - start_time

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Response Time", f"{elapsed:.1f}s")
        with col2:
            st.metric("Chunks Retrieved", len(chunks))
        with col3:
            arxiv_count = sum(
                1 for c in chunks
                if c["metadata"].get("source") == "arxiv"
            )
            st.metric("arXiv Sources", arxiv_count)
        with col4:
            st.metric("Model", "llama-3.3-70b")

        # Display answer
        st.markdown("### PromptForge Answer")
        st.markdown(answer)

        # Display sources
        render_sources(chunks)

    elif search_clicked and not query.strip():
        st.warning("Please enter a query.")

    # Query history
    if "history" not in st.session_state:
        st.session_state.history = []

    if search_clicked and query.strip():
        st.session_state.history.append(query.strip())

    if st.session_state.get("history"):
        with st.expander("Query History", expanded=False):
            for i, past_query in enumerate(
                reversed(st.session_state.history[-10:]), 1
            ):
                st.markdown(f"{i}. {past_query}")


if __name__ == "__main__":
    main()