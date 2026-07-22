"""
app.py — Streamlit Web Interface for Data Analysis RAG Mentor
Powered by OpenRouter LLM + HuggingFace local embeddings + ChromaDB.
Uses LangChain LCEL (0.3.x+ compatible — no RetrievalQA).
"""

import os

import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    build_rag_chain,
    format_sources,
    DEFAULT_MODEL,
    EMBEDDING_MODEL
)


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="📚 Data Analysis RAG Mentor",
    page_icon="📚",
    layout="centered"
)


# ─────────────────────────────────────────────
# CACHED RESOURCE LOADER
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Loading AI Mentor (first load only)...")
def load_cached_rag_chain():
    """
    Load and cache the full RAG chain on first run.
    Subsequent user interactions reuse the cached chain — zero re-initialization.
    """
    # Streamlit Cloud Secrets → fallback to .env for local dev
    api_key = st.secrets.get("OPENROUTER_API_KEY", None)
    if not api_key:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key or api_key.startswith("sk-or-v1-your_openrouter"):
        st.error(
            "❌ **OPENROUTER_API_KEY not found.**\n\n"
            "**Local:** Add `OPENROUTER_API_KEY=sk-or-v1-...` to your `.env` file.\n\n"
            "**Streamlit Cloud:** Go to App Settings → Secrets → add:\n"
            "```toml\nOPENROUTER_API_KEY = \"sk-or-v1-...\"\n```"
        )
        st.stop()

    if not os.path.exists("./chroma_db"):
        st.error(
            "❌ **ChromaDB vector store not found.**\n\n"
            "Run the following command locally first:\n"
            "```bash\npython ingest.py\n```\n"
            "Then commit the `chroma_db/` folder and redeploy."
        )
        st.stop()

    try:
        return build_rag_chain(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG chain: {str(e)}")
        st.stop()


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # Header
    st.title("📚 Data Analysis RAG Mentor")
    st.markdown(
        "Ask any Data Analysis question and receive a structured, "
        "mentorship-style answer drawn directly from your uploaded reference books."
    )

    # Sidebar — system info
    with st.sidebar:
        st.header("⚙️ System Info")
        st.markdown(f"**🤖 LLM Provider:** OpenRouter")
        st.markdown(f"**🧠 LLM Model:** `{DEFAULT_MODEL}`")
        st.markdown(f"**🤗 Embeddings:** HuggingFace (local)")
        st.markdown(f"**📦 Embed Model:** `{EMBEDDING_MODEL}`")
        st.markdown(f"**💾 Vector Store:** ChromaDB (persistent)")
        st.divider()
        st.markdown("**📖 Knowledge Base**")
        st.caption(
            "Add PDFs to `./books/` and run `python ingest.py` "
            "to update the knowledge base."
        )
        st.divider()
        st.markdown(
            "🔑 Set your key:\n"
            "- **Local:** `.env` → `OPENROUTER_API_KEY`\n"
            "- **Cloud:** Streamlit Secrets"
        )

    st.divider()

    # Load RAG chain (cached — runs only once per session)
    chain = load_cached_rag_chain()

    # Question input
    question = st.text_input(
        "💡 Enter your Data Analysis question:",
        placeholder="e.g., Explain the difference between INNER JOIN and LEFT JOIN with examples...",
        key="question_input"
    )

    ask_button = st.button(
        "🚀 Ask Question",
        type="primary",
        use_container_width=True
    )

    # Process and display answer
    if ask_button and question:
        with st.spinner("🧠 Retrieving context and generating response via OpenRouter..."):
            try:
                # LCEL chain: input key is "input", answer is result["answer"]
                result = chain.invoke({"input": question})
                answer = result.get("answer", "No answer generated.")
                source_docs = result.get("context", [])

                st.divider()
                st.markdown("### 🤖 AI Mentor Response")
                st.markdown(answer)

                # Source citations
                sources = format_sources(source_docs, icon=True)
                with st.expander("📚 Source Citations", expanded=False):
                    if sources:
                        for src in sources:
                            st.markdown(f"- {src}")
                    else:
                        st.caption("No specific source pages retrieved for this query.")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info(
                    "Possible causes:\n"
                    "- Invalid or expired **OPENROUTER_API_KEY**\n"
                    "- OpenRouter rate limit reached (free tier)\n"
                    "- No internet connection"
                )

    elif ask_button and not question:
        st.warning("⚠️ Please enter a question first.")

    # Footer
    st.divider()
    st.caption(
        "Built with LangChain 0.3 • OpenRouter • HuggingFace • ChromaDB • Streamlit"
    )


if __name__ == "__main__":
    main()
