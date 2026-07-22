"""
app.py — Streamlit Web Interface for Data Analysis RAG Mentor
Powered by Groq LLM + HuggingFace local embeddings + ChromaDB.
"""

import os
import streamlit as st

# 1. إعدادات الصفحة الأساسية (يجب أن تكون في بداية الملف دائماً)
st.set_page_config(
    page_title="📚 Data Analysis RAG Mentor",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 🎨 2. تطبيق التصميم (خلفية فاتحة، بوكس كبير، أزرق، خطوط واضحة)
st.markdown("""
    <style>
        /* خلفية التطبيق فاتحة مريحة */
        .stApp {
            background-color: #f8f9fa !important;
        }

        /* العنوان الرئيسي بأزرق مميز */
        h1 {
            color: #0d6efd !important;
            font-weight: 800 !important;
        }

        /* تصميم كارت الإدخال كبير وبارز */
        div[data-testid="stForm"] {
            background-color: #ffffff !important;
            padding: 30px !important;
            border-radius: 16px !important;
            border: 2px solid #e2e8f0 !important;
            box-shadow: 0px 10px 25px rgba(13, 110, 253, 0.08) !important;
        }

        /* تكبير بوكس الإدخال والخط الداخلي */
        div[data-baseweb="input"] input {
            font-size: 1.2rem !important;
            padding: 14px 18px !important;
            background-color: #f8fafc !important;
            color: #0f172a !important;
            border: 2px solid #0d6efd !important;
            border-radius: 10px !important;
        }

        /* زر السؤال أزرق كبير وواضح */
        div[data-testid="stForm"] button {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            font-size: 1.25rem !important;
            font-weight: bold !important;
            padding: 14px !important;
            border-radius: 10px !important;
            border: none !important;
        }

        div[data-testid="stForm"] button:hover {
            background-color: #0b5ed7 !important;
        }
    </style>
""", unsafe_allow_html=True)

# استيراد الوظائف من rag_core بعد ضبط الواجهة
try:
    from rag_core import (
        build_rag_chain,
        format_sources,
        verify_environment,
        DEFAULT_MODEL,
        EMBEDDING_MODEL
    )
except Exception as e:
    st.error(f"❌ خطأ في استيراد ملف rag_core.py: {str(e)}")
    st.stop()


# ─────────────────────────────────────────────
# CACHED RESOURCE LOADER
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ جارٍ تحميل نموذج الذكاء الاصطناعي والمحرك (للمرة الأولى فقط)...")
def load_cached_rag_chain():
    """تحميل السلسلة وتخزينها في الذاكرة المؤقتة."""
    api_key = verify_environment(exit_on_fail=False)
    return build_rag_chain(api_key=api_key)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # Header
    st.title("📚 Data Analysis RAG Mentor")
    st.markdown(
        "<p style='font-size: 1.15rem; color: #475569;'>"
        "Ask any Data Analysis question and receive a structured, "
        "mentorship-style answer drawn directly from your uploaded reference books."
        "</p>",
        unsafe_allow_html=True
    )

    # Sidebar — system info
    with st.sidebar:
        st.header("⚙️ System Architecture")
        st.caption("Data Analysis RAG Engine Status")
        
        with st.container(border=True):
            st.markdown(f"**🤖 LLM Provider:** `Groq`")
            st.markdown(f"**🧠 LLM Model:** `{DEFAULT_MODEL}`")
            st.markdown(f"**🤗 Embeddings:** `HuggingFace`")
            st.markdown(f"**📦 Embed Model:** `{EMBEDDING_MODEL}`")
            st.markdown(f"**💾 Vector Store:** `ChromaDB`")
        
        st.divider()
        st.markdown("**📖 Knowledge Base Guide**")
        st.caption("To update knowledge base: add PDFs to `./books/` and run `python ingest.py`.")
        
        st.divider()
        st.markdown("🔑 **API Key Setup:**\n- Local: `.env` → `GROQ_API_KEY`\n- Cloud: Streamlit Secrets")

    st.divider()

    # تحميل النموذج مع التعامل مع أي خطأ قد يظهر
    try:
        chain = load_cached_rag_chain()
    except Exception as e:
        st.error(f"❌ تعذر تهيئة المحرك: {str(e)}")
        st.info("💡 تأكد من إضافة GROQ_API_KEY في ملف .env أو في Streamlit Secrets.")
        st.stop()

    # Form السؤال
    with st.form(key="qa_form", clear_on_submit=False):
        question = st.text_input(
            "💡 Enter your Data Analysis question:",
            placeholder="e.g., Explain the difference between INNER JOIN and LEFT JOIN with examples...",
            key="question_input"
        )
        
        ask_button = st.form_submit_button(
            "🚀 Ask Question",
            type="primary",
            use_container_width=True
        )

    # الإجابة معالجة
    if ask_button and question.strip():
        with st.spinner("🧠 Retrieving context and generating response via Groq..."):
            try:
                result = chain.invoke({"input": question.strip()})
                answer = result.get("answer", "No answer generated.")
                source_docs = result.get("context", [])

                st.divider()
                st.markdown("<h3 style='color: #0d6efd;'>🤖 AI Mentor Response</h3>", unsafe_allow_html=True)
                st.markdown(answer)

                sources = format_sources(source_docs, icon=True)
                with st.expander("📚 Source Citations", expanded=False):
                    if sources:
                        for src in sources:
                            st.markdown(f"- {src}")
                    else:
                        st.caption("No specific source pages retrieved for this query.")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

    elif ask_button and not question.strip():
        st.warning("⚠️ Please enter a question first.")

    st.divider()
    st.caption("Built with LangChain LCEL • Groq API • HuggingFace • ChromaDB • Streamlit")


if __name__ == "__main__":
    main()