# ─────────────────────────────────────────────
# MAIN APP INTERFACE
# ─────────────────────────────────────────────

def main():
    # 🎨 إدخال تنسيقات CSS المخصصة (خلفية فاتحة، خطوط كبيرة، ولون أزرق)
    st.markdown("""
        <style>
            /* 1. خلفية الصفحة فاتحة */
            .stApp {
                background-color: #f8f9fa !important;
            }

            /* 2. العنوان الرئيسي باللون الأزرق وحجم كبير */
            h1 {
                color: #0d6efd !important;
                font-size: 2.8rem !important;
                font-weight: 800 !important;
            }

            /* 3. تكبير كارت الإدخال وتحديد خلفيته باللون الأبيض الناصع */
            div[data-testid="stForm"] {
                background-color: #ffffff !important;
                padding: 35px !important;
                border-radius: 18px !important;
                border: 2px solid #e3e8ee !important;
                box-shadow: 0px 8px 20px rgba(13, 110, 253, 0.08) !important;
            }

            /* 4. تكبير عنوان حقل الإدخال والخط الأزرق */
            div[data-testid="stForm"] label p {
                font-size: 1.3rem !important;
                font-weight: bold !important;
                color: #0b5ed7 !important;
            }

            /* 5. تكبير بوكس الإدخال (Input Box) والخط الداخلي */
            div[data-baseweb="input"] input {
                font-size: 1.25rem !important;
                padding: 16px 20px !important;
                background-color: #f4f7fa !important;
                color: #0d233a !important;
                border: 2px solid #0d6efd !important;
                border-radius: 12px !important;
            }

            /* 6. تكبير زر الإرسال ولونه الأزرق المميز */
            div[data-testid="stForm"] button {
                background-color: #0d6efd !important;
                color: #ffffff !important;
                font-size: 1.3rem !important;
                font-weight: bold !important;
                padding: 16px !important;
                border-radius: 12px !important;
                border: none !important;
                transition: all 0.3s ease !important;
            }

            /* تأثير تحويم الماوس على الزر */
            div[data-testid="stForm"] button:hover {
                background-color: #0b5ed7 !important;
                box-shadow: 0px 4px 12px rgba(13, 110, 253, 0.3) !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # 1. العنوان والوصف الرئيسي
    st.title("📚 Data Analysis RAG Mentor")
    st.caption("✨ Powered by Groq LLM & LangChain LCEL")
    
    st.markdown(
        "<p style='font-size: 1.15rem; color: #495057;'>"
        "Ask any Data Analysis question and receive a structured, "
        "mentorship-style answer drawn directly from your uploaded reference books."
        "</p>", 
        unsafe_allow_html=True
    )

    st.divider()

    # 2. تحميل الـ Chain
    chain = load_cached_rag_chain()

    # 3. البوكس الكبير والنموذج
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

    # 4. معالجة السؤال وعرض النتيجة
    if ask_button and question.strip():
        with st.spinner("🧠 Retrieving context and generating response via Groq..."):
            try:
                result = chain.invoke({"input": question.strip()})
                answer = result.get("answer", "No answer generated.")
                source_docs = result.get("context", [])

                st.divider()
                st.markdown("<h3 style='color: #0d6efd;'>🤖 AI Mentor Response</h3>", unsafe_allow_html=True)
                st.markdown(answer)

                # المصادر والمراجع
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