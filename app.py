# ─────────────────────────────────────────────
# MAIN APP INTERFACE
# ─────────────────────────────────────────────

def main():
    # 1. العنوان والوصف الرئيسي
    st.title("📚 Data Analysis RAG Mentor")
    st.caption("✨ Powered by Groq LLM & LangChain LCEL")
    
    st.markdown(
        "Ask any Data Analysis question and receive a structured, "
        "mentorship-style answer drawn directly from your uploaded reference books."
    )

    st.divider()

    # 2. تحميل الـ Chain
    chain = load_cached_rag_chain()

    # 3. نموذج السؤال وإدخال البيانات (استخدام Form يتيح الإرسال بزر Enter)
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
                st.markdown("### 🤖 AI Mentor Response")
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