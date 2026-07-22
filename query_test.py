"""
query_test.py — Terminal/CLI Testing Script
Interactive RAG query loop using ChromaDB (HuggingFace embeddings)
+ OpenRouter LLM via LangChain LCEL.
"""

import os
import sys

from rag_core import (
    verify_environment,
    build_rag_chain,
    format_sources,
    DEFAULT_MODEL,
)


def main():
    print("=" * 60)
    print("📚 DATA ANALYSIS RAG MENTOR — CLI Interface")
    print(f"   LLM:        OpenRouter → {DEFAULT_MODEL}")
    print(f"   Embeddings: HuggingFace (all-MiniLM-L6-v2, local)")
    print("=" * 60)
    print("Type your question and press Enter.")
    print("Type 'quit', 'exit', or 'q' to stop.\n")

    # Verify environment
    api_key = verify_environment(exit_on_fail=True)
    print("✅ OPENROUTER_API_KEY loaded.\n")

    # Load RAG chain
    print("⏳ Loading HuggingFace embeddings and RAG chain...")
    try:
        chain = build_rag_chain(api_key=api_key)
    except RuntimeError as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)
    print("✅ RAG chain ready.\n")

    # Interactive loop
    while True:
        try:
            question = input("🧑‍💻 Your Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye! Keep learning Data Analysis!")
            break

        if not question:
            print("⚠️  Please enter a question.\n")
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye! Keep learning Data Analysis!")
            break

        print("\n⏳ Retrieving context and querying OpenRouter...\n")

        try:
            # LCEL chain execution: "input" -> returns "answer" & "context"
            result = chain.invoke({"input": question})
            answer = result.get("answer", "No answer generated.")
            source_docs = result.get("context", [])

            print("─" * 60)
            print("🤖 AI MENTOR RESPONSE:")
            print("─" * 60)
            print(answer)
            print("\n" + "─" * 60)

            sources = format_sources(source_docs, icon=True)
            if sources:
                print("📚 SOURCE CITATIONS:")
                for src in sources:
                    print(f"   {src}")
            else:
                print("📚 SOURCE CITATIONS: None retrieved.")
            print("─" * 60)
            print()

        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            print("   Check your OPENROUTER_API_KEY and internet connection.\n")


if __name__ == "__main__":
    main()
