"""
ingest.py — PDF Processing & Vector Embedding Script
Reads all PDFs from ./books/, chunks them, embeds via HuggingFace (local),
and persists vectors in ./chroma_db using ChromaDB.

No API key is required for embedding — the model runs locally on CPU.
"""

import os
import sys
import time
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from rag_core import (
    get_embeddings,
    verify_environment,
    CHROMA_DIR,
    BOOKS_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)


def main():
    print("=" * 60)
    print("📚 DATA ANALYSIS RAG — INGESTION PIPELINE")
    print("   Embeddings: HuggingFace (local) — no API key needed")
    print("=" * 60)

    # Step 1: Load & verify environment (only needed to confirm project setup)
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-or-v1-your_openrouter"):
        print("⚠️  WARNING: OPENROUTER_API_KEY not set in .env.")
        print("   Ingestion will continue (embeddings are local),")
        print("   but querying will fail until you set the key.\n")
    else:
        print("✅ OPENROUTER_API_KEY detected.")

    # Step 2: Warn if old ChromaDB exists (different embedding model)
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        print(f"\n⚠️  Existing ChromaDB found at '{CHROMA_DIR}'.")
        print("   If you migrated from Google embeddings, this must be deleted.")
        answer = input("   Delete and rebuild? [y/N]: ").strip().lower()
        if answer == "y":
            shutil.rmtree(CHROMA_DIR)
            print(f"   🗑️  Deleted '{CHROMA_DIR}'.")
        else:
            print("   ⏩ Skipping deletion. Adding new documents to existing store.")

    # Step 3: Verify books directory
    books_path = Path(BOOKS_DIR)
    if not books_path.exists():
        books_path.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Created '{BOOKS_DIR}' directory.")
        print("   Please add your Data Analysis PDF books/slides and re-run.")
        sys.exit(0)

    pdf_files = list(books_path.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ ERROR: No PDF files found in '{BOOKS_DIR}/'.")
        print("   Please add your Data Analysis PDF books/slides and re-run.")
        sys.exit(1)

    print(f"\n📄 Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"   • {pdf.name} ({size_mb:.1f} MB)")

    # Step 4: Load PDFs
    print("\n⏳ Loading PDF documents...")
    start = time.time()
    loader = PyPDFDirectoryLoader(BOOKS_DIR)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages in {time.time() - start:.2f}s.")

    if not documents:
        print("❌ ERROR: No content extracted from PDFs. Check file integrity.")
        sys.exit(1)

    # Step 5: Chunk documents
    print("\n⏳ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    avg_size = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"✅ Created {len(chunks)} text chunks.")
    print(f"   • Average chunk size: {avg_size} characters")

    # Step 6: Initialize HuggingFace embeddings (local, free)
    print(f"\n⏳ Loading HuggingFace Embeddings model: {EMBEDDING_MODEL}")
    print("   (First run downloads ~90MB — subsequent runs use local cache)")
    embed_start = time.time()
    embeddings = get_embeddings()
    print(f"✅ HuggingFace Embeddings ready in {time.time() - embed_start:.2f}s.")

    # Step 7: Build and persist ChromaDB vectorstore
    print(f"\n⏳ Building vector store in '{CHROMA_DIR}'...")
    print("   (Embedding all chunks locally — no API calls required)")
    store_start = time.time()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    total_time = time.time() - store_start
    print(f"✅ Vector store persisted in {total_time:.2f}s.")
    print(f"   • Location:      {os.path.abspath(CHROMA_DIR)}")
    print(f"   • Total vectors: {len(chunks)}")

    # Summary
    print("\n" + "=" * 60)
    print("🎉 INGESTION COMPLETE!")
    print("=" * 60)
    print(f"   📄 PDFs processed:     {len(pdf_files)}")
    print(f"   📃 Pages loaded:       {len(documents)}")
    print(f"   🧩 Chunks created:     {len(chunks)}")
    print(f"   💾 Vectors stored:     {len(chunks)}")
    print(f"   🤗 Embedding model:    {EMBEDDING_MODEL} (local)")
    print(f"   🤖 LLM provider:       OpenRouter (set key in .env)")
    print(f"\n   Next steps:")
    print(f"   • Test CLI:       python query_test.py")
    print(f"   • Launch Web UI:  streamlit run app.py")
    print(f"   • Launch API:     uvicorn api:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
