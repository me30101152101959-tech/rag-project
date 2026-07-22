"""
rag_core.py — Shared Core Module for Data Analysis RAG
Centralizes environment verification, LLM initialization (Groq API),
embeddings initialization, and modern LCEL RAG chain construction.

Pure LCEL implementation:
  No dependency on legacy `langchain.chains` modules.

Chain I/O contract:
  Input:  chain.invoke({"input": question_string})
  Output: result["answer"]       — the LLM response string
          result["context"]      — list of retrieved Document objects
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CHROMA_DIR = "./chroma_db"
BOOKS_DIR = "./books"
COLLECTION_NAME = "data_analysis_rag"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HF_REPO_ID = "mostafaeltaweel/data-analysis-chromadb"


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
[SYSTEM ROLE & GOAL]
You are a World-Class Senior Data Analyst, Business Intelligence Lead, and Academic AI Tutor. \
Your mission is to mentor the user into a Senior Data Analyst using strictly the reference \
materials provided in the context.

[TEACHING PROTOCOL - FOLLOW THIS FOR EVERY ANSWER]

1. STEP 1: Intuition & Zero-Math Business Analogy
   - Explain the concept using plain language and a real-life business scenario.

2. STEP 2: Visual ASCII Architecture & Data Workflow
   - Provide ASCII diagrams to visualize data flows, SQL JOINs, or Pandas operations
     with explicit input/output shapes (e.g., Input: (1000, 5) -> Output: (100, 2)).

3. STEP 3: Statistical/Mathematical Breakdown (10-Year-Old Explanation Rule)
   - Break down every formula element by element using intuitive arithmetic and
     concrete worked numerical examples.

4. STEP 4: Hands-On Code (Python/SQL) & Interactive Debugging
   - Provide clean Python (Pandas/Seaborn) or SQL code with an interactive task
     (code completion, bug fix, or data challenge).

5. STEP 5: Real-World Business Case Study & Micro-Quiz
   - Present an industry business problem and ask ONE single targeted question
     to test understanding.

[RAG CONTEXT]
Context from uploaded books:
{context}

[USER QUESTION]
{input}
"""


# ─────────────────────────────────────────────
# ENVIRONMENT VERIFICATION
# ─────────────────────────────────────────────

def verify_environment(exit_on_fail: bool = True) -> str:
    """
    Load .env and verify GROQ_API_KEY is present and valid.
    Checks Streamlit Secrets as fallback for Streamlit Cloud deployment.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    # Streamlit Cloud Fallback Check
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"].strip()
                os.environ["GROQ_API_KEY"] = api_key
        except Exception:
            pass

    if not api_key or api_key.startswith("gsk_your_groq_api_key"):
        msg = (
            "GROQ_API_KEY is not set or is still the placeholder value.\n"
            "  • Local:           Edit .env → set GROQ_API_KEY=gsk_...\n"
            "  • Streamlit Cloud: Add GROQ_API_KEY to your app Secrets.\n"
            "  • Render.com:      Add GROQ_API_KEY in Environment Variables."
        )
        if exit_on_fail:
            print(f"❌ ERROR: {msg}")
            sys.exit(1)
        raise RuntimeError(msg)

    return api_key


def verify_chroma_exists(exit_on_fail: bool = True) -> None:
    """Confirm the ChromaDB directory exists; download from Hugging Face if missing."""
    if not os.path.exists(CHROMA_DIR):
        print(f"🌐 ChromaDB not found at '{CHROMA_DIR}'. Downloading from Hugging Face ({HF_REPO_ID})...")
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                local_dir=CHROMA_DIR,
                ignore_patterns=[".git*"]
            )
            print("✅ ChromaDB downloaded successfully from Hugging Face!")
        except Exception as e:
            msg = (
                f"Failed to download ChromaDB from Hugging Face ({HF_REPO_ID}): {str(e)}\n"
                "  → Ensure internet connectivity or run 'python upload_to_hf.py' first."
            )
            if exit_on_fail:
                print(f"❌ ERROR: {msg}")
                sys.exit(1)
            raise RuntimeError(msg)


# ─────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return HuggingFaceEmbeddings instance using all-MiniLM-L6-v2."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ─────────────────────────────────────────────
# LLM — GROQ API via ChatOpenAI
# ─────────────────────────────────────────────

def get_llm(api_key: str, model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Return ChatOpenAI instance configured for Groq API."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0.2,
        max_tokens=2048,
    )


# ─────────────────────────────────────────────
# VECTORSTORE LOADER
# ─────────────────────────────────────────────

def get_vectorstore() -> Chroma:
    """Load the persisted ChromaDB vectorstore."""
    verify_chroma_exists(exit_on_fail=False)
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


# ─────────────────────────────────────────────
# RAG CHAIN BUILDER (PURE NATIVE LCEL)
# ─────────────────────────────────────────────

def build_rag_chain(api_key: str, model: str = DEFAULT_MODEL):
    """
    Build and return a modern LCEL RAG chain.
    
    Guarantees compatibility with the existing calling convention:
      result = chain.invoke({"input": question})
      answer = result["answer"]
      docs   = result["context"]
    """
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = get_llm(api_key=api_key, model=model)
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)

    # Helper function to combine document contents for the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 1. Fetch relevant documents based on the input question
    retrieval_step = RunnableParallel({
        "context": (lambda x: x["input"]) | retriever,
        "input": lambda x: x["input"]
    })

    # 2. Complete LCEL pipeline mapping inputs and context to the desired dict output structure
    chain = retrieval_step | RunnableParallel({
        "answer": (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(x["context"])
            )
            | prompt
            | llm
            | StrOutputParser()
        ),
        "context": lambda x: x["context"],
        "input": lambda x: x["input"]
    })

    return chain


# ─────────────────────────────────────────────
# SOURCE CITATION FORMATTER
# ─────────────────────────────────────────────

def format_sources(source_documents: list, icon: bool = True) -> list[str]:
    """Extract unique citation strings from retrieved documents."""
    sources = []
    seen = set()
    prefix = "📖 " if icon else ""

    for doc in source_documents:
        source_name = doc.metadata.get("source", "Unknown Source")
        page_number = doc.metadata.get("page", "N/A")
        filename = os.path.basename(source_name)
        page_display = (
            int(page_number) + 1
            if isinstance(page_number, (int, float))
            else page_number
        )
        citation = f"{prefix}{filename} — Page {page_display}"
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)

    return sources