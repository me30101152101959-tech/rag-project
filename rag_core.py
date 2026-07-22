"""
rag_core.py — Shared Core Module for Data Analysis RAG
Centralizes environment verification, LLM initialization (Groq API),
embeddings initialization, and RAG chain construction using modern LangChain LCEL.

LangChain 0.3.0+ native replacement:
  create_retrieval_chain + create_stuff_documents_chain (LCEL-native).

Chain I/O contract:
  Input:  chain.invoke({"input": question_string})
  Output: result["answer"]       — the LLM response string
          result["context"]      — list of retrieved Document objects
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CHROMA_DIR = "./chroma_db"
BOOKS_DIR = "./books"
COLLECTION_NAME = "data_analysis_rag"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# Note: LCEL create_stuff_documents_chain requires the variable
# for the user question to be named {input} and retrieved chunks {context}.
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
    Returns the API key string on success.
    Calls sys.exit(1) on failure if exit_on_fail=True (CLI),
    or raises RuntimeError if exit_on_fail=False (server/app usage).
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
    """Confirm the ChromaDB directory exists before loading."""
    if not os.path.exists(CHROMA_DIR):
        msg = (
            f"ChromaDB not found at '{CHROMA_DIR}'.\n"
            "  → Run: python ingest.py"
        )
        if exit_on_fail:
            print(f"❌ ERROR: {msg}")
            sys.exit(1)
        raise RuntimeError(msg)


# ─────────────────────────────────────────────
# EMBEDDINGS — FREE LOCAL HUGGINGFACE MODEL
# ─────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFaceEmbeddings instance using all-MiniLM-L6-v2.
    Model is downloaded once and cached locally by sentence-transformers.
    Runs entirely on CPU — no GPU or API key required.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ─────────────────────────────────────────────
# LLM — GROQ API via ChatOpenAI
# ─────────────────────────────────────────────

def get_llm(api_key: str, model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance configured for Groq API.
    Groq is OpenAI-API-compatible — we override base_url and use GROQ_API_KEY.
    """
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
    """Load the persisted ChromaDB vectorstore with HuggingFace embeddings."""
    verify_chroma_exists(exit_on_fail=False)
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


# ─────────────────────────────────────────────
# RAG CHAIN BUILDER  (LCEL — LangChain 0.3.x+)
# ─────────────────────────────────────────────

def build_rag_chain(api_key: str, model: str = DEFAULT_MODEL):
    """
    Build and return a LangChain LCEL retrieval chain using:
      - ChromaDB vectorstore (HuggingFace embeddings)
      - Groq LLM API (via ChatOpenAI client)
      - 5-step teaching protocol system prompt

    Caller contract:
      result = chain.invoke({"input": question})
      answer = result["answer"]
      docs   = result["context"]   # list of retrieved Document objects

    Args:
        api_key: Groq API key string.
        model:   Groq model identifier (default: llama-3.3-70b-versatile).

    Returns:
        Configured LCEL chain (create_retrieval_chain).
    """
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = get_llm(api_key=api_key, model=model)

    # LCEL prompt — variables must be {context} and {input}
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)

    # Stage 1: stuff retrieved docs into the prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Stage 2: wire the retriever to the document chain
    chain = create_retrieval_chain(retriever, document_chain)

    return chain


# ─────────────────────────────────────────────
# SOURCE CITATION FORMATTER
# ─────────────────────────────────────────────

def format_sources(source_documents: list, icon: bool = True) -> list[str]:
    """
    Extract unique, human-readable source citations from retrieved documents.

    Args:
        source_documents: List of LangChain Document objects (from result["context"]).
        icon:              Prepend a book emoji for CLI/UI display.

    Returns:
        Deduplicated list of citation strings like "📖 book.pdf — Page 42".
    """
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