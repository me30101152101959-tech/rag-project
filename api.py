"""
api.py — FastAPI Backend Server for Data Analysis RAG
Uses OpenRouter (ChatOpenAI) as LLM + HuggingFace local embeddings.
LangChain LCEL (0.3.x+ compatible — no RetrievalQA).
Includes CORS Middleware for seamless frontend integration.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_core import (
    verify_environment,
    build_rag_chain,
    format_sources,
    DEFAULT_MODEL,
    EMBEDDING_MODEL
)


# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────

rag_chain = None
startup_error: str | None = None


# ─────────────────────────────────────────────
# LIFESPAN CONTEXT MANAGER
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline once on server startup."""
    global rag_chain, startup_error
    print("🚀 Initializing Data Analysis RAG pipeline...")
    print(f"   LLM:         OpenRouter → {DEFAULT_MODEL}")
    print(f"   Embeddings: HuggingFace ({EMBEDDING_MODEL}, local)")
    try:
        api_key = verify_environment(exit_on_fail=False)
        rag_chain = build_rag_chain(api_key=api_key)
        print("✅ RAG pipeline ready.")
    except Exception as e:
        startup_error = str(e)
        rag_chain = None
        print(f"❌ Startup error: {startup_error}")
    yield
    print("👋 Shutting down Data Analysis RAG API.")


# ─────────────────────────────────────────────
# FASTAPI APP INSTANCE
# ─────────────────────────────────────────────

app = FastAPI(
    title="Data Analysis RAG API",
    description=(
        "Production-grade RAG API for Data Analysis mentoring. "
        "Powered by OpenRouter LLM and HuggingFace local embeddings."
    ),
    version="2.1.0",
    lifespan=lifespan
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The data analysis question to ask the RAG mentor.",
        json_schema_extra={
            "example": "What is the difference between a LEFT JOIN and an INNER JOIN?"
        }
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="OpenRouter model identifier to use for this request."
    )


class AnswerResponse(BaseModel):
    question: str = Field(description="The original question asked.")
    answer: str = Field(description="The AI mentor's structured 5-step response.")
    sources: list[str] = Field(description="Source citations (book name + page number).")
    model_used: str = Field(description="The OpenRouter model that generated the answer.")


class HealthResponse(BaseModel):
    status: str
    message: str
    rag_loaded: bool
    llm_provider: str
    llm_model: str
    embedding_model: str
    startup_error: str | None


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check — verify API status and pipeline readiness."""
    return HealthResponse(
        status="online" if rag_chain is not None else "degraded",
        message=(
            "Data Analysis RAG API is running and ready."
            if rag_chain is not None
            else "API running but RAG pipeline failed to initialize. See startup_error."
        ),
        rag_loaded=rag_chain is not None,
        llm_provider="OpenRouter",
        llm_model=DEFAULT_MODEL,
        embedding_model=EMBEDDING_MODEL,
        startup_error=startup_error
    )


@app.post("/ask", response_model=AnswerResponse, tags=["RAG Query"])
async def ask_question(request: QuestionRequest):
    """
    Submit a data analysis question and receive a structured mentorship response.
    Uses LCEL create_retrieval_chain (LangChain 0.3.x+ compatible).
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"RAG pipeline is not initialized. "
                f"Startup error: {startup_error or 'Unknown'}. "
                f"Ensure OPENROUTER_API_KEY is set and ChromaDB exists."
            )
        )

    try:
        # Use a fresh chain instance if a non-default OpenRouter model is requested
        if request.model != DEFAULT_MODEL:
            api_key = verify_environment(exit_on_fail=False)
            active_chain = build_rag_chain(api_key=api_key, model=request.model)
        else:
            active_chain = rag_chain

        # LCEL execution (Input key: "input", Output keys: "answer" & "context")
        result = active_chain.invoke({"input": request.question})
        answer = result.get("answer", "No answer could be generated.")
        source_docs = result.get("context", [])
        sources = format_sources(source_docs, icon=False)

        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            model_used=request.model
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your question: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
