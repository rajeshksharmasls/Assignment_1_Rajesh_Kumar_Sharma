"""
API module — Multi-tenant RAG Chatbot (FastAPI).
Supports:
- User-level namespace isolation
- Document versioning
- Lifecycle-aware retrieval
- Duplicate-safe generation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ingestion import ingest_document
from embedding import upsert_chunks, delete_previous_versions
from retrieval import search
from reranker import rerank
from generation import generate_answer

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Classic Chatbot (Multi-Tenant)",
    description="Multi-user Retrieval-Augmented Generation system with versioned documents",
    version="2.0.0",
)

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    user_id: str
    file_path: str
    document_id: str
    version: str = "v1"


class IngestResponse(BaseModel):
    user_id: str
    document_id: str
    version: str
    chunks: int
    message: str


class ChatRequest(BaseModel):
    user_id: str
    question: str
    use_reranker: bool = True
    version: Optional[str] = None
    debug: bool = False


class SearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 10
    use_reranker: bool = True
    version: Optional[str] = None


class GenerateRequest(BaseModel):
    user_id: str
    question: str
    top_k: int = 10
    top_n: int = 5
    use_reranker: bool = True
    version: Optional[str] = None


class SourceChunk(BaseModel):
    id: str
    score: float
    source: str
    pages: str
    chunk_text: str
    citation: str
    document_id: str
    version: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    pipeline: str


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion (User-Isolated + Versioned)
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest):
    try:
        records = ingest_document(req.file_path)

        # Version update strategy:
        # Delete previous versions before inserting new one
        delete_previous_versions(req.user_id, req.document_id)

        upserted = upsert_chunks(
            user_id=req.user_id,
            document_id=req.document_id,
            records=records,
            version=req.version,
        )

        return IngestResponse(
            user_id=req.user_id,
            document_id=req.document_id,
            version=req.version,
            chunks=upserted,
            message="Document ingested successfully",
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Search Only
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/search")
def search_endpoint(req: SearchRequest):
    try:
        if req.use_reranker:
            results = rerank(
                user_id=req.user_id,
                query=req.query,
                top_k=req.top_k,
                version=req.version,
            )
        else:
            results = search(
                user_id=req.user_id,
                query=req.query,
                top_k=req.top_k,
                version=req.version,
            )

        return {
            "user_id": req.user_id,
            "query": req.query,
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Chat (Full RAG)
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        # Step 1: Retrieve
        retrieved_chunks = search(
            user_id=req.user_id,
            query=req.question,
            version=req.version,
        )

        # Step 2: Rerank (optional)
        if req.use_reranker:
            chunks = rerank(
                user_id=req.user_id,
                query=req.question,
                version=req.version,
            )
            pipeline = "retrieve → rerank → generate"
        else:
            chunks = retrieved_chunks
            pipeline = "retrieve → generate"

        if not chunks:
            return ChatResponse(
                answer="No relevant information found.",
                sources=[],
                pipeline=pipeline,
            )

        # Step 3: Generate answer
        answer = generate_answer(
            user_id=req.user_id,
            query=req.question,
            chunks=chunks,
        )

        # Step 4: Format sources
        sources = [
            SourceChunk(
                id=c["id"],
                score=c["score"],
                source=c["source"],
                pages=c.get("pages", ""),
                chunk_text=c["chunk_text"][:200] + "...",
                citation=f"[{i}]",
                document_id=c.get("document_id", ""),
                version=c.get("version", ""),
            )
            for i, c in enumerate(chunks, 1)
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            pipeline=pipeline,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Generate
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/generate", response_model=ChatResponse)
def generate_endpoint(req: GenerateRequest):
    try:
        retrieved_chunks = search(
            user_id=req.user_id,
            query=req.question,
            top_k=req.top_k,
            version=req.version,
        )

        if req.use_reranker:
            chunks = rerank(
                user_id=req.user_id,
                query=req.question,
                top_k=req.top_k,
                top_n=req.top_n,
                version=req.version,
            )
            pipeline = "retrieve → rerank → generate"
        else:
            chunks = retrieved_chunks[: req.top_n]
            pipeline = "retrieve → generate"

        if not chunks:
            return ChatResponse(
                answer="No relevant information found.",
                sources=[],
                pipeline=pipeline,
            )

        answer = generate_answer(
            user_id=req.user_id,
            query=req.question,
            chunks=chunks,
        )

        sources = [
            SourceChunk(
                id=c["id"],
                score=c["score"],
                source=c["source"],
                pages=c.get("pages", ""),
                chunk_text=c["chunk_text"][:200] + "...",
                citation=f"[{i}]",
                document_id=c.get("document_id", ""),
                version=c.get("version", ""),
            )
            for i, c in enumerate(chunks, 1)
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            pipeline=pipeline,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
