"""
RAG API routes — Add these to your FastAPI app.
Provides endpoints to query the knowledge base from the dashboard.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Global retriever instance (initialized on first use)
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        try:
            from rag.retriever import RAGRetriever
            _retriever = RAGRetriever(use_local=True)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"RAG not initialized. Run: python -m rag.indexer --local. Error: {str(e)}"
            )
    return _retriever


class RAGQuery(BaseModel):
    question: str
    n_results: int = 3


class DiagnosisQuery(BaseModel):
    slice_name: str
    issue_type: str
    metrics: Dict


@router.get("/status")
async def rag_status():
    """Check if RAG system is initialized."""
    try:
        r = get_retriever()
        return {"status": "ok", **r.status}
    except HTTPException:
        return {"status": "not_initialized", "message": "Run: python -m rag.indexer --local"}


@router.post("/query")
async def rag_query(q: RAGQuery):
    """Query the knowledge base with a natural language question."""
    r = get_retriever()
    results = r.query(q.question, n_results=q.n_results)
    context = r.get_context_for_prompt(q.question, n_results=q.n_results)
    return {
        "results": results,
        "context": context,
        "query": q.question,
    }


@router.post("/diagnose")
async def rag_diagnose(q: DiagnosisQuery):
    """Get diagnosis context for a specific slice issue."""
    r = get_retriever()
    context = r.query_for_diagnosis(q.slice_name, q.issue_type, q.metrics)
    return {
        "context": context,
        "slice": q.slice_name,
        "issue": q.issue_type,
    }
