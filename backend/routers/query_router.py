from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.rag_pipeline import RAGPipeline

router = APIRouter()
rag_pipeline = RAGPipeline()

class Query(BaseModel):
    question: str
    domain: str = None
    role: str = None

@router.post("/ask")
async def ask_question(query: Query):
    """Process a question using the RAG pipeline"""
    try:
        # Get answer using RAG pipeline
        response = await rag_pipeline.process_query(
            question=query.question,
            domain=query.domain,
            role=query.role
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))