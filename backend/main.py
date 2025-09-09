from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import os

from routers import document_router, query_router
from core.config import Settings

load_dotenv()
settings = Settings()

app = FastAPI(
    title="Domain-Agnostic RAG System",
    description="AI-powered document processing and question answering system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_router.router, prefix="/api/documents", tags=["documents"])
app.include_router(query_router.router, prefix="/api/query", tags=["query"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)