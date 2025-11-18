"""
Pydantic models for FastAPI request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Chunk Information Model
class ChunkInfo(BaseModel):
    """Detailed information about a retrieved chunk."""
    chunk_id: str = Field(..., description="The ID from Azure Search index")
    source_file: str = Field(..., description="Source file name")
    chunk_index: int = Field(..., description="Chunk position in the document")
    score: float = Field(..., description="Search score (vector similarity + BM25 if hybrid)")
    content_preview: str = Field(..., description="First 200 characters of the chunk")
    full_content: str = Field(..., description="Full chunk text for transparency")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "doc_pdf_chunk_5",
                "source_file": "document.pdf",
                "chunk_index": 5,
                "score": 0.8542,
                "content_preview": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience...",
                "full_content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves."
            }
        }


# Request Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=1, description="User's question")
    session_id: str = Field(..., min_length=1, description="Session identifier provided by backend")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    user_id: Optional[str] = Field(None, description="Optional user identifier for tracking")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "session_id": "session_12345",
                "top_k": 5,
                "user_id": "user_67890",
                "metadata": {"source": "web_app"}
            }
        }


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Status message")
    file_name: str = Field(..., description="Name of uploaded file")
    chunks_created: int = Field(..., description="Number of chunks created")
    chunks_indexed: int = Field(..., description="Number of chunks indexed")
    processing_time_seconds: float = Field(..., description="Total processing time")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document processed successfully",
                "file_name": "document.pdf",
                "chunks_created": 25,
                "chunks_indexed": 25,
                "processing_time_seconds": 12.5
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    success: bool = Field(..., description="Whether query was successful")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    intent: str = Field(..., description="Identified intent")
    session_id: str = Field(..., description="Session identifier echoed back for tracking")
    conversation_turn: int = Field(..., description="Turn number in this session")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    retrieved_chunks: List[ChunkInfo] = Field(..., description="Detailed chunk information")
    sources: List[str] = Field(..., description="List of source files")
    processing_time_seconds: float = Field(..., description="Query processing time")
    guardrail_passed: bool = Field(..., description="Whether question passed guardrail check")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence...",
                "intent": "definition",
                "session_id": "session_12345",
                "conversation_turn": 1,
                "chunks_retrieved": 5,
                "retrieved_chunks": [
                    {
                        "chunk_id": "doc_pdf_chunk_5",
                        "source_file": "document.pdf",
                        "chunk_index": 5,
                        "score": 0.8542,
                        "content_preview": "Machine learning is a subset of artificial intelligence...",
                        "full_content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
                    }
                ],
                "sources": ["document.pdf"],
                "processing_time_seconds": 2.3,
                "guardrail_passed": True
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    services: Dict[str, str] = Field(..., description="Status of dependent services")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "services": {
                    "azure_search": "connected",
                    "azure_ai_foundry": "connected",
                    "document_intelligence": "connected"
                }
            }
        }


class IndexStatusResponse(BaseModel):
    """Response model for index status endpoint."""
    index_name: str = Field(..., description="Name of the search index")
    exists: bool = Field(..., description="Whether the index exists")
    document_count: int = Field(..., description="Number of documents in index")
    vector_search_enabled: bool = Field(..., description="Whether vector search is enabled")
    semantic_search_enabled: bool = Field(..., description="Whether semantic search is enabled")
    fields: Optional[List[str]] = Field(None, description="List of index fields")

    class Config:
        json_schema_extra = {
            "example": {
                "index_name": "rag-documents-index",
                "exists": True,
                "document_count": 150,
                "vector_search_enabled": True,
                "semantic_search_enabled": True,
                "fields": ["id", "content", "content_vector", "source_file"]
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool = Field(False, description="Always False for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "ProcessingError",
                "message": "Failed to process document",
                "detail": "Document format not supported"
            }
        }
