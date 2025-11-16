"""
Pydantic models for FastAPI request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Request Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=1, description="User's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "top_k": 5
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
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    sources: List[str] = Field(..., description="List of source files")
    processing_time_seconds: float = Field(..., description="Query processing time")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence...",
                "intent": "definition",
                "chunks_retrieved": 5,
                "sources": ["document.pdf"],
                "processing_time_seconds": 2.3
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
