"""
FastAPI route handlers for the Multi-Agent RAG API.
"""

import logging
import time
import tempfile
import os
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.models import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    HealthResponse,
    IndexStatusResponse,
    ErrorResponse
)
from src.data_pipeline.document_extractor import DocumentExtractor
from src.data_pipeline.text_chunker import TextChunker
from src.data_pipeline.embedder import Embedder
from src.data_pipeline.indexer import AzureSearchIndexer
from src.agents.rag_agent import MultiAgentRAG
from src.monitoring.logger import track_performance, track_event, track_metric
from config import get_config

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
@track_performance("health_check")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service health status and dependent services status
    """
    logger.info("Health check requested")

    config = get_config()

    # Check services (basic connectivity check)
    services_status = {
        "azure_search": "unknown",
        "azure_ai_foundry": "unknown",
        "document_intelligence": "unknown"
    }

    # Try to check Azure Search
    try:
        indexer = AzureSearchIndexer()
        stats = indexer.get_index_statistics()
        services_status["azure_search"] = "connected" if stats else "error"
    except Exception as e:
        logger.warning(f"Azure Search health check failed: {e}")
        services_status["azure_search"] = "error"

    # Azure AI Foundry and Document Intelligence are checked when used
    services_status["azure_ai_foundry"] = "configured"
    services_status["document_intelligence"] = "configured"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        services=services_status
    )


@router.get("/index/status", response_model=IndexStatusResponse)
@track_performance("index_status")
async def get_index_status():
    """
    Get Azure AI Search index status.

    Returns:
        Index statistics and configuration
    """
    logger.info("Index status requested")

    try:
        indexer = AzureSearchIndexer()
        stats = indexer.get_index_statistics()

        return IndexStatusResponse(
            index_name=stats.get("index_name", "unknown"),
            exists=stats.get("exists", stats.get("document_count", 0) >= 0),
            document_count=stats.get("document_count", 0),
            vector_search_enabled=stats.get("vector_search_enabled", False),
            semantic_search_enabled=stats.get("semantic_search_enabled", False),
            fields=stats.get("fields", [])
        )

    except Exception as e:
        logger.error(f"Error getting index status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index status: {str(e)}"
        )


@router.post("/upload", response_model=UploadResponse)
@track_performance("document_upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.

    This endpoint:
    1. Extracts text from PDF
    2. Chunks the text
    3. Generates embeddings
    4. Indexes to Azure AI Search

    Args:
        file: PDF file to upload

    Returns:
        Processing results and statistics
    """
    start_time = time.time()

    logger.info(f"Document upload started: {file.filename}")

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    # Track event
    track_event("document_upload_started", {"file_name": file.filename})

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        logger.info(f"File saved temporarily: {tmp_file_path}")

        try:
            # Step 1: Extract text
            logger.info("Step 1: Extracting text from PDF...")
            extractor = DocumentExtractor()
            extraction_result = extractor.extract_text_from_pdf(tmp_file_path)
            extracted_text = extraction_result["text"]
            metadata = extraction_result["metadata"]
            metadata["source_file"] = file.filename  # Use original filename

            logger.info(f"Extracted {len(extracted_text)} characters from {extraction_result['page_count']} pages")

            # Step 2: Chunk text
            logger.info("Step 2: Chunking text...")
            chunker = TextChunker()
            chunks = chunker.chunk_text(extracted_text, metadata=metadata)

            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Generate embeddings
            logger.info("Step 3: Generating embeddings...")
            embedder = Embedder()
            chunks_with_embeddings = embedder.embed_chunks(chunks)

            logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")

            # Step 4: Create index if needed and index documents
            logger.info("Step 4: Indexing documents...")
            indexer = AzureSearchIndexer()

            # Ensure index exists
            indexer.create_index(recreate=False)

            # Index documents
            indexing_result = indexer.index_documents(chunks_with_embeddings)

            logger.info(
                f"Indexed {indexing_result['indexed']} chunks "
                f"({indexing_result['failed']} failed)"
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Track metrics
            track_metric("chunks_created", len(chunks), {"file": file.filename})
            track_metric("chunks_indexed", indexing_result['indexed'], {"file": file.filename})
            track_metric("processing_time", processing_time, {"file": file.filename})
            track_event("document_upload_completed", {
                "file_name": file.filename,
                "chunks": len(chunks),
                "indexed": indexing_result['indexed']
            })

            return UploadResponse(
                success=True,
                message="Document processed successfully",
                file_name=file.filename,
                chunks_created=len(chunks),
                chunks_indexed=indexing_result['indexed'],
                processing_time_seconds=round(processing_time, 2)
            )

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        track_event("document_upload_failed", {
            "file_name": file.filename,
            "error": str(e)
        })

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
@track_performance("query_processing")
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question.

    This endpoint:
    1. Retrieves relevant chunks from Azure AI Search
    2. Identifies user intent
    3. Generates answer using multi-agent system

    Args:
        request: Query request with question and optional top_k

    Returns:
        Generated answer with metadata
    """
    start_time = time.time()

    logger.info(f"Query received: {request.question}")

    # Track event
    track_event("query_started", {"question": request.question[:100]})

    try:
        # Initialize RAG system
        rag_system = MultiAgentRAG(top_k=request.top_k or 5)

        # Process query
        result = rag_system.query(request.question)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Track metrics
        track_metric("query_processing_time", processing_time, {"intent": result['intent']})
        track_metric("chunks_retrieved", result['chunks_retrieved'], {"intent": result['intent']})
        track_event("query_completed", {
            "question": request.question[:100],
            "intent": result['intent'],
            "chunks_retrieved": result['chunks_retrieved']
        })

        return QueryResponse(
            success=True,
            question=result['question'],
            answer=result['answer'],
            intent=result['intent'],
            chunks_retrieved=result['chunks_retrieved'],
            sources=result['sources'],
            processing_time_seconds=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        track_event("query_failed", {
            "question": request.question[:100],
            "error": str(e)
        })

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@router.post("/index/create")
@track_performance("index_creation")
async def create_index(recreate: bool = False):
    """
    Create or recreate the Azure AI Search index.

    Args:
        recreate: If True, delete and recreate existing index

    Returns:
        Index creation status
    """
    logger.info(f"Index creation requested (recreate={recreate})")

    try:
        indexer = AzureSearchIndexer()
        created = indexer.create_index(recreate=recreate)

        message = "Index created successfully" if created else "Index already exists"

        track_event("index_creation", {"created": created, "recreate": recreate})

        return {
            "success": True,
            "message": message,
            "created": created
        }

    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create index: {str(e)}"
        )
