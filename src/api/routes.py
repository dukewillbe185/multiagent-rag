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
    ErrorResponse,
    ChunkInfo
)
from src.data_pipeline.document_extractor import DocumentExtractor
from src.data_pipeline.text_chunker import TextChunker
from src.data_pipeline.embedder import Embedder
from src.data_pipeline.indexer import AzureSearchIndexer
from src.agents.rag_agent import MultiAgentRAG, RagState
from src.monitoring.logger import track_performance, track_event, track_metric
from src.utils.session_memory import get_session_manager
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
    1. Validates question through guardrail agent
    2. Retrieves session context (previous Q&A if exists)
    3. Retrieves relevant chunks from Azure AI Search
    4. Identifies user intent
    5. Generates answer using multi-agent system with conversation memory
    6. Updates session with current Q&A

    Args:
        request: Query request with question, session_id, and optional top_k, user_id, metadata

    Returns:
        Generated answer with detailed chunk information and session context
    """
    start_time = time.time()
    session_id = request.session_id
    question = request.question

    logger.info(
        f"[{session_id}] Query received: {question[:100]}...",
        extra={"session_id": session_id, "question": question}
    )

    # Track event with session_id
    track_event("query_started", {
        "session_id": session_id,
        "question": question[:100],
        "user_id": request.user_id
    })

    try:
        # Get session manager
        config = get_config()
        session_timeout = config.get_optional_env("SESSION_TIMEOUT_MINUTES", "30")
        session_manager = get_session_manager(timeout_minutes=int(session_timeout))

        # Retrieve session context (previous turn)
        session_context = session_manager.get_session(session_id)

        previous_question = ""
        previous_answer = ""
        conversation_turn = 1

        if session_context:
            previous_question = session_context.get("previous_question", "")
            previous_answer = session_context.get("previous_answer", "")
            conversation_turn = session_context.get("turn_count", 0) + 1

            logger.info(
                f"[{session_id}] Session context loaded, turn: {conversation_turn}",
                extra={
                    "session_id": session_id,
                    "turn": conversation_turn,
                    "has_previous_context": bool(previous_question)
                }
            )
        else:
            logger.info(
                f"[{session_id}] New session started",
                extra={"session_id": session_id}
            )

        # Get guardrail settings from config
        guardrail_enabled = config.get_optional_env("GUARDRAIL_ENABLED", "true").lower() == "true"
        guardrail_strictness = config.get_optional_env("GUARDRAIL_STRICTNESS", "medium")

        # Initialize RAG system
        rag_system = MultiAgentRAG(
            top_k=request.top_k or 5,
            guardrail_enabled=guardrail_enabled,
            guardrail_strictness=guardrail_strictness
        )

        # Create initial state with session context
        initial_state = RagState(
            session_id=session_id,
            user_id=request.user_id or "",
            current_question=question,
            previous_question=previous_question,
            previous_answer=previous_answer,
            retrieved_chunks=[],
            retrieved_metadata=[],
            intent="",
            answer="",
            conversation_turn=conversation_turn,
            guardrail_passed=False,
            guardrail_reason=""
        )

        # Run the workflow
        logger.info(f"[{session_id}] Starting RAG workflow...")
        result = rag_system.graph.invoke(initial_state)

        # Check if guardrail passed
        guardrail_passed = result.get("guardrail_passed", True)
        guardrail_reason = result.get("guardrail_reason", "")

        if not guardrail_passed:
            # Guardrail rejected the question
            processing_time = time.time() - start_time

            rejection_messages = {
                "irrelevant": "Your question doesn't appear to be related to the indexed documents. Please ask questions relevant to the available content.",
                "unsafe": "Your question has been flagged as inappropriate or potentially unsafe. Please rephrase your question."
            }

            # Determine rejection type from reason
            rejection_type = "irrelevant"
            if "unsafe" in guardrail_reason.lower() or "jailbreak" in guardrail_reason.lower():
                rejection_type = "unsafe"

            answer = rejection_messages.get(rejection_type, "Your question could not be processed. Please try rephrasing.")

            logger.warning(
                f"[{session_id}] Guardrail rejected question: {guardrail_reason}",
                extra={
                    "session_id": session_id,
                    "guardrail_decision": "rejected",
                    "guardrail_reason": guardrail_reason,
                    "question": question
                }
            )

            track_event("query_guardrail_rejected", {
                "session_id": session_id,
                "question": question[:100],
                "reason": guardrail_reason,
                "rejection_type": rejection_type
            })

            return QueryResponse(
                success=False,
                question=question,
                answer=answer,
                intent="rejected",
                session_id=session_id,
                conversation_turn=conversation_turn,
                chunks_retrieved=0,
                retrieved_chunks=[],
                sources=[],
                processing_time_seconds=round(processing_time, 2),
                guardrail_passed=False
            )

        # Guardrail passed - process normally
        logger.info(
            f"[{session_id}] Guardrail passed",
            extra={"session_id": session_id, "guardrail_decision": "approved"}
        )

        # Build ChunkInfo objects from retrieved metadata
        retrieved_chunks_info = []
        for meta in result.get("retrieved_metadata", []):
            content = meta.get("content", "")
            chunk_info = ChunkInfo(
                chunk_id=meta.get("chunk_id", "unknown"),
                source_file=meta.get("source_file", "unknown"),
                chunk_index=meta.get("chunk_index", 0),
                score=meta.get("score", 0.0),
                content_preview=content[:200] if content else "",
                full_content=content
            )
            retrieved_chunks_info.append(chunk_info)

        # Extract sources
        sources = list(set(
            meta.get("source_file", "Unknown")
            for meta in result.get("retrieved_metadata", [])
        ))

        # Calculate processing time
        processing_time = time.time() - start_time

        # Update session with current Q&A
        session_manager.update_session(
            session_id=session_id,
            question=question,
            answer=result["answer"],
            user_id=request.user_id,
            metadata=request.metadata
        )

        logger.info(
            f"[{session_id}] Session updated with current Q&A",
            extra={"session_id": session_id, "turn": conversation_turn}
        )

        # Track metrics with session_id
        track_metric("query_processing_time", processing_time, {
            "session_id": session_id,
            "intent": result['intent'],
            "turn": conversation_turn
        })
        track_metric("chunks_retrieved", len(retrieved_chunks_info), {
            "session_id": session_id,
            "intent": result['intent']
        })

        # Calculate average score
        avg_score = sum(c.score for c in retrieved_chunks_info) / len(retrieved_chunks_info) if retrieved_chunks_info else 0.0

        track_event("query_completed", {
            "session_id": session_id,
            "question": question[:100],
            "intent": result['intent'],
            "chunks_retrieved": len(retrieved_chunks_info),
            "avg_retrieval_score": round(avg_score, 4),
            "turn": conversation_turn,
            "had_previous_context": bool(previous_question)
        })

        logger.info(
            f"[{session_id}] Query processing complete",
            extra={
                "session_id": session_id,
                "turn": conversation_turn,
                "chunks_retrieved": len(retrieved_chunks_info),
                "processing_time": round(processing_time, 2)
            }
        )

        return QueryResponse(
            success=True,
            question=question,
            answer=result['answer'],
            intent=result['intent'],
            session_id=session_id,
            conversation_turn=conversation_turn,
            chunks_retrieved=len(retrieved_chunks_info),
            retrieved_chunks=retrieved_chunks_info,
            sources=sources,
            processing_time_seconds=round(processing_time, 2),
            guardrail_passed=True
        )

    except Exception as e:
        logger.error(
            f"[{session_id}] Error processing query: {e}",
            extra={"session_id": session_id, "error": str(e)}
        )
        track_event("query_failed", {
            "session_id": session_id,
            "question": question[:100],
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
