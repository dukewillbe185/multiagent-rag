"""
FastAPI Application for Multi-Agent RAG System.

This is the main entry point for the REST API server.
"""

import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.monitoring.logger import setup_logging
from src.utils.session_memory import get_session_manager
from config import get_config, ConfigurationError

# Setup logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

# Background task control
_cleanup_task = None
_shutdown_event = asyncio.Event()

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent RAG System API",
    description="""
    Multi-Agent RAG (Retrieval-Augmented Generation) System using Azure Services.

    This API provides endpoints for:
    - Document upload and processing (PDF → chunks → embeddings → index)
    - Question answering using multi-agent workflow
    - Index management
    - Health monitoring

    ## Architecture

    The system uses a 3-agent workflow powered by LangGraph:
    1. **Supervisor Retrieval Agent** - Retrieves relevant document chunks
    2. **Intent Identifier Agent** - Identifies user's question intent
    3. **Answer Generator Agent** - Generates comprehensive answers

    ## Azure Services

    - **Azure AI Document Intelligence** - PDF text extraction
    - **Azure AI Foundry** - Embeddings and GPT-4 (deployed models)
    - **Azure AI Search** - Vector and hybrid search
    - **Azure Application Insights** - Monitoring and logging
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def session_cleanup_task():
    """
    Background task to periodically clean up expired sessions.

    Runs every 5 minutes to remove sessions that have been inactive
    for longer than the configured timeout period.
    """
    global _shutdown_event

    logger.info("Session cleanup task started")

    while not _shutdown_event.is_set():
        try:
            # Wait 5 minutes between cleanups (or until shutdown)
            await asyncio.wait_for(_shutdown_event.wait(), timeout=300)  # 5 minutes
        except asyncio.TimeoutError:
            # Timeout means we should clean up (not shutdown)
            try:
                session_manager = get_session_manager()
                cleaned = session_manager.cleanup_expired_sessions()

                if cleaned > 0:
                    logger.info(f"Session cleanup: {cleaned} expired sessions removed")

                # Log current session count
                active_sessions = session_manager.get_session_count()
                logger.info(f"Active sessions: {active_sessions}")

            except Exception as e:
                logger.error(f"Error during session cleanup: {e}")

    logger.info("Session cleanup task stopped")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global _cleanup_task

    logger.info("=" * 60)
    logger.info("Multi-Agent RAG API Starting Up")
    logger.info("=" * 60)

    try:
        # Load and validate configuration
        config = get_config()
        logger.info("Configuration loaded successfully")

        # Log startup info
        logger.info(f"API Version: 1.0.0")
        logger.info(f"Search Index: {config.search_index_name}")
        logger.info(f"Chunk Size: {config.chunk_size}")
        logger.info(f"Retrieval Top-K: {config.retrieval_top_k}")

        # Get guardrail settings
        guardrail_enabled = config.get_optional_env("GUARDRAIL_ENABLED", "true")
        session_timeout = config.get_optional_env("SESSION_TIMEOUT_MINUTES", "30")

        logger.info(f"Guardrail Enabled: {guardrail_enabled}")
        logger.info(f"Session Timeout: {session_timeout} minutes")

        # Start session cleanup background task
        _cleanup_task = asyncio.create_task(session_cleanup_task())
        logger.info("Session cleanup background task started")

        logger.info("=" * 60)
        logger.info("API is ready to accept requests")
        logger.info("Documentation available at: /docs")
        logger.info("=" * 60)

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file and ensure all required variables are set")
        raise
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _cleanup_task, _shutdown_event

    logger.info("=" * 60)
    logger.info("Multi-Agent RAG API Shutting Down")
    logger.info("=" * 60)

    # Signal cleanup task to stop
    _shutdown_event.set()

    # Wait for cleanup task to finish
    if _cleanup_task:
        try:
            await asyncio.wait_for(_cleanup_task, timeout=5.0)
            logger.info("Session cleanup task stopped gracefully")
        except asyncio.TimeoutError:
            logger.warning("Session cleanup task did not stop in time")
            _cleanup_task.cancel()

    logger.info("Shutdown complete")


# Include routers
app.include_router(router, prefix="/api/v1", tags=["Multi-Agent RAG"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multi-Agent RAG System API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/api/v1/health"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Run with uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
