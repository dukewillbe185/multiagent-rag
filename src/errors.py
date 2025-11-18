"""
Custom exception classes for the Multi-Agent RAG system.

Provides specific error types for better error handling and debugging.
"""


class RAGSystemError(Exception):
    """Base exception for all RAG system errors."""
    pass


class GuardrailRejectionError(RAGSystemError):
    """Raised when a question is rejected by the guardrail agent."""
    def __init__(self, message: str, reason: str, rejection_type: str = "irrelevant"):
        self.reason = reason
        self.rejection_type = rejection_type  # "irrelevant" or "unsafe"
        super().__init__(message)


class RetrievalError(RAGSystemError):
    """Raised when document retrieval fails."""
    pass


class GenerationError(RAGSystemError):
    """Raised when answer generation fails."""
    pass


class EmbeddingError(RAGSystemError):
    """Raised when embedding generation fails."""
    pass


class IndexingError(RAGSystemError):
    """Raised when document indexing fails."""
    pass


class SessionError(RAGSystemError):
    """Raised when session management operations fail."""
    pass


class ConfigurationError(RAGSystemError):
    """Raised when there's a configuration issue."""
    pass


class DocumentExtractionError(RAGSystemError):
    """Raised when document text extraction fails."""
    pass


class ChunkingError(RAGSystemError):
    """Raised when text chunking fails."""
    pass


class ValidationError(RAGSystemError):
    """Raised when input validation fails."""
    pass


def get_user_friendly_message(error: Exception) -> str:
    """
    Convert system errors to user-friendly messages.

    Args:
        error: Exception that occurred

    Returns:
        User-friendly error message
    """
    error_messages = {
        GuardrailRejectionError: lambda e: str(e),  # Use the custom message
        RetrievalError: lambda e: "Unable to retrieve relevant information. Please try again.",
        GenerationError: lambda e: "Unable to generate an answer at this time. Please try again.",
        EmbeddingError: lambda e: "Error processing your question. Please try again.",
        IndexingError: lambda e: "Error indexing the document. Please try again.",
        SessionError: lambda e: "Session error occurred. Please refresh and try again.",
        ConfigurationError: lambda e: "System configuration error. Please contact support.",
        DocumentExtractionError: lambda e: "Error extracting text from document. Please check the file format.",
        ChunkingError: lambda e: "Error processing document content. Please try again.",
        ValidationError: lambda e: str(e),  # Use the validation error message directly
    }

    error_type = type(error)
    if error_type in error_messages:
        return error_messages[error_type](error)

    # Default message for unknown errors
    return "An unexpected error occurred. Please try again or contact support."


def get_error_category(error: Exception) -> str:
    """
    Get the error category for logging and metrics.

    Args:
        error: Exception that occurred

    Returns:
        Error category string
    """
    if isinstance(error, GuardrailRejectionError):
        return "guardrail_rejection"
    elif isinstance(error, RetrievalError):
        return "retrieval_error"
    elif isinstance(error, GenerationError):
        return "generation_error"
    elif isinstance(error, EmbeddingError):
        return "embedding_error"
    elif isinstance(error, IndexingError):
        return "indexing_error"
    elif isinstance(error, SessionError):
        return "session_error"
    elif isinstance(error, ConfigurationError):
        return "configuration_error"
    elif isinstance(error, DocumentExtractionError):
        return "extraction_error"
    elif isinstance(error, ChunkingError):
        return "chunking_error"
    elif isinstance(error, ValidationError):
        return "validation_error"
    else:
        return "unknown_error"
