"""
Text Chunking Module using LangChain's RecursiveCharacterTextSplitter.

This module splits long text documents into smaller, overlapping chunks
suitable for embedding and retrieval.
"""

import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import get_config

logger = logging.getLogger(__name__)


class TextChunkingError(Exception):
    """Raised when text chunking fails."""
    pass


class TextChunker:
    """
    Splits text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

    The splitter tries to split on natural boundaries (paragraphs, sentences, words)
    while maintaining the specified chunk size and overlap.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap_ratio: float = None):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
                       If None, uses value from config (default: 1200)
            chunk_overlap_ratio: Ratio of overlap between chunks (0.0 to 1.0).
                                If None, uses value from config (default: 0.2)
        """
        config = get_config()

        # Use provided values or fall back to config
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        self.chunk_overlap_ratio = chunk_overlap_ratio if chunk_overlap_ratio is not None else config.chunk_overlap

        # Calculate overlap in characters
        self.chunk_overlap = int(self.chunk_size * self.chunk_overlap_ratio)

        # Initialize LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on natural boundaries
        )

        logger.info(
            f"TextChunker initialized with chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap} ({self.chunk_overlap_ratio*100:.0f}%)"
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries, each containing:
                - content: The chunk text
                - metadata: Chunk metadata (position, source, etc.)
                - chunk_id: Unique identifier for the chunk

        Raises:
            TextChunkingError: If chunking fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        logger.info(f"Starting text chunking. Input length: {len(text)} characters")

        try:
            # Split text using LangChain splitter
            text_chunks = self.splitter.split_text(text)

            logger.info(f"Text split into {len(text_chunks)} chunks")

            # Create chunk dictionaries with metadata
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = metadata.copy() if metadata else {}

                # Add chunk-specific metadata
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "chunk_size": len(chunk_text),
                    "is_first_chunk": i == 0,
                    "is_last_chunk": i == len(text_chunks) - 1
                })

                chunk = {
                    "content": chunk_text,
                    "metadata": chunk_metadata,
                    # Generate a unique chunk ID
                    "chunk_id": self._generate_chunk_id(metadata, i)
                }

                chunks.append(chunk)

            # Log statistics
            chunk_sizes = [len(c["content"]) for c in chunks]
            logger.info(
                f"Chunking complete. Chunks: {len(chunks)}, "
                f"Avg size: {sum(chunk_sizes)/len(chunks):.0f} chars, "
                f"Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}"
            )

            return chunks

        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise TextChunkingError(f"Chunking failed: {e}")

    def _generate_chunk_id(self, metadata: Dict[str, Any], chunk_index: int) -> str:
        """
        Generate a unique ID for a chunk.

        Args:
            metadata: Source metadata
            chunk_index: Index of the chunk

        Returns:
            Unique chunk identifier
        """
        # Try to use source file name if available
        if metadata and "source_file" in metadata:
            source_name = metadata["source_file"].replace(".", "_")
            return f"{source_name}_chunk_{chunk_index}"
        elif metadata and "source_url" in metadata:
            # Use last part of URL as identifier
            url_parts = metadata["source_url"].split("/")
            source_name = url_parts[-1].replace(".", "_") if url_parts else "unknown"
            return f"{source_name}_chunk_{chunk_index}"
        else:
            # Fallback to generic ID
            return f"chunk_{chunk_index}"

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics about the chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }

        chunk_sizes = [len(c["content"]) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "configured_chunk_size": self.chunk_size,
            "configured_overlap": self.chunk_overlap
        }


def chunk_document(text: str, metadata: Dict[str, Any] = None,
                   chunk_size: int = None, chunk_overlap_ratio: float = None) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk a document.

    Args:
        text: The text to chunk
        metadata: Optional metadata to attach to each chunk
        chunk_size: Maximum chunk size (uses config default if None)
        chunk_overlap_ratio: Overlap ratio (uses config default if None)

    Returns:
        List of chunk dictionaries

    Example:
        >>> chunks = chunk_document(
        ...     text="Long document text...",
        ...     metadata={"source_file": "document.pdf"}
        ... )
        >>> print(f"Created {len(chunks)} chunks")
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap_ratio=chunk_overlap_ratio)
    return chunker.chunk_text(text, metadata)


if __name__ == "__main__":
    # Test the chunker
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Sample text for testing
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create
    intelligent machines that can perform tasks that typically require human intelligence.

    Machine learning is a subset of AI that allows systems to learn and improve from
    experience without being explicitly programmed.

    Deep learning is a type of machine learning that uses artificial neural networks
    to process large amounts of data and recognize complex patterns.

    Natural Language Processing (NLP) is a field of AI that focuses on enabling
    computers to understand, interpret, and generate human language.
    """ * 10  # Repeat to make it longer

    try:
        chunker = TextChunker(chunk_size=200, chunk_overlap_ratio=0.2)
        chunks = chunker.chunk_text(sample_text, metadata={"source": "test"})

        print("\n" + "=" * 60)
        print("CHUNKING RESULT")
        print("=" * 60)

        stats = chunker.get_chunk_statistics(chunks)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Avg chunk size: {stats['avg_chunk_size']:.0f} characters")
        print(f"Min/Max: {stats['min_chunk_size']}/{stats['max_chunk_size']}")

        print("\n" + "=" * 60)
        print("SAMPLE CHUNKS:")
        print("=" * 60)
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i} (ID: {chunk['chunk_id']}):")
            print(f"Length: {len(chunk['content'])} chars")
            print(f"Content: {chunk['content'][:100]}...")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
