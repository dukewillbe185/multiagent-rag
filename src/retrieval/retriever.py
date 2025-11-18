"""
Retrieval Module for Azure AI Search.

This module implements hybrid search (vector + keyword) retrieval
using Azure AI Search.
"""

import logging
from typing import List, Dict, Any, Optional
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from config import get_config
from src.data_pipeline.embedder import Embedder

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when retrieval operations fail."""
    pass


class AzureSearchRetriever:
    """
    Retrieves relevant documents using Azure AI Search hybrid search.

    Combines:
    - Vector search (semantic similarity using embeddings)
    - Keyword search (BM25 text matching)
    """

    def __init__(self, index_name: str = None, top_k: int = None):
        """
        Initialize the retriever.

        Args:
            index_name: Name of the search index. If None, uses config default.
            top_k: Number of results to retrieve. If None, uses config default.
        """
        config = get_config()

        self.index_name = index_name or config.search_index_name
        self.top_k = top_k or config.retrieval_top_k

        # Initialize search client
        self.search_client = SearchClient(
            endpoint=config.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(config.search_admin_key)
        )

        # Initialize embedder for query vectorization
        self.embedder = Embedder()

        logger.info(
            f"AzureSearchRetriever initialized for index: {self.index_name}, "
            f"top_k: {self.top_k}"
        )

    def retrieve(self, query: str, top_k: int = None,
                 use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results to retrieve (overrides default)
            use_hybrid: If True, use hybrid search (vector + keyword).
                       If False, use vector search only.

        Returns:
            List of retrieved documents with metadata and scores

        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        k = top_k or self.top_k

        logger.info(
            f"Retrieving documents for query: '{query[:50]}...' "
            f"(top_k={k}, hybrid={use_hybrid})"
        )

        try:
            # Generate query embedding
            logger.info("Generating query embedding...")
            query_embedding = self.embedder.embed_text(query)

            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=k,
                fields="content_vector"
            )

            # Perform search
            if use_hybrid:
                # Hybrid search: vector + keyword
                logger.info("Performing hybrid search (vector + keyword)...")
                results = self.search_client.search(
                    search_text=query,  # Keyword search
                    vector_queries=[vector_query],  # Vector search
                    top=k,
                    select=["id", "content", "source_file", "chunk_index", "metadata"]
                )
            else:
                # Vector search only
                logger.info("Performing vector search only...")
                results = self.search_client.search(
                    search_text=None,  # No keyword search
                    vector_queries=[vector_query],
                    top=k,
                    select=["id", "content", "source_file", "chunk_index", "metadata"]
                )

            # Process results
            documents = self._process_results(results)

            logger.info(f"Retrieved {len(documents)} documents")

            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise RetrievalError(f"Retrieval failed: {e}")

    def retrieve_with_filter(self, query: str, filter_expr: str,
                           top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents with filtering.

        Args:
            query: Search query
            filter_expr: OData filter expression
                        Example: "source_file eq 'document.pdf'"
            top_k: Number of results to retrieve

        Returns:
            List of retrieved documents

        Raises:
            RetrievalError: If retrieval fails
        """
        k = top_k or self.top_k

        logger.info(f"Retrieving with filter: {filter_expr}")

        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)

            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=k,
                fields="content_vector"
            )

            # Perform filtered search
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                filter=filter_expr,
                top=k,
                select=["id", "content", "source_file", "chunk_index", "metadata"]
            )

            # Process results
            documents = self._process_results(results)

            logger.info(f"Retrieved {len(documents)} filtered documents")

            return documents

        except Exception as e:
            logger.error(f"Error retrieving filtered documents: {e}")
            raise RetrievalError(f"Filtered retrieval failed: {e}")

    def _process_results(self, results) -> List[Dict[str, Any]]:
        """
        Process search results into a structured format.

        Args:
            results: Search results from Azure AI Search

        Returns:
            List of document dictionaries
        """
        import json

        documents = []

        for result in results:
            # Extract score from Azure Search results
            # Try different possible score field names
            score = None
            if hasattr(result, '__getitem__'):  # Dictionary-like object
                # Azure Search returns score with @ prefix
                score = result.get('@search.score') or result.get('search_score') or result.get('score')
            else:  # Object with attributes
                score = getattr(result, 'score', None)

            # Log if score is still None for debugging
            if score is None:
                logger.debug(f"No score found for result: {result.get('id', 'unknown')}")

            # Parse metadata JSON
            metadata_str = result.get('metadata', '{}')
            try:
                metadata = json.loads(metadata_str)
            except:
                metadata = {}

            # Create document dictionary
            doc = {
                "id": result.get('id'),
                "content": result.get('content'),
                "source_file": result.get('source_file'),
                "chunk_index": result.get('chunk_index'),
                "score": score if score is not None else 0.0,  # Default to 0.0 if no score
                "metadata": metadata
            }

            documents.append(doc)

        return documents

    def get_context_for_query(self, query: str, top_k: int = None) -> str:
        """
        Retrieve documents and format them as context for LLM.

        Args:
            query: Search query
            top_k: Number of results to retrieve

        Returns:
            Formatted context string ready for LLM consumption

        Example:
            >>> context = retriever.get_context_for_query("What is AI?")
            >>> print(context)
        """
        documents = self.retrieve(query, top_k=top_k)

        if not documents:
            return "No relevant documents found."

        # Format documents as context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('source_file', 'Unknown')
            content = doc.get('content', '')

            context_parts.append(
                f"[Document {i}] (Source: {source})\n{content}"
            )

        context = "\n\n".join(context_parts)
        return context


def retrieve_documents(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve documents.

    Args:
        query: Search query
        top_k: Number of results to retrieve

    Returns:
        List of retrieved documents

    Example:
        >>> docs = retrieve_documents("What is machine learning?", top_k=3)
        >>> for doc in docs:
        ...     print(doc['content'][:100])
    """
    retriever = AzureSearchRetriever(top_k=top_k)
    return retriever.retrieve(query)


def get_context(query: str, top_k: int = None) -> str:
    """
    Convenience function to get formatted context for a query.

    Args:
        query: Search query
        top_k: Number of results to retrieve

    Returns:
        Formatted context string

    Example:
        >>> context = get_context("What is AI?")
        >>> print(context)
    """
    retriever = AzureSearchRetriever(top_k=top_k)
    return retriever.get_context_for_query(query)


if __name__ == "__main__":
    # Test the retriever
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

        try:
            retriever = AzureSearchRetriever()

            print("\n" + "=" * 60)
            print("RETRIEVAL TEST")
            print("=" * 60)
            print(f"Query: {query}")

            # Retrieve documents
            documents = retriever.retrieve(query, top_k=5)

            print(f"\nRetrieved {len(documents)} documents:")
            print("=" * 60)

            for i, doc in enumerate(documents, 1):
                print(f"\n[{i}] Score: {doc.get('score', 'N/A'):.4f}" if doc.get('score') else f"\n[{i}]")
                print(f"Source: {doc.get('source_file', 'Unknown')}")
                print(f"Content: {doc.get('content', '')[:200]}...")
                print("-" * 60)

            # Get formatted context
            print("\n" + "=" * 60)
            print("FORMATTED CONTEXT")
            print("=" * 60)
            context = retriever.get_context_for_query(query, top_k=3)
            print(context[:500] + "..." if len(context) > 500 else context)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python retriever.py <query>")
        print("Example: python retriever.py What is machine learning?")
        sys.exit(1)
