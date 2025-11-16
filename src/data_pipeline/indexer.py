"""
Azure AI Search Indexing Module.

This module creates and manages Azure AI Search indexes, including:
- Programmatic index creation with vector search configuration
- Schema definition for hybrid search (keyword + vector)
- Batch upload of document chunks with embeddings
- Index validation and management

IMPORTANT: The index is created programmatically if it doesn't exist.
"""

import logging
from typing import List, Dict, Any, Optional
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SimpleField,
    SearchableField
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError

from config import get_config

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Raised when indexing operations fail."""
    pass


class AzureSearchIndexer:
    """
    Manages Azure AI Search index creation and document indexing.

    Features:
    - Automatic index creation with proper schema
    - Vector search configuration (HNSW algorithm)
    - Semantic search configuration for hybrid search
    - Batch document upload
    """

    def __init__(self, index_name: str = None):
        """
        Initialize the Azure Search indexer.

        Args:
            index_name: Name of the index. If None, uses config default.
        """
        config = get_config()

        self.index_name = index_name or config.search_index_name
        self.endpoint = config.search_endpoint
        self.credential = AzureKeyCredential(config.search_admin_key)

        # Initialize clients
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )

        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

        logger.info(f"AzureSearchIndexer initialized for index: {self.index_name}")

    def create_index(self, recreate: bool = False) -> bool:
        """
        Create the search index with vector search configuration.

        If the index already exists and recreate=False, this is a no-op.
        If recreate=True, the existing index will be deleted and recreated.

        Args:
            recreate: If True, delete and recreate existing index

        Returns:
            True if index was created, False if it already existed

        Raises:
            IndexingError: If index creation fails
        """
        logger.info(f"Checking if index '{self.index_name}' exists...")

        try:
            # Check if index exists
            existing_index = None
            try:
                existing_index = self.index_client.get_index(self.index_name)
                logger.info(f"Index '{self.index_name}' already exists")

                if recreate:
                    logger.warning(f"Deleting existing index '{self.index_name}'...")
                    self.index_client.delete_index(self.index_name)
                    logger.info("Index deleted successfully")
                else:
                    logger.info("Using existing index")
                    return False

            except ResourceNotFoundError:
                logger.info(f"Index '{self.index_name}' does not exist, will create it")

            # Define the index schema
            logger.info("Creating index schema...")
            index = self._create_index_schema()

            # Create the index
            logger.info(f"Creating index '{self.index_name}'...")
            result = self.index_client.create_index(index)
            logger.info(f"Index '{self.index_name}' created successfully")

            return True

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise IndexingError(f"Index creation failed: {e}")

    def _create_index_schema(self) -> SearchIndex:
        """
        Create the index schema with fields and vector search configuration.

        Returns:
            SearchIndex object with complete configuration
        """
        # Define fields
        fields = [
            # ID field (unique key)
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),

            # Content field (searchable text)
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="en.microsoft"
            ),

            # Content vector field (for vector search)
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # text-embedding-ada-002 dimension
                vector_search_profile_name="vector-profile"
            ),

            # Metadata fields
            SearchableField(
                name="source_file",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),

            SimpleField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True
            ),

            SimpleField(
                name="chunk_size",
                type=SearchFieldDataType.Int32,
                filterable=True
            ),

            # Additional metadata as JSON string
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=False
            )
        ]

        # Configure vector search with HNSW algorithm
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters={
                        "m": 4,  # Number of bi-directional links
                        "efConstruction": 400,  # Size of dynamic candidate list for construction
                        "efSearch": 500,  # Size of dynamic candidate list for search
                        "metric": "cosine"  # Similarity metric
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )

        # Configure semantic search for hybrid search
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[
                    SemanticField(field_name="content")
                ],
                keywords_fields=[
                    SemanticField(field_name="source_file")
                ]
            )
        )

        semantic_search = SemanticSearch(
            configurations=[semantic_config]
        )

        # Create the index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )

        logger.info("Index schema created with vector and semantic search configuration")
        return index

    def index_documents(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index document chunks with embeddings to Azure AI Search.

        Args:
            chunks: List of chunks with embeddings (from Embedder)

        Returns:
            Dictionary with indexing statistics

        Raises:
            IndexingError: If indexing fails
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return {"indexed": 0, "failed": 0}

        logger.info(f"Starting indexing of {len(chunks)} chunks...")

        try:
            # Prepare documents for indexing
            documents = self._prepare_documents(chunks)

            # Upload documents in batches
            batch_size = 100
            total_indexed = 0
            total_failed = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size

                logger.info(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} documents)")

                try:
                    result = self.search_client.upload_documents(documents=batch)

                    # Count successes and failures
                    succeeded = sum(1 for r in result if r.succeeded)
                    failed = len(result) - succeeded

                    total_indexed += succeeded
                    total_failed += failed

                    if failed > 0:
                        logger.warning(f"Batch {batch_num}: {failed} documents failed to index")

                except Exception as e:
                    logger.error(f"Error uploading batch {batch_num}: {e}")
                    total_failed += len(batch)

            logger.info(
                f"Indexing complete. Indexed: {total_indexed}, Failed: {total_failed}"
            )

            return {
                "indexed": total_indexed,
                "failed": total_failed,
                "total": len(chunks)
            }

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise IndexingError(f"Indexing failed: {e}")

    def _prepare_documents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare chunks for indexing by converting to search document format.

        Args:
            chunks: List of chunks with embeddings

        Returns:
            List of documents ready for indexing
        """
        import json

        documents = []

        for chunk in chunks:
            # Validate chunk has required fields
            if "chunk_id" not in chunk:
                raise ValueError("Chunk missing 'chunk_id' field")
            if "content" not in chunk:
                raise ValueError("Chunk missing 'content' field")
            if "embedding" not in chunk:
                raise ValueError("Chunk missing 'embedding' field")

            metadata = chunk.get("metadata", {})

            # Create search document
            doc = {
                "id": chunk["chunk_id"],
                "content": chunk["content"],
                "content_vector": chunk["embedding"],
                "source_file": metadata.get("source_file", "unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "chunk_size": metadata.get("chunk_size", len(chunk["content"])),
                "metadata": json.dumps(metadata)  # Store as JSON string
            }

            documents.append(doc)

        logger.info(f"Prepared {len(documents)} documents for indexing")
        return documents

    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with index statistics

        Raises:
            IndexingError: If retrieval fails
        """
        try:
            # Get index info
            index = self.index_client.get_index(self.index_name)

            # Get document count (approximate)
            # Note: This is an approximate count and may not be exact
            search_results = self.search_client.search(
                search_text="*",
                include_total_count=True,
                top=0
            )

            doc_count = search_results.get_count() if hasattr(search_results, 'get_count') else 0

            return {
                "index_name": self.index_name,
                "document_count": doc_count,
                "fields": [f.name for f in index.fields],
                "vector_search_enabled": index.vector_search is not None,
                "semantic_search_enabled": index.semantic_search is not None
            }

        except ResourceNotFoundError:
            return {
                "index_name": self.index_name,
                "exists": False,
                "document_count": 0
            }
        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            raise IndexingError(f"Failed to get index statistics: {e}")

    def delete_index(self) -> bool:
        """
        Delete the index.

        Returns:
            True if deleted successfully

        Raises:
            IndexingError: If deletion fails
        """
        try:
            logger.warning(f"Deleting index '{self.index_name}'...")
            self.index_client.delete_index(self.index_name)
            logger.info(f"Index '{self.index_name}' deleted successfully")
            return True

        except ResourceNotFoundError:
            logger.warning(f"Index '{self.index_name}' does not exist")
            return False
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise IndexingError(f"Index deletion failed: {e}")


if __name__ == "__main__":
    # Test the indexer
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        indexer = AzureSearchIndexer()

        # Check/create index
        print("\n" + "=" * 60)
        print("CREATING INDEX")
        print("=" * 60)

        created = indexer.create_index(recreate=False)
        print(f"Index created: {created}")

        # Get statistics
        print("\n" + "=" * 60)
        print("INDEX STATISTICS")
        print("=" * 60)

        stats = indexer.get_index_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
