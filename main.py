#!/usr/bin/env python3
"""
Multi-Agent RAG System - CLI Orchestration Script

This script provides a command-line interface for:
- Processing documents through the pipeline
- Testing retrieval
- Testing multi-agent query
- Managing the index
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data_pipeline.document_extractor import DocumentExtractor
from src.data_pipeline.text_chunker import TextChunker
from src.data_pipeline.embedder import Embedder
from src.data_pipeline.indexer import AzureSearchIndexer
from src.retrieval.retriever import AzureSearchRetriever
from src.agents.rag_agent import MultiAgentRAG
from src.monitoring.logger import setup_logging
from config import get_config, ConfigurationError


def setup_cli_logging():
    """Setup logging for CLI."""
    setup_logging(logging.INFO)
    return logging.getLogger(__name__)


def process_document(pdf_path: str, logger):
    """
    Process a document through the full pipeline.

    Args:
        pdf_path: Path to PDF file
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("DOCUMENT PROCESSING PIPELINE")
    logger.info("=" * 60)

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"File not found: {pdf_path}")
        sys.exit(1)

    try:
        # Step 1: Extract text
        logger.info("\n[1/5] Extracting text from PDF...")
        extractor = DocumentExtractor()
        extraction_result = extractor.extract_text_from_pdf(str(pdf_file))
        extracted_text = extraction_result["text"]
        metadata = extraction_result["metadata"]

        logger.info(f"✓ Extracted {len(extracted_text)} characters from {extraction_result['page_count']} pages")

        # Step 2: Chunk text
        logger.info("\n[2/5] Chunking text...")
        chunker = TextChunker()
        chunks = chunker.chunk_text(extracted_text, metadata=metadata)

        stats = chunker.get_chunk_statistics(chunks)
        logger.info(f"✓ Created {stats['total_chunks']} chunks")
        logger.info(f"  - Avg size: {stats['avg_chunk_size']:.0f} chars")
        logger.info(f"  - Min/Max: {stats['min_chunk_size']}/{stats['max_chunk_size']}")

        # Step 3: Generate embeddings
        logger.info("\n[3/5] Generating embeddings...")
        embedder = Embedder()
        chunks_with_embeddings = embedder.embed_chunks(chunks)

        logger.info(f"✓ Generated embeddings for {len(chunks_with_embeddings)} chunks")

        # Step 4: Create/verify index
        logger.info("\n[4/5] Setting up search index...")
        indexer = AzureSearchIndexer()
        created = indexer.create_index(recreate=False)

        if created:
            logger.info("✓ Index created successfully")
        else:
            logger.info("✓ Using existing index")

        # Step 5: Index documents
        logger.info("\n[5/5] Indexing documents...")
        result = indexer.index_documents(chunks_with_embeddings)

        logger.info(f"✓ Indexed {result['indexed']} chunks")
        if result['failed'] > 0:
            logger.warning(f"⚠ {result['failed']} chunks failed to index")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Source file: {pdf_file.name}")
        logger.info(f"Chunks created: {len(chunks)}")
        logger.info(f"Chunks indexed: {result['indexed']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        sys.exit(1)


def test_retrieval(query: str, top_k: int, logger):
    """
    Test retrieval functionality.

    Args:
        query: Search query
        top_k: Number of results
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("TESTING RETRIEVAL")
    logger.info("=" * 60)
    logger.info(f"Query: {query}")
    logger.info(f"Top-K: {top_k}")

    try:
        retriever = AzureSearchRetriever(top_k=top_k)
        documents = retriever.retrieve(query)

        logger.info(f"\nRetrieved {len(documents)} documents:")
        logger.info("-" * 60)

        for i, doc in enumerate(documents, 1):
            logger.info(f"\n[{i}] Score: {doc.get('score', 'N/A')}")
            logger.info(f"Source: {doc.get('source_file', 'Unknown')}")
            logger.info(f"Content: {doc.get('content', '')[:200]}...")

        logger.info("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Error testing retrieval: {e}")
        sys.exit(1)


def test_query(question: str, top_k: int, logger):
    """
    Test multi-agent query.

    Args:
        question: User question
        top_k: Number of chunks to retrieve
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("TESTING MULTI-AGENT RAG QUERY")
    logger.info("=" * 60)
    logger.info(f"Question: {question}")

    try:
        rag_system = MultiAgentRAG(top_k=top_k)
        response = rag_system.query(question)

        logger.info("\n" + "=" * 60)
        logger.info("RESPONSE")
        logger.info("=" * 60)
        logger.info(f"Intent: {response['intent']}")
        logger.info(f"Chunks Retrieved: {response['chunks_retrieved']}")
        logger.info(f"Sources: {', '.join(response['sources'])}")
        logger.info("\n" + "-" * 60)
        logger.info("Answer:")
        logger.info("-" * 60)
        logger.info(response['answer'])
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error testing query: {e}")
        sys.exit(1)


def manage_index(action: str, logger):
    """
    Manage the search index.

    Args:
        action: Action to perform (create, status, delete)
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info(f"INDEX MANAGEMENT: {action.upper()}")
    logger.info("=" * 60)

    try:
        indexer = AzureSearchIndexer()

        if action == "create":
            created = indexer.create_index(recreate=False)
            if created:
                logger.info("✓ Index created successfully")
            else:
                logger.info("ℹ Index already exists")

        elif action == "recreate":
            logger.warning("This will delete all existing data!")
            confirm = input("Are you sure? (yes/no): ")
            if confirm.lower() == "yes":
                indexer.create_index(recreate=True)
                logger.info("✓ Index recreated successfully")
            else:
                logger.info("Operation cancelled")

        elif action == "status":
            stats = indexer.get_index_statistics()
            logger.info(f"Index Name: {stats.get('index_name')}")
            logger.info(f"Exists: {stats.get('exists', True)}")
            logger.info(f"Document Count: {stats.get('document_count')}")
            logger.info(f"Vector Search: {stats.get('vector_search_enabled')}")
            logger.info(f"Semantic Search: {stats.get('semantic_search_enabled')}")
            if 'fields' in stats:
                logger.info(f"Fields: {', '.join(stats['fields'])}")

        elif action == "delete":
            logger.warning("This will delete all indexed data!")
            confirm = input("Are you sure? (yes/no): ")
            if confirm.lower() == "yes":
                indexer.delete_index()
                logger.info("✓ Index deleted successfully")
            else:
                logger.info("Operation cancelled")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error managing index: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a document
  python main.py process doc.pdf

  # Test retrieval
  python main.py retrieve "What is machine learning?" --top-k 5

  # Test multi-agent query
  python main.py query "What is artificial intelligence?"

  # Manage index
  python main.py index status
  python main.py index create
  python main.py index delete
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a document")
    process_parser.add_argument("pdf_path", help="Path to PDF file")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Test retrieval")
    retrieve_parser.add_argument("query", help="Search query")
    retrieve_parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")

    # Query command
    query_parser = subparsers.add_parser("query", help="Test multi-agent query")
    query_parser.add_argument("question", help="User question")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks (default: 5)")

    # Index command
    index_parser = subparsers.add_parser("index", help="Manage search index")
    index_parser.add_argument(
        "action",
        choices=["create", "recreate", "status", "delete"],
        help="Index action"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_cli_logging()

    # Load configuration
    try:
        config = get_config()
        logger.info("Configuration loaded successfully")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file")
        sys.exit(1)

    # Execute command
    if args.command == "process":
        process_document(args.pdf_path, logger)

    elif args.command == "retrieve":
        test_retrieval(args.query, args.top_k, logger)

    elif args.command == "query":
        test_query(args.question, args.top_k, logger)

    elif args.command == "index":
        manage_index(args.action, logger)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
