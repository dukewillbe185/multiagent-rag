# Multi-Agent RAG System with Azure Services

A production-ready Multi-Agent RAG (Retrieval-Augmented Generation) system built with Azure services, featuring a complete data pipeline, multi-agent collaboration using LangGraph, and FastAPI backend.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI Interface](#cli-interface)
  - [FastAPI Server](#fastapi-server)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project implements a sophisticated RAG system that:
1. Extracts text from PDF documents using Azure AI Document Intelligence
2. Chunks text using LangChain's RecursiveCharacterTextSplitter
3. Generates embeddings using Azure AI Foundry (text-embedding-ada-002)
4. Indexes documents in Azure AI Search with hybrid search capabilities
5. Answers questions using a 4-agent workflow powered by LangGraph and GPT-4

## Architecture

### Multi-Agent Workflow

The system uses **LangGraph** to orchestrate a 4-agent workflow with guardrails:

```
User Question
     ↓
[Azure AI Content Safety] → Blocks unsafe input
     ↓ (if safe)
[Supervisor Retrieval Agent]
     ↓ (retrieve relevant chunks)
[Guardrail Agent] → Validates relevance against retrieved evidence
     ↓ (if relevant)
[Intent Identifier Agent]
     ↓ (identify user intent)
[Answer Generator Agent] → Uses conversation memory
     ↓
Final Answer
```

**Agent Roles:**
1. **Supervisor Retrieval Agent** - Retrieves relevant document chunks from Azure AI Search
2. **Guardrail Agent** - Validates relevance against retrieved chunks with stable seeded sampling
3. **Intent Identifier Agent** - Identifies user's question intent (definition, explanation, etc.)
4. **Answer Generator Agent** - Generates comprehensive answer using retrieved context and conversation history

### Data Pipeline

```
PDF Document
     ↓
[Azure AI Document Intelligence] → Extract Text
     ↓
[LangChain Text Splitter] → Create Chunks (1200 chars, 20% overlap)
     ↓
[Azure AI Foundry Embedding] → Generate Vectors (1536-dim)
     ↓
[Azure AI Search] → Index with Hybrid Search (Vector + Keyword)
```

## Features

### Core Functionality
- **PDF Text Extraction** - Azure AI Document Intelligence with Layout model
- **Intelligent Chunking** - LangChain RecursiveCharacterTextSplitter with configurable size and overlap
- **Vector Embeddings** - Azure AI Foundry text-embedding-ada-002 (1536 dimensions)
- **Hybrid Search** - Combines vector similarity and keyword matching (BM25)
- **Multi-Agent RAG** - LangGraph-powered 4-agent workflow with guardrails
- **Intent Recognition** - Automatic classification of user intent
- **Source Citation** - Answers include source references with detailed chunk information

### 🆕 New Production Features

#### Guardrail System
- **Input Safety** - Azure AI Content Safety blocks unsafe input before retrieval
- **Evidence-Grounded Relevance** - Filters unrelated questions after consulting indexed documents
- **Configurable Strictness** - Low, medium, or high guardrail sensitivity
- **Stable Classification** - Relevance checks use the GPT-5-supported default temperature with a fixed best-effort seed
- **Detailed Diagnostics** - Guardrail decisions and reasons are logged and returned by the API
- **Rejection Messages** - User-friendly messages for rejected queries

#### Session Management & Conversation Memory
- **Session Tracking** - Every query includes a session_id for conversation continuity
- **Multi-Turn Conversations** - Remembers previous question and answer (N-1 turn)
- **Follow-up Support** - Handle questions like "tell me more" or "what did I just ask?"
- **Automatic Cleanup** - Sessions expire after 30 minutes of inactivity
- **In-Memory Storage** - Fast, efficient session management without external databases

#### Enhanced Retrieval Transparency
- **Detailed Chunk Information** - Every retrieved chunk includes:
  - Chunk ID from Azure Search index
  - Source file name
  - Chunk position (index)
  - Search score (hybrid similarity)
  - Content preview (200 chars)
  - Full content for transparency
- **Retrieval Metrics** - Average scores, chunk counts tracked per query

#### Error Handling & Reliability
- **Retry Logic** - Exponential backoff for Azure API calls (max 3 retries)
- **Specific Error Types** - GuardrailRejectionError, RetrievalError, GenerationError, etc.
- **User-Friendly Messages** - Clear error messages distinguishing:
  - Guardrail rejection
  - Retrieval failures
  - Generation errors

### Infrastructure
- **Automatic Index Creation** - Programmatically creates Azure AI Search index if not exists
- **Monitoring** - Azure Application Insights integration with session_id tracking
- **REST API** - FastAPI backend with Swagger documentation
- **CLI Tools** - Command-line interface for all operations
- **Security** - Environment-based configuration, no hardcoded secrets
- **Background Tasks** - Automatic session cleanup every 5 minutes

## Technology Stack

### Azure Services
- **Azure AI Document Intelligence** - PDF text extraction
- **Azure AI Foundry** - Embedding model (text-embedding-ada-002) and GPT-4
  - ⚠️ **Important**: Using Azure AI Foundry, NOT Azure OpenAI Service
  - Each deployed model has separate endpoint, deployment name, and API key
- **Azure AI Search** - Vector and hybrid search with HNSW algorithm
- **Azure Application Insights** - Monitoring and logging

### Python Libraries
- **LangChain** - Text splitting and AI orchestration
- **LangGraph** - Multi-agent workflow management
- **FastAPI** - REST API framework
- **Pydantic** - Data validation
- **OpenAI Python SDK** - Azure AI Foundry API client

## Prerequisites

### Azure Resources Required

You need the following Azure resources created:

1. **Azure AI Document Intelligence**
   - Get: Endpoint and API Key

2. **Azure AI Foundry** (with deployed models)
   - Embedding Model: text-embedding-ada-002
     - Get: Endpoint, Deployment Name, API Key
   - GPT-4 Model: gpt-4
     - Get: Endpoint, Deployment Name, API Key

3. **Azure AI Search**
   - Get: Endpoint and Admin API Key
   - Note: Index will be created programmatically

4. **Azure Application Insights**
   - Get: Connection String

### Local Requirements
- Python 3.9 or higher
- pip (Python package manager)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd multiagent-rag
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### 1. Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Fill in Your Azure Credentials

Edit `.env` and fill in all values:

```bash
# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-doc-intelligence.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your_key_here

# Azure AI Foundry - Embedding Model
AZURE_AI_FOUNDRY_EMBEDDING_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
AZURE_AI_FOUNDRY_EMBEDDING_API_KEY=your_key_here
AZURE_AI_FOUNDRY_EMBEDDING_API_VERSION=2024-08-01-preview

# Azure AI Foundry - GPT-4 Model
AZURE_AI_FOUNDRY_GPT4_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_AI_FOUNDRY_GPT4_DEPLOYMENT_NAME=gpt-4
AZURE_AI_FOUNDRY_GPT4_API_KEY=your_key_here
AZURE_AI_FOUNDRY_GPT4_API_VERSION=2024-08-01-preview

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_ADMIN_KEY=your_admin_key_here
AZURE_SEARCH_INDEX_NAME=rag-documents-index

# Azure Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...

# Application Settings
CHUNK_SIZE=1200
CHUNK_OVERLAP=0.2
RETRIEVAL_TOP_K=5
LLM_TEMPERATURE=0.7

# 🆕 Guardrail Settings
GUARDRAIL_ENABLED=true  # false skips the relevance classifier; Content Safety remains separate
GUARDRAIL_STRICTNESS=medium  # low, medium, or high

# 🆕 Session Management
SESSION_TIMEOUT_MINUTES=30
CONVERSATION_MEMORY_TURNS=1  # Currently only supports 1 (previous turn)
```

### 3. Important Notes on Azure AI Foundry

⚠️ **Critical Configuration Details:**

- **NOT Azure OpenAI Service**: This system uses Azure AI Foundry, which has different endpoints
- **Separate Credentials**: Each deployed model (embedding and GPT-4) has its own:
  - Endpoint URL
  - Deployment name
  - API key
- **Find Your Credentials**:
  - Go to Azure AI Foundry portal
  - Navigate to "Deployments"
  - Select each deployment to get its specific credentials

### 4. Verify Configuration

```bash
# Test configuration loading
python -c "from config import get_config; get_config(); print('✓ Configuration loaded successfully')"
```

## Usage

### CLI Interface

The `main.py` script provides a command-line interface for all operations.

#### 1. Process a Document

```bash
# Process doc.pdf through the full pipeline
python main.py process doc.pdf
```

This will:
1. Extract text from PDF
2. Chunk the text
3. Generate embeddings
4. Create index (if needed)
5. Index all chunks

#### 2. Test Retrieval

```bash
# Retrieve relevant chunks for a query
python main.py retrieve "What is machine learning?" --top-k 5
```

#### 3. Test Multi-Agent Query

```bash
# Ask a question using the full multi-agent system
python main.py query "What is artificial intelligence?" --top-k 5
```

#### 4. Manage Index

```bash
# Check index status
python main.py index status

# Create index
python main.py index create

# Recreate index (deletes all data!)
python main.py index recreate

# Delete index
python main.py index delete
```

### FastAPI Server

#### Start the Server

```bash
# Start with uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python src/api/main.py
```

The server will start at: `http://localhost:8000`

#### Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Documentation

### Endpoints

#### Health Check
```http
GET /api/v1/health
```

Returns service health status and connectivity.

#### Index Status
```http
GET /api/v1/index/status
```

Returns index statistics and configuration.

#### Upload Document
```http
POST /api/v1/upload
Content-Type: multipart/form-data
```

Upload and process a PDF document.

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@doc.pdf"
```

**Response:**
```json
{
  "success": true,
  "message": "Document processed successfully",
  "file_name": "doc.pdf",
  "chunks_created": 25,
  "chunks_indexed": 25,
  "processing_time_seconds": 12.5
}
```

#### Query System 🆕 Enhanced
```http
POST /api/v1/query
Content-Type: application/json
```

Ask a question using the multi-agent RAG system with session management and conversation memory.

**Request:**
```json
{
  "question": "What is machine learning?",
  "session_id": "session_12345",
  "top_k": 5,
  "user_id": "user_67890",
  "metadata": {"source": "web_app"}
}
```

**Required Fields:**
- `question` (string): User's question
- `session_id` (string): Session identifier (provided by backend)

**Optional Fields:**
- `top_k` (integer, 1-20, default: 5): Number of chunks to retrieve
- `user_id` (string): Optional user identifier for tracking
- `metadata` (object): Additional context

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "session_id": "session_12345",
    "top_k": 5
  }'
```

**Response (Success):**
```json
{
  "success": true,
  "question": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "intent": "definition",
  "session_id": "session_12345",
  "conversation_turn": 1,
  "chunks_retrieved": 5,
  "retrieved_chunks": [
    {
      "chunk_id": "doc_pdf_chunk_5",
      "source_file": "doc.pdf",
      "chunk_index": 5,
      "score": 0.8542,
      "content_preview": "Machine learning is a subset of artificial intelligence...",
      "full_content": "Machine learning is a subset of artificial intelligence that enables..."
    }
  ],
  "sources": ["doc.pdf"],
  "processing_time_seconds": 2.3,
  "guardrail_passed": true,
  "guardrail_reason": "The retrieved evidence directly supports the question.",
  "input_guardrail": "nothing detected",
  "output_guardrail": "nothing detected"
}
```

**Response (Guardrail Rejection):**
```json
{
  "success": false,
  "question": "What's the weather today?",
  "answer": "Your question doesn't appear to be related to the indexed documents.",
  "intent": "rejected",
  "session_id": "session_12345",
  "conversation_turn": 1,
  "chunks_retrieved": 0,
  "retrieved_chunks": [],
  "sources": [],
  "processing_time_seconds": 0.5,
  "guardrail_passed": false,
  "guardrail_reason": "The retrieved chunks do not address weather.",
  "input_guardrail": "nothing detected",
  "output_guardrail": "not evaluated"
}
```

**Follow-up Question Example:**
```bash
# First question
curl -X POST "http://localhost:8000/api/v1/query" \
  -d '{"question": "What is AI?", "session_id": "session_123"}'

# Follow-up question (uses conversation memory)
curl -X POST "http://localhost:8000/api/v1/query" \
  -d '{"question": "Tell me more about that", "session_id": "session_123"}'
```

#### Create Index
```http
POST /api/v1/index/create?recreate=false
```

Create or recreate the search index.

## Project Structure

```
multiagent-rag/
├── config/
│   ├── __init__.py
│   └── azure_config.py          # Azure service configuration
├── src/
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── document_extractor.py    # PDF text extraction
│   │   ├── text_chunker.py          # Text chunking
│   │   ├── embedder.py              # Embedding generation
│   │   └── indexer.py               # Azure AI Search indexing
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py             # Hybrid search retrieval
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py            # Base agent class
│   │   └── rag_agent.py             # Multi-agent RAG system
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── logger.py                # Azure Application Insights
│   └── api/
│       ├── __init__.py
│       ├── main.py                  # FastAPI application
│       ├── models.py                # Pydantic models
│       └── routes.py                # API endpoints
├── doc.pdf                          # Test document
├── langgraph_test.py               # Multi-agent blueprint
├── main.py                         # CLI orchestration script
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore
└── README.md
```

## Security Considerations

### 🔒 API Key Management

**NEVER commit secrets to version control!**

- ✅ All secrets in `.env` file (which is in `.gitignore`)
- ✅ `.env.example` contains only variable names, no actual values
- ✅ Environment variables validated at startup
- ✅ Secrets masked in logs (show first/last 4 characters only)

### 🏭 Production Recommendations

**For production deployments, use Azure Managed Identity instead of API keys:**

```python
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
```

**Benefits:**
- No API keys to manage or rotate
- Automatic credential management
- Fine-grained RBAC (Role-Based Access Control)
- Works seamlessly in Azure-hosted environments

**See:** [Azure Identity Documentation](https://learn.microsoft.com/azure/developer/python/sdk/authentication-overview)

### 🛡️ Admin API Key

The Azure AI Search Admin Key has full management permissions:
- Can create/delete indexes
- Can read/write all data
- Must be carefully protected
- Consider using Query Keys for read-only operations in production

## Troubleshooting

### Configuration Errors

**Error: Required environment variable not set**
```
Solution: Check your .env file and ensure all variables from .env.example are filled in
```

**Error: Invalid endpoint URL**
```
Solution: Ensure endpoints start with https:// and are from correct Azure resources
```

### Azure AI Foundry Issues

**Error: Model not found or deployment not available**
```
Solution:
1. Verify models are deployed in Azure AI Foundry
2. Check deployment names match exactly
3. Ensure API keys are for the correct deployments
4. Verify API version is correct (2024-08-01-preview)
```

### Index Creation Issues

**Error: Index creation failed**
```
Solution:
1. Verify Azure Search Admin Key has management permissions
2. Check search service tier supports vector search
3. Ensure index name follows naming rules (lowercase, no special chars)
```

### Import Errors

**Error: No module named 'azure.ai.documentintelligence'**
```
Solution: Install dependencies
pip install -r requirements.txt
```

### Connection Issues

**Error: Connection timeout or refused**
```
Solution:
1. Check internet connectivity
2. Verify Azure service endpoints are correct
3. Check firewall/network settings
4. Ensure Azure services are in same region (recommended)
```

## Monitoring

### Azure Application Insights

All operations are logged to Azure Application Insights:

- **Metrics**: Processing times, chunk counts, retrieval performance
- **Events**: Document uploads, queries, index operations
- **Exceptions**: Automatic error tracking
- **Dependencies**: Azure service call tracking

**View in Azure Portal:**
1. Go to your Application Insights resource
2. Navigate to "Logs" or "Metrics"
3. Query for custom events and metrics

### Local Logs

Logs are also output to console with format:
```
2024-01-15 10:30:00 - module.name - INFO - Message
```

## Testing

### Quick Test Workflow

1. **Process test document:**
   ```bash
   python main.py process doc.pdf
   ```

2. **Test retrieval:**
   ```bash
   python main.py retrieve "test query"
   ```

3. **Test multi-agent query:**
   ```bash
   python main.py query "What is the main topic?"
   ```

4. **Test API:**
   ```bash
   # Start server
   uvicorn src.api.main:app --reload

   # In another terminal, test upload
   curl -X POST "http://localhost:8000/api/v1/upload" \
     -F "file=@doc.pdf"

   # Test query
   curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this about?"}'
   ```

## Performance Optimization

### Chunk Size Tuning

Adjust in `.env`:
```bash
CHUNK_SIZE=1200          # Larger = more context, fewer chunks
CHUNK_OVERLAP=0.2        # Higher = more overlap, better continuity
```

### Retrieval Tuning

```bash
RETRIEVAL_TOP_K=5        # Number of chunks to retrieve
```

### LLM Temperature

```bash
LLM_TEMPERATURE=0.7      # Lower = more focused, Higher = more creative
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Support

For issues and questions:
- Create an issue in the repository
- Check Azure service health status
- Review Azure documentation for service-specific issues

## Acknowledgments

- Built with LangChain and LangGraph
- Powered by Azure AI services
- FastAPI for REST API
- OpenAI for embedding and language models

---

**Happy RAG Building! 🚀**
