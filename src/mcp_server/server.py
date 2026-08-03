"""
MCP server for the multi-agent RAG system.

This wraps the existing production pipeline -- Azure AI Search retrieval
(``src.retrieval.retriever.AzureSearchRetriever``) and the 4-agent LangGraph
workflow (``src.agents.rag_agent.MultiAgentRAG``) -- and exposes it over the
Model Context Protocol via FastMCP. It is a teaching example of all three
MCP primitives layered on top of a *real* system, rather than toy handlers:

- Tools (actions an LLM can invoke):      search_documents, ask_rag
- Resources (read-only context to load):  rag://index/status
- Prompts (reusable prompt templates):     cited_answer

Any MCP client -- the MCP Inspector, Claude Desktop, or a custom client --
can use this server with zero custom integration code once it speaks MCP.

Run it directly:
    python -m src.mcp_server                    # streamable-http transport
    python -m src.mcp_server --transport stdio  # stdio transport

Or point the Inspector at this module without running anything yourself:
    mcp dev src/mcp_server/server.py

IMPORTANT: importing this module must never touch Azure or require any
environment variables. AzureSearchRetriever, MultiAgentRAG, and
AzureSearchIndexer all construct Azure clients (and validate config) in
their constructors, so they are wrapped in lazy singletons below and only
built the first time a tool/resource actually needs them. This keeps
`mcp dev` and tool/resource/prompt listing working even with no .env
present.
"""

from __future__ import annotations

import json
import logging
import os

from mcp.server.fastmcp import FastMCP

from src.agents.rag_agent import MultiAgentRAG
from src.data_pipeline.indexer import AzureSearchIndexer
from src.retrieval.retriever import AzureSearchRetriever

logger = logging.getLogger(__name__)

# Host/port defaults, overridable via environment (and, in __main__.py, via
# --host/--port CLI flags). Reading os.environ here is safe for import: these
# are optional, generic settings, not the Azure credentials that
# config.get_config() would require.
DEFAULT_HOST = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MCP_SERVER_PORT", "8100"))

mcp = FastMCP(
    "Multi-Agent RAG",
    instructions=(
        "Exposes a production-style multi-agent RAG system over Azure AI "
        "Search: hybrid (vector + keyword) retrieval plus a 4-agent "
        "LangGraph pipeline (guardrail -> retrieval -> intent -> answer). "
        "Use the `search_documents` tool for raw retrieval when you just "
        "want matching chunks, or `ask_rag` to run the full guarded "
        "pipeline and get a synthesized answer. Read the "
        "`rag://index/status` resource to check what is actually indexed "
        "before querying. Use the `cited_answer` prompt when you want a "
        "reusable template for retrieval-grounded, cited answers."
    ),
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    json_response=True,
)


# --- Lazy singletons ---------------------------------------------------
#
# Each of these classes calls config.get_config() during __init__ (directly,
# or transitively via BaseAgent._initialize_llm), which raises
# ConfigurationError if required Azure env vars are missing. Deferring
# construction until first use means importing this module -- and listing
# its tools/resources/prompts -- never requires Azure credentials. Errors
# only surface when a tool is actually called without a configured .env.

_retriever: AzureSearchRetriever | None = None
_rag: MultiAgentRAG | None = None
_indexer: AzureSearchIndexer | None = None


def _get_retriever() -> AzureSearchRetriever:
    """Return the process-wide AzureSearchRetriever, constructing it on first use."""
    global _retriever
    if _retriever is None:
        _retriever = AzureSearchRetriever()
    return _retriever


def _get_rag() -> MultiAgentRAG:
    """Return the process-wide MultiAgentRAG, constructing it on first use."""
    global _rag
    if _rag is None:
        _rag = MultiAgentRAG()
    return _rag


def _get_indexer() -> AzureSearchIndexer:
    """Return the process-wide AzureSearchIndexer, constructing it on first use."""
    global _indexer
    if _indexer is None:
        _indexer = AzureSearchIndexer()
    return _indexer


# --- Tools ---------------------------------------------------------------


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search the indexed document corpus with hybrid (vector + keyword) search.

    This runs only the retrieval step -- the same Azure AI Search hybrid
    search used inside the RAG pipeline -- and returns the raw matching
    chunks with no LLM involved. Use this when you want to see exactly what
    would be retrieved for a query, or to build your own answer from the
    chunks yourself (see the `cited_answer` prompt for a template that does
    this). Contrast with `ask_rag`, which spends LLM calls to run the full
    4-agent pipeline (guardrail, intent, answer generation) on top of this
    same retrieval step and returns a synthesized answer instead of chunks.

    Args:
        query: Natural language search query.
        top_k: Maximum number of chunks to return (default 5).

    Returns:
        A list of matching chunks ordered most-relevant-first. Each item is
        a dict with:
            - id: chunk identifier
            - source_file: name of the document the chunk came from
            - chunk_index: position of the chunk within its source document
            - score: hybrid search relevance score
            - content_preview: first 200 characters of the chunk text
            - content: full chunk text
    """
    documents = _get_retriever().retrieve(query, top_k=top_k)
    results = []
    for doc in documents:
        content = doc.get("content") or ""
        results.append(
            {
                "id": doc.get("id"),
                "source_file": doc.get("source_file"),
                "chunk_index": doc.get("chunk_index"),
                "score": doc.get("score"),
                "content_preview": content[:200],
                "content": content,
            }
        )
    return results


@mcp.tool()
def ask_rag(question: str) -> dict:
    """Answer a question using the full 4-agent multi-agent RAG workflow.

    Runs the complete LangGraph pipeline instead of raw retrieval: a
    guardrail agent first checks the question is safe and relevant to the
    corpus; if it passes, a retrieval agent fetches supporting chunks, an
    intent agent classifies what kind of question it is, and an answer
    agent synthesizes a final answer grounded in the retrieved chunks. This
    is the production entry point (`MultiAgentRAG.query`), so -- unlike
    `search_documents`, which just returns raw chunks for free -- this tool
    spends real LLM calls and returns a ready-to-read answer plus the
    guardrail/intent decisions behind it.

    Args:
        question: Natural language question to ask of the document corpus.

    Returns:
        A dict with:
            - question: the original question
            - answer: the generated answer (or a rejection message if the
              guardrail agent blocked the question)
            - intent: identified question intent
            - guardrail_passed: whether the guardrail agent allowed the question
            - guardrail_reason: the guardrail agent's explanation
            - chunks_retrieved: number of chunks retrieved
            - sources: list of source file names used
    """
    result = _get_rag().query(question)
    return {
        "question": result.get("question", question),
        "answer": result.get("answer", ""),
        "intent": result.get("intent", ""),
        "guardrail_passed": result.get("guardrail_passed", False),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "chunks_retrieved": result.get("chunks_retrieved", 0),
        "sources": result.get("sources", []),
    }


# --- Resources -------------------------------------------------------------


@mcp.resource("rag://index/status")
def index_status() -> str:
    """Current Azure AI Search index status.

    Reports the index name, approximate document count, which search
    capabilities are configured (vector / semantic), and the indexed
    fields. This is a *resource*, not a tool: it is inert, readable context
    about the system's current state for a client to load alongside a
    conversation -- not an action with side effects or arguments for an LLM
    to decide to invoke.
    """
    stats = _get_indexer().get_index_statistics()
    return json.dumps(stats, indent=2, default=str)


# --- Prompts ---------------------------------------------------------------


@mcp.prompt()
def cited_answer(question: str) -> str:
    """Reusable prompt template for a retrieval-grounded, cited answer.

    Instructs the calling LLM to retrieve before answering, ground its
    answer strictly in what was retrieved, cite every claim to its source
    file, and explicitly admit when the corpus has nothing relevant --
    rather than falling back on outside/parametric knowledge.
    """
    return (
        f"Question: {question}\n\n"
        "Instructions:\n"
        "1. Call the `search_documents` tool with this question as the "
        "query before answering anything.\n"
        "2. Answer using ONLY information found in the retrieved chunks. "
        "Do not use outside or parametric knowledge.\n"
        "3. Cite every factual claim with its source, formatted as "
        "[source_file].\n"
        "4. If `search_documents` returns no chunks, or none of the "
        "retrieved content actually addresses the question, say plainly "
        "that the answer is not in the corpus instead of guessing.\n"
    )
