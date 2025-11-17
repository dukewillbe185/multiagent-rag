"""
Multi-Agent RAG System using LangGraph.

Implements a 3-agent workflow based on the blueprint:
1. Supervisor Retrieval Agent - Retrieves relevant chunks
2. Intent Identifier Agent - Identifies user intent
3. Answer Generator Agent - Generates final answer

Uses LangGraph for orchestrating the agent workflow.
"""

import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from src.agents.base_agent import BaseAgent
from src.retrieval.retriever import AzureSearchRetriever

logger = logging.getLogger(__name__)


# Define state for the agent workflow
class RagState(TypedDict):
    """State shared between agents in the RAG workflow."""
    question: str
    retrieved_chunks: List[str]
    retrieved_metadata: List[Dict[str, Any]]
    intent: str
    answer: str


class SupervisorRetrievalAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant document chunks.

    Connects to Azure AI Search and retrieves documents based on the query.
    """

    def __init__(self, top_k: int = 5):
        """
        Initialize the supervisor retrieval agent.

        Args:
            top_k: Number of chunks to retrieve
        """
        super().__init__(
            agent_name="Supervisor Retrieval",
            system_prompt="You are a retrieval supervisor responsible for finding relevant information."
        )
        self.retriever = AzureSearchRetriever(top_k=top_k)
        self.log_info(f"Initialized with top_k={top_k}")

    def execute(self, state: RagState) -> RagState:
        """
        Retrieve relevant chunks for the question.

        Args:
            state: Current RAG state

        Returns:
            Updated state with retrieved chunks
        """
        question = state["question"]

        self.log_info(f"Retrieving documents for: '{question[:50]}...'")

        try:
            # Retrieve documents
            documents = self.retriever.retrieve(question)

            # Extract chunks and metadata
            chunks = [doc["content"] for doc in documents]
            metadata = [
                {
                    "source_file": doc.get("source_file"),
                    "chunk_index": doc.get("chunk_index"),
                    "score": doc.get("score")
                }
                for doc in documents
            ]

            self.log_info(f"Retrieved {len(chunks)} chunks")

            # Update state
            state["retrieved_chunks"] = chunks
            state["retrieved_metadata"] = metadata

        except Exception as e:
            self.log_error(f"Retrieval failed: {e}")
            state["retrieved_chunks"] = []
            state["retrieved_metadata"] = []

        return state


class IntentIdentifierAgent(BaseAgent):
    """
    Agent responsible for identifying user intent.

    Uses Azure AI Foundry GPT-4 to classify the question intent.
    """

    def __init__(self):
        """Initialize the intent identifier agent."""
        super().__init__(
            agent_name="Intent Identifier",
            system_prompt=(
                "You are an intent classification expert. "
                "Identify the user's intent from their question. "
                "Classify into one of these categories: "
                "definition, comparison, explanation, how-to, factual, opinion, other. "
                "Respond with ONLY the intent label, nothing else."
            )
        )

    def execute(self, state: RagState) -> RagState:
        """
        Identify the intent of the question.

        Args:
            state: Current RAG state

        Returns:
            Updated state with identified intent
        """
        question = state["question"]

        self.log_info(f"Identifying intent for: '{question[:50]}...'")

        try:
            # Create prompt for intent identification
            prompt = f"""Question: {question}

What is the user's intent? Respond with ONLY one word from:
definition, comparison, explanation, how-to, factual, opinion, other"""

            # Get intent from LLM
            intent = self.invoke_llm(prompt)
            intent = intent.strip().lower()

            self.log_info(f"Identified intent: {intent}")

            # Update state
            state["intent"] = intent

        except Exception as e:
            self.log_error(f"Intent identification failed: {e}")
            state["intent"] = "general"

        return state


class AnswerGeneratorAgent(BaseAgent):
    """
    Agent responsible for generating the final answer.

    Uses retrieved chunks and identified intent to generate a comprehensive answer.
    """

    def __init__(self):
        """Initialize the answer generator agent."""
        super().__init__(
            agent_name="Answer Generator",
            system_prompt=(
                "You are a helpful AI assistant that answers questions based on "
                "provided context. Generate clear, accurate, and helpful answers. "
                "Always cite your sources when possible."
            )
        )

    def execute(self, state: RagState) -> RagState:
        """
        Generate answer based on retrieved chunks and intent.

        Args:
            state: Current RAG state

        Returns:
            Updated state with generated answer
        """
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        metadata = state.get("retrieved_metadata", [])
        intent = state.get("intent", "general")

        self.log_info(f"Generating answer for intent: {intent}")

        try:
            if not chunks:
                state["answer"] = (
                    "I couldn't find any relevant information to answer your question. "
                    "Please try rephrasing or asking a different question."
                )
                self.log_warning("No chunks available for answer generation")
                return state

            # Format retrieved chunks with sources
            context_parts = []
            for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1):
                source = meta.get("source_file", "Unknown")
                context_parts.append(f"[Source {i}: {source}]\n{chunk}")

            context = "\n\n".join(context_parts)

            # Create prompt for answer generation
            prompt = f"""Based on the following retrieved information, answer the user's question.

User's Intent: {intent}

Retrieved Information:
{context}

User's Question: {question}

Instructions:
- Provide a clear, accurate, and helpful answer
- Use information from the retrieved sources
- Cite sources where appropriate (e.g., "According to Source 1...")
- If the information doesn't fully answer the question, acknowledge this
- Keep the answer concise but comprehensive

Answer:"""

            # Generate answer
            answer = self.invoke_llm(prompt)

            self.log_info("Answer generated successfully")

            # Update state
            state["answer"] = answer

        except Exception as e:
            self.log_error(f"Answer generation failed: {e}")
            state["answer"] = (
                "I encountered an error while generating the answer. "
                "Please try again later."
            )

        return state


class MultiAgentRAG:
    """
    Multi-Agent RAG system orchestrator using LangGraph.

    Coordinates the workflow between three agents:
    1. Supervisor Retrieval Agent
    2. Intent Identifier Agent
    3. Answer Generator Agent
    """

    def __init__(self, top_k: int = 5):
        """
        Initialize the multi-agent RAG system.

        Args:
            top_k: Number of chunks to retrieve
        """
        self.top_k = top_k

        # Initialize agents
        self.supervisor_agent = SupervisorRetrievalAgent(top_k=top_k)
        self.intent_agent = IntentIdentifierAgent()
        self.answer_agent = AnswerGeneratorAgent()

        # Create workflow graph
        self.graph = self._create_graph()

        logger.info("MultiAgentRAG initialized with LangGraph workflow")

    def _create_graph(self) -> StateGraph:
        """
        Create the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        # Create workflow
        workflow = StateGraph(RagState)

        # Add agent nodes
        workflow.add_node("supervisor_retrieval", self.supervisor_agent.execute)
        workflow.add_node("intent_identifier", self.intent_agent.execute)
        workflow.add_node("answer_generator", self.answer_agent.execute)

        # Define workflow edges (sequential flow)
        workflow.set_entry_point("supervisor_retrieval")
        workflow.add_edge("supervisor_retrieval", "intent_identifier")
        workflow.add_edge("intent_identifier", "answer_generator")
        workflow.add_edge("answer_generator", END)

        # Compile graph
        graph = workflow.compile()

        logger.info("LangGraph workflow created: retrieval → intent → answer → END")

        return graph

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the multi-agent workflow.

        Args:
            question: User's question

        Returns:
            Dictionary containing:
                - question: Original question
                - answer: Generated answer
                - intent: Identified intent
                - chunks_retrieved: Number of chunks retrieved
                - sources: List of source files
        """
        logger.info("=" * 60)
        logger.info(f"Processing query: {question}")
        logger.info("=" * 60)

        try:
            # Create initial state
            initial_state = RagState(
                question=question,
                retrieved_chunks=[],
                retrieved_metadata=[],
                intent="",
                answer=""
            )

            # Run the workflow
            result = self.graph.invoke(initial_state)

            # Extract sources
            sources = list(set(
                meta.get("source_file", "Unknown")
                for meta in result.get("retrieved_metadata", [])
            ))

            # Prepare response
            response = {
                "question": result["question"],
                "answer": result["answer"],
                "intent": result["intent"],
                "chunks_retrieved": len(result.get("retrieved_chunks", [])),
                "sources": sources
            }

            logger.info("=" * 60)
            logger.info("Query processing complete")
            logger.info("=" * 60)

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "question": question,
                "answer": f"Error processing query: {e}",
                "intent": "error",
                "chunks_retrieved": 0,
                "sources": []
            }


def query_rag_system(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Convenience function to query the RAG system.

    Args:
        question: User's question
        top_k: Number of chunks to retrieve

    Returns:
        Response dictionary with answer and metadata

    Example:
        >>> response = query_rag_system("What is machine learning?")
        >>> print(response["answer"])
    """
    rag_system = MultiAgentRAG(top_k=top_k)
    return rag_system.query(question)


if __name__ == "__main__":
    # Test the multi-agent RAG system
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

        try:
            rag_system = MultiAgentRAG(top_k=5)
            response = rag_system.query(question)

            print("\n" + "=" * 60)
            print("MULTI-AGENT RAG RESPONSE")
            print("=" * 60)
            print(f"Question: {response['question']}")
            print(f"Intent: {response['intent']}")
            print(f"Chunks Retrieved: {response['chunks_retrieved']}")
            print(f"Sources: {', '.join(response['sources'])}")
            print("\n" + "-" * 60)
            print("Answer:")
            print("-" * 60)
            print(response['answer'])
            print("=" * 60)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python rag_agent.py <question>")
        print("Example: python rag_agent.py What is artificial intelligence?")
        sys.exit(1)
