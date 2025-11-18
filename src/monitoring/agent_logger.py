"""
Agent-level structured logging for LangGraph workflow.

Provides detailed visibility into each agent's execution with proper
Application Insights integration and trace separation.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class AgentExecutionLogger:
    """
    Structured logger for agent execution with Application Insights integration.

    Ensures each agent execution is logged with:
    - Agent name
    - Session ID
    - Turn number
    - Input/output
    - Duration
    - Handoff information
    """

    def __init__(self, agent_name: str, session_id: str, turn: int):
        """
        Initialize agent execution logger.

        Args:
            agent_name: Name of the agent being executed
            session_id: Current session identifier
            turn: Conversation turn number
        """
        self.agent_name = agent_name
        self.session_id = session_id
        self.turn = turn
        self.start_time = None
        self.execution_id = str(uuid.uuid4())[:8]

    def log_start(self, input_data: Any):
        """
        Log the start of agent execution.

        Args:
            input_data: Input received by the agent
        """
        self.start_time = time.time()

        separator = "=" * 60
        logger.info(f"\n{separator}")
        logger.info(f"🤖 AGENT: {self.agent_name}")
        logger.info(separator)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Turn: {self.turn}")
        logger.info(f"Execution ID: {self.execution_id}")
        logger.info(f"Input: {str(input_data)[:200]}...")
        logger.info(f"Status: STARTED")
        logger.info(separator)

        # Structured log for Application Insights
        logger.info(
            f"[{self.agent_name}] Execution started",
            extra={
                "agent_name": self.agent_name,
                "session_id": self.session_id,
                "turn": self.turn,
                "execution_id": self.execution_id,
                "event_type": "agent_start"
            }
        )

    def log_action(self, action: str, details: Dict[str, Any] = None):
        """
        Log an action taken by the agent.

        Args:
            action: Description of the action
            details: Additional details about the action
        """
        logger.info(f"[{self.agent_name}] Action: {action}")
        if details:
            for key, value in details.items():
                logger.info(f"  - {key}: {value}")

        # Structured log
        logger.info(
            f"[{self.agent_name}] {action}",
            extra={
                "agent_name": self.agent_name,
                "session_id": self.session_id,
                "turn": self.turn,
                "execution_id": self.execution_id,
                "action": action,
                "details": details or {},
                "event_type": "agent_action"
            }
        )

    def log_decision(self, decision: str, reason: str, confidence: Optional[str] = None):
        """
        Log a decision made by the agent.

        Args:
            decision: The decision made
            reason: Reason for the decision
            confidence: Confidence level (if applicable)
        """
        logger.info(f"[{self.agent_name}] Decision: {decision}")
        logger.info(f"  - Reason: {reason}")
        if confidence:
            logger.info(f"  - Confidence: {confidence}")

        # Structured log
        logger.info(
            f"[{self.agent_name}] Decision: {decision}",
            extra={
                "agent_name": self.agent_name,
                "session_id": self.session_id,
                "turn": self.turn,
                "execution_id": self.execution_id,
                "decision": decision,
                "reason": reason,
                "confidence": confidence,
                "event_type": "agent_decision"
            }
        )

    def log_complete(
        self,
        output_summary: str,
        next_agent: str = "END",
        metadata: Dict[str, Any] = None
    ):
        """
        Log the completion of agent execution.

        Args:
            output_summary: Summary of the agent's output
            next_agent: Next agent in the workflow
            metadata: Additional metadata about the execution
        """
        duration = time.time() - self.start_time if self.start_time else 0

        separator = "-" * 60
        logger.info(separator)
        logger.info(f"Output: {output_summary}")
        logger.info(f"Handoff to: {next_agent}")
        logger.info(f"Duration: {duration:.2f}s")
        if metadata:
            logger.info("Metadata:")
            for key, value in metadata.items():
                logger.info(f"  - {key}: {value}")
        logger.info(separator + "\n")

        # Structured log for Application Insights
        logger.info(
            f"[{self.agent_name}] Execution complete",
            extra={
                "agent_name": self.agent_name,
                "session_id": self.session_id,
                "turn": self.turn,
                "execution_id": self.execution_id,
                "duration_seconds": duration,
                "next_agent": next_agent,
                "metadata": metadata or {},
                "event_type": "agent_complete"
            }
        )


@contextmanager
def log_agent_execution(agent_name: str, session_id: str, turn: int, input_data: Any):
    """
    Context manager for logging agent execution.

    Usage:
        with log_agent_execution("GuardrailAgent", session_id, turn, question) as agent_log:
            # ... agent logic ...
            agent_log.log_action("Evaluating question")
            agent_log.log_decision("PASSED", "Question is relevant")

    Args:
        agent_name: Name of the agent
        session_id: Session identifier
        turn: Conversation turn
        input_data: Input to the agent

    Yields:
        AgentExecutionLogger instance
    """
    agent_log = AgentExecutionLogger(agent_name, session_id, turn)
    agent_log.log_start(input_data)

    try:
        yield agent_log
    except Exception as e:
        logger.error(
            f"[{agent_name}] Execution failed: {e}",
            extra={
                "agent_name": agent_name,
                "session_id": session_id,
                "turn": turn,
                "error": str(e),
                "event_type": "agent_error"
            }
        )
        raise


def log_request_start(session_id: str, turn: int, question: str, request_id: str):
    """
    Log the start of an API request.

    Args:
        session_id: Session identifier
        turn: Conversation turn
        question: User's question
        request_id: Unique request identifier
    """
    separator = "=" * 80
    logger.info(f"\n{separator}")
    logger.info("🚀 NEW QUERY REQUEST")
    logger.info(separator)
    logger.info(f"Request ID: {request_id}")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Turn Number: {turn}")
    logger.info(f"Question: {question}")
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    logger.info(separator + "\n")

    # Structured log for Application Insights
    logger.info(
        "Query request started",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "turn": turn,
            "question": question[:200],
            "event_type": "request_start"
        }
    )


def log_request_complete(
    request_id: str,
    session_id: str,
    duration: float,
    guardrail_passed: bool,
    agents_executed: list,
    chunks_retrieved: int,
    success: bool,
    error: Optional[str] = None
):
    """
    Log the completion of an API request.

    Args:
        request_id: Unique request identifier
        session_id: Session identifier
        duration: Total request duration
        guardrail_passed: Whether guardrail check passed
        agents_executed: List of agents that were executed
        chunks_retrieved: Number of chunks retrieved
        success: Whether the request succeeded
        error: Error message if failed
    """
    separator = "=" * 80
    logger.info(f"\n{separator}")
    logger.info("✅ REQUEST COMPLETE" if success else "❌ REQUEST FAILED")
    logger.info(separator)
    logger.info(f"Request ID: {request_id}")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Total Duration: {duration:.2f}s")
    logger.info(f"Guardrail: {'PASSED' if guardrail_passed else 'REJECTED'}")
    logger.info(f"Agents Executed: {', '.join(agents_executed)}")
    logger.info(f"Chunks Retrieved: {chunks_retrieved}")
    logger.info(f"Success: {success}")
    if error:
        logger.info(f"Error: {error}")
    logger.info(separator + "\n")

    # Structured log for Application Insights
    logger.info(
        "Query request completed",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "duration_seconds": duration,
            "guardrail_passed": guardrail_passed,
            "agents_executed": agents_executed,
            "chunks_retrieved": chunks_retrieved,
            "success": success,
            "error": error,
            "event_type": "request_complete"
        }
    )
