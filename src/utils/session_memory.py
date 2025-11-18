"""
Session Memory Manager for Multi-Agent RAG System

Manages in-memory session storage for conversation context.
- Stores only the previous turn (N-1) for each session
- Automatic cleanup of expired sessions
- Thread-safe operations
"""

import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SessionMemoryManager:
    """
    In-memory session manager for conversation context.

    Each session stores:
    - previous_question: Last question asked in this session
    - previous_answer: Last answer generated in this session
    - turn_count: Number of conversation turns
    - last_activity: Timestamp of last activity
    """

    def __init__(self, timeout_minutes: int = 30):
        """
        Initialize the session memory manager.

        Args:
            timeout_minutes: Session expiration timeout in minutes (default: 30)
        """
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._timeout_minutes = timeout_minutes
        logger.info(f"SessionMemoryManager initialized with {timeout_minutes}min timeout")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session context if it exists and hasn't expired.

        Args:
            session_id: Unique session identifier

        Returns:
            Dictionary with session context or None if session doesn't exist/expired
            Format: {
                "previous_question": str,
                "previous_answer": str,
                "turn_count": int,
                "last_activity": datetime
            }
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.info(f"Session not found: {session_id}")
                return None

            session = self._sessions[session_id]

            # Check if session has expired
            if self._is_expired(session):
                logger.info(f"Session expired: {session_id}")
                del self._sessions[session_id]
                return None

            # Update last activity
            session["last_activity"] = datetime.now()
            logger.info(f"Session retrieved: {session_id}, turn: {session['turn_count']}")
            return session.copy()

    def update_session(
        self,
        session_id: str,
        question: str,
        answer: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update session with current question and answer (becomes previous for next turn).

        Args:
            session_id: Unique session identifier
            question: Current question being asked
            answer: Generated answer for the current question
            user_id: Optional user identifier
            metadata: Optional additional metadata to store
        """
        with self._lock:
            if session_id in self._sessions:
                # Existing session - increment turn count
                self._sessions[session_id] = {
                    "previous_question": question,
                    "previous_answer": answer,
                    "turn_count": self._sessions[session_id]["turn_count"] + 1,
                    "last_activity": datetime.now(),
                    "user_id": user_id,
                    "metadata": metadata or {}
                }
                logger.info(f"Session updated: {session_id}, turn: {self._sessions[session_id]['turn_count']}")
            else:
                # New session
                self._sessions[session_id] = {
                    "previous_question": question,
                    "previous_answer": answer,
                    "turn_count": 1,
                    "last_activity": datetime.now(),
                    "user_id": user_id,
                    "metadata": metadata or {}
                }
                logger.info(f"New session created: {session_id}")

    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions from memory.

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            expired_sessions = [
                sid for sid, session in self._sessions.items()
                if self._is_expired(session)
            ]

            for sid in expired_sessions:
                del self._sessions[sid]

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions: {expired_sessions}")

            return len(expired_sessions)

    def get_session_count(self) -> int:
        """Get the current number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def delete_session(self, session_id: str) -> bool:
        """
        Manually delete a session.

        Args:
            session_id: Session to delete

        Returns:
            True if session was deleted, False if it didn't exist
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Session manually deleted: {session_id}")
                return True
            return False

    def _is_expired(self, session: Dict[str, Any]) -> bool:
        """
        Check if a session has expired based on last activity.

        Args:
            session: Session dictionary with 'last_activity' field

        Returns:
            True if session has expired, False otherwise
        """
        timeout_delta = timedelta(minutes=self._timeout_minutes)
        return datetime.now() - session["last_activity"] > timeout_delta

    def get_all_sessions_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active sessions (for debugging/monitoring).

        Returns:
            Dictionary mapping session_id to session info
        """
        with self._lock:
            return {
                sid: {
                    "turn_count": session["turn_count"],
                    "last_activity": session["last_activity"].isoformat(),
                    "user_id": session.get("user_id"),
                    "minutes_since_activity": (datetime.now() - session["last_activity"]).total_seconds() / 60
                }
                for sid, session in self._sessions.items()
            }


# Singleton instance
_session_manager: Optional[SessionMemoryManager] = None
_manager_lock = threading.Lock()


def get_session_manager(timeout_minutes: int = 30) -> SessionMemoryManager:
    """
    Get or create the singleton SessionMemoryManager instance.

    Args:
        timeout_minutes: Session timeout in minutes (only used on first call)

    Returns:
        SessionMemoryManager singleton instance
    """
    global _session_manager

    if _session_manager is None:
        with _manager_lock:
            if _session_manager is None:
                _session_manager = SessionMemoryManager(timeout_minutes=timeout_minutes)

    return _session_manager
