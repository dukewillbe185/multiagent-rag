"""
Base Agent Module.

Provides base functionality for all agents in the multi-agent system.
"""

import logging
from typing import Dict, Any, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_config

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all agents.

    Provides common functionality:
    - Azure AI Foundry LLM client initialization
    - Logging
    - Error handling
    """

    def __init__(self, agent_name: str, system_prompt: str = None):
        """
        Initialize base agent.

        Args:
            agent_name: Name of the agent (for logging)
            system_prompt: Optional system prompt for the agent
        """
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.llm = self._initialize_llm()

        logger.info(f"[{self.agent_name}] Agent initialized")

    def _initialize_llm(self) -> AzureChatOpenAI:
        """
        Initialize Azure AI Foundry LLM client.

        Returns:
            Configured AzureChatOpenAI instance

        Note:
            Uses Azure AI Foundry endpoint, NOT Azure OpenAI Service
        """
        config = get_config()

        try:
            llm = AzureChatOpenAI(
                azure_endpoint=config.ai_foundry_gpt4_endpoint,
                azure_deployment=config.ai_foundry_gpt4_deployment,
                api_key=config.ai_foundry_gpt4_key,
                api_version=config.ai_foundry_gpt4_api_version,
                temperature=config.llm_temperature
            )

            logger.info(
                f"[{self.agent_name}] LLM initialized with deployment: "
                f"{config.ai_foundry_gpt4_deployment}"
            )

            return llm

        except Exception as e:
            logger.error(f"[{self.agent_name}] Failed to initialize LLM: {e}")
            raise

    def invoke_llm(self, user_message: str, system_message: str = None) -> str:
        """
        Invoke the LLM with a message.

        Args:
            user_message: User/human message
            system_message: Optional system message (overrides default)

        Returns:
            LLM response text

        Raises:
            Exception: If LLM invocation fails
        """
        try:
            messages = []

            # Add system message
            sys_msg = system_message or self.system_prompt
            if sys_msg:
                messages.append(SystemMessage(content=sys_msg))

            # Add user message
            messages.append(HumanMessage(content=user_message))

            logger.info(f"[{self.agent_name}] Invoking LLM...")

            # Invoke LLM
            response = self.llm.invoke(messages)

            logger.info(f"[{self.agent_name}] LLM response received")

            return response.content

        except Exception as e:
            logger.error(f"[{self.agent_name}] LLM invocation failed: {e}")
            raise

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent logic.

        This should be overridden by subclasses.

        Args:
            state: Current state dictionary

        Returns:
            Updated state dictionary
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def log_info(self, message: str):
        """Log info message with agent name prefix."""
        logger.info(f"[{self.agent_name}] {message}")

    def log_warning(self, message: str):
        """Log warning message with agent name prefix."""
        logger.warning(f"[{self.agent_name}] {message}")

    def log_error(self, message: str):
        """Log error message with agent name prefix."""
        logger.error(f"[{self.agent_name}] {message}")
