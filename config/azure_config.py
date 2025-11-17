"""
Azure Services Configuration Module

This module loads and validates all Azure service credentials and endpoints
from environment variables. It implements security best practices:
- No hardcoded secrets
- Environment variable validation at startup
- Masked logging of sensitive data
- Support for Azure Managed Identity (recommended for production)

Security Note:
    In production environments, use Azure Managed Identity instead of API keys.
    This provides automatic credential rotation and eliminates the need to
    store secrets in environment variables.
    See: https://learn.microsoft.com/azure/developer/python/sdk/authentication-overview
"""

import os
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def _mask_secret(secret: str, show_chars: int = 4) -> str:
    """
    Mask a secret string for safe logging.

    Args:
        secret: The secret string to mask
        show_chars: Number of characters to show at start and end

    Returns:
        Masked string showing only first and last few characters
    """
    if not secret or len(secret) <= show_chars * 2:
        return "***"
    return f"{secret[:show_chars]}...{secret[-show_chars:]}"


def _get_required_env(var_name: str) -> str:
    """
    Get a required environment variable or raise an error.

    Args:
        var_name: Name of the environment variable

    Returns:
        Value of the environment variable

    Raises:
        ConfigurationError: If the environment variable is not set
    """
    value = os.getenv(var_name)
    if not value:
        raise ConfigurationError(
            f"Required environment variable '{var_name}' is not set. "
            f"Please check your .env file."
        )
    return value


def _get_optional_env(var_name: str, default: str) -> str:
    """
    Get an optional environment variable with a default value.

    Args:
        var_name: Name of the environment variable
        default: Default value if not set

    Returns:
        Value of the environment variable or default
    """
    return os.getenv(var_name, default)


class AzureConfig:
    """
    Azure Services Configuration.

    This class loads and validates all Azure service credentials and settings.
    All values are loaded from environment variables for security.
    """

    def __init__(self):
        """Initialize and validate Azure configuration."""
        try:
            # Document Intelligence
            self.doc_intelligence_endpoint = _get_required_env("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            self.doc_intelligence_key = _get_required_env("AZURE_DOCUMENT_INTELLIGENCE_KEY")

            # Azure AI Foundry - Embedding Model
            # CRITICAL: These are separate from Azure OpenAI Service
            self.ai_foundry_embedding_endpoint = _get_required_env("AZURE_AI_FOUNDRY_EMBEDDING_ENDPOINT")
            self.ai_foundry_embedding_deployment = _get_required_env("AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME")
            self.ai_foundry_embedding_key = _get_required_env("AZURE_AI_FOUNDRY_EMBEDDING_API_KEY")
            self.ai_foundry_embedding_api_version = _get_optional_env(
                "AZURE_AI_FOUNDRY_EMBEDDING_API_VERSION",
                "2024-08-01-preview"
            )

            # Azure AI Foundry - GPT-4 Model
            # CRITICAL: These are separate from Azure OpenAI Service
            self.ai_foundry_gpt4_endpoint = _get_required_env("AZURE_AI_FOUNDRY_GPT4_ENDPOINT")
            self.ai_foundry_gpt4_deployment = _get_required_env("AZURE_AI_FOUNDRY_GPT4_DEPLOYMENT_NAME")
            self.ai_foundry_gpt4_key = _get_required_env("AZURE_AI_FOUNDRY_GPT4_API_KEY")
            self.ai_foundry_gpt4_api_version = _get_optional_env(
                "AZURE_AI_FOUNDRY_GPT4_API_VERSION",
                "2024-08-01-preview"
            )

            # Azure AI Search
            self.search_endpoint = _get_required_env("AZURE_SEARCH_ENDPOINT")
            self.search_admin_key = _get_required_env("AZURE_SEARCH_ADMIN_KEY")
            self.search_index_name = _get_optional_env("AZURE_SEARCH_INDEX_NAME", "rag-documents-index")

            # Azure Application Insights
            self.app_insights_connection_string = _get_required_env("APPLICATIONINSIGHTS_CONNECTION_STRING")

            # Application Settings
            self.chunk_size = int(_get_optional_env("CHUNK_SIZE", "1200"))
            self.chunk_overlap = float(_get_optional_env("CHUNK_OVERLAP", "0.2"))
            self.retrieval_top_k = int(_get_optional_env("RETRIEVAL_TOP_K", "5"))
            self.llm_temperature = float(_get_optional_env("LLM_TEMPERATURE", "0.7"))

            # Validate configuration
            self._validate()

            logger.info("Azure configuration loaded successfully")
            self._log_config()

        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _validate(self):
        """Validate configuration values."""
        # Validate endpoints are URLs
        endpoints = [
            self.doc_intelligence_endpoint,
            self.ai_foundry_embedding_endpoint,
            self.ai_foundry_gpt4_endpoint,
            self.search_endpoint
        ]
        for endpoint in endpoints:
            if not endpoint.startswith(("http://", "https://")):
                raise ConfigurationError(f"Invalid endpoint URL: {endpoint}")

        # Validate numeric ranges
        if not 100 <= self.chunk_size <= 10000:
            raise ConfigurationError(f"CHUNK_SIZE must be between 100 and 10000, got {self.chunk_size}")

        if not 0.0 <= self.chunk_overlap <= 1.0:
            raise ConfigurationError(f"CHUNK_OVERLAP must be between 0.0 and 1.0, got {self.chunk_overlap}")

        if not 1 <= self.retrieval_top_k <= 100:
            raise ConfigurationError(f"RETRIEVAL_TOP_K must be between 1 and 100, got {self.retrieval_top_k}")

        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ConfigurationError(f"LLM_TEMPERATURE must be between 0.0 and 2.0, got {self.llm_temperature}")

    def _log_config(self):
        """Log configuration (with masked secrets) for debugging."""
        logger.info("=" * 60)
        logger.info("Azure Configuration:")
        logger.info("-" * 60)
        logger.info(f"Document Intelligence Endpoint: {self.doc_intelligence_endpoint}")
        logger.info(f"Document Intelligence Key: {_mask_secret(self.doc_intelligence_key)}")
        logger.info("-" * 60)
        logger.info(f"AI Foundry Embedding Endpoint: {self.ai_foundry_embedding_endpoint}")
        logger.info(f"AI Foundry Embedding Deployment: {self.ai_foundry_embedding_deployment}")
        logger.info(f"AI Foundry Embedding Key: {_mask_secret(self.ai_foundry_embedding_key)}")
        logger.info(f"AI Foundry Embedding API Version: {self.ai_foundry_embedding_api_version}")
        logger.info("-" * 60)
        logger.info(f"AI Foundry GPT-4 Endpoint: {self.ai_foundry_gpt4_endpoint}")
        logger.info(f"AI Foundry GPT-4 Deployment: {self.ai_foundry_gpt4_deployment}")
        logger.info(f"AI Foundry GPT-4 Key: {_mask_secret(self.ai_foundry_gpt4_key)}")
        logger.info(f"AI Foundry GPT-4 API Version: {self.ai_foundry_gpt4_api_version}")
        logger.info("-" * 60)
        logger.info(f"Azure Search Endpoint: {self.search_endpoint}")
        logger.info(f"Azure Search Admin Key: {_mask_secret(self.search_admin_key)}")
        logger.info(f"Azure Search Index Name: {self.search_index_name}")
        logger.info("-" * 60)
        logger.info(f"App Insights Connection String: {_mask_secret(self.app_insights_connection_string, 10)}")
        logger.info("-" * 60)
        logger.info(f"Chunk Size: {self.chunk_size}")
        logger.info(f"Chunk Overlap: {self.chunk_overlap}")
        logger.info(f"Retrieval Top K: {self.retrieval_top_k}")
        logger.info(f"LLM Temperature: {self.llm_temperature}")
        logger.info("=" * 60)


# Global configuration instance
# This will be initialized when the module is imported
_config: Optional[AzureConfig] = None


def get_config() -> AzureConfig:
    """
    Get the global Azure configuration instance.

    Returns:
        AzureConfig: The global configuration instance

    Raises:
        ConfigurationError: If configuration fails to load
    """
    global _config
    if _config is None:
        _config = AzureConfig()
    return _config


def reload_config() -> AzureConfig:
    """
    Reload the configuration from environment variables.

    Useful for testing or when environment variables change.

    Returns:
        AzureConfig: The new configuration instance
    """
    global _config
    _config = AzureConfig()
    return _config


# Production Security Note:
# ========================
# For production deployments, consider using Azure Managed Identity instead of API keys:
#
# from azure.identity import DefaultAzureCredential
#
# credential = DefaultAzureCredential()
#
# This approach:
# - Eliminates the need to store API keys
# - Provides automatic credential rotation
# - Enables fine-grained RBAC
# - Works seamlessly in Azure-hosted environments (App Service, Container Apps, VMs, etc.)
#
# Example usage with Azure Search:
# from azure.search.documents import SearchClient
# search_client = SearchClient(
#     endpoint=search_endpoint,
#     index_name=index_name,
#     credential=credential
# )
