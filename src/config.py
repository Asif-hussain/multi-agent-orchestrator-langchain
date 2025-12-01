"""
Centralized configuration management for the Multi-Agent Support System.
All configuration is read from environment variables with sensible defaults.
"""

import os
from typing import Optional


class Config:
    """Configuration manager for the multi-agent system."""

    # API Configuration
    @staticmethod
    def get_openrouter_api_key() -> str:
        """Get OpenRouter API key from environment."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Please add it to your .env file."
            )
        return api_key

    @staticmethod
    def get_openrouter_base_url() -> str:
        """Get OpenRouter base URL from environment."""
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Model Configuration
    @staticmethod
    def get_embedding_model() -> str:
        """Get embedding model name from environment."""
        return os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

    @staticmethod
    def get_llm_model() -> str:
        """Get LLM model name from environment."""
        return os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    @staticmethod
    def get_evaluator_model() -> str:
        """Get evaluator model name from environment."""
        return os.getenv("EVALUATOR_MODEL", "gpt-3.5-turbo")

    # RAG Configuration
    @staticmethod
    def get_chunk_size() -> int:
        """Get document chunk size from environment."""
        return int(os.getenv("CHUNK_SIZE", "1000"))

    @staticmethod
    def get_chunk_overlap() -> int:
        """Get document chunk overlap from environment."""
        return int(os.getenv("CHUNK_OVERLAP", "200"))

    @staticmethod
    def get_retrieval_k() -> int:
        """Get number of chunks to retrieve from environment."""
        return int(os.getenv("RETRIEVAL_K", "4"))

    # Model Parameters
    @staticmethod
    def get_temperature() -> float:
        """Get model temperature from environment."""
        return float(os.getenv("TEMPERATURE", "0.1"))

    # Langfuse Configuration
    @staticmethod
    def get_langfuse_public_key() -> Optional[str]:
        """Get Langfuse public key from environment."""
        return os.getenv("LANGFUSE_PUBLIC_KEY")

    @staticmethod
    def get_langfuse_secret_key() -> Optional[str]:
        """Get Langfuse secret key from environment."""
        return os.getenv("LANGFUSE_SECRET_KEY")

    @staticmethod
    def get_langfuse_host() -> str:
        """Get Langfuse host from environment."""
        return os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Logging Configuration
    @staticmethod
    def get_log_level() -> str:
        """Get logging level from environment."""
        return os.getenv("LOG_LEVEL", "INFO").upper()

    @classmethod
    def validate(cls) -> bool:
        """
        Validate that all required configuration is present.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If required configuration is missing
        """
        # Check required API key
        cls.get_openrouter_api_key()

        return True

    @classmethod
    def display(cls) -> None:
        """Display current configuration (masks sensitive values)."""
        print("Current Configuration:")
        print(f"  OpenRouter Base URL: {cls.get_openrouter_base_url()}")
        print(f"  Embedding Model: {cls.get_embedding_model()}")
        print(f"  LLM Model: {cls.get_llm_model()}")
        print(f"  Evaluator Model: {cls.get_evaluator_model()}")
        print(f"  Temperature: {cls.get_temperature()}")
        print(f"  Chunk Size: {cls.get_chunk_size()}")
        print(f"  Chunk Overlap: {cls.get_chunk_overlap()}")
        print(f"  Retrieval K: {cls.get_retrieval_k()}")
        print(f"  Log Level: {cls.get_log_level()}")

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if api_key:
            print(f"  OpenRouter API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else ''}")
        else:
            print("  OpenRouter API Key: NOT SET")
