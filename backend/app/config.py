"""
Configuration management for NL-to-Insights Chat Engine.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ClickHouse Database Configuration
    clickhouse_host: str = "10.1.1.22"
    clickhouse_port: int = 8123
    clickhouse_database: str = "veedol_sales"
    clickhouse_username: str = "default"
    clickhouse_password: str = ""
    query_timeout_seconds: int = 30

    # LLM Configuration
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    use_openrouter: bool = True  # Set to True to use OpenRouter
    llm_model: str = "google/gemini-3-flash-preview"
    reasoning_effort: str = "high"  # none | minimal | low | medium | high | xhigh
    max_tokens: int = 16000

    # RAG Embedding Configuration
    embedding_provider: str = "openai"  # "openai" or "sentence-transformers"
    embedding_model: str = "text-embedding-3-small"  # OpenAI: text-embedding-3-small/large, ST: all-MiniLM-L6-v2
    embedding_dimensions: int = 1536  # text-embedding-3-small=1536, all-MiniLM-L6-v2=384

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Business Context
    business_context_path: str = "../metadata/business_context.txt"

    # ChromaDB Configuration
    chroma_persist_directory: str = "../data/chroma"

    class Config:
        env_file = "../.env"  # .env file is in project root, one level up from backend/
        case_sensitive = False


# Global settings instance
settings = Settings()
