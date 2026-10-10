from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Vault Document System"
    database_url: str = "postgresql+psycopg://vault:change_me@localhost:5432/vault"
    ai_mode: Literal["mock", "ollama", "real"] = "mock"
    max_upload_mb: int = 20
    storage_dir: str = "storage"
    chunk_size: int = 800      # characters per chunk
    chunk_overlap: int = 150   # characters shared between neighbouring chunks
    # --- background processing ---
    # inline: process inside the request (simple, blocks) | background: FastAPI background task (no Redis)
    # | celery: real queue (Redis + a worker process)
    task_backend: Literal["inline", "background", "celery"] = "inline"
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"   # comma-separated origins allowed to call the API from a browser
    auto_insights: bool = False   # queued processing also generates insights once a document completes
    # --- Ollama (local models) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:latest"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_seconds: float = 180.0        # first call can be slow while the model loads
    ollama_embed_doc_prefix: str = "search_document: "   # nomic-embed-text expects these prefixes;
    ollama_embed_query_prefix: str = "search_query: "    # set both to "" for models that do not
    vector_min_similarity: float = 0.5           # cosine cut-off, NOT calibrated: tune with POST /search
    retrieval_top_k: int = 5   # how many chunks to retrieve per question
    min_coverage: float = 0.5  # share of the question's key terms a passage must match


@lru_cache
def get_settings() -> Settings:
    return Settings()
