from functools import lru_cache
from typing import Literal
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Vault Document System"
    database_url: str = "postgresql+psycopg://vault:change_me@localhost:5432/vault"
    # mock: keyword search, no AI | real: embeddings + LLM, each from CHAT_PROVIDER / EMBED_PROVIDER (ollama or openai)
    # | ollama: the original name of "real" with local models, kept so older .env files still work
    ai_mode: Literal["mock", "ollama", "real"] = "mock"
    max_upload_mb: int = 20
    storage_dir: str = "storage"
    chunk_size: int = 800      # characters per chunk
    chunk_overlap: int = 150   # characters shared between neighbouring chunks
    # own: sliding window | langchain: RecursiveCharacterTextSplitter (sentence-aware). Re-run reprocess_all after changing.
    chunker: Literal["own", "langchain"] = "own"
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
    # --- hosted chat model (optional). Only the ANSWER step moves; retrieval and embeddings stay local ---
    chat_provider: Literal["ollama", "openai"] = "ollama"   # used when AI_MODE=ollama
    openai_api_key: SecretStr = SecretStr("")                # put it in .env only, never in code or git
    embed_provider: Literal["ollama", "openai"] = "ollama"   # which model turns text into vectors (re-run reprocess_all after changing)
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_base_url: str = ""                                # blank = api.openai.com; set for a company gateway / Azure-style endpoint
    openai_timeout_seconds: float = 60.0
    openai_temperature: str = "0"                            # blank = do not send it (some newer models only accept their default)
    vector_min_similarity: float = 0.5           # cosine cut-off, NOT calibrated: tune with POST /search
    retrieval_top_k: int = 5   # how many chunks to retrieve per question
    min_coverage: float = 0.5  # share of the question's key terms a passage must match

    @property
    def real_ai(self) -> bool:
        """True when embeddings and an LLM are used (AI_MODE=real, or its older name ollama)."""
        return self.ai_mode in ("real", "ollama")


@lru_cache
def get_settings() -> Settings:
    return Settings()
