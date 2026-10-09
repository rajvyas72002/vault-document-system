from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Vault Document System"
    database_url: str = "postgresql+psycopg://vault:change_me@localhost:5432/vault"
    ai_mode: Literal["mock", "real"] = "mock"
    max_upload_mb: int = 20
    storage_dir: str = "storage"
    chunk_size: int = 800      # characters per chunk
    chunk_overlap: int = 150   # characters shared between neighbouring chunks


@lru_cache
def get_settings() -> Settings:
    return Settings()
