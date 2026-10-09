"""FastAPI dependencies that build the retriever / embedder / provider from settings.
Tests override these to inject fakes, so no real Ollama is ever called in tests."""
from typing import Optional

from fastapi import Depends

from app.config import Settings, get_settings
from app.embeddings import Embedder, OllamaEmbedder
from app.ollama import OllamaClient
from app.providers import LLMProvider, get_llm_provider
from app.retrieval import HybridRetriever, LexicalRetriever, Retriever, VectorRetriever


def get_embedder(settings: Settings = Depends(get_settings)) -> Optional[Embedder]:
    if settings.ai_mode == "ollama":
        client = OllamaClient(settings.ollama_base_url, settings.ollama_timeout_seconds)
        return OllamaEmbedder(client, settings.ollama_embed_model,
                              settings.ollama_embed_doc_prefix, settings.ollama_embed_query_prefix)
    return None


def get_retriever(settings: Settings = Depends(get_settings), embedder: Optional[Embedder] = Depends(get_embedder)) -> Retriever:
    lexical = LexicalRetriever()
    if embedder is None:
        return lexical
    return HybridRetriever(lexical, VectorRetriever(embedder, settings.vector_min_similarity))


def get_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    return get_llm_provider(settings)
