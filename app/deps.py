"""FastAPI dependencies that build the retriever / embedder / provider from settings.
Tests override these to inject fakes, so no real Ollama is ever called in tests."""
from typing import Optional

from fastapi import BackgroundTasks, Depends

from app.config import Settings, get_settings
from app.conversation import OllamaRewriter
from app.insights import InsightGenerator, get_insight_generator_for
from app.embeddings import Embedder, OllamaEmbedder, build_openai_embedder
from app.ollama import OllamaClient
from app.providers import LLMProvider, get_llm_provider
from app.retrieval import HybridRetriever, LexicalRetriever, Retriever, VectorRetriever


def get_embedder(settings: Settings = Depends(get_settings)) -> Optional[Embedder]:
    if settings.real_ai and settings.embed_provider == "openai":
        return build_openai_embedder(settings)
    if settings.real_ai:
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


def get_rewriter(settings: Settings = Depends(get_settings)) -> Optional[OllamaRewriter]:
    """Rewrites chat follow-ups into standalone questions. Only available with a real LLM."""
    if settings.real_ai:
        return OllamaRewriter(OllamaClient(settings.ollama_base_url, settings.ollama_timeout_seconds), settings.ollama_chat_model)
    return None


def get_insight_generator(settings: Settings = Depends(get_settings)) -> InsightGenerator:
    return get_insight_generator_for(settings)


def get_session_factory():
    """Background jobs need their own database session (the request's session is closed by then)."""
    from app.db import SessionLocal
    return SessionLocal


def get_dispatcher(background: BackgroundTasks, settings: Settings = Depends(get_settings),
                   session_factory=Depends(get_session_factory)):
    """None means 'process inline'. Imported lazily to avoid a circular import with app.tasks."""
    from app.tasks import BackgroundDispatcher, CeleryDispatcher
    if settings.task_backend == "background":
        return BackgroundDispatcher(background, session_factory, settings)
    if settings.task_backend == "celery":
        return CeleryDispatcher()
    return None
