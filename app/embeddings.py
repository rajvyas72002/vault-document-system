from typing import List, Optional, Protocol, Sequence

from app.config import Settings
from app.errors import AIServiceError
from app.ollama import OllamaClient


class EmbeddingError(AIServiceError):
    pass


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class OllamaEmbedder:
    BATCH_SIZE = 16

    def __init__(self, client: OllamaClient, model: str, doc_prefix: str = "", query_prefix: str = ""):
        self.client = client
        self.model_name = model
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        data = self.client.request("POST", "/api/embed", {"model": self.model_name, "input": inputs},
                                   EmbeddingError, self.model_name)
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise EmbeddingError("Ollama returned an unexpected number of embeddings.")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        prefixed = [self.doc_prefix + t for t in texts]
        for i in range(0, len(prefixed), self.BATCH_SIZE):
            out.extend(self._embed(prefixed[i:i + self.BATCH_SIZE]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([self.query_prefix + text])[0]


class OpenAIEmbedder:
    """OpenAI embeddings through LangChain's OpenAIEmbeddings (or any object with the same two methods)."""

    def __init__(self, client, model: str):
        self.client = client
        self.model_name = f"openai:{model}"   # recorded on every chunk, so vectors from different models are never mixed

    def _call(self, fn, arg):
        try:
            return fn(arg)
        except Exception as exc:  # never include str(exc): provider errors can echo part of the API key
            status = getattr(exc, "status_code", None)
            raise EmbeddingError(f"OpenAI embedding request failed ({type(exc).__name__}{f', HTTP {status}' if status else ''}). "
                                 "Check the API key, the embedding model name and the network.") from None

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._call(self.client.embed_documents, list(texts))
        if len(vectors) != len(texts):
            raise EmbeddingError("OpenAI returned an unexpected number of embeddings.")
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._call(self.client.embed_query, text)


def build_openai_embedder(settings: Settings) -> OpenAIEmbedder:
    from app.providers import ConfigurationError   # imported here: providers imports this module's siblings
    key = settings.openai_api_key.get_secret_value()
    if not key:
        raise ConfigurationError("EMBED_PROVIDER=openai but OPENAI_API_KEY is empty. Add it to your .env file.")
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ConfigurationError("EMBED_PROVIDER=openai needs the package: pip install langchain-openai") from None
    options = {"model": settings.openai_embed_model, "api_key": key, "timeout": settings.openai_timeout_seconds, "max_retries": 1}
    if settings.openai_base_url:
        options["base_url"] = settings.openai_base_url
        options["check_embedding_ctx_length"] = False   # a gateway may not understand pre-tokenised input
    return OpenAIEmbedder(OpenAIEmbeddings(**options), settings.openai_embed_model)
