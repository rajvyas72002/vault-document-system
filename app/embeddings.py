from typing import List, Optional, Protocol, Sequence

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
