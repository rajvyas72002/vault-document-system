"""OpenAI embeddings through LangChain. No network and no key: a fake client stands in."""
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.embeddings import EmbeddingError, OpenAIEmbedder, build_openai_embedder
from app.providers import ConfigurationError


class FakeClient:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(("docs", list(texts)))
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        self.calls.append(("query", text))
        return [float(len(text)), 1.0]


def test_embedder_returns_one_vector_per_text_and_names_its_model():
    client = FakeClient()
    emb = OpenAIEmbedder(client, "text-embedding-3-small")
    assert emb.model_name == "openai:text-embedding-3-small"          # never equal to an Ollama model name
    assert emb.embed_documents(["ab", "abcd"]) == [[2.0, 1.0], [4.0, 1.0]]
    assert emb.embed_query("abc") == [3.0, 1.0]
    assert client.calls[0] == ("docs", ["ab", "abcd"]) and client.calls[1] == ("query", "abc")   # no search_document: prefix


def test_failure_is_a_clear_error_that_never_echoes_the_key():
    class Failing:
        def embed_documents(self, texts):
            err = RuntimeError("Incorrect API key provided: sk-secret-1234")
            err.status_code = 401
            raise err

        embed_query = embed_documents

    with pytest.raises(EmbeddingError) as info:
        OpenAIEmbedder(Failing(), "m").embed_query("x")
    assert "sk-secret" not in str(info.value) and "HTTP 401" in str(info.value)


def test_wrong_number_of_vectors_is_an_error():
    class Short(FakeClient):
        def embed_documents(self, texts):
            return [[1.0]]

    with pytest.raises(EmbeddingError, match="unexpected number"):
        OpenAIEmbedder(Short(), "m").embed_documents(["a", "b"])


def test_missing_key_is_a_clear_configuration_error():
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_openai_embedder(Settings(ai_mode="ollama", embed_provider="openai", openai_api_key=SecretStr("")))


def test_builder_makes_an_openai_embedder_when_the_package_is_installed():
    pytest.importorskip("langchain_openai")
    emb = build_openai_embedder(Settings(ai_mode="ollama", embed_provider="openai", openai_api_key=SecretStr("sk-test"),
                                         openai_embed_model="text-embedding-3-small"))
    assert emb.model_name == "openai:text-embedding-3-small"


def test_default_embedder_is_still_ollama():
    from app.deps import get_embedder
    from app.embeddings import OllamaEmbedder
    assert isinstance(get_embedder(Settings(ai_mode="ollama", embed_provider="ollama")), OllamaEmbedder)
    assert get_embedder(Settings(ai_mode="mock")) is None
