"""The hosted-model path (LangChain). No network and no API key: a fake LangChain chat model stands in."""
import json
import uuid

import pytest

pytest.importorskip("langchain_core")
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.config import Settings  # noqa: E402
from app.providers import (ConfigurationError, LangChainChatProvider, ProviderError,  # noqa: E402
                           OllamaProvider, build_openai_llm, get_llm_provider)
from app.retrieval import RetrievedChunk, SearchResult  # noqa: E402


def retrieved(text="Initial waiting period: 30 days from the policy start date."):
    chunk = RetrievedChunk(uuid.uuid4(), uuid.uuid4(), "policy.pdf", 1, 0, text, 1.0, 1.0)
    return SearchResult(chunks=[chunk], term_weights={})


def fake(content):
    return GenericFakeChatModel(messages=iter([AIMessage(content=content)]))


def test_verified_answer_from_a_langchain_model_is_accepted():
    raw = json.dumps({"supported": True, "answer": "The waiting period is 30 days.",
                      "quotes": [{"passage": 1, "text": "Initial waiting period: 30 days"}]})
    result = LangChainChatProvider(fake(raw), "openai/test").generate("How long is the waiting period?", retrieved(), 0.5)
    assert result.text == "The waiting period is 30 days."
    assert result.generated_by == "openai/test (hosted LLM via LangChain)"
    assert result.evidence[0].sentence == "Initial waiting period: 30 days"


def test_invented_quote_from_a_langchain_model_is_still_rejected():
    raw = json.dumps({"supported": True, "answer": "It is 60 days.",
                      "quotes": [{"passage": 1, "text": "waiting period of 60 days"}]})
    assert LangChainChatProvider(fake(raw), "openai/test").generate("q", retrieved(), 0.5) is None


def test_no_retrieved_passages_means_no_model_call():
    class Boom:
        def invoke(self, messages):
            raise AssertionError("must not be called")
    assert LangChainChatProvider(Boom(), "x").generate("q", SearchResult(chunks=[], term_weights={}), 0.5) is None


def test_provider_failure_becomes_a_clear_error_that_never_echoes_the_key():
    class Failing:
        def invoke(self, messages):
            err = RuntimeError("Incorrect API key provided: sk-secret-1234")
            err.status_code = 401
            raise err

    with pytest.raises(ProviderError) as info:
        LangChainChatProvider(Failing(), "openai/test").generate("q", retrieved(), 0.5)
    text = str(info.value)
    assert "sk-secret" not in text and "HTTP 401" in text and "RuntimeError" in text


def test_missing_key_is_a_clear_configuration_error():
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_openai_llm(Settings(ai_mode="ollama", chat_provider="openai", openai_api_key=SecretStr("")))


def test_default_provider_is_still_ollama_and_openai_is_opt_in():
    assert isinstance(get_llm_provider(Settings(ai_mode="ollama", chat_provider="ollama")), OllamaProvider)
    pytest.importorskip("langchain_openai")
    s = Settings(ai_mode="ollama", chat_provider="openai", openai_api_key=SecretStr("sk-test"), openai_chat_model="gpt-4o-mini")
    provider = get_llm_provider(s)
    assert isinstance(provider, LangChainChatProvider) and provider.label == "openai/gpt-4o-mini"
    assert "sk-test" not in repr(s)          # the key is not printed when settings are logged


def test_ai_mode_real_and_its_older_name_ollama_both_turn_the_real_pipeline_on():
    from app.deps import get_embedder
    from app.embeddings import OllamaEmbedder
    for mode in ("real", "ollama"):
        s = Settings(ai_mode=mode, chat_provider="ollama", embed_provider="ollama")
        assert s.real_ai and isinstance(get_llm_provider(s), OllamaProvider) and isinstance(get_embedder(s), OllamaEmbedder)
    mock = Settings(ai_mode="mock")
    assert not mock.real_ai and get_embedder(mock) is None


def test_real_mode_with_openai_uses_langchain_for_answers():
    pytest.importorskip("langchain_openai")
    s = Settings(ai_mode="real", chat_provider="openai", openai_api_key=SecretStr("sk-test"))
    assert isinstance(get_llm_provider(s), LangChainChatProvider)
