"""The LangChain-built hybrid retriever (app/lc_retrieval.py). Fake embedder only: no model, no network."""
import json

from app.config import Settings
from app.deps import make_retriever
from app.lc_retrieval import LangChainHybridRetriever
from app.main import app
from app.providers import OllamaProvider
from app.qa import answer_question
from app.retrieval import HybridRetriever
from tests.fakes import FakeEmbedder, FakeOllamaChat, passage_number
from tests.test_hybrid import upload_all


def test_finds_the_right_page_for_a_keyword_question(client):
    upload_all(client, FakeEmbedder())
    with client.session_factory() as db:
        top = LangChainHybridRetriever(FakeEmbedder(), 0.5).search(db, "How many days of annual leave do employees get?", None, 5).chunks
    assert top and top[0].document_name == "employee_leave_policy.pdf"
    assert all(c.coverage >= 0 for c in top) and len(top) <= 5


def test_finds_a_synonym_match_through_the_vector_side(client):
    upload_all(client, FakeEmbedder())
    with client.session_factory() as db:
        def top(min_similarity):
            found = LangChainHybridRetriever(FakeEmbedder(), min_similarity).search(db, "Is surgery covered?", None, 5).chunks
            return [(c.document_name, c.page_number) for c in found]
        assert top(0.5)[0] == ("health_insurance_policy.pdf", 2)      # "surgery" is not in that text; matched by meaning
        assert ("health_insurance_policy.pdf", 2) not in top(2.0)      # vector side switched off: keywords alone miss it


def test_can_be_limited_to_some_documents(client):
    ids = upload_all(client, FakeEmbedder())
    import uuid
    only = [uuid.UUID(ids["refund_cancellation_policy.pdf"])]
    with client.session_factory() as db:
        chunks = LangChainHybridRetriever(FakeEmbedder(), 0.5).search(db, "How many days of annual leave?", only, 5).chunks
    assert chunks and {c.document_name for c in chunks} == {"refund_cancellation_policy.pdf"}


def test_stored_vectors_are_reused_not_embedded_again(client):
    embedder = FakeEmbedder()
    upload_all(client, embedder)
    calls_before = embedder.calls
    with client.session_factory() as db:
        LangChainHybridRetriever(embedder, 0.5).search(db, "cosmetic surgery", None, 5)
    assert embedder.calls == calls_before          # embed_documents (the chunks) was not called, only embed_query


def test_vectors_from_another_model_are_ignored(client):
    upload_all(client, FakeEmbedder("old-model"))
    with client.session_factory() as db:
        found = LangChainHybridRetriever(FakeEmbedder("new-model"), 0.5).search(db, "xyzzy plugh", None, 5).chunks
    # no vector from the new model exists; a keyword-less question finds nothing from the vector side either
    assert all("xyzzy" not in c.text for c in found)


def test_empty_database_and_empty_question_return_nothing(client):
    with client.session_factory() as db:
        assert LangChainHybridRetriever(FakeEmbedder()).search(db, "anything").chunks == []
    upload_all(client, FakeEmbedder())
    with client.session_factory() as db:
        assert LangChainHybridRetriever(FakeEmbedder()).search(db, "the of and").chunks == []


def test_factory_picks_the_requested_retriever():
    s, e = Settings(), FakeEmbedder()
    assert isinstance(make_retriever("own", s, e), HybridRetriever)
    assert isinstance(make_retriever("langchain", s, e), LangChainHybridRetriever)
    assert make_retriever("langchain", s, e).default_tokenizer is False            # our tokenizer feeds BM25
    plain = make_retriever("langchain-default-tokenizer", s, e)
    assert isinstance(plain, LangChainHybridRetriever) and plain.default_tokenizer is True
    assert not isinstance(make_retriever("langchain", s, None), LangChainHybridRetriever)   # no embedder: keyword only


def test_works_inside_the_full_answer_pipeline_and_verification_still_applies(client):
    embedder = FakeEmbedder()
    upload_all(client, embedder)
    quote = "Cosmetic procedures, self-inflicted injury and experimental treatments are excluded."

    def good(user):
        return json.dumps({"supported": True, "answer": "No. Cosmetic procedures are excluded.",
                           "quotes": [{"passage": passage_number(user, "Cosmetic procedures"), "text": quote}]})

    retriever = LangChainHybridRetriever(embedder, 0.5)
    settings = Settings(ai_mode="real")
    with client.session_factory() as db:
        ok = answer_question(db, "Does the policy cover cosmetic surgery?", None, settings, retriever,
                             OllamaProvider(FakeOllamaChat(good).client(), "llama3.1:latest"))
        bad = answer_question(db, "Does the policy cover cosmetic surgery?", None, settings, retriever,
                              OllamaProvider(FakeOllamaChat(lambda u: json.dumps(
                                  {"supported": True, "answer": "Yes, up to 500.",
                                   "quotes": [{"passage": 1, "text": "Cosmetic surgery is covered up to 500"}]})).client(), "llama3.1:latest"))
    assert ok.answered and ok.citations[0].page_number == 2
    assert not bad.answered


def test_our_tokenizer_finds_more_relevant_chunks_than_langchains_default_splitting(client):
    """Default splitting is case-sensitive and keeps punctuation, so 'LEAVE?' never matches 'leave'."""
    upload_all(client, FakeEmbedder())
    with client.session_factory() as db:
        ours = LangChainHybridRetriever(FakeEmbedder(), 2.0).search(db, "Annual LEAVE?", None, 3).chunks             # vector side off
        plain = LangChainHybridRetriever(FakeEmbedder(), 2.0, default_tokenizer=True).search(db, "Annual LEAVE?", None, 3).chunks
    assert [c.document_name for c in ours[:2]] == ["employee_leave_policy.pdf"] * 2
    assert plain[1].document_name != "employee_leave_policy.pdf"      # its second hit is just filler


def test_a_question_whose_embedding_is_all_zeros_does_not_crash(client):
    upload_all(client, FakeEmbedder())
    with client.session_factory() as db:      # FakeEmbedder gives an all-zero vector to words it does not know
        found = LangChainHybridRetriever(FakeEmbedder(), 0.5).search(db, "Reporting accidents", None, 3).chunks
    assert isinstance(found, list)
