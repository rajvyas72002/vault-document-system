"""The LangChain pipeline must behave exactly like the plain-function pipeline."""
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from app.providers import ProviderAnswer
from app.qa import NOT_FOUND_TEXT, answer_question
from app.retrieval import RetrievedChunk, SearchResult
from app.providers import Evidence


def make_chunk(text, page=1, name="policy.pdf"):
    return RetrievedChunk(uuid.uuid4(), uuid.uuid4(), name, page, 0, text, 0.9, 0.8)


class FakeRetriever:
    method = "fake"

    def __init__(self, chunks, weights=None):
        self.chunks, self.weights, self.calls = chunks, weights or {"leave": 1.5}, []

    def search(self, db, query, document_ids, top_k):
        self.calls.append((query, document_ids, top_k))
        return SearchResult(list(self.chunks), self.weights)


class FakeProvider:
    def __init__(self, answer=True):
        self.answer, self.seen = answer, None

    def generate(self, question, retrieved, min_coverage, history=()):
        self.seen = (question, retrieved, min_coverage, list(history))
        if not self.answer or not retrieved.chunks:
            return None
        c = retrieved.chunks[0]
        return ProviderAnswer("24 days.", [Evidence(c, "24 days of paid leave")], "fake", "note")


def settings(pipeline):
    return SimpleNamespace(pipeline=pipeline, retrieval_top_k=5, min_coverage=0.3)


def run(pipeline, chunks, answer=True, **kw):
    retriever, provider = FakeRetriever(chunks), FakeProvider(answer)
    result = answer_question(None, "How many leave days?", None, settings(pipeline), retriever, provider, **kw)
    return result, retriever, provider


def test_both_pipelines_give_the_same_answer_and_citations():
    chunks = [make_chunk("Employees get 24 days of paid leave."), make_chunk("Other text.", page=2)]
    own, *_ = run("own", chunks)
    lc, *_ = run("langchain", chunks)
    assert lc == own
    assert lc.answered and lc.citations[0].quote == "24 days of paid leave"


def test_nothing_retrieved_gives_the_same_not_found_answer():
    own, *_ = run("own", [])
    lc, *_ = run("langchain", [])
    assert lc == own and lc.answer == NOT_FOUND_TEXT and not lc.answered


def test_provider_refusal_gives_the_same_not_found_answer():
    own, *_ = run("own", [make_chunk("x")], answer=False)
    lc, *_ = run("langchain", [make_chunk("x")], answer=False)
    assert lc.answer == own.answer == NOT_FOUND_TEXT


def test_provider_receives_identical_chunks_weights_and_history():
    chunks = [make_chunk("Employees get 24 days of paid leave."), make_chunk("More.", page=3)]
    _, _, p_own = run("own", chunks, history=["h"], retrieval_query="leave days")
    _, r_lc, p_lc = run("langchain", chunks, history=["h"], retrieval_query="leave days")
    assert [c.text for c in p_lc.seen[1].chunks] == [c.text for c in p_own.seen[1].chunks]
    assert [c.chunk_id for c in p_lc.seen[1].chunks] == [c.chunk_id for c in p_own.seen[1].chunks]
    assert p_lc.seen[1].term_weights == p_own.seen[1].term_weights == {"leave": 1.5}
    assert p_lc.seen[3] == ["h"] and p_lc.seen[0] == "How many leave days?"
    assert r_lc.calls[0][0] == "leave days"           # the follow-up rewrite is what gets searched


def test_vault_retriever_is_a_real_langchain_retriever_with_metadata():
    from langchain_core.retrievers import BaseRetriever
    from app.lc_pipeline import VaultRetriever
    chunk = make_chunk("Employees get 24 days of paid leave.", page=4, name="leave.pdf")
    vault = VaultRetriever(retriever=FakeRetriever([chunk]), db=None, top_k=3)
    assert isinstance(vault, BaseRetriever)
    docs = vault.invoke("leave")
    assert docs[0].page_content == chunk.text
    assert docs[0].metadata["document_name"] == "leave.pdf" and docs[0].metadata["page_number"] == 4
