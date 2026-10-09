import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.deps import get_embedder, get_provider, get_retriever
from app.main import app
from app.models import Document, DocumentChunk
from app.providers import OllamaProvider
from app.retrieval import HybridRetriever, LexicalRetriever, VectorRetriever, cosine, reciprocal_rank_fusion
from tests.fakes import BrokenEmbedder, FakeEmbedder, FakeOllamaChat, passage_number

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"


def test_rank_fusion_rewards_agreement_between_rankings():
    a, b, c = (uuid.uuid4() for _ in range(3))
    fused = reciprocal_rank_fusion([[a, b], [b, c]])
    assert max(fused, key=fused.get) == b          # found by both lists
    assert fused[a] > fused[c]                     # a was ranked above c, each by one list


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0


def upload_all(client, embedder):
    app.dependency_overrides[get_embedder] = lambda: embedder
    ids = {}
    for f in sorted(POLICIES.glob("*.pdf")):
        r = client.post("/documents/upload", files={"file": (f.name, f.read_bytes(), "application/pdf")})
        ids[f.name] = r.json()["id"]
        assert client.post(f"/documents/{ids[f.name]}/process").json()["status"] == "completed"
    return ids


def test_processing_stores_embeddings_with_model_name(client):
    ids = upload_all(client, FakeEmbedder("fake-embed"))
    with client.session_factory() as db:
        chunks = db.scalars(select(DocumentChunk)).all()
        assert chunks and all(c.embedding_model == "fake-embed" and len(c.embedding) > 0 for c in chunks)


def test_embedding_failure_is_not_fatal_and_keyword_search_still_works(client):
    app.dependency_overrides[get_embedder] = lambda: BrokenEmbedder()
    f = POLICIES / "employee_leave_policy.pdf"
    doc_id = client.post("/documents/upload", files={"file": (f.name, f.read_bytes(), "application/pdf")}).json()["id"]
    body = client.post(f"/documents/{doc_id}/process").json()
    assert body["status"] == "completed"
    assert "without embeddings" in body["error_message"]
    with client.session_factory() as db:
        assert all(c.embedding is None and c.embedding_model is None for c in db.scalars(select(DocumentChunk)))
    # mock-mode keyword search still answers
    app.dependency_overrides.pop(get_embedder)
    assert client.post("/ask", json={"question": "How many annual leave days are provided per year?"}).json()["answered"] is True
    # a later successful retry clears the warning
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    retry = client.post(f"/documents/{doc_id}/process").json()
    assert retry["status"] == "completed" and retry["error_message"] is None


def test_vector_retriever_ignores_vectors_from_a_different_model(client):
    upload_all(client, FakeEmbedder("old-model"))
    with client.session_factory() as db:
        assert VectorRetriever(FakeEmbedder("new-model"), 0.5).search(db, "cosmetic surgery").chunks == []
        assert VectorRetriever(FakeEmbedder("old-model"), 0.5).search(db, "cosmetic surgery").chunks != []


def test_hybrid_finds_a_synonym_match_that_keywords_alone_rank_poorly(client):
    upload_all(client, FakeEmbedder("fake-embed"))
    with client.session_factory() as db:
        hybrid = HybridRetriever(LexicalRetriever(), VectorRetriever(FakeEmbedder("fake-embed"), 0.5))
        top = hybrid.search(db, "Does the policy cover cosmetic surgery?", None, 3).chunks[0]
        assert top.document_name == "health_insurance_policy.pdf" and top.page_number == 2
        assert "procedures" in top.text and "surgery" not in top.text   # matched by meaning, not by the word


@pytest.fixture()
def ollama_app(client):
    """Whole pipeline in ollama mode, with a scripted fake standing in for Ollama's chat endpoint."""
    embedder = FakeEmbedder("fake-embed")
    ids = upload_all(client, embedder)

    def install(reply):
        chat = FakeOllamaChat(reply)
        provider = OllamaProvider(chat.client(), "llama3.1:latest")
        app.dependency_overrides[get_provider] = lambda: provider
        app.dependency_overrides[get_retriever] = lambda: HybridRetriever(LexicalRetriever(), VectorRetriever(embedder, 0.5))
        return chat

    client.install = install
    client.ids = ids
    return client


def ask(client, q, **extra):
    r = client.post("/ask", json={"question": q, **extra})
    assert r.status_code == 200, r.text
    return r.json()


COSMETIC = "Does the policy cover cosmetic surgery?"
QUOTE = "Cosmetic procedures, self-inflicted injury and experimental treatments are excluded."


def test_end_to_end_answer_with_verified_citation(ollama_app):
    import json

    def reply(user):
        return json.dumps({"supported": True, "answer": "No. Cosmetic procedures are excluded.",
                           "quotes": [{"passage": passage_number(user, "Cosmetic procedures"), "text": QUOTE}]})

    chat = ollama_app.install(reply)
    body = ask(ollama_app, COSMETIC)
    assert body["answered"] is True and "excluded" in body["answer"]
    assert body["generated_by"].startswith("ollama/llama3.1")
    cite = body["citations"][0]
    assert (cite["document_name"], cite["page_number"], cite["quote"]) == ("health_insurance_policy.pdf", 2, QUOTE)
    assert len(chat.requests) == 1


def test_fabricated_quote_is_rejected_end_to_end(ollama_app):
    import json
    ollama_app.install(lambda user: json.dumps({"supported": True, "answer": "Yes, up to 500.",
                                                "quotes": [{"passage": 1, "text": "Cosmetic surgery is covered up to 500"}]}))
    body = ask(ollama_app, COSMETIC)
    assert body["answered"] is False and "do not specify" in body["answer"] and body["citations"] == []


def test_model_that_refuses_gives_not_found(ollama_app):
    import json
    ollama_app.install(lambda user: json.dumps({"supported": False, "answer": "", "quotes": []}))
    assert ask(ollama_app, "What is the support phone number for the fictional retailer?")["answered"] is False


def test_unrelated_question_never_reaches_the_llm(ollama_app):
    chat = ollama_app.install(lambda user: "{}")
    assert ask(ollama_app, "What is the capital of France?")["answered"] is False
    assert chat.requests == []


def test_ollama_down_returns_503_with_helpful_message(ollama_app):
    import httpx
    from app.ollama import OllamaClient

    def down(request):
        raise httpx.ConnectError("refused")

    ollama_app.install(lambda user: "{}")
    app.dependency_overrides[get_provider] = lambda: OllamaProvider(OllamaClient("http://x", 1, httpx.MockTransport(down)), "m")
    r = ollama_app.post("/ask", json={"question": COSMETIC})
    assert r.status_code == 503 and "Cannot reach Ollama" in r.json()["detail"]


def test_search_reports_hybrid_method(ollama_app):
    ollama_app.install(lambda user: "{}")
    body = ollama_app.post("/search", json={"question": COSMETIC, "top_k": 3}).json()
    assert body["retrieval_method"].startswith("hybrid")
    assert body["passages"][0]["document_name"] == "health_insurance_policy.pdf"
