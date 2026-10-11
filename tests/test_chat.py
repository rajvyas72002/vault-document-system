import json
from pathlib import Path

import pytest

from app.conversation import MAX_HISTORY_MESSAGES, LangChainRewriter, OllamaRewriter, build_retrieval_query, clean_rewrite, is_follow_up, to_history
from app.deps import get_embedder, get_provider, get_retriever, get_rewriter
from app.errors import AIServiceError
from app.main import app
from app.providers import ChatTurn, OllamaProvider
from app.retrieval import HybridRetriever, LexicalRetriever, VectorRetriever
from tests.fakes import FakeEmbedder, FakeOllamaChat, passage_number

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"
HEALTH_Q = "What is the claim acknowledgement target in the health insurance policy?"


def upload_all(client):
    ids = {}
    for f in sorted(POLICIES.glob("*.pdf")):
        r = client.post("/documents/upload", files={"file": (f.name, f.read_bytes(), "application/pdf")})
        ids[f.name] = r.json()["id"]
        assert client.post(f"/documents/{ids[f.name]}/process").json()["status"] == "completed"
    return ids


@pytest.fixture()
def loaded(client):
    return upload_all(client)


def new_chat(client, **body):
    r = client.post("/chats", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def say(client, chat_id, question):
    r = client.post(f"/chats/{chat_id}/messages", json={"question": question})
    assert r.status_code == 200, r.text
    return r.json()


# ----------------------------------------------------------------------------- lifecycle (mock mode)

def test_start_chat_and_history_starts_empty(client, loaded):
    chat = client.post("/chats", json={}).json()
    assert chat["title"] is None and chat["document_ids"] is None
    history = client.get(f"/chats/{chat['id']}/history").json()
    assert history["messages"] == []


def test_question_is_answered_and_stored_with_citations(client, loaded):
    chat_id = new_chat(client)
    body = say(client, chat_id, "How many annual leave days are provided per year?")
    assert body["answered"] is True and "24 days" in body["answer"]

    history = client.get(f"/chats/{chat_id}/history").json()
    user, assistant = history["messages"]
    assert (user["seq"], user["role"]) == (1, "user") and user["citations"] == []
    assert (assistant["seq"], assistant["role"], assistant["answered"]) == (2, "assistant", True)
    assert assistant["citations"][0]["document_name"] == "employee_leave_policy.pdf"
    assert assistant["citations"][0]["page_number"] == 1
    assert assistant["id"] == body["message_id"]
    assert history["session"]["title"].startswith("How many annual leave days")   # auto title from first question


def test_history_keeps_order_across_turns(client, loaded):
    chat_id = new_chat(client)
    say(client, chat_id, "How many annual leave days are provided per year?")
    say(client, chat_id, "How soon should an accident be reported?")
    messages = client.get(f"/chats/{chat_id}/history").json()["messages"]
    assert [m["seq"] for m in messages] == [1, 2, 3, 4]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert "48 hours" in messages[3]["content"]


def test_refusal_is_stored_without_citations(client, loaded):
    chat_id = new_chat(client)
    body = say(client, chat_id, "What is the capital of France?")
    assert body["answered"] is False
    assistant = client.get(f"/chats/{chat_id}/history").json()["messages"][1]
    assert assistant["answered"] is False and assistant["citations"] == []


def test_chat_scope_limits_which_documents_are_searched(client, loaded):
    vehicle_only = new_chat(client, document_ids=[loaded["vehicle_insurance_policy.pdf"]])
    assert say(client, vehicle_only, "What is the initial waiting period for non-accidental illness?")["answered"] is False
    assert say(client, vehicle_only, "How soon should an accident be reported?")["answered"] is True


def test_unknown_chat_and_bad_scope_are_clear_errors(client, loaded):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/chats/{missing}/history").status_code == 404
    assert client.post(f"/chats/{missing}/messages", json={"question": "hello there"}).status_code == 404
    assert client.post("/chats", json={"document_ids": [missing]}).status_code == 404
    chat_id = new_chat(client)
    assert client.post(f"/chats/{chat_id}/messages", json={"question": "hi"}).status_code == 422   # too short


def test_list_chats_newest_activity_first(client, loaded):
    first, second = new_chat(client, title="first"), new_chat(client, title="second")
    say(client, first, "How many annual leave days are provided per year?")   # touches `first`
    ids = [c["id"] for c in client.get("/chats").json()]
    assert ids[0] == first and set(ids) == {first, second}


def test_failed_turn_saves_nothing(client, loaded):
    class DownProvider:
        def generate(self, *a, **k):
            raise AIServiceError("Cannot reach Ollama")

    app.dependency_overrides[get_provider] = lambda: DownProvider()
    chat_id = new_chat(client)
    r = client.post(f"/chats/{chat_id}/messages", json={"question": "How many annual leave days are provided per year?"})
    assert r.status_code == 503
    assert client.get(f"/chats/{chat_id}/history").json()["messages"] == []


# ----------------------------------------------------------------------------- follow-up logic (pure functions)

@pytest.mark.parametrize("question, expected", [
    ("What about the vehicle policy?", True),
    ("And for the other one?", True),
    ("Does it apply to dependants?", True),
    ("Why?", True),
    ("What is the initial waiting period for non-accidental illness?", False),
    ("How soon should an accident be reported?", False),
])
def test_is_follow_up(question, expected):
    assert is_follow_up(question) is expected


def test_retrieval_query_adds_previous_question_only_for_follow_ups():
    history = [ChatTurn("user", HEALTH_Q), ChatTurn("assistant", "3 business days.")]
    assert build_retrieval_query("What about the vehicle policy?", history) == f"{HEALTH_Q} What about the vehicle policy?"
    standalone = "How many annual leave days are provided per year?"
    assert build_retrieval_query(standalone, history) == standalone
    assert build_retrieval_query("What about the vehicle policy?", []) == "What about the vehicle policy?"


def test_history_is_trimmed_and_truncated():
    class M:
        def __init__(self, i):
            self.role, self.content = ("user" if i % 2 == 0 else "assistant"), "x" * 2000
    turns = to_history([M(i) for i in range(20)])
    assert len(turns) == MAX_HISTORY_MESSAGES and all(len(t.content) <= 500 for t in turns)


# ----------------------------------------------------------------------------- multi-turn with a scripted LLM

@pytest.fixture()
def llm_chat(client):
    embedder = FakeEmbedder("fake-embed")
    app.dependency_overrides[get_embedder] = lambda: embedder
    ids = upload_all(client)

    def install(reply):
        chat = FakeOllamaChat(reply)
        provider = OllamaProvider(chat.client(), "llama3.1:latest")
        app.dependency_overrides[get_provider] = lambda: provider
        app.dependency_overrides[get_retriever] = lambda: HybridRetriever(LexicalRetriever(), VectorRetriever(embedder, 0.5))
        return chat

    client.install, client.ids = install, ids
    return client


def test_follow_up_sees_history_and_is_searched_with_earlier_context(llm_chat):
    client = llm_chat
    three = "within 3 business days of receiving it"
    two = "within 2 business days of receiving it"

    def reply(user):
        wanted, doc = (two, "vehicle") if "vehicle" in user.split("Question:")[-1].lower() else (three, "health")
        quote = f"Claim acknowledgement target: the insurer will acknowledge every claim {wanted}."
        return json.dumps({"supported": True, "answer": f"The {doc} policy acknowledges claims {wanted}.",
                           "quotes": [{"passage": passage_number(user, wanted), "text": quote}]})

    llm = client.install(reply)
    chat_id = new_chat(client)
    first = say(client, chat_id, HEALTH_Q)
    assert "3 business days" in first["answer"] and first["searched_for"] == HEALTH_Q

    second = say(client, chat_id, "What about the vehicle policy?")
    assert second["searched_for"] == f"{HEALTH_Q} What about the vehicle policy?"
    assert second["answered"] is True and "2 business days" in second["answer"]
    assert second["citations"][0]["document_name"] == "vehicle_insurance_policy.pdf"

    sent = llm.requests[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]   # earlier turn travelled with the question
    assert sent[1]["content"] == HEALTH_Q
    assert sent[-1]["content"].endswith("Question: What about the vehicle policy?")


def test_assistant_history_does_not_let_a_fabricated_quote_through(llm_chat):
    """History changes what the model sees, never what we accept: quotes are still verified."""
    client = llm_chat
    client.install(lambda user: json.dumps({"supported": True, "answer": "It is 99 days.",
                                            "quotes": [{"passage": 1, "text": "waiting period is 99 days"}]}))
    chat_id = new_chat(client)
    body = say(client, chat_id, "What is the initial waiting period for non-accidental illness?")
    assert body["answered"] is False and body["citations"] == []


# ----------------------------------------------------------------------------- LLM question rewriting

VEHICLE_Q = "What is the claim acknowledgement target in the vehicle insurance policy?"


@pytest.mark.parametrize("raw, expected", [
    (json.dumps({"standalone_question": "  What is the   waiting period? "}), "What is the waiting period?"),
    ("not json", None),
    (json.dumps(["list"]), None),
    (json.dumps({"standalone_question": 42}), None),
    (json.dumps({"standalone_question": ""}), None),
    (json.dumps({"standalone_question": "x" * 301}), None),
])
def test_clean_rewrite_accepts_only_sensible_strings(raw, expected):
    assert clean_rewrite(raw) == expected


def install_rewriter(client, llm_reply, rewrite_reply):
    """One scripted fake plays both the answering model and the rewriting model."""
    embedder = FakeEmbedder("fake-embed")

    def reply(user):
        return rewrite_reply(user) if "Latest question:" in user else llm_reply(user)

    chat = FakeOllamaChat(reply)
    provider = OllamaProvider(chat.client(), "llama3.1:latest")
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_rewriter] = lambda: OllamaRewriter(chat.client(), "llama3.1:latest")
    app.dependency_overrides[get_retriever] = lambda: HybridRetriever(LexicalRetriever(), VectorRetriever(embedder, 0.5))
    return chat


def answer_from(user):
    wanted, doc = ("within 2 business days of receiving it", "vehicle") if "vehicle" in user.split("Question:")[-1].lower() \
        else ("within 3 business days of receiving it", "health")
    quote = f"Claim acknowledgement target: the insurer will acknowledge every claim {wanted}."
    return json.dumps({"supported": True, "answer": f"The {doc} policy acknowledges claims {wanted}.",
                       "quotes": [{"passage": passage_number(user, wanted), "text": quote}]})


def test_rewritten_follow_up_is_searched_and_answered_as_a_standalone_question(client):
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder("fake-embed")
    upload_all(client)
    llm = install_rewriter(client, answer_from, lambda user: json.dumps({"standalone_question": VEHICLE_Q}))
    chat_id = new_chat(client)
    say(client, chat_id, HEALTH_Q)
    body = say(client, chat_id, "What about the vehicle policy?")

    assert body["searched_for"] == VEHICLE_Q                      # not "health ... What about the vehicle"
    assert body["citations"][0]["document_name"] == "vehicle_insurance_policy.pdf"
    assert "2 business days" in body["answer"]
    answering_call = llm.requests[-1]["messages"]
    assert [m["role"] for m in answering_call] == ["system", "user"]   # standalone question: no history needed
    assert answering_call[-1]["content"].endswith(f"Question: {VEHICLE_Q}")
    stored = client.get(f"/chats/{chat_id}/history").json()["messages"]
    assert stored[2]["content"] == "What about the vehicle policy?"    # the user's own words are what we store


def test_bad_rewrite_falls_back_to_the_keyword_heuristic(client):
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder("fake-embed")
    upload_all(client)
    install_rewriter(client, answer_from, lambda user: "this is not json")
    chat_id = new_chat(client)
    say(client, chat_id, HEALTH_Q)
    body = say(client, chat_id, "What about the vehicle policy?")
    assert body["searched_for"] == f"{HEALTH_Q} What about the vehicle policy?"


def test_standalone_questions_are_never_rewritten(client):
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder("fake-embed")
    upload_all(client)
    llm = install_rewriter(client, answer_from, lambda user: json.dumps({"standalone_question": "SHOULD NOT BE USED"}))
    chat_id = new_chat(client)
    say(client, chat_id, HEALTH_Q)
    body = say(client, chat_id, VEHICLE_Q)
    assert body["searched_for"] == VEHICLE_Q
    assert not any("Latest question:" in m["content"] for call in llm.requests for m in call["messages"])


# ----------------------------------------------------------------------------- hosted (OpenAI) rewriter

class FakeHostedLLM:
    """Stands in for a LangChain chat model: records the messages and returns a scripted reply."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return type("Reply", (), {"content": self.reply})()


HISTORY = [ChatTurn("user", HEALTH_Q), ChatTurn("assistant", "3 business days.")]


def test_hosted_rewriter_returns_the_standalone_question():
    llm = FakeHostedLLM(json.dumps({"standalone_question": VEHICLE_Q}))
    assert LangChainRewriter(llm).rewrite("What about the vehicle policy?", HISTORY) == VEHICLE_Q
    system, human = llm.calls[0]
    assert "standalone question" in system.content and "JSON" in system.content    # JSON mode needs the word JSON in the prompt
    assert HEALTH_Q in human.content and human.content.endswith("Latest question: What about the vehicle policy?")


@pytest.mark.parametrize("reply", ["this is not json", json.dumps({"standalone_question": ""}), RuntimeError("secret sk-123 leaked")])
def test_hosted_rewriter_falls_back_to_none_on_bad_output_or_errors(reply):
    assert LangChainRewriter(FakeHostedLLM(reply)).rewrite("What about the vehicle policy?", HISTORY) is None


def test_follow_up_is_rewritten_by_the_hosted_model(client):
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder("fake-embed")
    upload_all(client)
    chat = FakeOllamaChat(answer_from)                       # answering model
    provider = OllamaProvider(chat.client(), "llama3.1:latest")
    app.dependency_overrides[get_provider] = lambda: provider
    hosted = FakeHostedLLM(json.dumps({"standalone_question": VEHICLE_Q}))
    app.dependency_overrides[get_rewriter] = lambda: LangChainRewriter(hosted)
    app.dependency_overrides[get_retriever] = lambda: HybridRetriever(LexicalRetriever(), VectorRetriever(FakeEmbedder("fake-embed"), 0.5))
    chat_id = new_chat(client)
    say(client, chat_id, HEALTH_Q)
    body = say(client, chat_id, "What about the vehicle policy?")
    assert body["searched_for"] == VEHICLE_Q and "2 business days" in body["answer"]
    assert len(hosted.calls) == 1                            # the first question has no history, so it is not rewritten


def test_rewriter_follows_the_chat_provider():
    from app.config import Settings
    from app.deps import get_rewriter as factory
    hosted = factory(Settings(ai_mode="real", chat_provider="openai", openai_api_key="test-key"))
    local = factory(Settings(ai_mode="real", chat_provider="ollama"))
    assert isinstance(hosted, LangChainRewriter) and isinstance(local, OllamaRewriter)
    assert factory(Settings(ai_mode="mock")) is None
