from pathlib import Path

import pytest

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"
NOT_FOUND = "do not specify"


@pytest.fixture()
def loaded(client):
    """Upload and process all five fictional policies. Returns {filename: document_id}."""
    ids = {}
    for f in sorted(POLICIES.glob("*.pdf")):
        r = client.post("/documents/upload", files={"file": (f.name, f.read_bytes(), "application/pdf")})
        ids[f.name] = r.json()["id"]
        assert client.post(f"/documents/{ids[f.name]}/process").json()["status"] == "completed"
    return ids


def ask(client, question, **extra):
    r = client.post("/ask", json={"question": question, **extra})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize(
    "question, expected_text, expected_doc, expected_page",
    [
        ("What is the initial waiting period for non-accidental illness?", "30 days", "health_insurance_policy.pdf", 1),
        ("How soon should an accident be reported?", "48 hours", "vehicle_insurance_policy.pdf", 1),
        ("What is the return window for eligible items?", "14 days", "refund_cancellation_policy.pdf", 1),
        ("How many annual leave days are provided per year?", "24 days", "employee_leave_policy.pdf", 1),
        ("How often should access to sensitive shared folders be reviewed?", "quarter", "information_security_policy.pdf", 1),
    ],
)
def test_answers_present_in_documents_with_correct_citation(loaded, client, question, expected_text, expected_doc, expected_page):
    body = ask(client, question)
    assert body["answered"] is True
    assert expected_text in body["answer"]
    cite = body["citations"][0]
    assert (cite["document_name"], cite["page_number"]) == (expected_doc, expected_page)
    assert expected_text in cite["quote"]


@pytest.mark.parametrize(
    "question",
    [
        "What is the support phone number for the fictional retailer?",
        "What is the exact retention period for Restricted records?",
        "What is the capital of France?",
    ],
)
def test_missing_information_is_declared_not_invented(loaded, client, question):
    body = ask(client, question)
    assert body["answered"] is False
    assert NOT_FOUND in body["answer"]
    assert body["citations"] == []


def test_comparison_cites_both_insurance_policies(loaded, client):
    body = ask(client, "Compare claim acknowledgement targets in the two insurance policies.")
    assert body["answered"] is True
    assert "3 business days" in body["answer"] and "2 business days" in body["answer"]
    assert {c["document_name"] for c in body["citations"]} == {"health_insurance_policy.pdf", "vehicle_insurance_policy.pdf"}


def test_scoping_to_one_document_prevents_answers_from_others(loaded, client):
    q = "What is the initial waiting period for non-accidental illness?"
    assert ask(client, q, document_ids=[loaded["health_insurance_policy.pdf"]])["answered"] is True
    assert ask(client, q, document_ids=[loaded["vehicle_insurance_policy.pdf"]])["answered"] is False


def test_response_is_honest_about_mock_mode(loaded, client):
    body = ask(client, "How many annual leave days are provided per year?")
    assert "no LLM" in body["generated_by"] and "NOT written by an LLM" in body["note"]


def test_search_endpoint_ranks_relevant_document_first(loaded, client):
    r = client.post("/search", json={"question": "annual leave days", "top_k": 3})
    body = r.json()
    assert "not semantic" in body["retrieval_method"]
    assert body["passages"][0]["document_name"] == "employee_leave_policy.pdf"
    scores = [p["score"] for p in body["passages"]]
    assert scores == sorted(scores, reverse=True)


def test_ask_validation_and_errors(loaded, client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422             # too short
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422       # too long
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.post("/ask", json={"question": "anything?", "document_ids": [missing]}).status_code == 404


def test_unprocessed_document_cannot_be_queried(client):
    r = client.post("/documents/upload", files={"file": ("raw.pdf", b"%PDF-1.4 raw", "application/pdf")})
    doc_id = r.json()["id"]
    assert client.post("/ask", json={"question": "anything?", "document_ids": [doc_id]}).status_code == 409


def test_with_no_processed_documents_answer_is_not_found(client):
    assert ask(client, "What is the waiting period?")["answered"] is False


@pytest.mark.xfail(strict=True, reason="Known limitation: keyword search cannot link 'surgery' to 'procedures'. "
                                       "Needs embeddings / semantic retrieval (a real LLM provider would also help).")
def test_synonym_question_is_a_known_lexical_limitation(loaded, client):
    body = ask(client, "Does the policy cover cosmetic surgery?")
    assert body["answered"] is True and "excluded" in body["answer"]
