from pathlib import Path

import pytest

from app.models import Document, ProcessingJob
from datetime import datetime, timedelta, timezone

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"


def upload(client, name, content=None):
    data = content if content is not None else (POLICIES / name).read_bytes()
    return client.post("/documents/upload", files={"file": (name, data, "application/pdf")}).json()["id"]


def test_empty_system_reports_zeros_not_errors(client):
    docs = client.get("/metrics/documents").json()
    assert docs["total_documents"] == 0 and docs["avg_chunks_per_document"] == 0.0
    assert docs["by_status"] == {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
    proc = client.get("/metrics/processing").json()
    assert proc["total_jobs"] == 0 and proc["success_rate"] is None and proc["p95_duration_ms"] is None
    system = client.get("/metrics/system").json()
    assert system["answers_total"] == 0 and system["answer_rate"] is None


def test_document_statistics(client):
    ids = [upload(client, "health_insurance_policy.pdf"), upload(client, "employee_leave_policy.pdf")]
    pending_only = client.get("/metrics/documents").json()
    assert pending_only["by_status"]["pending"] == 2 and pending_only["total_chunks"] == 0
    for doc_id in ids:
        client.post(f"/documents/{doc_id}/process")
    client.post(f"/documents/{ids[0]}/insights")
    body = client.get("/metrics/documents").json()
    assert body["total_documents"] == 2 and body["by_status"]["completed"] == 2
    assert body["by_type"] == {"pdf": 2}
    assert body["total_chunks"] >= 2 and body["total_pages"] >= 2
    assert body["avg_chunks_per_document"] == pytest.approx(body["total_chunks"] / 2, abs=0.01)
    assert body["documents_with_insights"] == 1 and body["uploaded_last_24h"] == 2
    assert body["total_size_bytes"] > 0 and body["chunks_with_embeddings"] == 0   # mock mode embeds nothing


def test_every_processing_run_is_recorded_with_duration_and_chunks(client):
    doc_id = upload(client, "health_insurance_policy.pdf")
    client.post(f"/documents/{doc_id}/process")
    client.post(f"/documents/{doc_id}/process")        # a retry is a second job, not an overwrite
    body = client.get("/metrics/processing").json()
    assert body["total_jobs"] == 2 and body["by_status"]["succeeded"] == 2
    assert body["success_rate"] == 1.0 and body["avg_duration_ms"] is not None and body["p95_duration_ms"] is not None
    job = body["recent_jobs"][0]
    assert job["status"] == "succeeded" and job["chunk_count"] >= 1 and job["document_id"] == doc_id


def test_failures_are_counted_with_a_user_safe_reason(client):
    good = upload(client, "health_insurance_policy.pdf")
    broken = upload(client, "broken.pdf", b"%PDF-1.4 this is not really a pdf")
    client.post(f"/documents/{good}/process")
    assert client.post(f"/documents/{broken}/process").json()["status"] == "failed"
    body = client.get("/metrics/processing").json()
    assert body["by_status"]["succeeded"] == 1 and body["by_status"]["failed"] == 1
    assert body["success_rate"] == 0.5
    reason = body["top_failure_reasons"][0]
    assert reason["count"] == 1 and "Traceback" not in reason["reason"] and "/" not in reason["reason"].split("(")[0]


def test_stuck_documents_are_detected(client):
    doc_id = upload(client, "health_insurance_policy.pdf")
    with client.session_factory() as db:
        doc = db.get(Document, doc_id if not isinstance(doc_id, str) else __import__("uuid").UUID(doc_id))
        doc.status = "processing"
        doc.updated_at = datetime.now(timezone.utc) - timedelta(minutes=60)
        db.commit()
    body = client.get("/metrics/processing").json()
    assert body["stuck_documents"] == 1 and body["stuck_threshold_minutes"] == 15


def test_answer_metrics_track_refusals_latency_and_generator(client):
    for name in ("employee_leave_policy.pdf", "health_insurance_policy.pdf"):
        client.post(f"/documents/{upload(client, name)}/process")
    chat = client.post("/chats", json={}).json()["id"]
    client.post(f"/chats/{chat}/messages", json={"question": "How many annual leave days are provided per year?"})
    client.post(f"/chats/{chat}/messages", json={"question": "What is the capital of France?"})
    body = client.get("/metrics/system").json()
    assert body["chat_sessions"] == 1 and body["chat_messages"] == 4
    assert (body["answers_total"], body["answers_answered"], body["answers_refused"]) == (2, 1, 1)
    assert body["answer_rate"] == 0.5 and body["avg_answer_latency_ms"] is not None
    assert body["ai_mode"] == "mock" and sum(body["answers_by_generator"].values()) == 2
    assert client.get(f"/chats/{chat}/history").json()["messages"][1]["latency_ms"] is not None
