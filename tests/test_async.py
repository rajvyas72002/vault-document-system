import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.config import Settings, get_settings
from app.deps import get_dispatcher, get_session_factory
from app.main import app
from app.models import Document, DocumentChunk, DocumentInsight, ProcessingJob
from app.tasks import CeleryDispatcher, QueueUnavailable, run_processing_job

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"


def make_settings(tmp_path, **extra):
    return Settings(storage_dir=str(tmp_path / "storage"), max_upload_mb=1, ai_mode="mock", **extra)


@pytest.fixture()
def background_client(client, tmp_path):
    """Same app, but processing is queued and run by a background task."""
    settings = make_settings(tmp_path, task_backend="background")
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session_factory] = lambda: client.session_factory
    return client


def upload(client, name="health_insurance_policy.pdf"):
    return client.post("/documents/upload", files={"file": (name, (POLICIES / name).read_bytes(), "application/pdf")}).json()["id"]


class RecordingDispatcher:
    """Stands in for Redis: remembers job ids but does not run them."""
    def __init__(self):
        self.job_ids = []

    def enqueue(self, job_id):
        self.job_ids.append(job_id)


class DownDispatcher:
    def enqueue(self, job_id):
        raise QueueUnavailable("Error 111 connecting to localhost:6379. Connection refused.")


# ----------------------------------------------------------------------------- background mode, end to end

def test_process_returns_202_immediately_and_the_job_completes(background_client):
    client = background_client
    doc_id = upload(client)
    r = client.post(f"/documents/{doc_id}/process")
    assert r.status_code == 202
    accepted = r.json()
    assert accepted["status"] == "queued" and accepted["status_url"] == f"/jobs/{accepted['job_id']}"

    job = client.get(accepted["status_url"]).json()      # TestClient has already run the background task
    assert job["status"] == "succeeded" and job["chunk_count"] >= 1 and job["duration_ms"] is not None
    assert client.get(f"/documents/{doc_id}").json()["status"] == "completed"
    assert client.get(f"/documents/{doc_id}/chunks").json()["total"] == job["chunk_count"]
    assert [j["id"] for j in client.get(f"/documents/{doc_id}/jobs").json()] == [accepted["job_id"]]
    assert client.get("/metrics/processing").json()["total_jobs"] == 1


def test_unknown_job_and_document_are_404(background_client):
    missing = str(uuid.uuid4())
    assert background_client.get(f"/jobs/{missing}").status_code == 404
    assert background_client.post(f"/documents/{missing}/process").status_code == 404
    assert background_client.get(f"/documents/{missing}/jobs").status_code == 404


# ----------------------------------------------------------------------------- queue behaviour with a fake queue

@pytest.fixture()
def queued_client(client, tmp_path):
    settings = make_settings(tmp_path, task_backend="celery")
    queue = RecordingDispatcher()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_dispatcher] = lambda: queue
    client.queue, client.settings = queue, settings
    return client


def test_job_waits_in_the_queue_until_a_worker_runs_it(queued_client):
    client = queued_client
    doc_id = upload(client)
    job_id = client.post(f"/documents/{doc_id}/process").json()["job_id"]
    assert client.queue.job_ids == [uuid.UUID(job_id)]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "queued"
    assert client.get(f"/documents/{doc_id}").json()["status"] == "pending"      # nothing has run yet

    run_processing_job(uuid.UUID(job_id), client.session_factory, client.settings)   # what the worker does
    assert client.get(f"/jobs/{job_id}").json()["status"] == "succeeded"
    assert client.get(f"/documents/{doc_id}").json()["status"] == "completed"


def test_a_redelivered_task_does_not_repeat_the_work(queued_client):
    client = queued_client
    doc_id = upload(client)
    job_id = uuid.UUID(client.post(f"/documents/{doc_id}/process").json()["job_id"])
    run_processing_job(job_id, client.session_factory, client.settings)
    after_first = client.get(f"/jobs/{job_id}").json()
    run_processing_job(job_id, client.session_factory, client.settings)           # at-least-once delivery
    assert client.get(f"/jobs/{job_id}").json() == after_first                    # finished_at / duration untouched: it was skipped
    with client.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(ProcessingJob)) == 1
    assert client.get(f"/documents/{doc_id}/chunks").json()["total"] == after_first["chunk_count"]


def test_second_request_while_a_job_is_active_gets_409(queued_client):
    client = queued_client
    doc_id = upload(client)
    first = client.post(f"/documents/{doc_id}/process")
    second = client.post(f"/documents/{doc_id}/process")
    assert (first.status_code, second.status_code) == (202, 409)
    assert first.json()["job_id"] in second.json()["detail"]
    assert len(client.queue.job_ids) == 1


def test_a_stuck_job_does_not_block_a_new_one_forever(queued_client):
    client = queued_client
    doc_id = upload(client)
    old = client.post(f"/documents/{doc_id}/process").json()["job_id"]
    with client.session_factory() as db:                      # pretend the worker died an hour ago
        db.get(ProcessingJob, uuid.UUID(old)).created_at = datetime.now(timezone.utc) - timedelta(hours=1)
        db.commit()
    assert client.post(f"/documents/{doc_id}/process").status_code == 202


def test_unreachable_queue_returns_503_and_leaves_no_zombie_job(client, tmp_path):
    app.dependency_overrides[get_settings] = lambda: make_settings(tmp_path, task_backend="celery")
    app.dependency_overrides[get_dispatcher] = lambda: DownDispatcher()
    doc_id = upload(client)
    r = client.post(f"/documents/{doc_id}/process")
    assert r.status_code == 503 and "Redis" in r.json()["detail"]
    job = client.get(f"/documents/{doc_id}/jobs").json()[0]
    assert job["status"] == "failed" and job["error_message"] == "Task queue unavailable"
    app.dependency_overrides[get_dispatcher] = lambda: RecordingDispatcher()      # Redis is back
    assert client.post(f"/documents/{doc_id}/process").status_code == 202        # not blocked by the failed job


def test_celery_dispatcher_turns_any_broker_error_into_queue_unavailable(monkeypatch):
    pytest.importorskip("celery")
    from app import worker
    def boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(worker.process_document_task, "apply_async", boom)       # never touches a real Redis
    with pytest.raises(QueueUnavailable):
        CeleryDispatcher().enqueue(uuid.uuid4())


# ----------------------------------------------------------------------------- worker details

def test_job_for_a_deleted_document_is_handled_without_crashing(client, tmp_path):
    doc_id = upload(client)
    with client.session_factory() as db:
        job = ProcessingJob(document_id=uuid.UUID(doc_id), status="queued")
        db.add(job)
        db.commit()
        job_id = job.id
        db.delete(db.get(Document, uuid.UUID(doc_id)))      # deleted while the job waited (cascade removes the job too)
        db.commit()
    run_processing_job(job_id, client.session_factory, make_settings(tmp_path))       # must not raise


def test_celery_task_runs_the_same_job_function(client, tmp_path, monkeypatch):
    pytest.importorskip("celery")
    from app import tasks, worker
    monkeypatch.setattr(tasks, "SessionLocal", client.session_factory)
    monkeypatch.setattr(tasks, "get_settings", lambda: make_settings(tmp_path))
    doc_id = upload(client)
    with client.session_factory() as db:
        job = ProcessingJob(document_id=uuid.UUID(doc_id), status="queued")
        db.add(job)
        db.commit()
        job_id = job.id
    worker.process_document_task.apply(args=[str(job_id)]).get()                      # runs in-process, no broker
    assert client.get(f"/jobs/{job_id}").json()["status"] == "succeeded"
    assert worker.process_document_task.name == "vault.process_document"
    assert worker.celery_app.conf.task_acks_late is True


def test_auto_insights_are_generated_after_processing(queued_client):
    client = queued_client
    settings = make_settings(Path(client.settings.storage_dir).parent, auto_insights=True)
    doc_id = upload(client)
    job_id = uuid.UUID(client.post(f"/documents/{doc_id}/process").json()["job_id"])
    run_processing_job(job_id, client.session_factory, settings)
    insights = client.get(f"/documents/{doc_id}/insights")
    assert insights.status_code == 200 and insights.json()["category"] == "insurance"


def test_failing_insights_do_not_fail_the_processing_job(queued_client, monkeypatch):
    from app import tasks
    client = queued_client
    class Broken:
        def generate(self, *a, **k):
            raise RuntimeError("model exploded")
    monkeypatch.setattr(tasks, "get_insight_generator_for", lambda s: Broken())
    settings = make_settings(Path(client.settings.storage_dir).parent, auto_insights=True)
    doc_id = upload(client)
    job_id = uuid.UUID(client.post(f"/documents/{doc_id}/process").json()["job_id"])
    run_processing_job(job_id, client.session_factory, settings)
    assert client.get(f"/jobs/{job_id}").json()["status"] == "succeeded"
    assert client.get(f"/documents/{doc_id}").json()["status"] == "completed"
    assert client.get(f"/documents/{doc_id}/insights").status_code == 404
