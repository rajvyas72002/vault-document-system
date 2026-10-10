"""The background job: run one queued processing job. Used by the Celery worker and by the
in-process 'background' mode, so both behave identically.

Why this is safe to run twice (a queue may deliver a task again after a crash):
  * a job that already finished is skipped
  * process_document deletes the old chunks before writing new ones"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol

from fastapi import BackgroundTasks

from app.config import Settings, get_settings
from app.db import SessionLocal
from app.deps import get_embedder
from app.insight_service import store_insights
from app.insights import InsightOptions, get_insight_generator_for
from app.models import ProcessingJob
from app.processing import process_document

log = logging.getLogger(__name__)


class QueueUnavailable(Exception):
    pass


class Dispatcher(Protocol):
    def enqueue(self, job_id: uuid.UUID) -> None: ...


def run_processing_job(job_id: uuid.UUID, session_factory: Optional[Callable] = None,
                       settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    factory = session_factory or SessionLocal
    embedder = get_embedder(settings)
    with factory() as db:
        job = db.get(ProcessingJob, job_id)
        if job is None:
            log.error("Job %s not found", job_id)
            return
        if job.status in ("succeeded", "failed"):
            log.info("Job %s already finished (%s); skipping", job_id, job.status)
            return
        try:
            doc = process_document(db, job.document_id, settings, embedder, job_id=job.id)
        except LookupError:
            job.status, job.finished_at = "failed", datetime.now(timezone.utc)
            job.error_message = "Document no longer exists."
            db.commit()
            return
        if settings.auto_insights and doc.status == "completed":
            try:
                store_insights(db, doc, get_insight_generator_for(settings), InsightOptions())
            except Exception:   # insights are a bonus: never fail the processing job because of them
                db.rollback()
                log.exception("Automatic insights failed for document %s", doc.id)


class BackgroundDispatcher:
    """Runs the job after the HTTP response is sent, in the same server process. No Redis needed.
    Good for demos; a crash of the server loses queued work, which is why Celery exists."""

    def __init__(self, background: BackgroundTasks, session_factory: Callable, settings: Settings):
        self.background, self.session_factory, self.settings = background, session_factory, settings

    def enqueue(self, job_id: uuid.UUID) -> None:
        self.background.add_task(run_processing_job, job_id, self.session_factory, self.settings)


class CeleryDispatcher:
    """Sends the job id to Redis; a separate worker process picks it up."""

    def enqueue(self, job_id: uuid.UUID) -> None:
        try:
            from app.worker import process_document_task
            process_document_task.apply_async(args=[str(job_id)], retry=False)
        except Exception as exc:   # Redis down, broker misconfigured, celery not installed ...
            raise QueueUnavailable(str(exc)) from exc
