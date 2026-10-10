"""Celery worker. Start it with (Windows needs --pool=solo):

    celery -A app.worker worker --pool=solo --loglevel=info

Only job ids travel through Redis. All state (status, errors, timings) lives in PostgreSQL, so the
queue can be lost without losing history, and a stuck job is visible in GET /metrics/processing."""
import uuid

from celery import Celery

from app.config import get_settings
from app.tasks import run_processing_job

settings = get_settings()

celery_app = Celery("vault", broker=settings.redis_url)   # no result backend: the database is the source of truth
celery_app.conf.update(
    task_acks_late=True,                 # a task is acknowledged only after it finished, so a crash re-delivers it
    worker_prefetch_multiplier=1,        # one long job must not hide other jobs behind it
    broker_connection_timeout=2,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"max_retries": 1, "socket_connect_timeout": 2},
)


@celery_app.task(name="vault.process_document", soft_time_limit=300, time_limit=330)
def process_document_task(job_id: str) -> None:
    run_processing_job(uuid.UUID(job_id))
