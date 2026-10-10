"""Shared rules about processing jobs."""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ProcessingJob

# A job still queued/running after this long is treated as stuck (worker died, queue lost the task).
# Metrics report such documents, and a new job may be queued for them.
STUCK_AFTER_MINUTES = 15


def find_active_job(db: Session, document_id: uuid.UUID) -> Optional[ProcessingJob]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STUCK_AFTER_MINUTES)
    return db.scalar(
        select(ProcessingJob)
        .where(ProcessingJob.document_id == document_id,
               ProcessingJob.status.in_(("queued", "running")),
               ProcessingJob.created_at >= cutoff)
        .order_by(ProcessingJob.created_at.desc())
    )
