import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ProcessingJob
from app.schemas import JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
    """Status of one background job: queued -> running -> succeeded | failed."""
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job
