import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.deps import get_dispatcher, get_embedder
from app.embeddings import Embedder
from app.jobs import find_active_job
from app.models import Document, DocumentChunk, ProcessingJob
from app.processing import process_document
from app.schemas import ChunkList, DocumentList, DocumentOut, DocumentStatus, JobAccepted, JobOut
from app.tasks import Dispatcher
from app.storage import save_upload

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _get_or_404(db: Session, document_id: uuid.UUID) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found.")
    return doc


# Plain `def` (not async): FastAPI runs it in a thread pool, so blocking file and DB work
# does not freeze the event loop.
@router.post("/upload", response_model=DocumentOut, status_code=201)
def upload_document(
    response: Response,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    saved = save_upload(file.file, file.filename, settings.storage_dir, settings.max_upload_mb * 1024 * 1024)

    # Same bytes uploaded before? Return the existing record instead of duplicating work.
    existing = db.scalar(select(Document).where(Document.sha256 == saved.sha256))
    if existing is not None:
        Path(saved.stored_path).unlink(missing_ok=True)
        response.status_code = 200
        return existing

    doc = Document(
        original_filename=saved.original_filename,
        stored_path=saved.stored_path,
        content_type=saved.content_type,
        file_size_bytes=saved.size_bytes,
        sha256=saved.sha256,
        status="pending",
    )
    try:
        db.add(doc)
        db.commit()
    except Exception:
        db.rollback()
        Path(saved.stored_path).unlink(missing_ok=True)  # no orphan file if the DB write fails
        raise
    db.refresh(doc)
    return doc


@router.get("", response_model=DocumentList)
def list_documents(
    status: Optional[Literal["pending", "processing", "completed", "failed"]] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    query = select(Document)
    count = select(func.count()).select_from(Document)
    if status:
        query = query.where(Document.status == status)
        count = count.where(Document.status == status)
    items = db.scalars(query.order_by(Document.created_at.desc()).limit(limit).offset(offset)).all()
    return DocumentList(items=items, total=db.scalar(count) or 0, limit=limit, offset=offset)


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    return _get_or_404(db, document_id)


@router.get("/{document_id}/status", response_model=DocumentStatus)
def get_document_status(document_id: uuid.UUID, db: Session = Depends(get_db)):
    return _get_or_404(db, document_id)


@router.post(
    "/{document_id}/process",
    response_model=DocumentOut,
    responses={
        202: {"model": JobAccepted, "description": "Queued (TASK_BACKEND=background or celery). Poll GET /jobs/{job_id}."},
        409: {"description": "A job for this document is already queued or running."},
        503: {"description": "The task queue is unreachable."},
    },
)
def process(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    embedder: Optional[Embedder] = Depends(get_embedder),
    dispatcher: Optional[Dispatcher] = Depends(get_dispatcher),
):
    """Extract text, build chunks and (in Ollama mode) embed them. Also works as a retry for failed documents.

    TASK_BACKEND=inline: runs now and returns the document (200).
    TASK_BACKEND=background|celery: creates a job, returns 202 immediately, a worker does the rest.
    """
    doc = _get_or_404(db, document_id)
    if dispatcher is None:
        if doc.status == "processing":
            raise HTTPException(409, "Document is already being processed.")
        return process_document(db, doc.id, settings, embedder)

    active = find_active_job(db, doc.id)
    if active is not None:
        raise HTTPException(409, f"Document already has a job in progress (job {active.id}).")
    job = ProcessingJob(document_id=doc.id, kind="process", status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    try:
        dispatcher.enqueue(job.id)
    except Exception as exc:
        log.warning("Could not queue job %s: %s", job.id, exc)
        job.status, job.finished_at = "failed", datetime.now(timezone.utc)
        job.error_message = "Task queue unavailable"
        db.commit()
        raise HTTPException(503, "The task queue is not reachable. Start Redis and the worker, or set TASK_BACKEND=background.")
    return JSONResponse(
        status_code=202,
        content=JobAccepted(job_id=job.id, document_id=doc.id, status="queued", status_url=f"/jobs/{job.id}").model_dump(mode="json"),
    )


@router.get("/{document_id}/jobs", response_model=list[JobOut])
def list_document_jobs(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """Processing history of one document, newest first."""
    _get_or_404(db, document_id)
    return db.scalars(
        select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at.desc()).limit(20)
    ).all()


@router.get("/{document_id}/chunks", response_model=ChunkList)
def list_chunks(
    document_id: uuid.UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    _get_or_404(db, document_id)
    where = DocumentChunk.document_id == document_id
    items = db.scalars(
        select(DocumentChunk).where(where).order_by(DocumentChunk.chunk_index).limit(limit).offset(offset)
    ).all()
    total = db.scalar(select(func.count()).select_from(DocumentChunk).where(where)) or 0
    return ChunkList(items=items, total=total, limit=limit, offset=offset)
