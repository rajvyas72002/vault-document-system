import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.chunking import chunk_pages
from app.config import Settings
from app.embeddings import Embedder
from app.errors import AIServiceError
from app.extraction import ExtractionError, extract_pages
from app.models import Document, DocumentChunk, ProcessingJob

log = logging.getLogger(__name__)


def process_document(db: Session, document_id: uuid.UUID, settings: Settings,
                     embedder: Optional[Embedder] = None, job_id: Optional[uuid.UUID] = None) -> Document:
    """pending -> processing -> completed | failed. Safe to run again (it replaces old chunks).

    Every run is recorded in processing_jobs (status, duration, chunk count, failure reason).
    A queue worker passes the job_id it created when the task was queued; otherwise a job is created here.
    Kept free of FastAPI so a Celery task can call it unchanged.
    """
    doc = db.get(Document, document_id)
    if doc is None:
        raise LookupError(f"Document {document_id} not found")

    job = db.get(ProcessingJob, job_id) if job_id else None
    if job is None:
        job = ProcessingJob(document_id=doc.id, kind="process")
        db.add(job)
    started = time.perf_counter()
    job.status, job.started_at = "running", datetime.now(timezone.utc)
    doc.status = "processing"
    doc.error_message = None
    db.commit()
    job_pk = job.id

    try:
        pages = extract_pages(doc.stored_path)
        if not any(p.text.strip() for p in pages):
            raise ExtractionError(
                "No extractable text found. The file may be a scanned PDF; OCR is not supported yet."
            )
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap, settings.chunker)

        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))  # makes retries safe
        rows = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                text=c.text,
                char_count=len(c.text),
            )
            for c in chunks
        ]
        warning = None
        if embedder is not None:
            try:
                vectors = embedder.embed_documents([r.text for r in rows])
                for row, vector in zip(rows, vectors):
                    row.embedding, row.embedding_model = vector, embedder.model_name
            except AIServiceError as exc:
                # Not fatal: keyword search still works. Re-run /process to try the embeddings again.
                log.warning("Embedding failed for document %s: %s", document_id, exc)
                warning = f"Completed without embeddings (keyword search still works): {exc}"
        db.add_all(rows)
        doc.error_message = warning
        doc.page_count = len(pages) if doc.stored_path.lower().endswith(".pdf") else None
        doc.status = "completed"
        job.status, job.finished_at = "succeeded", datetime.now(timezone.utc)
        job.duration_ms, job.chunk_count = int((time.perf_counter() - started) * 1000), len(rows)
        db.commit()
    except Exception as exc:
        db.rollback()
        log.exception("Processing failed for document %s", document_id)  # full detail stays in server logs
        message = str(exc) if isinstance(exc, ExtractionError) else (
            f"Could not read the file ({type(exc).__name__}). It may be corrupt or unsupported."
        )
        doc = db.get(Document, document_id)
        doc.status = "failed"
        doc.error_message = message[:500]  # user-safe: no server paths or stack traces
        job = db.get(ProcessingJob, job_pk)
        job.status, job.finished_at = "failed", datetime.now(timezone.utc)
        job.duration_ms, job.error_message = int((time.perf_counter() - started) * 1000), message[:500]
        db.commit()
    db.refresh(doc)
    return doc
