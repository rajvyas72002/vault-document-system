import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.models import Document
from app.schemas import DocumentList, DocumentOut, DocumentStatus
from app.storage import save_upload

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
