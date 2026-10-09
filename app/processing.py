import logging
import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.chunking import chunk_pages
from app.config import Settings
from app.extraction import ExtractionError, extract_pages
from app.models import Document, DocumentChunk

log = logging.getLogger(__name__)


def process_document(db: Session, document_id: uuid.UUID, settings: Settings) -> Document:
    """pending -> processing -> completed | failed. Safe to run again (it replaces old chunks).

    Kept free of FastAPI so a Celery task can call it unchanged later.
    """
    doc = db.get(Document, document_id)
    if doc is None:
        raise LookupError(f"Document {document_id} not found")

    doc.status = "processing"
    doc.error_message = None
    db.commit()

    try:
        pages = extract_pages(doc.stored_path)
        if not any(p.text.strip() for p in pages):
            raise ExtractionError(
                "No extractable text found. The file may be a scanned PDF; OCR is not supported yet."
            )
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)

        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))  # makes retries safe
        db.add_all(
            DocumentChunk(
                document_id=doc.id,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                text=c.text,
                char_count=len(c.text),
            )
            for c in chunks
        )
        doc.page_count = len(pages) if doc.stored_path.lower().endswith(".pdf") else None
        doc.status = "completed"
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
        db.commit()
    db.refresh(doc)
    return doc
