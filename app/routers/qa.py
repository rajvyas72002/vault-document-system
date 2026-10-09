from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.models import Document
from app.providers import ConfigurationError
from app.qa import answer_question
from app.retrieval import LexicalRetriever
from app.schemas import AskRequest, AskResponse, CitationOut, PassageOut, SearchRequest, SearchResponse

router = APIRouter(tags=["question answering"])


def validate_scope(db: Session, document_ids: Optional[List[uuid.UUID]]) -> None:
    """Every requested document must exist and be fully processed."""
    if not document_ids:
        return
    found = {d.id: d for d in db.scalars(select(Document).where(Document.id.in_(document_ids)))}
    missing = [str(i) for i in document_ids if i not in found]
    if missing:
        raise HTTPException(404, f"Document(s) not found: {', '.join(missing)}")
    unprocessed = [str(d.id) for d in found.values() if d.status != "completed"]
    if unprocessed:
        raise HTTPException(409, f"Document(s) not processed yet: {', '.join(unprocessed)}")


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, db: Session = Depends(get_db)):
    """Retrieval only (no answer). Useful for seeing exactly what the answerer is given."""
    validate_scope(db, body.document_ids)
    result = LexicalRetriever().search(db, body.question, body.document_ids, body.top_k)
    return SearchResponse(
        retrieval_method="lexical-bm25 (keyword matching, not semantic)",
        passages=[PassageOut(**{k: getattr(c, k) for k in PassageOut.model_fields}) for c in result.chunks],
    )


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    """One-off question, no chat history. Chat sessions (next step) reuse the same pipeline."""
    validate_scope(db, body.document_ids)
    try:
        result = answer_question(db, body.question, body.document_ids, settings)
    except ConfigurationError as exc:
        raise HTTPException(503, str(exc))
    return AskResponse(
        answer=result.answer,
        answered=result.answered,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        generated_by=result.generated_by,
        note=result.note,
    )
