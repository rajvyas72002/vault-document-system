import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.deps import get_provider, get_retriever
from app.models import Document
from app.providers import LLMProvider
from app.qa import answer_question
from app.retrieval import Retriever
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
def search(body: SearchRequest, db: Session = Depends(get_db), retriever: Retriever = Depends(get_retriever)):
    """Retrieval only (no answer). Shows exactly which passages the answerer would be given."""
    validate_scope(db, body.document_ids)
    result = retriever.search(db, body.question, body.document_ids, body.top_k)
    return SearchResponse(
        retrieval_method=retriever.method,
        passages=[PassageOut(**{k: getattr(c, k) for k in PassageOut.model_fields}) for c in result.chunks],
    )


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever),
    provider: LLMProvider = Depends(get_provider),
):
    """One-off question, no chat history. Chat sessions reuse the same pipeline."""
    validate_scope(db, body.document_ids)
    result = answer_question(db, body.question, body.document_ids, settings, retriever, provider)
    return AskResponse(
        answer=result.answer,
        answered=result.answered,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        generated_by=result.generated_by,
        note=result.note,
    )
