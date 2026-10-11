import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_insight_generator
from app.insight_service import store_insights
from app.insights import InsightGenerator, InsightOptions
from app.models import Document, DocumentInsight, InsightFeedback
from app.schemas import FeedbackIn, InsightFeedbackOut, InsightOut, InsightRequest

router = APIRouter(prefix="/documents", tags=["insights"])


def _with_feedback(db: Session, insight: DocumentInsight) -> InsightOut:
    out = InsightOut.model_validate(insight)
    row = db.scalar(select(InsightFeedback).where(InsightFeedback.document_id == insight.document_id))
    if row is not None:
        out.feedback, out.feedback_comment = row.rating, row.comment
    return out


def _insight_or_404(db: Session, document_id: uuid.UUID) -> None:
    if db.get(Document, document_id) is None:
        raise HTTPException(404, "Document not found.")
    if db.scalar(select(DocumentInsight.id).where(DocumentInsight.document_id == document_id)) is None:
        raise HTTPException(404, "No insights yet, so there is nothing to rate.")


@router.post("/{document_id}/insights", response_model=InsightOut)
def generate_insights(
    document_id: uuid.UUID,
    body: Optional[InsightRequest] = None,
    db: Session = Depends(get_db),
    generator: InsightGenerator = Depends(get_insight_generator),
):
    """Generate (or regenerate) the summary, key points, category and tags. Options customise the summary."""
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found.")
    if doc.status != "completed":
        raise HTTPException(409, "Document is not processed yet. Call POST /documents/{id}/process first.")
    options = body or InsightRequest()
    return _with_feedback(db, store_insights(db, doc, generator, InsightOptions(options.length, options.focus, options.tone)))


@router.get("/{document_id}/insights", response_model=InsightOut)
def get_insights(document_id: uuid.UUID, db: Session = Depends(get_db)):
    if db.get(Document, document_id) is None:
        raise HTTPException(404, "Document not found.")
    insight = db.scalar(select(DocumentInsight).where(DocumentInsight.document_id == document_id))
    if insight is None:
        raise HTTPException(404, "No insights yet. Call POST /documents/{id}/insights to generate them.")
    return _with_feedback(db, insight)


@router.put("/{document_id}/insights/feedback", response_model=InsightFeedbackOut)
def rate_insight(document_id: uuid.UUID, body: FeedbackIn, db: Session = Depends(get_db)):
    """Thumbs up or down on the document's current summary. Sending it again replaces the earlier rating."""
    _insight_or_404(db, document_id)
    row = db.scalar(select(InsightFeedback).where(InsightFeedback.document_id == document_id))
    if row is None:
        row = InsightFeedback(document_id=document_id, rating=body.rating, comment=body.comment)
        db.add(row)
    else:
        row.rating, row.comment = body.rating, body.comment
    db.commit()
    return InsightFeedbackOut(document_id=document_id, rating=row.rating, comment=row.comment)


@router.delete("/{document_id}/insights/feedback", status_code=204)
def clear_insight_rating(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """Take a rating back. Safe to call when there is none."""
    _insight_or_404(db, document_id)
    row = db.scalar(select(InsightFeedback).where(InsightFeedback.document_id == document_id))
    if row is not None:
        db.delete(row)
        db.commit()
    return Response(status_code=204)
