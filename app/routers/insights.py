import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_insight_generator
from app.insight_service import store_insights
from app.insights import InsightGenerator, InsightOptions
from app.models import Document, DocumentInsight
from app.schemas import InsightOut, InsightRequest

router = APIRouter(prefix="/documents", tags=["insights"])


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
    return store_insights(db, doc, generator, InsightOptions(options.length, options.focus, options.tone))


@router.get("/{document_id}/insights", response_model=InsightOut)
def get_insights(document_id: uuid.UUID, db: Session = Depends(get_db)):
    if db.get(Document, document_id) is None:
        raise HTTPException(404, "Document not found.")
    insight = db.scalar(select(DocumentInsight).where(DocumentInsight.document_id == document_id))
    if insight is None:
        raise HTTPException(404, "No insights yet. Call POST /documents/{id}/insights to generate them.")
    return insight
