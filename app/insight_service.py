from sqlalchemy import select
from sqlalchemy.orm import Session

from app.insights import InsightGenerator, InsightOptions
from app.models import Document, DocumentChunk, DocumentInsight, InsightFeedback


def store_insights(db: Session, doc: Document, generator: InsightGenerator, options: InsightOptions) -> DocumentInsight:
    """Generate insights for a processed document and save them (replacing any earlier ones)."""
    passages = db.scalars(
        select(DocumentChunk.text).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)
    ).all()
    data = generator.generate(doc.original_filename, passages, options)
    insight = db.scalar(select(DocumentInsight).where(DocumentInsight.document_id == doc.id))
    if insight is None:
        insight = DocumentInsight(document_id=doc.id)
        db.add(insight)
    insight.summary, insight.key_points = data.summary, data.key_points
    insight.category, insight.tags = data.category, data.tags
    insight.options = {"length": options.length, "focus": options.focus, "tone": options.tone}
    insight.generated_by, insight.note = data.generated_by, data.note
    old = db.scalar(select(InsightFeedback).where(InsightFeedback.document_id == doc.id))
    if old is not None:      # a rating belongs to the summary it was given to; this is a new summary
        db.delete(old)
    db.commit()
    db.refresh(insight)
    return insight
