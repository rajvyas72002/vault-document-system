import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.jobs import STUCK_AFTER_MINUTES
from app.models import ChatMessage, ChatSession, Document, DocumentChunk, DocumentInsight, ProcessingJob
from app.schemas import DocumentMetrics, JobOut, ProcessingMetrics, SystemMetrics

router = APIRouter(prefix="/metrics", tags=["metrics"])



def percentile(values: List[int], pct: float) -> Optional[int]:
    """Nearest-rank percentile. Fine for monitoring; no numpy needed."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def _count(db: Session, query) -> int:
    return db.scalar(query) or 0


@router.get("/documents", response_model=DocumentMetrics)
def document_metrics(db: Session = Depends(get_db)):
    """Document statistics: how many, in which state, how big, how much was indexed."""
    total = _count(db, select(func.count()).select_from(Document))
    by_status = dict(db.execute(select(Document.status, func.count()).group_by(Document.status)).all())
    by_type: dict = {}
    for filename, count in db.execute(select(Document.original_filename, func.count()).group_by(Document.original_filename)).all():
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
        by_type[ext] = by_type.get(ext, 0) + count
    chunks = _count(db, select(func.count()).select_from(DocumentChunk))
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    return DocumentMetrics(
        total_documents=total,
        by_status={s: by_status.get(s, 0) for s in ("pending", "processing", "completed", "failed")},
        by_type=by_type,
        total_size_bytes=_count(db, select(func.coalesce(func.sum(Document.file_size_bytes), 0))),
        total_pages=_count(db, select(func.coalesce(func.sum(Document.page_count), 0))),
        total_chunks=chunks,
        chunks_with_embeddings=_count(db, select(func.count()).select_from(DocumentChunk).where(DocumentChunk.embedding.is_not(None))),
        documents_with_insights=_count(db, select(func.count()).select_from(DocumentInsight)),
        uploaded_last_24h=_count(db, select(func.count()).select_from(Document).where(Document.created_at >= since)),
        avg_chunks_per_document=round(chunks / total, 2) if total else 0.0,
    )


@router.get("/processing", response_model=ProcessingMetrics)
def processing_metrics(db: Session = Depends(get_db)):
    """Processing pipeline health: success rate, durations, why jobs fail, anything stuck."""
    by_status = dict(db.execute(select(ProcessingJob.status, func.count()).group_by(ProcessingJob.status)).all())
    durations = [d for (d,) in db.execute(
        select(ProcessingJob.duration_ms).where(ProcessingJob.duration_ms.is_not(None)).order_by(ProcessingJob.created_at.desc()).limit(500)
    ).all()]
    finished = by_status.get("succeeded", 0) + by_status.get("failed", 0)
    reasons = db.execute(
        select(ProcessingJob.error_message, func.count()).where(ProcessingJob.status == "failed")
        .group_by(ProcessingJob.error_message).order_by(func.count().desc()).limit(5)
    ).all()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STUCK_AFTER_MINUTES)
    stuck = _count(db, select(func.count()).select_from(Document).where(Document.status == "processing", Document.updated_at < cutoff))
    recent = db.scalars(select(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(10)).all()
    return ProcessingMetrics(
        total_jobs=sum(by_status.values()),
        by_status={s: by_status.get(s, 0) for s in ("queued", "running", "succeeded", "failed")},
        success_rate=round(by_status.get("succeeded", 0) / finished, 4) if finished else None,
        avg_duration_ms=round(sum(durations) / len(durations), 1) if durations else None,
        p95_duration_ms=percentile(durations, 95),
        top_failure_reasons=[{"reason": r or "unknown", "count": c} for r, c in reasons],
        stuck_documents=stuck,
        stuck_threshold_minutes=STUCK_AFTER_MINUTES,
        recent_jobs=[JobOut.model_validate(j) for j in recent],
    )


@router.get("/system", response_model=SystemMetrics)
def system_metrics(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    """Question-answering performance: volume, refusal rate, latency, which generator answered."""
    answers = select(ChatMessage).where(ChatMessage.role == "assistant")
    total = _count(db, select(func.count()).select_from(answers.subquery()))
    answered = _count(db, select(func.count()).select_from(ChatMessage).where(ChatMessage.role == "assistant", ChatMessage.answered.is_(True)))
    latencies = [v for (v,) in db.execute(select(ChatMessage.latency_ms).where(ChatMessage.role == "assistant", ChatMessage.latency_ms.is_not(None))).all()]
    by_generator = dict(db.execute(
        select(ChatMessage.generated_by, func.count()).where(ChatMessage.role == "assistant").group_by(ChatMessage.generated_by)
    ).all())
    return SystemMetrics(
        ai_mode=settings.ai_mode,
        chat_sessions=_count(db, select(func.count()).select_from(ChatSession)),
        chat_messages=_count(db, select(func.count()).select_from(ChatMessage)),
        answers_total=total,
        answers_answered=answered,
        answers_refused=total - answered,
        answer_rate=round(answered / total, 4) if total else None,
        avg_answer_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else None,
        p95_answer_latency_ms=percentile(latencies, 95),
        answers_by_generator={(k or "unknown"): v for k, v in by_generator.items()},
    )
