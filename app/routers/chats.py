import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.conversation import MAX_HISTORY_MESSAGES, OllamaRewriter, build_retrieval_query, is_follow_up, to_history
from app.db import get_db
from app.deps import get_provider, get_retriever, get_rewriter
from app.models import ChatMessage, ChatSession, MessageFeedback
from app.providers import LLMProvider
from app.qa import answer_question
from app.retrieval import Retriever
from app.routers.qa import validate_scope
from app.schemas import (
    ChatAnswer,
    ChatCreate,
    ChatHistory,
    ChatMessageOut,
    ChatSessionOut,
    CitationOut,
    FeedbackIn,
    FeedbackOut,
    MessageRequest,
)

router = APIRouter(prefix="/chats", tags=["chat"])


def get_chat_or_404(db: Session, chat_id: uuid.UUID) -> ChatSession:
    chat = db.get(ChatSession, chat_id)
    if chat is None:
        raise HTTPException(404, "Chat session not found")
    return chat


@router.post("", response_model=ChatSessionOut, status_code=201)
def start_chat(body: ChatCreate, db: Session = Depends(get_db)):
    """Start a conversation, optionally limited to some documents."""
    validate_scope(db, body.document_ids)
    chat = ChatSession(
        title=body.title,
        document_ids=[str(i) for i in body.document_ids] if body.document_ids else None,
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("", response_model=List[ChatSessionOut])
def list_chats(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    return db.scalars(select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit).offset(offset)).all()


@router.get("/{chat_id}/history", response_model=ChatHistory)
def chat_history(chat_id: uuid.UUID, db: Session = Depends(get_db)):
    chat = get_chat_or_404(db, chat_id)
    messages = db.scalars(select(ChatMessage).where(ChatMessage.session_id == chat.id).order_by(ChatMessage.seq)).all()
    ratings = {
        f.message_id: f
        for f in db.scalars(select(MessageFeedback).where(MessageFeedback.message_id.in_([m.id for m in messages])))
    } if messages else {}
    out = []
    for m in messages:
        item = ChatMessageOut.model_validate(m)
        if m.id in ratings:
            item.feedback, item.feedback_comment = ratings[m.id].rating, ratings[m.id].comment
        out.append(item)
    return ChatHistory(session=ChatSessionOut.model_validate(chat), messages=out)


@router.post("/{chat_id}/messages", response_model=ChatAnswer)
def ask_in_chat(
    chat_id: uuid.UUID,
    body: MessageRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever),
    provider: LLMProvider = Depends(get_provider),
    rewriter: Optional[OllamaRewriter] = Depends(get_rewriter),
):
    """Ask a question inside a conversation. Earlier turns are used to understand follow-ups."""
    chat = get_chat_or_404(db, chat_id)
    scope = [uuid.UUID(i) for i in chat.document_ids] if chat.document_ids else None
    validate_scope(db, scope)

    recent = db.scalars(
        select(ChatMessage).where(ChatMessage.session_id == chat.id).order_by(ChatMessage.seq.desc()).limit(MAX_HISTORY_MESSAGES)
    ).all()
    history = to_history(list(reversed(recent)))

    # A follow-up ("What about the vehicle policy?") cannot be searched as written. With a real LLM we
    # rewrite it into a standalone question; if that is unavailable or fails, we use the keyword heuristic.
    started = time.perf_counter()
    standalone = rewriter.rewrite(body.question, history) if rewriter and history and is_follow_up(body.question) else None

    # If retrieval or the model fails this raises (HTTP 503) BEFORE anything is saved,
    # so a failed turn never leaves half a conversation behind.
    if standalone:
        searched_for = standalone
        result = answer_question(db, standalone, scope, settings, retriever, provider)
    else:
        searched_for = build_retrieval_query(body.question, history)
        result = answer_question(db, body.question, scope, settings, retriever, provider, history, searched_for)

    latency_ms = int((time.perf_counter() - started) * 1000)
    citations = [CitationOut(**c.__dict__) for c in result.citations]
    last_seq = db.scalar(select(func.max(ChatMessage.seq)).where(ChatMessage.session_id == chat.id)) or 0
    db.add(ChatMessage(session_id=chat.id, seq=last_seq + 1, role="user", content=body.question))
    assistant = ChatMessage(
        session_id=chat.id, seq=last_seq + 2, role="assistant", content=result.answer, answered=result.answered,
        citations=[c.model_dump(mode="json") for c in citations], generated_by=result.generated_by,
        latency_ms=latency_ms,
    )
    db.add(assistant)
    if not chat.title:
        chat.title = body.question[:80]
    chat.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(assistant)
    return ChatAnswer(
        message_id=assistant.id, answer=result.answer, answered=result.answered, citations=citations,
        generated_by=result.generated_by, note=result.note, searched_for=searched_for,
    )


def _answer_or_404(db: Session, chat_id: uuid.UUID, message_id: uuid.UUID) -> ChatMessage:
    get_chat_or_404(db, chat_id)
    message = db.get(ChatMessage, message_id)
    if message is None or message.session_id != chat_id:
        raise HTTPException(404, "Message not found in this chat")
    if message.role != "assistant":
        raise HTTPException(400, "Only the assistant's answers can be rated")
    return message


@router.put("/{chat_id}/messages/{message_id}/feedback", response_model=FeedbackOut)
def rate_answer(chat_id: uuid.UUID, message_id: uuid.UUID, body: FeedbackIn, db: Session = Depends(get_db)):
    """Thumbs up or down on an answer. Sending it again replaces the earlier rating (one rating per answer)."""
    _answer_or_404(db, chat_id, message_id)
    row = db.scalar(select(MessageFeedback).where(MessageFeedback.message_id == message_id))
    if row is None:
        row = MessageFeedback(message_id=message_id, rating=body.rating, comment=body.comment)
        db.add(row)
    else:
        row.rating, row.comment = body.rating, body.comment
    db.commit()
    return FeedbackOut(message_id=message_id, rating=row.rating, comment=row.comment)


@router.delete("/{chat_id}/messages/{message_id}/feedback", status_code=204)
def clear_rating(chat_id: uuid.UUID, message_id: uuid.UUID, db: Session = Depends(get_db)):
    """Take a rating back. Safe to call when there is none."""
    _answer_or_404(db, chat_id, message_id)
    row = db.scalar(select(MessageFeedback).where(MessageFeedback.message_id == message_id))
    if row is not None:
        db.delete(row)
        db.commit()
    return Response(status_code=204)
