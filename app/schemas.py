import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    content_type: str
    file_size_bytes: int
    sha256: str
    status: str
    error_message: Optional[str] = None
    page_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    items: List[DocumentOut]
    total: int
    limit: int
    offset: int


class DocumentStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    error_message: Optional[str] = None
    updated_at: datetime


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int
    page_number: Optional[int] = None
    char_count: int
    text: str


class ChunkList(BaseModel):
    items: List[ChunkOut]
    total: int
    limit: int
    offset: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    document_ids: Optional[List[uuid.UUID]] = None  # omit to search all processed documents


class SearchRequest(AskRequest):
    top_k: int = Field(5, ge=1, le=20)


class CitationOut(BaseModel):
    document_id: uuid.UUID
    document_name: str
    page_number: Optional[int] = None
    chunk_id: uuid.UUID
    quote: str


class AskResponse(BaseModel):
    answer: str
    answered: bool
    citations: List[CitationOut]
    generated_by: str
    note: str


class PassageOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    page_number: Optional[int] = None
    score: float
    coverage: float
    text: str


class SearchResponse(BaseModel):
    retrieval_method: str
    passages: List[PassageOut]


# ----------------------------------------------------------------------------- chat
class ChatCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    document_ids: Optional[List[uuid.UUID]] = None  # omit to chat across all processed documents


class ChatSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    title: Optional[str] = None
    document_ids: Optional[List[uuid.UUID]] = None
    created_at: datetime
    updated_at: datetime


class MessageRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class ChatAnswer(AskResponse):
    message_id: uuid.UUID
    searched_for: str  # the query actually used for retrieval (differs from the question for follow-ups)


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    seq: int
    role: str
    content: str
    answered: Optional[bool] = None
    citations: List[CitationOut] = []
    generated_by: Optional[str] = None
    latency_ms: Optional[int] = None
    created_at: datetime

    @field_validator("citations", mode="before")
    @classmethod
    def none_means_empty(cls, value):
        return value or []


class ChatHistory(BaseModel):
    session: ChatSessionOut
    messages: List[ChatMessageOut]


# ----------------------------------------------------------------------------- insights
class InsightRequest(BaseModel):
    length: Literal["short", "medium", "long"] = "medium"
    focus: Optional[str] = Field(None, max_length=200, description="Topic the summary should concentrate on")
    tone: Literal["neutral", "executive", "simple"] = "neutral"


class InsightOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    document_id: uuid.UUID
    summary: str
    key_points: List[str]
    category: str
    tags: List[str]
    options: dict
    generated_by: str
    note: str
    created_at: datetime
    updated_at: datetime


# ----------------------------------------------------------------------------- metrics
class DocumentMetrics(BaseModel):
    total_documents: int
    by_status: dict
    by_type: dict
    total_size_bytes: int
    total_pages: int
    total_chunks: int
    chunks_with_embeddings: int
    documents_with_insights: int
    uploaded_last_24h: int
    avg_chunks_per_document: float


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    document_id: uuid.UUID
    kind: str = "process"
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None


class ProcessingMetrics(BaseModel):
    total_jobs: int
    by_status: dict
    success_rate: Optional[float] = None       # succeeded / (succeeded + failed); None until a job has finished
    avg_duration_ms: Optional[float] = None
    p95_duration_ms: Optional[int] = None
    top_failure_reasons: List[dict]
    stuck_documents: int                       # "processing" for longer than the stuck threshold
    stuck_threshold_minutes: int
    recent_jobs: List[JobOut]


class SystemMetrics(BaseModel):
    ai_mode: str
    chat_sessions: int
    chat_messages: int
    answers_total: int
    answers_answered: int
    answers_refused: int
    answer_rate: Optional[float] = None
    avg_answer_latency_ms: Optional[float] = None
    p95_answer_latency_ms: Optional[int] = None
    answers_by_generator: dict


class JobAccepted(BaseModel):
    job_id: uuid.UUID
    document_id: uuid.UUID
    status: str
    status_url: str
