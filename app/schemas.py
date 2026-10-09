import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


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
