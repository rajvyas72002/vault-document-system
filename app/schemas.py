import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


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
