import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from app.config import Settings
from app.providers import LLMProvider, get_llm_provider
from app.retrieval import LexicalRetriever, Retriever

NOT_FOUND_TEXT = "The uploaded documents do not specify an answer to this question."


@dataclass
class CitationData:
    document_id: uuid.UUID
    document_name: str
    page_number: Optional[int]
    chunk_id: uuid.UUID
    quote: str


@dataclass
class AnswerResult:
    answer: str
    answered: bool
    citations: List[CitationData] = field(default_factory=list)
    generated_by: str = ""
    note: str = ""


def answer_question(
    db: Session,
    question: str,
    document_ids: Optional[Sequence[uuid.UUID]],
    settings: Settings,
    retriever: Optional[Retriever] = None,
    provider: Optional[LLMProvider] = None,
) -> AnswerResult:
    """retrieve -> (enough evidence?) -> generate -> attach citations.

    If nothing relevant is found we say so; we never ask the provider to guess."""
    retriever = retriever or LexicalRetriever()
    provider = provider or get_llm_provider(settings.ai_mode)

    retrieved = retriever.search(db, question, document_ids, settings.retrieval_top_k)
    result = provider.generate(question, retrieved, settings.min_coverage) if retrieved.chunks else None
    if result is None:
        return AnswerResult(NOT_FOUND_TEXT, False, [], "no answer generated", "No passage matched the question well enough.")

    citations = [
        CitationData(e.chunk.document_id, e.chunk.document_name, e.chunk.page_number, e.chunk.chunk_id, e.sentence)
        for e in result.evidence
    ]
    return AnswerResult(result.text, True, citations, result.generated_by, result.note)
