import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from app.config import Settings
from app.providers import ChatTurn, LLMProvider
from app.retrieval import Retriever

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
    retriever: Retriever,
    provider: LLMProvider,
    history: Sequence[ChatTurn] = (),
) -> AnswerResult:
    """retrieve -> (enough evidence?) -> generate -> attach citations.

    If nothing relevant is found, or the provider cannot ground its answer, we say so."""
    retrieved = retriever.search(db, question, document_ids, settings.retrieval_top_k)
    result = provider.generate(question, retrieved, settings.min_coverage, history) if retrieved.chunks else None
    if result is None:
        return AnswerResult(NOT_FOUND_TEXT, False, [], "no answer generated",
                            "No passage supported an answer to the question.")

    citations = [
        CitationData(e.chunk.document_id, e.chunk.document_name, e.chunk.page_number, e.chunk.chunk_id, e.sentence)
        for e in result.evidence
    ]
    return AnswerResult(result.text, True, citations, result.generated_by, result.note)
