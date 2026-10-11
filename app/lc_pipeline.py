"""The question-answering pipeline expressed with LangChain building blocks.

    question -> VaultRetriever (a LangChain BaseRetriever) -> Documents -> answer step -> AnswerResult

What is LangChain here: the retriever interface (BaseRetriever / Document) and the chain (LCEL Runnables), so the
steps can be traced, swapped or extended with any LangChain component.
What is NOT: the retrieval algorithm (our BM25 + vector + rank fusion) and the verification of quotes and numbers
(app/providers.py). Those stay our own code on purpose; they are what the evaluation measures.
Behaviour is meant to be identical to qa.answer_question; python -m app.evaluate must give the same score with
PIPELINE=own and PIPELINE=langchain.
"""
import uuid
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import ConfigDict

from app.retrieval import RetrievedChunk, SearchResult


class VaultRetriever(BaseRetriever):
    """Exposes any of our retrievers (lexical, vector, hybrid) as a LangChain retriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any                                  # app.retrieval.Retriever
    db: Any                                         # SQLAlchemy session
    document_ids: Optional[List[uuid.UUID]] = None  # None = search every document
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        found = self.retriever.search(self.db, query, self.document_ids, self.top_k)
        return [
            Document(
                page_content=c.text,
                metadata={
                    "chunk_id": c.chunk_id, "document_id": c.document_id, "document_name": c.document_name,
                    "page_number": c.page_number, "chunk_index": c.chunk_index, "score": c.score,
                    "coverage": c.coverage, "term_weights": found.term_weights,
                },
            )
            for c in found.chunks
        ]


def documents_to_search_result(docs: Sequence[Document]) -> SearchResult:
    """Back to the structure the answer providers (and the verifier) work with, losing nothing."""
    chunks = [
        RetrievedChunk(m["chunk_id"], m["document_id"], m["document_name"], m["page_number"], m["chunk_index"],
                       d.page_content, m["score"], m["coverage"])
        for d in docs for m in [d.metadata]
    ]
    weights = docs[0].metadata.get("term_weights", {}) if docs else {}
    return SearchResult(chunks, weights)


def build_answer_chain(db, document_ids, settings, retriever, provider) -> Runnable:
    """Input: {"question", "retrieval_query" (optional), "history" (optional)}  ->  Output: AnswerResult."""
    from app.qa import NOT_FOUND_TEXT, AnswerResult, CitationData   # imported late: qa.py imports this module

    vault = VaultRetriever(retriever=retriever, db=db, document_ids=list(document_ids) if document_ids else None,
                           top_k=settings.retrieval_top_k)

    def retrieve(x: Dict[str, Any]) -> List[Document]:
        return vault.invoke(x.get("retrieval_query") or x["question"])

    def generate(x: Dict[str, Any]):
        docs = x["docs"]
        if not docs:
            return None
        return provider.generate(x["question"], documents_to_search_result(docs), settings.min_coverage,
                                 x.get("history", ()))

    def to_result(result) -> "AnswerResult":
        if result is None:
            return AnswerResult(NOT_FOUND_TEXT, False, [], "no answer generated",
                                "No passage supported an answer to the question.")
        citations = [
            CitationData(e.chunk.document_id, e.chunk.document_name, e.chunk.page_number, e.chunk.chunk_id, e.sentence)
            for e in result.evidence
        ]
        return AnswerResult(result.text, True, citations, result.generated_by, result.note)

    return (
        RunnablePassthrough.assign(docs=RunnableLambda(retrieve).with_config(run_name="retrieve"))
        | RunnableLambda(generate).with_config(run_name="generate_and_verify")
        | RunnableLambda(to_result).with_config(run_name="build_answer")
    )
