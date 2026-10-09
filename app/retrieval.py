"""Lexical (keyword) retrieval with BM25 scoring.

IMPORTANT: this is NOT semantic search. It matches words, not meaning, so it misses synonyms
("car" vs "vehicle"). It sits behind the Retriever interface so an embedding / pgvector
retriever can replace it without touching the rest of the app.
"""
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.textutil import tokenize

K1, B = 1.5, 0.75


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    page_number: Optional[int]
    chunk_index: int
    text: str
    score: float      # BM25 relevance (only comparable within one search)
    coverage: float   # 0..1, share of the question's (rarity-weighted) key terms found in this chunk


@dataclass
class SearchResult:
    chunks: List[RetrievedChunk]
    term_weights: Dict[str, float] = field(default_factory=dict)  # query stem -> rarity weight


class Retriever(Protocol):
    def search(
        self, db: Session, query: str, document_ids: Optional[Sequence[uuid.UUID]], top_k: int
    ) -> SearchResult: ...


class LexicalRetriever:
    """Loads chunks of completed documents and ranks them in Python.
    Fine for a demo; for thousands of documents use PostgreSQL full-text search or pgvector."""

    def search(self, db, query, document_ids=None, top_k=5) -> SearchResult:
        rows = db.execute(
            select(DocumentChunk, Document.original_filename)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "completed")
        ).all()
        if not rows:
            return SearchResult([])

        tokenised = [tokenize(chunk.text) for chunk, _ in rows]
        n = len(rows)
        df: Counter = Counter()
        for toks in tokenised:
            df.update(set(toks))
        avg_len = (sum(len(t) for t in tokenised) / n) or 1.0

        def idf(term: str) -> float:
            d = df.get(term, 0) or 1  # a word nobody uses is treated as rare, not infinitely rare
            return math.log(1 + (n - d + 0.5) / (d + 0.5))

        query_terms = set(tokenize(query))
        if not query_terms:
            return SearchResult([])
        weights = {t: idf(t) for t in query_terms}
        total_weight = sum(weights.values())
        wanted = set(document_ids) if document_ids else None

        scored: List[RetrievedChunk] = []
        for (chunk, name), toks in zip(rows, tokenised):
            if wanted is not None and chunk.document_id not in wanted:
                continue
            tf = Counter(toks)
            score = 0.0
            matched_weight = 0.0
            for term in query_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                score += weights[term] * (f * (K1 + 1)) / (f + K1 * (1 - B + B * len(toks) / avg_len))
                matched_weight += weights[term]
            if score > 0:
                scored.append(
                    RetrievedChunk(
                        chunk.id, chunk.document_id, name, chunk.page_number, chunk.chunk_index,
                        chunk.text, round(score, 4), round(matched_weight / total_weight, 4),
                    )
                )
        scored.sort(key=lambda c: c.score, reverse=True)
        return SearchResult(scored[:top_k], weights)
