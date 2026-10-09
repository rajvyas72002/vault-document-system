"""Lexical (keyword) retrieval with BM25 scoring.

IMPORTANT: this is NOT semantic search. It matches words, not meaning, so it misses synonyms
("car" vs "vehicle"). It sits behind the Retriever interface so an embedding / pgvector
retriever can replace it without touching the rest of the app.
"""
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import Embedder
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
    method: str

    def search(
        self, db: Session, query: str, document_ids: Optional[Sequence[uuid.UUID]], top_k: int
    ) -> SearchResult: ...


class LexicalRetriever:
    """Loads chunks of completed documents and ranks them in Python.
    Fine for a demo; for thousands of documents use PostgreSQL full-text search or pgvector."""

    method = "lexical-bm25 (keyword matching, not semantic)"

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


def coverage_of(text: str, weights: Dict[str, float]) -> float:
    """Share of the question's rarity-weighted key terms that occur in `text`."""
    total = sum(weights.values()) or 1.0
    terms = set(tokenize(text))
    return round(sum(w for t, w in weights.items() if t in terms) / total, 4)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorRetriever:
    """Semantic search: compare the question's embedding with each stored chunk embedding.
    Only chunks embedded by the SAME model are compared. Pure Python is fine for thousands of
    chunks; beyond that use pgvector or a vector database."""

    def __init__(self, embedder: Embedder, min_similarity: float = 0.5):
        self.embedder = embedder
        self.min_similarity = min_similarity
        self.method = f"vector ({embedder.model_name})"

    def search(self, db, query, document_ids=None, top_k=5) -> SearchResult:
        rows = db.execute(
            select(DocumentChunk, Document.original_filename)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "completed", DocumentChunk.embedding_model == self.embedder.model_name)
        ).all()
        if not rows:
            return SearchResult([])
        wanted = set(document_ids) if document_ids else None
        query_vector = self.embedder.embed_query(query)
        scored = []
        for chunk, name in rows:
            if (wanted is not None and chunk.document_id not in wanted) or not chunk.embedding:
                continue
            sim = cosine(query_vector, chunk.embedding)
            if sim >= self.min_similarity:
                scored.append(RetrievedChunk(chunk.id, chunk.document_id, name, chunk.page_number,
                                             chunk.chunk_index, chunk.text, round(sim, 4), 0.0))
        scored.sort(key=lambda c: c.score, reverse=True)
        return SearchResult(scored[:top_k])


def reciprocal_rank_fusion(rankings: Sequence[Sequence[uuid.UUID]], k: int = 60) -> Dict[uuid.UUID, float]:
    """Merge several ranked lists using ranks only: score = sum(1 / (k + rank)).
    Ranks are used because BM25 scores and cosine similarities are on different scales."""
    fused: Dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """Keyword + semantic search, merged with Reciprocal Rank Fusion."""

    def __init__(self, lexical: LexicalRetriever, vector: VectorRetriever, pool: int = 20):
        self.lexical, self.vector, self.pool = lexical, vector, pool
        self.method = f"hybrid: lexical-bm25 + {vector.method}, fused with reciprocal rank fusion"

    def search(self, db, query, document_ids=None, top_k=5) -> SearchResult:
        lex = self.lexical.search(db, query, document_ids, self.pool)
        vec = self.vector.search(db, query, document_ids, self.pool)
        by_id = {c.chunk_id: c for c in lex.chunks}
        for c in vec.chunks:
            by_id.setdefault(c.chunk_id, c)
        fused = reciprocal_rank_fusion([[c.chunk_id for c in lex.chunks], [c.chunk_id for c in vec.chunks]])
        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        chunks = []
        for chunk_id in ranked:
            c = by_id[chunk_id]
            chunks.append(RetrievedChunk(c.chunk_id, c.document_id, c.document_name, c.page_number, c.chunk_index,
                                         c.text, round(fused[chunk_id], 6), coverage_of(c.text, lex.term_weights)))
        return SearchResult(chunks, lex.term_weights)
