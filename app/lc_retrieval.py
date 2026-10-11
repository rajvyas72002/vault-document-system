"""Hybrid retrieval assembled from LangChain's own parts, as an alternative to app.retrieval.HybridRetriever.

    BM25Retriever (keywords)  +  InMemoryVectorStore retriever (embeddings)  ->  EnsembleRetriever (rank fusion)

Same interface as our own retrievers (search(db, query, document_ids, top_k)), so everything else (the answer
pipeline, the verifier, the evaluation) works unchanged. Select it with RETRIEVER=langchain.

BM25Retriever splits text on spaces by default (no lower-casing, no stemming, no stop words). Measured on the 50 retrieval
questions that made it clearly worse than our retriever (first result right 38/50 against 46/50), so it is given our
tokenizer (app/textutil.py) instead, and then it matches ours exactly (46/50). Kind "langchain-default-tokenizer" keeps
LangChain's own splitting, for comparison.

Other differences from our own version, left as LangChain does them:
  * BM25Retriever returns its top-k even when a chunk shares no word with the question (ours drops score 0).
  * Chunk vectors are NOT re-embedded: the vectors already stored in PostgreSQL are placed in the vector store, so only
    the question is embedded per search.
The "coverage" of each chunk (share of the question's key terms it contains) is still computed with our helper because the
answer step uses it.
"""
import logging
import math
import uuid
from collections import Counter
from typing import Any, Callable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from sqlalchemy import select

from app.embeddings import Embedder
from app.models import Document as DocumentRow
from app.models import DocumentChunk
from app.retrieval import RetrievedChunk, SearchResult, coverage_of
from app.textutil import tokenize

RRF_C = 60  # same constant as our own rank fusion

# LangChain logs a warning every time a question has no vector above the similarity cut-off (normal for unanswerable questions).
logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class _EmbedderAdapter(Embeddings):
    """Lets a LangChain vector store use one of our embedders (so the question gets the right model and prefix)."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)


class _CosineStore(InMemoryVectorStore):
    """InMemoryVectorStore already returns cosine similarity; say so, so a score threshold can be used."""

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return lambda score: score

    def similarity_search_with_score_by_vector(self, embedding, k: int = 4, **kwargs):
        if not any(embedding):          # an all-zero vector has no direction; cosine would be NaN and raise
            return []
        return super().similarity_search_with_score_by_vector(embedding, k, **kwargs)


def _term_weights(query: str, corpus_tokens: Sequence[Sequence[str]]) -> dict:
    n = len(corpus_tokens) or 1
    df: Counter = Counter()
    for toks in corpus_tokens:
        df.update(set(toks))
    return {t: math.log(1 + (n - (df.get(t, 0) or 1) + 0.5) / ((df.get(t, 0) or 1) + 0.5)) for t in set(tokenize(query))}


class LangChainHybridRetriever:
    def __init__(self, embedder: Embedder, min_similarity: float = 0.5, pool: int = 20, default_tokenizer: bool = False):
        self.embedder = embedder
        self.min_similarity = min_similarity
        self.pool = pool
        self.default_tokenizer = default_tokenizer
        self.method = ("langchain: BM25Retriever + InMemoryVectorStore, EnsembleRetriever"
                       + (" (BM25 with LangChain's default word splitting)" if default_tokenizer else " (BM25 fed by our tokenizer)"))

    def search(self, db, query: str, document_ids: Optional[Sequence[uuid.UUID]] = None, top_k: int = 5) -> SearchResult:
        from langchain_classic.retrievers import EnsembleRetriever   # imported here: optional dependencies
        from langchain_community.retrievers import BM25Retriever

        rows = db.execute(
            select(DocumentChunk, DocumentRow.original_filename)
            .join(DocumentRow, DocumentRow.id == DocumentChunk.document_id)
            .where(DocumentRow.status == "completed")
        ).all()
        wanted = set(document_ids) if document_ids else None
        rows = [(c, n) for c, n in rows if wanted is None or c.document_id in wanted]
        if not rows or not tokenize(query):
            return SearchResult([])

        def to_doc(chunk: Any, name: str) -> Document:
            return Document(page_content=chunk.text, id=str(chunk.id), metadata={
                "chunk_id": str(chunk.id), "document_id": str(chunk.document_id), "document_name": name,
                "page_number": chunk.page_number, "chunk_index": chunk.chunk_index})

        docs = [to_doc(c, n) for c, n in rows]
        bm25_kwargs = {} if self.default_tokenizer else {"preprocess_func": tokenize}
        bm25 = BM25Retriever.from_documents(docs, k=self.pool, **bm25_kwargs)

        store = _CosineStore(embedding=_EmbedderAdapter(self.embedder))
        for (chunk, _), doc in zip(rows, docs):   # reuse the vectors we already stored; nothing is embedded again
            if chunk.embedding and any(chunk.embedding) and chunk.embedding_model == self.embedder.model_name:
                store.store[doc.id] = {"id": doc.id, "vector": chunk.embedding, "text": doc.page_content,
                                       "metadata": doc.metadata}
        vector = store.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"k": self.pool, "score_threshold": self.min_similarity})

        ensemble = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5], c=RRF_C, id_key="chunk_id")
        found = ensemble.invoke(query)[:top_k]

        weights = _term_weights(query, [tokenize(c.text) for c, _ in rows])
        chunks = [
            RetrievedChunk(uuid.UUID(d.metadata["chunk_id"]), uuid.UUID(d.metadata["document_id"]), d.metadata["document_name"],
                           d.metadata["page_number"], d.metadata["chunk_index"], d.page_content,
                           round(1.0 / (RRF_C + rank), 6), coverage_of(d.page_content, weights))
            for rank, d in enumerate(found, start=1)
        ]
        return SearchResult(chunks, weights)
