"""LLM provider abstraction.

Today only the mock exists. A real provider (OpenAI / Claude) will implement the same
`generate` method, so qa.py never needs to change.
"""
import uuid
from dataclasses import dataclass
from typing import List, Optional, Protocol

from app.retrieval import RetrievedChunk, SearchResult
from app.textutil import expects_quantity, has_quantity, split_sentences, tokenize, wants_comparison


class ConfigurationError(Exception):
    """Raised when a real provider is selected but cannot be used."""


@dataclass
class Evidence:
    chunk: RetrievedChunk
    sentence: str


@dataclass
class ProviderAnswer:
    text: str
    evidence: List[Evidence]  # what the answer was built from (becomes the citations)
    generated_by: str
    note: str


class LLMProvider(Protocol):
    def generate(self, question: str, retrieved: SearchResult, min_coverage: float) -> Optional[ProviderAnswer]:
        """Return an answer grounded in `retrieved`, or None if the passages do not contain one."""


MOCK_NAME = "mock-extractive (no LLM)"
MOCK_NOTE = (
    "Development mode: this answer is a sentence copied from the retrieved passages, chosen by "
    "keyword overlap. It was NOT written by an LLM and is not semantic search."
)


class ExtractiveMockProvider:
    """Answers by quoting the best-matching sentence. Never invents text."""

    def generate(self, question, retrieved, min_coverage) -> Optional[ProviderAnswer]:
        weights = retrieved.term_weights
        total = sum(weights.values()) or 1.0
        need_quantity = expects_quantity(question)

        candidates: List[Evidence] = []
        scored = {}
        for rank, chunk in enumerate(c for c in retrieved.chunks if c.coverage >= min_coverage):
            for sentence in split_sentences(chunk.text):
                terms = set(tokenize(sentence))
                coverage = sum(w for t, w in weights.items() if t in terms) / total
                if coverage < min_coverage:
                    continue
                if need_quantity and not has_quantity(sentence):
                    continue  # the question wants a number/duration but this sentence has none
                ev = Evidence(chunk, sentence)
                candidates.append(ev)
                scored[id(ev)] = (coverage, -rank)
        if not candidates:
            return None
        candidates.sort(key=lambda e: scored[id(e)], reverse=True)

        if wants_comparison(question):
            best_per_document = {}
            for ev in candidates:
                best_per_document.setdefault(ev.chunk.document_id, ev)
            picked = list(best_per_document.values())[:3]
            text = "\n".join(
                f"{e.chunk.document_name}" + (f" (page {e.chunk.page_number})" if e.chunk.page_number else "") + f": {e.sentence}"
                for e in picked
            )
            return ProviderAnswer(text, picked, MOCK_NAME, MOCK_NOTE)

        best = candidates[0]
        return ProviderAnswer(best.sentence, [best], MOCK_NAME, MOCK_NOTE)


def get_llm_provider(ai_mode: str) -> LLMProvider:
    if ai_mode == "mock":
        return ExtractiveMockProvider()
    raise ConfigurationError(
        "AI_MODE=real is selected but no real LLM provider is implemented yet. "
        "Set AI_MODE=mock, or add a provider (and its API key in .env) first."
    )
