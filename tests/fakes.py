"""Test doubles. Nothing here talks to a real model or the network."""
import json
import re
from typing import List, Sequence

import httpx

from app.errors import AIServiceError
from app.ollama import OllamaClient

# Words that mean the same thing share a dimension, a crude stand-in for what a real embedding model learns.
CONCEPTS = {
    "surgery": "procedure", "procedure": "procedure", "procedures": "procedure",
    "cosmetic": "cosmetic", "excluded": "excluded", "exclusions": "excluded",
    "leave": "leave", "annual": "leave", "claim": "claim", "claims": "claim",
    "accident": "accident", "accidental": "accident", "return": "return", "returned": "return",
    "waiting": "waiting", "capital": "geo", "france": "geo",
}
DIMENSIONS = sorted(set(CONCEPTS.values()))


class FakeEmbedder:
    def __init__(self, model_name: str = "fake-embed"):
        self.model_name = model_name
        self.calls = 0

    def _vector(self, text: str) -> List[float]:
        words = set(re.findall(r"[a-z]+", text.lower()))
        present = {CONCEPTS[w] for w in words if w in CONCEPTS}
        return [1.0 if d in present else 0.0 for d in DIMENSIONS]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class BrokenEmbedder:
    model_name = "broken-embed"

    def embed_documents(self, texts):
        raise AIServiceError("Cannot reach Ollama at http://localhost:11434. Start the Ollama app and try again.")

    def embed_query(self, text):
        raise AIServiceError("Cannot reach Ollama")


def passage_number(user_content: str, needle: str) -> int:
    """Which numbered passage in the prompt contains `needle`?"""
    for match in re.finditer(r"\[(\d+)\] source:.*?\n(.*?)(?=\n\n\[\d+\] source:|\n</context>)", user_content, re.S):
        if needle in match.group(2):
            return int(match.group(1))
    raise AssertionError(f"no passage contains {needle!r}")


class FakeOllamaChat:
    """Plays the part of Ollama's /api/chat. `reply(user_content, question)` returns the model's JSON."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        user_content = body["messages"][-1]["content"]
        content = self.reply(user_content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})

    def client(self) -> OllamaClient:
        return OllamaClient("http://fake-ollama", 5, httpx.MockTransport(self))
