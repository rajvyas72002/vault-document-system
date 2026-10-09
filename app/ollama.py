"""Tiny HTTP client for a local Ollama server (https://github.com/ollama/ollama)."""
from typing import Optional, Type

import httpx

from app.errors import AIServiceError


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 120.0, transport: Optional[httpx.BaseTransport] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport  # tests inject a fake transport so no real network is used

    def request(self, method: str, path: str, payload: Optional[dict] = None,
                error_cls: Type[AIServiceError] = AIServiceError, model: str = "") -> dict:
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.request(method, self.base_url + path, json=payload)
        except httpx.TimeoutException:
            raise error_cls(f"Ollama at {self.base_url} timed out after {self.timeout:.0f}s "
                            "(the model may still be loading; try again).")
        except httpx.TransportError:
            raise error_cls(f"Cannot reach Ollama at {self.base_url}. "
                            "Start the Ollama app (or run `ollama serve`) and try again.")
        if response.status_code == 404:
            raise error_cls(f"Model '{model}' was not found in Ollama. Run: ollama pull {model}")
        if response.status_code >= 400:
            raise error_cls(f"Ollama returned HTTP {response.status_code}.")
        try:
            return response.json()
        except ValueError:
            raise error_cls("Ollama returned a response that is not valid JSON.")
