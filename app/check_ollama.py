"""Smoke test for YOUR Ollama setup. Run:  python -m app.check_ollama"""
import json

from app.config import get_settings
from app.embeddings import OllamaEmbedder
from app.errors import AIServiceError
from app.ollama import OllamaClient


def line(ok: bool, text: str) -> None:
    print(("PASS  " if ok else "FAIL  ") + text)


def main() -> None:
    s = get_settings()
    client = OllamaClient(s.ollama_base_url, s.ollama_timeout_seconds)
    print(f"Ollama URL: {s.ollama_base_url} | chat model: {s.ollama_chat_model} | embed model: {s.ollama_embed_model}")
    try:
        tags = client.request("GET", "/api/tags")
    except AIServiceError as exc:
        line(False, f"cannot reach Ollama: {exc}")
        return
    installed = {m["name"] for m in tags.get("models", [])}
    for wanted in (s.ollama_chat_model, s.ollama_embed_model):
        present = wanted in installed or f"{wanted}:latest" in installed
        line(present, f"model installed: {wanted}" + ("" if present else f"   -> run: ollama pull {wanted}"))
    try:
        emb = OllamaEmbedder(client, s.ollama_embed_model, s.ollama_embed_doc_prefix, s.ollama_embed_query_prefix)
        vectors = emb.embed_documents(["Initial waiting period: 30 days."])
        line(True, f"embedding works, vector length {len(vectors[0])}")
        a, b = emb.embed_query("how long is the waiting period?"), vectors[0]
        dot = sum(x * y for x, y in zip(a, b))
        sim = dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))
        print(f"      similarity of a related question and passage: {sim:.2f} (compare with vector_min_similarity={s.vector_min_similarity})")
    except AIServiceError as exc:
        line(False, f"embedding failed: {exc}")
    try:
        data = client.request("POST", "/api/chat", {
            "model": s.ollama_chat_model, "stream": False, "format": "json",
            "messages": [{"role": "user", "content": 'Reply with JSON {"ok": true} and nothing else.'}],
        }, AIServiceError, s.ollama_chat_model)
        line(True, "chat works, model replied: " + json.dumps(data.get("message", {}).get("content", ""))[:80])
    except AIServiceError as exc:
        line(False, f"chat failed: {exc}")


if __name__ == "__main__":
    main()
