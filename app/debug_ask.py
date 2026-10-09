"""Diagnose why /ask refused. Run:  python -m app.debug_ask "Does the policy cover cosmetic surgery?"
Shows what each retrieval stage found and what the LLM said BEFORE verification."""
import sys

from app.config import get_settings
from app.db import SessionLocal
from app.deps import get_embedder
from app.ollama import OllamaClient
from app.providers import ProviderError, build_messages, parse_and_verify
from app.retrieval import HybridRetriever, LexicalRetriever, VectorRetriever, cosine
from app.models import Document, DocumentChunk
from sqlalchemy import select


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Does the policy cover cosmetic surgery?"
    s = get_settings()
    print("AI_MODE:", s.ai_mode, "| min_similarity:", s.vector_min_similarity, "| min_coverage:", s.min_coverage)
    embedder = get_embedder(s)
    if embedder is None:
        print("Embedder is None -> AI_MODE is not ollama. Stop."); return
    with SessionLocal() as db:
        lex = LexicalRetriever().search(db, question, None, 5)
        print(f"\n[1] LEXICAL kept {len(lex.chunks)} chunk(s)")
        for c in lex.chunks:
            print(f"    {c.document_name} p{c.page_number} score={c.score}  {c.text[:70]!r}")

        qv = embedder.embed_query(question)
        rows = db.execute(select(DocumentChunk, Document.original_filename)
                          .join(Document, Document.id == DocumentChunk.document_id)
                          .where(DocumentChunk.embedding_model == embedder.model_name)).all()
        sims = sorted(((cosine(qv, ch.embedding), n, ch) for ch, n in rows if ch.embedding),
                      key=lambda x: x[0], reverse=True)
        print(f"\n[2] VECTOR: {len(sims)} embedded chunks. Top 5 similarities (before the {s.vector_min_similarity} cut):")
        for sim, name, ch in sims[:5]:
            print(f"    {sim:.3f}  {name} p{ch.page_number}  {ch.text[:60]!r}")

        hybrid = HybridRetriever(LexicalRetriever(), VectorRetriever(embedder, s.vector_min_similarity))
        res = hybrid.search(db, question, None, s.retrieval_top_k)
        print(f"\n[3] HYBRID kept {len(res.chunks)} chunk(s) after the similarity cut")
        if not res.chunks:
            print("    -> nothing retrieved, so the LLM is never called. Lower VECTOR_MIN_SIMILARITY."); return

        client = OllamaClient(s.ollama_base_url, s.ollama_timeout_seconds)
        payload = {"model": s.ollama_chat_model, "messages": build_messages(question, res.chunks, ()),
                   "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": 4096}}
        try:
            data = client.request("POST", "/api/chat", payload, ProviderError, s.ollama_chat_model)
        except ProviderError as exc:
            print("LLM error:", exc); return
        raw = (data.get("message") or {}).get("content", "")
        print("\n[4] RAW LLM OUTPUT:\n", raw)
        verified = parse_and_verify(raw, res.chunks, s.ollama_chat_model)
        print("\n[5] AFTER VERIFICATION:", "ACCEPTED -> " + verified.text if verified else "REJECTED (unsupported, bad quote, or number mismatch)")


if __name__ == "__main__":
    main()
