"""Score the system on a fixed question set. The questions live in app/eval_questions.py.

    python -m app.evaluate                       # both sets (physics chapter + the six policy documents)
    python -m app.evaluate --set policies        # only the policy questions (asked across ALL uploaded documents)
    python -m app.evaluate --set cbse            # only the physics chapter (asked inside keph101.pdf)
    python -m app.evaluate --only 8,110,129      # just these question numbers
    python -m app.evaluate --min-sim 0.4         # try another vector cut-off for this run only
    python -m app.evaluate --provider openai --model gpt-4o   # override the answer model for this run only
    python -m app.evaluate --retriever langchain              # use LangChain's own hybrid retriever (see app/lc_retrieval.py)
    python -m app.evaluate --retrieval-only --retriever own   # only check that search finds the right page: no answer-model calls

It uses the SAME retrieval + provider code as the API, so the score is the real system's score.
The pass check is keyword based (approximate): read the printed answers too.
Needs the documents uploaded and processed with the CURRENT embedding model (python -m app.reprocess_all).
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select

from app.config import get_settings
from app.db import SessionLocal
from app.deps import get_embedder, make_retriever
from app.eval_questions import SETS
from app.models import Document, DocumentChunk
from app.providers import get_llm_provider
from app.qa import answer_question


def diagnose(s, provider, question, chunks):
    """Show the model's raw reply and whether our verification accepts it (works for Ollama and OpenAI)."""
    from app.ollama import OllamaClient
    from app.providers import ProviderError, build_messages, parse_and_verify
    try:
        if hasattr(provider, "ask_raw"):
            raw = provider.ask_raw(question, chunks)
        else:
            client = OllamaClient(s.ollama_base_url, s.ollama_timeout_seconds)
            payload = {"model": s.ollama_chat_model, "messages": build_messages(question, chunks, ()),
                       "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": 4096}}
            raw = (client.request("POST", "/api/chat", payload, ProviderError, s.ollama_chat_model).get("message") or {}).get("content", "")
    except ProviderError as exc:
        return f"LLM error: {exc}"
    ok = parse_and_verify(raw, chunks, "diagnose", question)
    return f"raw LLM output: {raw[:900]!r}\n     verification: {'ACCEPTED' if ok else 'REJECTED'}"


def first_hit_rank(chunks, sources):
    """1-based position of the first chunk that comes from a place where the answer really is, or None."""
    for rank, c in enumerate(chunks, start=1):
        if any(part.lower() in c.document_name.lower() and (page is None or c.page_number == page) for part, page in sources):
            return rank
    return None


def found_source(chunks, sources) -> bool:
    """Did retrieval return at least one chunk from a place where the answer really is?"""
    return first_hit_rank(chunks, sources) is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=sorted(SETS))
    ap.add_argument("--only", default="")
    ap.add_argument("--model", default="", help="chat model name (Ollama model, or OpenAI model with --provider openai)")
    ap.add_argument("--provider", default="", choices=["", "ollama", "openai"], help="override CHAT_PROVIDER from .env")
    ap.add_argument("--min-sim", type=float, default=None, help="override VECTOR_MIN_SIMILARITY for this run only")
    ap.add_argument("--retriever", default="", choices=["", "own", "langchain", "langchain-default-tokenizer"],
                    help="override RETRIEVER from .env for this run only")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="only test that search finds the source page; the answer model is never called")
    args = ap.parse_args()

    s = get_settings()
    if args.provider:
        s = s.model_copy(update={"chat_provider": args.provider})
    if args.model:
        s = s.model_copy(update={"openai_chat_model" if s.chat_provider == "openai" else "ollama_chat_model": args.model})
    if args.min_sim is not None:
        s = s.model_copy(update={"vector_min_similarity": args.min_sim})
    chat_model = s.openai_chat_model if s.chat_provider == "openai" else s.ollama_chat_model
    if not s.real_ai:
        raise SystemExit(f"AI_MODE is '{s.ai_mode}'. Set AI_MODE=real in .env first.")
    embedder = get_embedder(s)
    retriever_kind = args.retriever or s.retriever
    retriever = make_retriever(retriever_kind, s, embedder)
    provider = None if args.retrieval_only else get_llm_provider(s)
    wanted = {int(x) for x in args.only.split(",") if x.strip()}

    with SessionLocal() as db:
        docs = db.scalars(select(Document).where(Document.status == "completed")).all()
        embedded = db.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.embedding_model == embedder.model_name))
        if not embedded:
            raise SystemExit(f"No chunks are embedded with '{embedder.model_name}'. Run:  python -m app.reprocess_all")
        print(f"Provider: {s.chat_provider} | Model: {chat_model} | Embeddings: {embedder.model_name} | "
              f"min similarity: {s.vector_min_similarity} | retriever: {retriever_kind} | {len(docs)} documents uploaded"
              + (" | RETRIEVAL ONLY (no answer-model calls)" if args.retrieval_only else "") + "\n")

        rows = []
        for num, kind, scope, question, need, sources in SETS[args.set]:
            if wanted and num not in wanted:
                continue
            if scope:
                ids = [d.id for d in docs if scope.lower() in d.original_filename.lower()]
                if not ids:
                    print(f"[{num:>3}] SKIPPED: no uploaded document with '{scope}' in its name\n")
                    continue
            else:
                ids = None   # search every uploaded document
            got = retriever.search(db, question, ids, s.retrieval_top_k)
            got_pages = [(c.document_name.replace(".pdf", "").replace(".docx", ""), c.page_number) for c in got.chunks]
            if args.retrieval_only:
                if not sources:
                    continue          # refusal questions have no source page to find
                rank = first_hit_rank(got.chunks, sources)
                print(f"[{num:>3}] {'FOUND, rank ' + str(rank) if rank else 'MISSED'}  top: {got_pages}")
                rows.append({"n": num, "kind": kind, "set": "cbse" if num < 100 else "policies", "question": question,
                             "retrieval_found": rank is not None, "rank": rank})
                continue
            t0 = time.time()
            try:
                r = answer_question(db, question, ids, s, retriever, provider)
                text, answered, cited = r.answer, r.answered, sorted({(c.document_name, c.page_number) for c in r.citations}, key=str)
            except Exception as exc:  # keep going; one failure should not stop the run
                text, answered, cited = f"ERROR: {exc}", False, []
            secs = time.time() - t0
            hit = all(re.search(g, text, re.I) for g in need)
            ok = {"answer": answered and hit, "either": (not answered) or hit, "refuse": not answered}[kind]
            why = "" if ok else {"answer": "refused" if not answered else "answer missing the expected facts",
                                 "either": "answered without the expected fact",
                                 "refuse": "answered but should have refused"}[kind]
            retrieval = None
            if sources:
                retrieval = found_source(got.chunks, sources)
            print(f"[{num:>3}] {'PASS' if ok else 'FAIL'}  {secs:4.1f}s  {('<- ' + why) if why else ''}")
            print(f"      Q: {question}")
            if retrieval is not None:
                print(f"      retrieval: {'FOUND' if retrieval else 'MISSED'} | top: {got_pages}")
            print(f"      A: {text[:420].replace(chr(10), ' ')}")
            if cited:
                print(f"      cited: {[(n, p) for n, p in cited]}")
            if not ok and kind != "refuse" and got.chunks and not text.startswith("ERROR"):
                print("      why: " + diagnose(s, provider, question, got.chunks))
            print()
            rows.append({"n": num, "kind": kind, "set": "cbse" if num < 100 else "policies", "question": question, "pass": bool(ok),
                         "answered": answered, "answer": text, "retrieval_found": retrieval, "seconds": round(secs, 1)})

    print("=" * 66)
    if args.retrieval_only:
        print(f"Retriever: {retriever_kind}")
        for name in ("cbse", "policies", "all"):
            part = [r for r in rows if name == "all" or r["set"] == name]
            if not part:
                continue
            found = sum(r["retrieval_found"] for r in part)
            mrr = sum((1 / r["rank"]) if r["rank"] else 0 for r in part) / len(part)
            top1 = sum(r["rank"] == 1 for r in part)
            print(f"{name.upper():9} found the source {found}/{len(part)} | first result is right {top1}/{len(part)} | mean reciprocal rank {mrr:.3f}")
        out = Path("eval_results")
        out.mkdir(exist_ok=True)
        f = out / f"{datetime.now():%Y%m%d_%H%M%S}_retrieval_{retriever_kind}.json"
        f.write_text(json.dumps({"retriever": retriever_kind, "embeddings": embedder.model_name,
                                 "min_similarity": s.vector_min_similarity, "rows": rows}, indent=2), encoding="utf-8")
        print(f"Saved: {f}")
        return
    total_ok = 0
    for name in ("cbse", "policies"):
        part = [r for r in rows if r["set"] == name]
        if not part:
            continue
        ans = [r for r in part if r["kind"] == "answer"]
        eit = [r for r in part if r["kind"] == "either"]
        ref = [r for r in part if r["kind"] == "refuse"]
        ret = [r for r in part if r["retrieval_found"] is not None]
        ok = sum(r["pass"] for r in part)
        total_ok += ok
        print(f"{name.upper():9} {ok}/{len(part)} correct | answerable {sum(r['pass'] for r in ans)}/{len(ans)}"
              + (f" | typo/definition {sum(r['pass'] for r in eit)}/{len(eit)}" if eit else "")
              + f" | correctly refused {sum(r['pass'] for r in ref)}/{len(ref)}"
              + f" | retrieval found the source {sum(bool(r['retrieval_found']) for r in ret)}/{len(ret)}")
    print(f"TOTAL     {total_ok}/{len(rows)}")
    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    f = out / f"{datetime.now():%Y%m%d_%H%M%S}_{s.chat_provider}_{chat_model.replace(':', '-').replace('/', '-')}.json"
    f.write_text(json.dumps({"provider": s.chat_provider, "model": chat_model, "embeddings": embedder.model_name,
                             "min_similarity": s.vector_min_similarity, "score": f"{total_ok}/{len(rows)}", "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved: {f}")


if __name__ == "__main__":
    main()
