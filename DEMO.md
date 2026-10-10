# Demo script

Goal: show document chat end to end, then show that it is honest and reliable.
About 8 minutes. Wording of LLM answers can vary slightly from run to run; the facts and citations should not.

## Before the demo (checklist)
- [ ] PostgreSQL running; `python -m app.init_db` done
- [ ] Ollama running with `llama3.1` and `nomic-embed-text`; `python -m app.check_ollama` shows all PASS
- [ ] `.env`: `AI_MODE=ollama`, `TASK_BACKEND=background` (or `celery` with Redis + worker running)
- [ ] The 5 sample PDFs from `data/policies/` uploaded and processed (`python -m app.reprocess_all` if needed)
- [ ] `pytest` is green
- [ ] `uvicorn app.main:app` running; browser open at `http://127.0.0.1:8000/docs`

## 1. Upload and process (the pipeline)
1. `POST /documents/upload` with `health_insurance_policy.pdf` → `201`, status `pending`.
2. Upload the same file again → `200` and the **same id**. Say: duplicates are detected by SHA-256.
3. `POST /documents/{id}/process` → `202` with a `job_id` (background/celery mode).
4. `GET /jobs/{job_id}` → `succeeded`, with duration and chunk count. Say: the request returned immediately and a worker did the work.
5. `GET /documents/{id}/chunks` → chunks with page numbers. Say: page numbers are what make citations possible.

## 2. Chat with citations (the core feature)
1. `POST /chats` with `{}` → copy the chat id.
2. `POST /chats/{chat_id}/messages`:
   - `What is the initial waiting period for non-accidental illness?` → **30 days**, citation with document, page, exact quote.
3. Say: the quote was checked to appear word for word in the PDF.

## 3. Multi-turn follow-up
1. Same chat: `What is the claim acknowledgement target in the health insurance policy?` → **3 business days**.
2. Same chat: `What about the vehicle policy?`
   - `searched_for` shows the rewritten standalone question.
   - Answer **2 business days**, cited from `vehicle_insurance_policy.pdf`.
3. `GET /chats/{chat_id}/history` → the full conversation with citations stored per answer.

## 4. Meaning, not just keywords
- `Does the policy cover cosmetic surgery?` → "No. Cosmetic procedures are excluded...". The document says "procedures", not "surgery". Say: keyword search alone fails here; this is why retrieval is hybrid (BM25 + embeddings).

## 5. Honesty (what makes it trustworthy)
- `What is the support phone number for the fictional retailer?` → **"The uploaded documents do not specify an answer to this question."** with no citations. The refund policy has no phone number, so any number would be invented.
- `What is the capital of France?` → also refused.
- Say: two layers protect this: the prompt, and the verification of quotes and numbers.

## 6. Comparison across documents
- `Compare claim acknowledgement targets in the two insurance policies.` → 3 vs 2 business days, with a citation from each document.

## 7. Insights
- `POST /documents/{health_id}/insights` with `{"length": "short", "focus": "exclusions", "tone": "executive"}`.
- Show summary, key points (verbatim sentences), category, tags, `generated_by`, and the `note`.

## 8. Monitoring
- `GET /metrics/documents`, `GET /metrics/processing`, `GET /metrics/system`.
- Say: success rate, p95 duration, failure reasons, stuck documents, refusal rate and answer latency.

## 9. Failure handling (optional, strong)
- Upload a `.txt` file → `415`. Upload a fake PDF (text renamed to `.pdf`) → rejected by magic-byte check.
- With `TASK_BACKEND=celery`: stop the worker → job stays `queued`; restart the worker → it picks the job up. Stop Redis (or Memurai on Windows) → `503` after ~4s with a clear message and a `failed` job record; start it again and the worker reconnects by itself. Start the worker a minute early (`--without-mingle --without-gossip`), it needs ~10s to be ready on Windows.
- Stop Ollama → `/ask` returns `503` "Cannot reach Ollama", not a crash.

## If something breaks live
- `python -m app.debug_ask "<question>"` shows what retrieval found and what the model returned before verification.
- Set `AI_MODE=mock` to show the system working without any AI (keyword answers only, clearly labelled).
