# Project log

Plain record of what was built, in order, and what went wrong. Newest information last.

## Day 1: foundation (Oct 9 to early Oct 10)
- Read the case study; chose FastAPI, PostgreSQL, Celery/Redis; created the 5 fictional policy PDFs as test data.
- Environment: Python venv, Git, PostgreSQL 17, VS Code. Created `vault` database and user.
- Built: settings, DB layer, `Document` model, upload endpoint (type allow-list, magic bytes, size limit, SHA-256 duplicate detection), list/detail/status endpoints.
- Built: PDF/DOCX text extraction, chunking with overlap and page numbers, processing state machine.
- Built: BM25 keyword retrieval, refusal rules, extractive mock answers with citations.
- Problems: PowerShell tried to run a `.env` line as a command; psql input typed into the wrong prompt; `.env` had `DATABASE_URL=` twice; connected to the wrong database; `fitz` deprecation warning. All fixed.

## Day 2: AI workflow (Oct 10)
- Switched to local Ollama (company API key not available yet): embeddings with `nomic-embed-text`, answers with `llama3.1`.
- Built: embeddings stored with model name, vector search, Reciprocal Rank Fusion, hybrid retrieval, verified LLM answers (exact quotes + number check), `check_ollama` smoke test, `reprocess_all`, `debug_ask`.
- Bug: a correct negative answer ("No. Cosmetic procedures are excluded") was rejected because the prompt's `supported` field was ambiguous. Prompt rewritten.
- Built: chat sessions, history, multi-turn. Bug found in a real chat: a vehicle-policy follow-up was answered from the health policy. Added LLM question rewriting with heuristic fallback.
- Built: document insights (summary, key points, category, tags, length/focus/tone) with verification against the document.
- Built: processing job history, metrics APIs (documents, processing, system), answer latency.
- Built: async queue with `inline`, `background` and `celery` modes, `202` + job status, redelivery-safe worker, queue-down `503`.
- Wrote README, AI_USAGE, DEMO, requirements checklist.
- Verified background mode by hand (202, then job succeeded). Docker Desktop could not start (WSL not installed), so used Memurai as the Redis server: Celery worker received and completed a job end to end.
- Test count: 131 passed, 1 xfailed (documented limitation).

## Still open
- Real LLM provider (`AI_MODE=real`): waiting for company API key and provider name.
- Rehearse the failure demos: stop the worker (job stays queued), stop Redis (503).
- Optional: simple frontend (Next.js), streaming, cost tracking.
