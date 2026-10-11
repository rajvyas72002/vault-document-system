# Requirements checklist (traced to the case study PDF)

Status: DONE = built and tested | PARTIAL = works but incomplete | MOCK = fake/placeholder | TODO = not started
Last updated: 2026-10-11 (207 tests; Celery and frontend verified by hand)

## Backend components
| PDF requirement | Status | Notes |
|---|---|---|
| Intelligent upload handler | DONE | extension allowlist, magic bytes, size limit, SHA-256 dedup, safe filenames |
| Smart document analysis (AI summary/insights) | DONE | summary, key points, category, tags; LLM output verified (verbatim key points, no invented numbers), extractive fallback. Not yet auto-run after processing |
| Asynchronous processing pipeline | DONE | inline / background / celery, 202 + GET /jobs/{id}, retries safe, queue-down => 503. Verified by hand with Celery + Memurai on Windows (one job end to end). Failure demos rehearsed by hand: worker down => job waits then completes; Redis down => 503 + failed job record; Redis back => worker reconnects by itself |
| Upload & ingestion API | DONE | POST /documents/upload |
| Document retrieval API | DONE | list, detail, status, chunks |
| AI insights API | DONE | POST + GET /documents/{id}/insights |
| Start chat session | DONE | POST /chats (optional document scope) |
| Ask question | DONE | POST /chats/{id}/messages (and one-off POST /ask), cited and verified |
| Get chat history | DONE | GET /chats/{id}/history, citations stored per answer |
| Multi-turn context | DONE | last 3 turns sent to the LLM; follow-ups rewritten into a standalone question by the LLM (keyword fallback if it fails); one level deep |
| Document statistics API | DONE | GET /metrics/documents |
| Processing metrics API | DONE | GET /metrics/processing (success rate, avg/p95 duration, failure reasons, stuck docs) |

## Data storage
| Need | Status |
|---|---|
| Document metadata and status | DONE |
| Chunks + embeddings | DONE (JSON column; pgvector is the scaling path) |
| AI summaries and insights | DONE (document_insights) |
| Chat sessions / messages with citations | DONE (chat_sessions, chat_messages) |
| Processing metrics and history | DONE (processing_jobs) |
| System performance data | DONE (answer latency per message, GET /metrics/system) |

## RAG quality
| Item | Status | Notes |
|---|---|---|
| Retrieval | DONE | keyword + embedding search merged by rank fusion: LangChain EnsembleRetriever by default, hand-written version kept (RETRIEVER=own); both 60/60 |
| Citations | DONE | document, page, exact quote, verified against source |
| Anti-hallucination | DONE | coverage gate, quote + number verification, "not specified" answers |
| Local LLM (Ollama) | DONE | llama3.1 verified on 3 questions; not a benchmark |
| Real provider (OpenAI) | DONE | AI_MODE=real, CHAT_PROVIDER/EMBED_PROVIDER=openai via LangChain; evaluation gpt-4o 60/60 (small set, CBSE part partly tuned) |
| LangChain / LlamaIndex | DONE | ChatOpenAI, OpenAIEmbeddings, BM25Retriever + vector store + EnsembleRetriever (as a BaseRetriever), LCEL chain. Verifier and word splitting are our own code. I first built retrieval by hand out of curiosity and compared: both 60/60 |

## Deliverables
| Item | Status |
|---|---|
| Working backend | DONE | verified with real Ollama, PostgreSQL, Celery + Redis |
| Git repository | DONE (local; GitHub push TODO) |
| README.md with architecture + assumptions | DONE | architecture diagrams, setup, API, assumptions, honest limitations |
| AI_USAGE.md | DONE | has ONE placeholder line for other AI tools Raj used: fill in or delete |
| Tests (pytest) | DONE: 207 passed, 1 xfailed (known synonym limitation in mock mode). Tests are isolated from .env (mock AI, inline tasks) |
| Demo of document chat | PARTIAL | DEMO.md script written; live rehearsal still to do |

## Optional / bonus
| Item | Status |
|---|---|
| Multi-document questions | DONE (document_ids optional filter) |
| Comparative analysis | DONE (verified on 2 insurance policies) |
| Follow-up question suggestions | TODO |
| Summary customisation | DONE (length, focus, tone) |
| Auto categorisation / tagging | DONE |
| Frontend (Next.js, optional) | DONE (minimal) | 4 pages: documents, chat, insights, metrics. Builds cleanly; clicked through in a real browser (documents, chat with citation, insights, metrics all work). No automated frontend tests |
| Streaming responses | TODO |
| Health endpoint | DONE (/health) |
| Cost / token tracking | TODO |
| Redis caching, rate limiting, versioning | TODO |
