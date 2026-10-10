# AI Usage

This project uses AI in two different ways, and they are described separately:

1. **AI as a development tool**: how AI assistants helped me build it.
2. **AI inside the product**: how the system itself uses language models, and how its output is controlled.

---

## 1. AI as a development tool

**Tools used**

- **Claude (Anthropic)**, used as a coding and design assistant throughout: planning, code, tests, debugging and explanations.
- **ChatGPT (OpenAI)**, used for background research: understanding the case-study requirements and concepts such as vector search before building. It did not write any of the project code.

**How I worked**

I am new to PostgreSQL, Redis/Celery and RAG, so I used the assistant as a mentor as well as a code generator. For each step I asked for the reason behind it,
ran the code on my own machine (Windows, VS Code, PostgreSQL 17, Ollama), pasted real results back, and only moved on when it worked.
Work was done in small steps: upload → processing → retrieval → LLM answers → chat → insights → metrics → async queue.

**What the assistant did**
- Proposed the 3-day plan, the architecture and the data model.
- Wrote the application code, the tests and the fictional sample policy documents.
- Explained concepts at each step (indexes, dependency injection, BM25, embeddings, Reciprocal Rank Fusion, Celery acknowledgement) and prepared interview questions.
- Helped debug from the real error messages I pasted.

**What I did**
- Installed and configured PostgreSQL, Python, Git and Ollama; created the database and `.env`.
- Ran every step, including the test suite, and exercised the API through Swagger with real documents and a real local model.
- Made the product decisions: use local Ollama until a company API key exists, go hybrid (BM25 + vectors) later, build RAG by hand to learn it, keep scope to a working subset.
- Reported failures back instead of accepting output on trust.

**Where the AI was wrong, and how it was caught.** These are real examples from this project:

| Problem | How it was found | Fix |
|---|---|---|
| The test suite would have called the real Ollama if `.env` said `AI_MODE=ollama` | Review of the test setup | Test client forces `ai_mode="mock"` |
| The LLM answered a "does it cover X?" question with `"supported": false` and `"answer": "no"`, so a correct negative answer was rejected | Running a real question through the local model and printing the raw output | Rewrote the prompt so `supported` means "a passage answers the question", not "the answer is yes" |
| A follow-up ("What about the vehicle policy?") was answered from the **health** policy, with a real, verified quote. Verification cannot catch an answer that is true but about the wrong document | Real two-turn chat in Swagger | The LLM rewrites follow-ups into standalone questions; keyword heuristic stays as fallback |
| The first follow-up heuristic treated any 2-word question as a follow-up | Failing tests | Narrowed the rule |
| A "skip finished jobs" guard was not actually covered by a test | Deliberately removing the guard and seeing no test fail | Strengthened the test; confirmed it now fails when the guard is removed |

Lesson: AI-generated code and prompts looked reasonable but needed running against real data to find the problems. Tests with fake models prove the plumbing; only real runs showed model behaviour.

**What I did not verify.** There is no systematic evaluation set for answer quality yet; real-model behaviour was checked by hand on a handful of questions. The Celery path with a live Redis was unit-tested with fakes.

---

## 2. AI inside the product

| Feature | Model | What it does | Control |
|---|---|---|---|
| Embeddings | `nomic-embed-text` (Ollama) | Turns chunks and questions into vectors for semantic search | Vector stored with the model name; only same-model vectors are compared |
| Question answering | `llama3.1` (Ollama) | Answers from numbered passages, returns JSON with exact quotes | Quotes verified word for word; numbers in the answer must appear in a quote; otherwise "not specified" |
| Follow-up rewriting | `llama3.1` | Turns "What about the vehicle policy?" into a standalone question | Only affects what is searched; never adds facts to an answer; falls back to a heuristic on bad output |
| Insights | `llama3.1` | Summary, key points, category, tags (length / focus / tone options) | Key points must be verbatim sentences; summary numbers must exist in the document; category from a fixed list; extractive fallback |

An offline **mock mode** (no AI, extractive) exists so the application and test suite run without a model. It is clearly labelled in every response (`generated_by`).

### Prompt engineering notes

Prompts are in `app/providers.py` (answering), `app/conversation.py` (rewriting) and `app/insights.py` (insights).

- **Ground and refuse.** "Use ONLY the numbered passages", "never guess numbers, dates, names or contact details", and an explicit way to say "not supported".
- **Machine-checkable output.** JSON only, with quotes that can be verified by string matching, instead of free text that has to be trusted.
- **Untrusted documents.** Document text sits inside `<context>` / `<document>` tags and the prompt says never to follow instructions found there. A test checks that document text never enters the system prompt.
- **Define ambiguous words.** The `supported` bug above came from a field name with two possible meanings. The final prompt defines it and gives a worked example ("No. Cosmetic procedures are excluded..." is a supported answer to "does it cover...?").
- **Full-sentence answers.** The model was returning `"no"`; the prompt now requires one complete sentence that restates the subject.
- **History is context only.** Earlier turns are for resolving words like "it"; the model must still answer the latest question from the passages in JSON.
- **Few-shot where it matters.** The rewrite prompt includes one example showing that a new subject replaces the old one instead of being combined with it.

### AI-driven automation
- Processing can automatically generate insights when a document completes (`AUTO_INSIGHTS=true`).
- Document category and tags are generated automatically.
- Follow-up questions are understood without the user restating context.

### Limits of the AI parts
- A local 8B-class model is weaker than a hosted frontier model; answer quality will change when the company API key is added (`AI_MODE=real` is **not implemented yet**).
- Verification proves a quote exists in the document, not that it is the right quote for the question.
- Summaries are checked for invented numbers, not for every kind of paraphrase error.
