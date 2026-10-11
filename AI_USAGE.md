# AI Usage

This project uses AI in two different ways, and they are described separately:

1. **AI inside the product**: how the system itself uses language models, and how its output is controlled.
2. **AI as a development tool**: how AI assistants helped me build it.

---

## 1. AI inside the product

| Feature | Model | What it does | Control |
|---|---|---|---|
| Embeddings | `text-embedding-3-small` (OpenAI, via LangChain) or `nomic-embed-text` (Ollama) | Turns chunks and questions into vectors for semantic search | Vector stored with the model name; only same-model vectors are compared |
| Question answering | `gpt-4o` (OpenAI, via LangChain `ChatOpenAI`) or `llama3.1` (Ollama) | Answers from numbered passages, returns JSON with exact quotes | Quotes must exist in the retrieved text; every number must be traceable (see below); otherwise "not specified" |
| Follow-up rewriting | same model as the answers (`gpt-4o` with OpenAI, `llama3.1` with Ollama) | Turns "What about the vehicle policy?" into a standalone question | Only affects what is searched; never adds facts to an answer; falls back to a heuristic on bad output |
| Insights | `llama3.1` (Ollama) | Summary, key points, category, tags (length / focus / tone options) | Key points must be verbatim sentences; summary numbers must exist in the document; category from a fixed list; extractive fallback |

An offline **mock mode** (no AI, extractive) exists so the application and test suite run without a model. It is clearly labelled in every response (`generated_by`).

**Number rule.** A number in an answer must appear in a cited passage, on the same page as a cited passage, in the question, or be the difference of two quoted figures.

### Prompt engineering notes

Prompts are in `app/providers.py` (answering), `app/conversation.py` (rewriting) and `app/insights.py` (insights).

- **Ground and refuse.** "Use ONLY the numbered passages", "never guess numbers, dates, names or contact details", and an explicit way to say "not supported".
- **Machine-checkable output.** JSON only, with quotes that can be verified by string matching, instead of free text that has to be trusted.
- **Untrusted documents.** Document text sits inside `<context>` / `<document>` tags and the prompt says never to follow instructions found there. A test checks that document text never enters the system prompt.
- **Define ambiguous words.** A `supported` bug came from a field name with two possible meanings. The final prompt defines it and gives a worked example.
- **Completeness and definitions.** When a passage lists items or steps, the model must name all of them; it must not answer a definition question with a circular sentence.
- **Full-sentence answers.** The model was returning `"no"`; the prompt now requires one complete sentence that restates the subject.
- **History is context only.** Earlier turns are for resolving words like "it"; the model must still answer the latest question from the passages in JSON.
- **Few-shot where it matters.** The rewrite prompt includes one example showing that a new subject replaces the old one.

### AI-driven automation
- Processing can automatically generate insights when a document completes (`AUTO_INSIGHTS=true`).
- Document category and tags are generated automatically.
- Follow-up questions are understood without the user restating context.

### Measured quality
`python -m app.evaluate` runs 60 questions (see README, Evaluation). With `gpt-4o` and OpenAI embeddings it scored 60/60; `gpt-4o-mini` 58 to 59/60. The question set is small and the CBSE part was partly tuned. The local `llama3.1` with `nomic-embed-text` scored 52/60 (about 17.6 s per answer on a laptop). Side-by-side card and chart: `docs/model_comparison_card.png`.

### Limits of the AI parts
- Verification proves a quote exists in the document, not that it is the right quote for the question. We saw this with a wrong "joule and newton" answer.
- A local 8B-class model is weaker and much slower than the hosted one.
- Summaries are checked for invented numbers, not for every kind of paraphrase error.
- Thumbs up / down feedback is collected and shown in Metrics, but it is not used to retrain or tune anything.

---

## 2. AI as a development tool

I used **Claude** as a coding and design assistant (planning, code, tests, debugging) and **ChatGPT** for background reading on the case study and on vector search. I ran every step on my own machine and checked the results with real documents, a real model and the evaluation set before moving on.
