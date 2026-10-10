"""Multi-turn helpers: which earlier turns the model sees, and how a follow-up question is searched.

A follow-up like "What about the vehicle policy?" has almost no searchable words of its own. We keep
the heuristic simple and explainable: if the question looks like a follow-up, search with the previous
user question added. (A production system could ask the LLM to rewrite the question; that costs one
extra model call per turn and can itself hallucinate, so we start deterministic.)"""
import json
import re
from typing import List, Optional, Sequence

from app.ollama import OllamaClient
from app.providers import ChatTurn, ProviderError
from app.textutil import tokenize

MAX_HISTORY_MESSAGES = 6      # last 3 question/answer pairs
MAX_TURN_CHARS = 500          # keeps the prompt inside the model's context window

REFERENCE_WORDS = {"it", "its", "that", "those", "they", "them", "their", "he", "she", "same", "both", "another"}
OPENERS = ("and ", "what about", "how about", "but ", "then ", "so ")


def is_follow_up(question: str) -> bool:
    q = question.strip().lower()
    if q.startswith(OPENERS):
        return True
    if any(word in REFERENCE_WORDS for word in re.findall(r"[a-z']+", q)):
        return True
    # "Why?" has no searchable words. Two words can be a full question ("accident report?"), so only 0-1 count.
    return len(tokenize(question)) <= 1


def to_history(messages: Sequence) -> List[ChatTurn]:
    """Messages (oldest first, anything with .role/.content) -> the recent turns the model gets."""
    return [ChatTurn(m.role, m.content[:MAX_TURN_CHARS]) for m in list(messages)[-MAX_HISTORY_MESSAGES:]]


def build_retrieval_query(question: str, history: Sequence[ChatTurn]) -> str:
    if not history or not is_follow_up(question):
        return question
    previous_questions = [t.content for t in history if t.role == "user"]
    return f"{previous_questions[-1]} {question}" if previous_questions else question


# ----------------------------------------------------------------------------- LLM question rewriting

REWRITE_PROMPT = """You rewrite a chat follow-up into ONE standalone question that makes sense without the conversation.

Rules:
1. Resolve words like "it", "that" and "the other one" using the conversation.
2. If the latest question names a DIFFERENT subject (another policy, document or topic), replace the old subject
   with the new one. Never mix the old and new subjects together.
3. Keep every specific detail the user asked about. Do not answer the question. Do not add facts.
4. If the latest question is already standalone, return it unchanged.

Example:
Conversation: User: What is the waiting period in the health policy?
Latest question: What about the vehicle policy?
Reply: {"standalone_question": "What is the waiting period in the vehicle policy?"}

Reply with JSON only: {"standalone_question": "..."}"""


def clean_rewrite(raw: str) -> Optional[str]:
    """Accept the model's rewrite only if it is a sensible short string; otherwise the caller falls back."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    question = data.get("standalone_question") if isinstance(data, dict) else None
    if not isinstance(question, str):
        return None
    question = " ".join(question.split())
    return question if 3 <= len(question) <= 300 else None


class OllamaRewriter:
    """The rewritten question is only used to search and to phrase the question. Answers are still
    built from retrieved passages and verified quotes, so a bad rewrite can mislead retrieval but
    can never put an unsupported fact into an answer."""

    def __init__(self, client: OllamaClient, model: str):
        self.client, self.model = client, model

    def rewrite(self, question: str, history: Sequence[ChatTurn]) -> Optional[str]:
        transcript = "\n".join(f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}" for t in history)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Conversation so far:\n{transcript}\n\nLatest question: {question}"},
            ],
            "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": 2048},
        }
        try:
            data = self.client.request("POST", "/api/chat", payload, ProviderError, self.model)
        except ProviderError:
            return None   # the answering call will report a real outage; here we just fall back
        return clean_rewrite((data.get("message") or {}).get("content", ""))
