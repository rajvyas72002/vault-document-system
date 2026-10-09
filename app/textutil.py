"""Small text helpers shared by retrieval and the mock answerer. No external NLP libraries."""
import re
from typing import List

STOPWORDS = set("""
a an the is are was were be been am do does did of for to in on at by with from as and or
what which who whom when where why how should would could can may must will shall it its
this that these those any each per there their they you your we our i me my not non
many much long soon often compare comparison difference differences between two both vs versus
tell about please give state
""".split())


def stem(word: str) -> str:
    """Very light suffix stripping so 'reported', 'reporting' and 'report' match.
    Deliberately simple; it is NOT a real stemmer (see README limitations)."""
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 6 and word.endswith("ment"):
        word = word[:-4]
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 6 and word.endswith("al"):
        word = word[:-2]
    elif len(word) > 5 and word.endswith("ly"):
        word = word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    """lowercase -> words -> drop stopwords -> stem"""
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


_QUANTITY_QUESTION = re.compile(
    r"\bhow (many|much|long|soon|often|quickly|frequently)\b"
    r"|\b(period|window|deadline|duration|limit|number|percent|percentage|exact)\b",
    re.I,
)
_QUANTITY_ANSWER = re.compile(
    r"\d|\b(quarterly|quarter|monthly|weekly|daily|annual|annually|yearly|hourly|immediately|"
    r"every|twice|once|within)\b",
    re.I,
)


def expects_quantity(question: str) -> bool:
    """True if the question asks for a number, duration or frequency."""
    return bool(_QUANTITY_QUESTION.search(question))


def has_quantity(sentence: str) -> bool:
    return bool(_QUANTITY_ANSWER.search(sentence))


def wants_comparison(question: str) -> bool:
    return bool(re.search(r"\b(compare|comparison|difference|differences|versus|vs)\b|\bbetween\b.*\band\b", question, re.I))
