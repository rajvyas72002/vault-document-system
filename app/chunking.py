import re
from dataclasses import dataclass
from typing import List, Optional

from app.extraction import PageText


@dataclass
class Chunk:
    chunk_index: int
    page_number: Optional[int]
    text: str


def normalize_text(text: str) -> str:
    """Collapse spaces and line wraps inside a paragraph, but keep blank-line paragraph breaks."""
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return "\n\n".join(p for p in paragraphs if p)


def _last_space(text: str, lo: int, hi: int) -> int:
    return max(text.rfind(" ", lo, hi), text.rfind("\n", lo, hi))


def _next_space(text: str, start: int) -> int:
    candidates = [i for i in (text.find(" ", start), text.find("\n", start)) if i != -1]
    return min(candidates) if candidates else -1


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sliding window over the text. Cuts at a space when possible so words stay whole."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = normalize_text(text)
    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            cut = _last_space(text, start + chunk_size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always move forward
        nxt = _next_space(text, start)         # start the next chunk on a word boundary
        if nxt != -1 and nxt < end:
            start = nxt + 1
    return pieces


def chunk_pages(pages: List[PageText], chunk_size: int, overlap: int) -> List[Chunk]:
    """Chunk page by page so every chunk belongs to exactly one page (clean citations)."""
    chunks: List[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(len(chunks), page.page_number, piece))
    return chunks
