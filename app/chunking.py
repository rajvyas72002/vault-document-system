import re
from dataclasses import dataclass
from typing import List, Optional

from app.extraction import PageText


@dataclass
class Chunk:
    chunk_index: int
    page_number: Optional[int]
    text: str


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sliding window over the text. Cuts at a space when possible so words stay whole."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = re.sub(r"\s+", " ", text).strip()
    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always move forward
        nxt = text.find(" ", start)            # start the next chunk on a word boundary
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
