from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pymupdf  # PyMuPDF
from docx import Document as DocxDocument


class ExtractionError(Exception):
    """A problem we can safely explain to the user."""


@dataclass
class PageText:
    page_number: Optional[int]  # 1-based for PDFs; None for DOCX (Word has no fixed pages)
    text: str


def extract_pages(path: str) -> List[PageText]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    raise ExtractionError(f"Unsupported file type '{ext}'.")


def _extract_pdf(path: str) -> List[PageText]:
    with pymupdf.open(path) as pdf:
        return [PageText(i + 1, page.get_text()) for i, page in enumerate(pdf)]


def _extract_docx(path: str) -> List[PageText]:
    doc = DocxDocument(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return [PageText(None, "\n".join(parts))]
