"""Show how a phrase is stored in the database. Run:  python -m app.peek_chunk "least precise measurement"
Prints the stored chunk text around the phrase, exactly (repr), so glued words or odd characters are visible."""
import sys

import pymupdf
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Document, DocumentChunk


def main() -> None:
    phrase = sys.argv[1] if len(sys.argv) > 1 else "least precise measurement"
    print("PyMuPDF", pymupdf.__version__)
    with SessionLocal() as db:
        rows = db.execute(select(DocumentChunk, Document.original_filename).join(Document, Document.id == DocumentChunk.document_id)).all()
        shown = 0
        for chunk, name in rows:
            i = chunk.text.lower().find(phrase.lower())
            if i >= 0 and shown < 3:
                shown += 1
                print(f"\n{name} page {chunk.page_number} chunk {chunk.chunk_index}:")
                print(repr(chunk.text[max(0, i - 70): i + len(phrase) + 90]))
        if not shown:
            print("phrase not found in any stored chunk")


if __name__ == "__main__":
    main()
