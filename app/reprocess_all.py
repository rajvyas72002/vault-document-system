"""Re-run processing (chunking + embeddings) for every document. Run:  python -m app.reprocess_all
Use it after switching AI_MODE (for example mock -> ollama) so existing documents get embeddings."""
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.deps import get_embedder
from app.models import Document
from app.processing import process_document


def main() -> None:
    settings = get_settings()
    embedder = get_embedder(settings)
    print("Embedder:", embedder.model_name if embedder else "none (mock mode)")
    with SessionLocal() as db:
        for doc in db.scalars(select(Document).order_by(Document.created_at)).all():
            result = process_document(db, doc.id, settings, embedder)
            note = f" | {result.error_message}" if result.error_message else ""
            print(f"{result.status:10} {result.original_filename}{note}")


if __name__ == "__main__":
    main()
