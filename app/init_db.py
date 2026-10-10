from sqlalchemy import inspect, text

from app import models  # noqa: F401  (importing registers the tables on Base)
from app.db import Base, engine


def main() -> None:
    with engine.connect() as conn:
        print("Connected to:", conn.execute(text("SELECT version()")).scalar())
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        # create_all never alters an existing table. This tiny idempotent step adds the embedding
        # columns to a table made earlier. A real project would use Alembic migrations instead.
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding JSON"))
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100)"))
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS latency_ms INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_model ON document_chunks (embedding_model)"))
    print("Tables:", inspect(engine).get_table_names())


if __name__ == "__main__":
    main()
