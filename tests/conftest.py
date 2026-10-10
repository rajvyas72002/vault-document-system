import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models  # noqa: F401
from app.config import Settings, get_settings
from app.db import Base, get_db
from app.main import app


@pytest.fixture()
def client(tmp_path):
    """Isolated app: in-memory SQLite + temp storage folder. No PostgreSQL, no Ollama.

    ai_mode is forced to "mock" so a developer's .env (for example AI_MODE=ollama) can never
    make the test suite call a real model. task_backend is forced to "inline" for the same reason
    (a .env with TASK_BACKEND=celery must not make tests need Redis)."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_settings] = lambda: Settings(
        storage_dir=str(tmp_path / "storage"), max_upload_mb=1, ai_mode="mock", task_backend="inline", auto_insights=False
    )
    test_client = TestClient(app)
    test_client.session_factory = TestSession  # lets tests inspect the database directly
    yield test_client
    app.dependency_overrides.clear()
