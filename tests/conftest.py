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
    """Isolated app: in-memory SQLite + temp storage folder. No PostgreSQL needed."""
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
    app.dependency_overrides[get_settings] = lambda: Settings(storage_dir=str(tmp_path / "storage"), max_upload_mb=1)
    yield TestClient(app)
    app.dependency_overrides.clear()
