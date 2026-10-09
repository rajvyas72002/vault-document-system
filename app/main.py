from fastapi import Depends, FastAPI

from app.config import Settings, get_settings
from app.routers import documents, qa

app = FastAPI(title="Vault Document System")
app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "ai_mode": settings.ai_mode}
