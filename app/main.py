from fastapi import Depends, FastAPI
from app.config import Settings, get_settings

app = FastAPI(title="Vault Document System")


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "ai_mode": settings.ai_mode}
