from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.errors import AIServiceError
from app.routers import chats, documents, insights, jobs, metrics, qa

app = FastAPI(title="Vault Document System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(documents.router)
app.include_router(qa.router)
app.include_router(chats.router)
app.include_router(insights.router)
app.include_router(metrics.router)
app.include_router(jobs.router)


@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request, exc: AIServiceError):
    """Embedding / LLM problems (Ollama down, model missing, bad config) become a clear 503."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "ai_mode": settings.ai_mode}
