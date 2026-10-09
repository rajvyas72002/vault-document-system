class AIServiceError(Exception):
    """A problem with an AI service (embedding model, LLM, configuration). Shown to clients as HTTP 503."""
