import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Configured once, here, at the application entrypoint - modules should
# never call logging.basicConfig() themselves. Without this, INFO-level
# logs anywhere in the app (including evaluators/speaking_audio.py's
# request/transcription/cache events) are silently suppressed, since
# Python's root logger defaults to WARNING and above only.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve project root and .env path

project_root = Path(__file__).parent.resolve()
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)

logger.info("OPENAI_API_KEY loaded: %s", os.getenv("OPENAI_API_KEY") is not None)

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from evaluators.api.writing import router as writing_router
from evaluators.api.reading import router as reading_router
from evaluators.api.listening import router as listening_router
from evaluators import speaking_audio
from evaluators.api.speaking import router as speaking_part_router
from evaluators.api.speaking_text import router as speaking_text_router

app = FastAPI(
    title="IELTS AI Evaluator API",
    description="AI-powered IELTS Writing, Reading, Listening & Speaking Evaluation API",
    version="1.0.0"
)


# Diagnostic only - a Pydantic validation failure (422) short-circuits
# BEFORE any endpoint body runs, so evaluators/api/writing.py's own
# request logging never fires for it. Logs the raw body only for
# /writing/* paths (scoped, not global) so a genuine schema mismatch from
# whatever real client is calling this API is visible here instead of
# only in that caller's own response. Response body/status code sent to
# the caller is completely unchanged - FastAPI's default 422 behavior.
@app.exception_handler(RequestValidationError)
async def log_writing_validation_errors(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/writing/"):
        raw_body = await request.body()
        logger.error(
            "422 on %s - errors=%s body=%s",
            request.url.path, exc.errors(), raw_body[:2000],
        )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# Include Routers
app.include_router(writing_router)
app.include_router(reading_router)
app.include_router(listening_router)
app.include_router(speaking_audio.router)
app.include_router(speaking_part_router)
app.include_router(speaking_text_router)

# Health Check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "IELTS AI Evaluator API"
    }

# Root
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "IELTS AI Evaluator API is live"
    }

if __name__ == "__main__":
    import uvicorn

    # A fixed default port, overridable via PORT - the previous approach
    # (bind a temp socket to port 0, read the OS-assigned port, close it,
    # then have uvicorn bind to that same port number) is a known race
    # condition on Windows: a just-closed socket doesn't always release
    # the port in time for immediate rebinding, which surfaces as
    # WinError 10013 (access forbidden), not a normal "port in use" error.
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting server on 127.0.0.1:%s", port)
    uvicorn.run(app, host="127.0.0.1", port=port)
