
# --- Explicitly load .env at startup ---
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root and .env path
project_root = Path(__file__).parent.resolve()
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Debug print: was the key loaded?
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY") is not None)

from fastapi import FastAPI

from evaluators.api.writing import router as writing_router
from evaluators.api.reading import router as reading_router
from evaluators.api.listening import router as listening_router
from evaluators.api.speaking_text import router as speaking_router
from evaluators.api.speaking import router as speaking_audio_router

app = FastAPI(
    title="IELTS AI Evaluator API",
    description="AI-powered IELTS Writing, Reading, Listening & Speaking Evaluation API",
    version="1.0.0"
)

# --------------------
# Include Routers
# --------------------
app.include_router(writing_router)
app.include_router(reading_router)
app.include_router(listening_router)
app.include_router(speaking_router)
app.include_router(speaking_audio_router)

# --------------------
# Health Check
# --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "IELTS AI Evaluator API"
    }

# --------------------
# Root
# --------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "IELTS AI Evaluator API is live"
    }

if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Dynamically find an available port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    print(f"Starting server on 127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
