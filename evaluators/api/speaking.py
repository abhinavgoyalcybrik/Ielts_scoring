from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import uuid4

from utils.audio_transcriber import transcribe_audio
from utils.audio_features import extract_audio_features
from evaluators.speaking import evaluate_speaking_part
from storage.speaking_store import SPEAKING_ATTEMPTS

router = APIRouter(prefix="/speaking", tags=["Speaking"])


@router.post("/part/{part}/audio")
async def upload_speaking_audio(
    part: int,
    file: UploadFile = File(...),
    attempt_id: str | None = None
):
    if part not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="Invalid part number")

    # Keep attempt_id optional and resilient across stateless/multi-instance deployments.
    attempt_id = (attempt_id or "").strip() or uuid4().hex

    try:
        # ---- AUDIO PROCESSING ----
        transcript = transcribe_audio(file)
        audio_metrics = extract_audio_features(file)

        # Speech rate (WPM)
        words = len(transcript.split())
        duration = audio_metrics.get("duration_sec", 1)
        speech_rate = round((words / duration) * 60) if duration > 0 else 0
        audio_metrics["speech_rate_wpm"] = speech_rate

        # ---- EVALUATION ----
        result = evaluate_speaking_part(
            part=part,
            transcript=transcript,
            audio_metrics=audio_metrics
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio evaluation failed for part {part}: {e}")

    # ---- STORE RESULT (best effort; non-critical for client-side aggregation) ----
    SPEAKING_ATTEMPTS[attempt_id]["parts"][part] = result

    return {
        "attempt_id": attempt_id,
        "part": part,
        "result": result
    }
