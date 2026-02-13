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

    # 🔑 RULE: attempt_id generate ONLY for Part 1
    if part == 1:
        attempt_id = attempt_id or uuid4().hex
    else:
        if not attempt_id:
            raise HTTPException(
                status_code=400,
                detail="attempt_id is required for Part 2 and Part 3"
            )
        if attempt_id not in SPEAKING_ATTEMPTS:
            raise HTTPException(status_code=400, detail="Invalid attempt_id")

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

    # ---- STORE RESULT (SAME attempt_id) ----
    SPEAKING_ATTEMPTS[attempt_id]["parts"][part] = result

    return {
        "attempt_id": attempt_id,
        "part": part,
        "result": result
    }
