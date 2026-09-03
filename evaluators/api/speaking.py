from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from uuid import uuid4

from utils.audio_transcriber import transcribe_audio
from utils.audio_features import compute_speech_rate_wpm, extract_audio_features
from utils.audio_normalizer import normalize_to_wav
from utils.eval_log import log_evaluation
from evaluators.speaking import evaluate_speaking_part
from storage.speaking_store import SPEAKING_ATTEMPTS

router = APIRouter(prefix="/speaking", tags=["Speaking"])


@router.post("/part/{part}/audio")
async def upload_speaking_audio(
    part: int,
    file: UploadFile = File(...),
    attempt_id: str | None = Form(None)
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
    wav_path = normalize_to_wav(file)
    transcript = transcribe_audio(wav_path)
    audio_metrics = extract_audio_features(wav_path)

    # Speech rate (WPM) - both denominators always computed (see
    # compute_speech_rate_wpm), "active" selects the one that actually
    # drives evaluate_speaking_part()'s scoring, per SPEAKING_LEGACY_VOICED_WPM.
    wpm = compute_speech_rate_wpm(transcript, audio_metrics)
    audio_metrics["speech_rate_wpm_raw"] = wpm["raw"]
    audio_metrics["speech_rate_wpm_voiced"] = wpm["voiced"]
    audio_metrics["speech_rate_wpm"] = wpm["active"]

    # ---- EVALUATION ----
    result = evaluate_speaking_part(
        part=part,
        transcript=transcript,
        audio_metrics=audio_metrics
    )

    # ---- STORE RESULT (SAME attempt_id) ----
    SPEAKING_ATTEMPTS[attempt_id]["parts"][part] = result

    # Logged under a distinct evaluator name (not "speaking") - this is a
    # different scoring engine (evaluators/speaking.py's
    # evaluate_speaking_part, not speaking_audio.py's generate_scores) from
    # a different live endpoint, and the log needs to be able to tell them
    # apart to answer which endpoint real traffic actually uses.
    log_evaluation({
        "evaluator": "speaking_legacy_part_audio",
        "task_or_part": f"part_{part}",
        "question": None,
        "input_text": transcript,
        "response": result,
        "model_default": "gpt-4o-mini",
        "flags": {},
        "audio_metrics": audio_metrics,
    })

    return {
        "attempt_id": attempt_id,
        "part": part,
        "result": result
    }
