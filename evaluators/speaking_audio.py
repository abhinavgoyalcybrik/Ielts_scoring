from fastapi import APIRouter, UploadFile, File
from utils.audio_normalizer import normalize_to_wav
from utils.audio_transcriber import transcribe_audio
from utils.audio_features import extract_audio_features
from evaluators.speaking import evaluate_speaking_part

router = APIRouter(prefix="/speaking", tags=["Speaking"])


def validate_part_duration(part: int, duration: float):
    rules = {
        1: (10, 30),
        2: (60, 120),
        3: (30, 60)
    }
    min_d, max_d = rules.get(part, (0, 999))
    return min_d <= duration <= max_d


@router.post("/part/{part}/audio")
async def upload_speaking_audio(part: int, file: UploadFile = File(...)):
    # ✅ SAVE AUDIO ONCE
    wav_path = normalize_to_wav(file)

    # ✅ REUSE WAV PATH
    transcript = transcribe_audio(wav_path)
    audio_metrics = extract_audio_features(wav_path)

    # Speech rate
    words = len(transcript.split())
    duration = audio_metrics["duration_sec"]
    audio_metrics["speech_rate_wpm"] = round((words / duration) * 60) if duration > 0 else 0

    # Duration validation
    audio_metrics["duration_valid"] = validate_part_duration(part, duration)

    result = evaluate_speaking_part(
        part=part,
        transcript=transcript,
        audio_metrics=audio_metrics
    )

    return {
        "part": part,
        "transcript": transcript,
        "audio_metrics": audio_metrics,
        "result": result
    }
