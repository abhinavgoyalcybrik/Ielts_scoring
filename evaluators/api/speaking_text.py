from fastapi import APIRouter, Request, HTTPException
from evaluator import evaluate_attempt
from evaluators.speaking import evaluate_speaking_part
from storage.speaking_store import SPEAKING_ATTEMPTS
from utils.audio_transcriber import transcribe_audio
from utils.audio_features import extract_audio_features
from uuid import uuid4

router = APIRouter(
    prefix="/speaking",
    tags=["Speaking"]
)

@router.post("/evaluate")
async def evaluate_speaking_text(request: Request):
    """
    TEXT-based Speaking Evaluation - PART-WISE ASSESSMENT
    Accepts multiple input formats and returns all 3 parts evaluated together.
    
    Input formats supported:
    1. Answers format: { "part_1": {"answers": [...]}, "part_2": {"answer": "..."}, "part_3": {"answers": [...]} }
    2. Transcript format: { "part_1": {"transcript": "..."}, ... }
    3. Nested: { "speaking": { "part_1": {...}, ... } }
    4. Single: { "transcript": "...", "part": 1 }
    
    Returns: Complete module assessment with part_1, part_2, part_3, overall_band, cefr_level, vocabulary_to_learn
    """
    
    content_type = request.headers.get("content-type", "")

    # Handle legacy multipart uploads sent to /speaking/evaluate
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="No audio file provided")

        try:
            part = int(form.get("part", 1))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid part number")

        attempt_id = form.get("attempt_id")
        if part not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="Invalid part number")
        attempt_id = (attempt_id or "").strip() or uuid4().hex

        try:
            transcript = transcribe_audio(upload)
            audio_metrics = extract_audio_features(upload)

            # Speech rate (WPM)
            words = len(transcript.split())
            duration = audio_metrics.get("duration_sec", 1)
            speech_rate = round((words / duration) * 60) if duration > 0 else 0
            audio_metrics["speech_rate_wpm"] = speech_rate

            result = evaluate_speaking_part(
                part=part,
                transcript=transcript,
                audio_metrics=audio_metrics
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio evaluation failed for part {part}: {e}")

        SPEAKING_ATTEMPTS[attempt_id]["parts"][part] = result

        return {
            "attempt_id": attempt_id,
            "part": part,
            "result": result
        }

    # JSON payload handling
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Debug: Log incoming data structure
    if isinstance(data, dict):
        print(f"[DEBUG] Speaking API received data keys: {list(data.keys())}")
    else:
        raise HTTPException(status_code=400, detail="Input should be a valid dictionary")
    
    def normalize_part_data(part_data):
        """Convert 'answers'/'answer' to 'transcript' and ensure audio_metrics exists"""
        if not part_data:
            return None
        
        normalized = dict(part_data) if isinstance(part_data, dict) else {}
        
        # Handle 'answers' array format (Part 1, Part 3)
        if "answers" in normalized and not "transcript" in normalized:
            if isinstance(normalized["answers"], list):
                normalized["transcript"] = " ".join(str(a) for a in normalized["answers"] if a)
                print(f"[DEBUG] Converted 'answers' array to transcript: {normalized['transcript'][:50]}")
            del normalized["answers"]
        
        # Handle 'answer' string format (Part 2)
        elif "answer" in normalized and not "transcript" in normalized:
            normalized["transcript"] = normalized.get("answer", "")
            print(f"[DEBUG] Converted 'answer' string to transcript: {normalized['transcript'][:50]}")
            del normalized["answer"]
        
        # Ensure audio_metrics exists
        if "audio_metrics" not in normalized:
            normalized["audio_metrics"] = {}
        
        return normalized if normalized.get("transcript") else None
    
    # Normalize input to standard format with part_1, part_2, part_3
    eval_data = {
        "test_type": "speaking",
        "part_1": None,
        "part_2": None,
        "part_3": None
    }
    
    # Format 1: Direct part_N keys with 'answers'/'answer' format (PRIMARY)
    direct_parts = [k for k in ["part_1", "part_2", "part_3"] if k in data and data[k]]
    if direct_parts:
        print(f"[DEBUG] Attempting to normalize direct parts: {direct_parts}")
        for part_key in ["part_1", "part_2", "part_3"]:
            if part_key in data and data[part_key]:
                normalized = normalize_part_data(data[part_key])
                if normalized:
                    eval_data[part_key] = normalized
                    # Extract time_seconds if present (for Part 2 WPM calculation)
                    if "time_seconds" in data[part_key]:
                        eval_data[part_key]["time_seconds"] = data[part_key]["time_seconds"]
                    print(f"[DEBUG] Successfully normalized {part_key}")
    
    # Format 2: Nested under "speaking" key
    if not any(eval_data[k] for k in ["part_1", "part_2", "part_3"]):
        if "speaking" in data and isinstance(data["speaking"], dict):
            print(f"[DEBUG] Attempting nested 'speaking' format")
            speaking_data = data["speaking"]
            for part_key in ["part_1", "part_2", "part_3"]:
                if part_key in speaking_data and speaking_data[part_key]:
                    normalized = normalize_part_data(speaking_data[part_key])
                    if normalized:
                        eval_data[part_key] = normalized
                        print(f"[DEBUG] Successfully normalized nested {part_key}")
    
    # Format 3: Single part (legacy) - transcript and part at top level
    if not any(eval_data[k] for k in ["part_1", "part_2", "part_3"]):
        if "transcript" in data:
            single_part = data.get("part", 1)
            print(f"[DEBUG] Detected Format 3 (single part {single_part})")
            part_key = f"part_{single_part}"
            part_data = {
                "transcript": data["transcript"],
                "audio_metrics": data.get("audio_metrics", {})
            }
            eval_data[part_key] = part_data
        else:
            # No transcript-like data found
            print(f"[DEBUG] No recognized transcript/answers format found")
            eval_data["part_1"] = {"transcript": "", "audio_metrics": {}}
    
    # Ensure at least part_1 exists with audio_metrics
    if not any(eval_data[k] for k in ["part_1", "part_2", "part_3"]):
        print(f"[DEBUG] No parts found, creating empty part_1")
        eval_data["part_1"] = {"transcript": "", "audio_metrics": {}}
    
    parts_summary = [(k, 'has_content' if eval_data[k] and eval_data[k].get('transcript') else 'empty') for k in ['part_1', 'part_2', 'part_3']]
    print(f"[DEBUG] Final eval_data parts: {parts_summary}")
    
    # Call evaluate_attempt with all parts data
    return evaluate_attempt(eval_data)
