from fastapi import APIRouter
from evaluator import evaluate_attempt
import json

router = APIRouter(
    prefix="/speaking",
    tags=["Speaking"]
)

@router.post("/evaluate")

def evaluate_speaking_text(data: dict):
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
    
    # Debug: Log incoming data structure
    print(f"[DEBUG] Speaking API received data keys: {list(data.keys())}")
    
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
