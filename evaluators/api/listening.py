from fastapi import APIRouter, HTTPException
from evaluators.listening import evaluate_listening
from utils.eval_log import log_evaluation

router = APIRouter(prefix="/listening", tags=["Listening"])

@router.post("/evaluate")
def evaluate_listening_api(data: dict):
    try:
        result = evaluate_listening(data)
        # Listening is deterministic (answer-key matching, no GPT call) -
        # same "model"/"flags" N/A reasoning as reading.py.
        log_evaluation({
            "evaluator": "listening",
            "task_or_part": data.get("test_type") or data.get("task_type"),
            "input": data,
            "response": result,
            "model": None,
            "flags": {},
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
