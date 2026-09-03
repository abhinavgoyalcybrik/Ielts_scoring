from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from evaluators.reading import evaluate_reading
from utils.eval_log import log_evaluation

router = APIRouter(prefix="/reading", tags=["Reading"])


@router.post("/evaluate")
def evaluate_reading_api(data: Dict[str, Any]):
    try:
        result = evaluate_reading(data)
        # Reading is a deterministic, rule-based scorer (answer-key
        # matching, no GPT call anywhere in it) - "model"/"flags" are N/A
        # here, unlike Writing/Speaking. Still logged for the same audit/
        # calibration reasons (e.g. verifying which answer key was used on
        # a disputed score).
        log_evaluation({
            "evaluator": "reading",
            "task_or_part": data.get("test_type") or data.get("task_type"),
            "input": data,
            "response": result,
            "model": None,
            "flags": {},
        })
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Reading evaluation failed due to server error"
        )
