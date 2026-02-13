from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from evaluators.reading import evaluate_reading

router = APIRouter(prefix="/reading", tags=["Reading"])


@router.post("/evaluate")
def evaluate_reading_api(data: Dict[str, Any]):
    try:
        return evaluate_reading(data)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Reading evaluation failed due to server error"
        )
