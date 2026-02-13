from fastapi import APIRouter, HTTPException
from evaluators.listening import evaluate_listening

router = APIRouter(prefix="/listening", tags=["Listening"])

@router.post("/evaluate")
def evaluate_listening_api(data: dict):
    try:
        result = evaluate_listening(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
