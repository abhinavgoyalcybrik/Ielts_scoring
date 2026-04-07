from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from evaluators.writing import evaluate_writing

router = APIRouter(prefix="/writing", tags=["Writing"])


# =========================
# REQUEST MODELS
# =========================
class WritingTask(BaseModel):
    question: str
    answer: str


class WritingRequest(BaseModel):
    task_1: WritingTask | None = None
    task_2: WritingTask


# =========================
# ENDPOINT
# =========================
@router.post("/evaluate")
def evaluate(data: WritingRequest):

    results = {}
    bands = []

    # =========================
    # TASK 1 (OPTIONAL)
    # =========================
    if data.task_1:
        try:
            r1 = evaluate_writing({
                "metadata": {
                    "task_type": "task_1",
                    "question": data.task_1.question
                },
                "user_answers": {
                    "text": data.task_1.answer
                }
            })

            results["task_1"] = r1
            bands.append(r1.get("overall_band", 0))

        except Exception as e:
            results["task_1"] = {"error": str(e)}

    # =========================
    # TASK 2 (REQUIRED)
    # =========================
    try:
        r2 = evaluate_writing({
            "metadata": {
                "task_type": "task_2",
                "question": data.task_2.question
            },
            "user_answers": {
                "text": data.task_2.answer
            }
        })

        results["task_2"] = r2
        bands.append(r2.get("overall_band", 0))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Task 2 evaluation failed: {str(e)}"
        )

    # =========================
    # OVERALL BAND CALCULATION
    # =========================
    if not bands:
        raise HTTPException(
            status_code=400,
            detail="No valid tasks evaluated"
        )

    overall = sum(bands) / len(bands)

    return {
        "module": "writing",
        "overall_writing_band": round(overall * 2) / 2,
        "tasks": results
    }