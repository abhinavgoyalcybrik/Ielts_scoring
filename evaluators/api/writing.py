from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from evaluators.writing import evaluate_writing

router = APIRouter(prefix="/writing", tags=["Writing"])


class WritingTask(BaseModel):
    question: str
    answer: str


class WritingRequest(BaseModel):
    task_1: WritingTask | None = None
    task_2: WritingTask


@router.post("/evaluate")
def evaluate(data: WritingRequest):

    results = {}
    bands = []

    if data.task_1:
        try:
            r1 = evaluate_writing({
                "metadata": {"task_type": "task_1", "question": data.task_1.question},
                "user_answers": {"text": data.task_1.answer}
            })
            results["task_1"] = r1
            bands.append(r1["overall_band"])
        except Exception as e:
            results["task_1"] = {"error": str(e)}

    try:
        r2 = evaluate_writing({
            "metadata": {"task_type": "task_2", "question": data.task_2.question},
            "user_answers": {"text": data.task_2.answer}
        })
        results["task_2"] = r2
        bands.append(r2["overall_band"])
    except Exception as e:
        raise HTTPException(500, f"Task 2 failed: {e}")

    overall = sum(bands) / len(bands)

    return {
        "module": "writing",
        "overall_writing_band": round(overall * 2) / 2,
        "tasks": results
    }
