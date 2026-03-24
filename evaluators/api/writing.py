from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from evaluators.writing import evaluate_writing

router = APIRouter(prefix="/writing", tags=["Writing"])


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip()).strip()
    return str(value).strip()


def _extract_task(raw_payload: dict, keys: tuple[str, ...], *, allow_top_level_answer: bool = False):
    raw_task = None
    for key in keys:
        if key in raw_payload:
            raw_task = raw_payload.get(key)
            break

    if raw_task is None and allow_top_level_answer and any(
        key in raw_payload
        for key in ("answer", "response", "essay", "essay_text", "text", "user_answer", "userAnswer")
    ):
        raw_task = raw_payload

    if not isinstance(raw_task, dict):
        return None

    question = _coerce_text(
        raw_task.get("question")
        or raw_task.get("prompt")
        or raw_task.get("task_question")
        or raw_task.get("taskQuestion")
    )
    answer = _coerce_text(
        raw_task.get("answer")
        or raw_task.get("response")
        or raw_task.get("essay")
        or raw_task.get("essay_text")
        or raw_task.get("text")
        or raw_task.get("user_answer")
        or raw_task.get("userAnswer")
    )

    if not question and not answer:
        return None
    return {"question": question, "answer": answer}


@router.post("/evaluate")
async def evaluate(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Input should be a valid dictionary")

    task_1 = _extract_task(payload, ("task_1", "task1"))
    task_2 = _extract_task(payload, ("task_2", "task2"), allow_top_level_answer=True)
    if not task_2:
        raise HTTPException(status_code=422, detail="task_2 is required")
    if not task_2.get("answer"):
        raise HTTPException(status_code=422, detail="task_2.answer is required")

    results = {}
    bands = []

    if task_1:
        if not task_1.get("answer"):
            results["task_1"] = {"error": "task_1.answer is required"}
        else:
            try:
                r1 = evaluate_writing(
                    {
                        "metadata": {"task_type": "task_1", "question": task_1["question"]},
                        "user_answers": {"text": task_1["answer"]},
                    }
                )
                results["task_1"] = r1
                bands.append(float(r1.get("overall_band", 0)))
            except Exception as e:
                results["task_1"] = {"error": str(e)}

    try:
        r2 = evaluate_writing(
            {
                "metadata": {"task_type": "task_2", "question": task_2["question"]},
                "user_answers": {"text": task_2["answer"]},
            }
        )
        results["task_2"] = r2
        bands.append(float(r2.get("overall_band", 0)))
    except ValueError as e:
        if "essay text missing" in str(e).lower():
            raise HTTPException(status_code=422, detail="task_2.answer is required")
        raise HTTPException(status_code=422, detail=f"Task 2 validation failed: {e}")
    except Exception as e:
        raise HTTPException(500, f"Task 2 failed: {e}")

    if not bands:
        raise HTTPException(status_code=500, detail="No writing tasks could be evaluated")

    overall = sum(bands) / len(bands)

    return {
        "module": "writing",
        "overall_writing_band": round(overall * 2) / 2,
        "tasks": results
    }
