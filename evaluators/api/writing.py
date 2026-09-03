import base64
import logging
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from evaluators.writing import evaluate_writing
from utils.ai_client import _writing_model
from utils.band import round_band
from utils.eval_log import log_evaluation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/writing", tags=["Writing"])


# =========================
# REQUEST MODELS
# =========================
class WritingTask1(BaseModel):
    question: str
    answer: str
    # Only meaningful for Academic Task 1 (a chart/graph/diagram/table) -
    # ignored for a General Training Task 1 letter, which has no image to
    # verify against (evaluate_writing() only forwards it to GPT when the
    # detected variant is "academic"). Optional so existing callers that
    # never send it are completely unaffected.
    image_url: str | None = None


class WritingTask2(BaseModel):
    question: str
    answer: str
    # No image_url field - Task 2 is always a pure essay response with no
    # source image to verify against, so it's not offered in the API
    # schema at all (not just ignored server-side).


class WritingRequest(BaseModel):
    task_1: WritingTask1 | None = None
    task_2: WritingTask2


# =========================
# TESTING HELPER - NO IMAGE STORAGE EXISTS YET
# =========================
# There is no image hosting/upload pipeline in this codebase - a
# candidate's Task 1 chart/graph/diagram has nowhere to be stored, so it
# can't be referenced by a real https:// URL the way WritingTask1.image_url
# expects. A file:// path (or any URL only reachable from the caller's own
# machine) can NEVER work here, because OpenAI's servers fetch image_url
# over the public internet, not the caller's local disk.
# This endpoint exists purely so an image can be tested end-to-end without
# building real storage: upload the file, get back a "data:image/...;
# base64,..." string, then paste that string into WritingTask1.image_url on
# a normal /writing/evaluate call - evaluate_writing()/call_gpt_writing()
# already accept a base64 data URL exactly like a hosted URL (OpenAI reads
# the embedded bytes directly, no network fetch involved). Nothing is
# persisted - the upload is read into memory, base64-encoded, and returned;
# the request body/response contract of /evaluate itself is unchanged.
@router.post("/task1-image-to-data-url")
async def task1_image_to_data_url(file: UploadFile = File(...)):
    content_type = file.content_type or "image/png"
    raw = await file.read()
    encoded = base64.b64encode(raw).decode("ascii")
    return {"image_url": f"data:{content_type};base64,{encoded}"}


# =========================
# ENDPOINT
# =========================
@router.post("/evaluate")
def evaluate(data: WritingRequest):
    # Diagnostic only - logs a short PREFIX of image_url (never the full
    # value, since a base64 data URL can be 100k+ chars) so a real caller's
    # actual payload is visible in the server log without guessing at
    # client code this repo doesn't contain. No behavior change.
    if data.task_1:
        img = data.task_1.image_url
        img_preview = f"{img[:60]}... (len={len(img)})" if img else None
        logger.info(
            "writing/evaluate: task_1 present, image_url=%s",
            img_preview,
        )
    else:
        logger.info("writing/evaluate: no task_1 in request")

    results = {}
    task1_band = None
    task2_band = None

    # =========================
    # TASK 1 (OPTIONAL)
    # =========================
    if data.task_1:
        try:
            _t1_start = time.perf_counter()
            r1 = evaluate_writing({
                "metadata": {
                    "task_type": "task_1",
                    "question": data.task_1.question,
                    "image_url": data.task_1.image_url
                },
                "user_answers": {
                    "text": data.task_1.answer
                }
            })
            _t1_latency = time.perf_counter() - _t1_start

            results["task_1"] = r1
            task1_band = r1.get("overall_band", 0)
            log_evaluation({
                "evaluator": "writing",
                "task_or_part": "task_1",
                "question": data.task_1.question,
                "input_text": data.task_1.answer,
                "response": r1,
                "model": _writing_model(),
                "flags": {"WRITING_INDEPENDENT_MODEL_ANSWER": os.getenv("WRITING_INDEPENDENT_MODEL_ANSWER", "false")},
                "latency_seconds": round(_t1_latency, 3),
            })

        except Exception as e:
            results["task_1"] = {"error": str(e)}

    # =========================
    # TASK 2 (REQUIRED)
    # =========================
    try:
        _t2_start = time.perf_counter()
        r2 = evaluate_writing({
            "metadata": {
                "task_type": "task_2",
                "question": data.task_2.question
            },
            "user_answers": {
                "text": data.task_2.answer
            }
        })
        _t2_latency = time.perf_counter() - _t2_start

        results["task_2"] = r2
        task2_band = r2.get("overall_band", 0)
        log_evaluation({
            "evaluator": "writing",
            "task_or_part": "task_2",
            "question": data.task_2.question,
            "input_text": data.task_2.answer,
            "response": r2,
            "model": _writing_model(),
            "flags": {"WRITING_INDEPENDENT_MODEL_ANSWER": os.getenv("WRITING_INDEPENDENT_MODEL_ANSWER", "false")},
            "latency_seconds": round(_t2_latency, 3),
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Task 2 evaluation failed: {str(e)}"
        )

    # =========================
    # OVERALL BAND CALCULATION
    # =========================
    if task1_band is None and task2_band is None:
        raise HTTPException(
            status_code=400,
            detail="No valid tasks evaluated"
        )

    # Official IELTS weighting: Task 2 counts for roughly twice as much of
    # the Writing score as Task 1 (Task 2 = 2/3, Task 1 = 1/3) - a straight
    # 1:1 average of the two task bands was used before, which is not how
    # IELTS actually combines them. Falls back to whichever task actually
    # has a real score if the other wasn't submitted or failed.
    if task1_band is not None and task2_band is not None:
        overall = (task1_band + 2 * task2_band) / 3
    else:
        overall = task1_band if task1_band is not None else task2_band

    return {
        "module": "writing",
        # round_band(), not Python's round(): round() uses round-half-to-
        # even, which silently rounds a .25/.75 average DOWN roughly half
        # the time (e.g. round(6.25*2)/2 = 6.0, when official IELTS
        # convention rounds .25 UP to 6.5) - the same rounding bug already
        # fixed elsewhere in this codebase (utils/band.py,
        # evaluators/speaking_audio.py's _ielts_round_half_up()), just not
        # yet applied here.
        "overall_writing_band": round_band(overall),
        "tasks": results
    }