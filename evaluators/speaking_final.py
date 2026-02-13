from fastapi import APIRouter, HTTPException
from storage.speaking_store import SPEAKING_ATTEMPTS
from utils.band import calculate_final_speaking_band

router = APIRouter(prefix="/speaking", tags=["Speaking"])


@router.get("/final/{attempt_id}")
def get_final_speaking_result(attempt_id: str):
    if attempt_id not in SPEAKING_ATTEMPTS:
        raise HTTPException(status_code=404, detail="Invalid attempt_id")

    parts = SPEAKING_ATTEMPTS[attempt_id]["parts"]

    if not all(p in parts for p in [1, 2, 3]):
        raise HTTPException(
            status_code=400,
            detail="All three parts (1, 2, 3) are required"
        )

    final_band = calculate_final_speaking_band(parts)

    return {
        "attempt_id": attempt_id,
        "part_results": parts,
        "final_speaking_band": final_band
    }
