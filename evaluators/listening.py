from utils.band import band_from_correct


def evaluate_listening(data):
    # -----------------------------
    # Input validation
    # -----------------------------
    if "user_answers" not in data or "answer_key" not in data:
        raise ValueError("user_answers and answer_key are required")

    user = data["user_answers"]
    key = data["answer_key"]

    total = len(key)
    correct = 0
    error_types = set()   # 👈 store error categories only

    # -----------------------------
    # Answer checking
    # -----------------------------
    for qid, ans in key.items():
        user_ans = user.get(qid)

        if user_ans == ans:
            correct += 1
        else:
            # Listening IELTS-style generic error classification
            error_types.add("Spelling / Distractor")

    # -----------------------------
    # Improvements based on errors
    # -----------------------------
    improvements = []

    if "Spelling / Distractor" in error_types:
        improvements.append(
            "Improve ability to catch specific details, spellings, and common distractors in listening sections."
        )

    # -----------------------------
    # Examiner feedback
    # -----------------------------
    band = band_from_correct(correct)

    examiner_feedback = (
        f"This is a Band {band} listening performance. "
        "Errors suggest difficulty with identifying precise details and managing distractors."
        if error_types else
        f"This is a Band {band} listening performance with a high level of accuracy."
    )

    # -----------------------------
    # Final response (FLAT FORMAT)
    # -----------------------------
    return {
        "module": "listening",
        "overall_band": band,
        "accuracy": f"{correct}/{total}",
        "improvements": improvements,
        "examiner_feedback": examiner_feedback
    }
