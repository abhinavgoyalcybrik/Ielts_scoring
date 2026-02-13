from utils.band import band_from_correct


def evaluate_reading(data: dict):
    # =========================
    # SAFE EXTRACTION
    # =========================
    questions = data.get("questions", [])
    user_answers = data.get("user_answers", {})

    if not questions or not user_answers:
        raise ValueError("Invalid reading input format")

    correct = 0
    wrong_question_types = set()  # 👈 store types of wrong answers

    # =========================
    # EVALUATION LOGIC
    # =========================
    for q in questions:
        qid = q.get("question_id")
        answer_key = q.get("answer_key")
        qtype = q.get("type", "UNKNOWN")

        if not qid or answer_key is None:
            continue

        user_ans = user_answers.get(qid)

        if user_ans == answer_key:
            correct += 1
        else:
            wrong_question_types.add(qtype)

    # =========================
    # BAND CALCULATION
    # =========================
    band = band_from_correct(correct)

    # =========================
    # IMPROVEMENTS BASED ON QUESTION TYPE
    # =========================
    improvements = []

    for qtype in wrong_question_types:
        if qtype == "TRUE_FALSE_NOT_GIVEN":
            improvements.append(
                "Improve ability to distinguish clearly between TRUE, FALSE and NOT GIVEN statements."
            )
        elif qtype == "MCQ":
            improvements.append(
                "Practise multiple-choice questions by identifying distractors more carefully."
            )
        elif qtype == "FILL_IN_THE_BLANKS":
            improvements.append(
                "Work on scanning and word-matching skills for fill in the blanks questions."
            )
        else:
            improvements.append(
                f"Improve accuracy in {qtype} type reading questions."
            )

    # =========================
    # IELTS-STYLE EXAMINER FEEDBACK
    # =========================
    examiner_feedback = (
        f"This is a Band {band} reading performance. "
        "Errors were observed in specific question types, indicating areas "
        "where targeted practice is required."
        if wrong_question_types else
        f"This is a Band {band} reading performance with a high level of accuracy."
    )

    # =========================
    # FINAL RESPONSE (NO MISTAKES ARRAY)
    # =========================
    return {
        "module": "reading",
        "overall_band": band,
        "accuracy": f"{correct}/{len(questions)}",
        "improvements": improvements,
        "examiner_feedback": examiner_feedback
    }
