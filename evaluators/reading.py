from utils.band import band_from_correct, general_reading_band, get_band_skill_description


# Academic and General Training Reading use two COMPLETELY SEPARATE raw-
# score -> band mappings (band_from_correct() / general_reading_band(),
# each backed by its own named constant in utils/band.py) that must never
# be mixed. The test type is resolved into exactly one of these two
# canonical values HERE, before any band lookup happens, so the routing
# below is a plain, exhaustive check with no fallback between the two
# maps. "general" is accepted as an alias for "general_training" (the
# value this project used before this canonical spelling was introduced)
# so an existing caller is never silently rescored as Academic just
# because the wire format's canonical spelling changed.
_GENERAL_TRAINING_TEST_TYPE_ALIASES = {"general", "general_training", "generaltraining", "gt"}


def _normalize_reading_test_type(raw_test_type) -> str:
    key = str(raw_test_type or "").strip().casefold().replace(" ", "_").replace("-", "_")
    if key in _GENERAL_TRAINING_TEST_TYPE_ALIASES:
        return "general_training"
    return "academic"


def _answers_match(user_ans, correct_ans) -> bool:
    """Case- and whitespace-insensitive comparison, per real IELTS Reading
    marking convention: capitalisation is not penalised (e.g. "true",
    "True", and "TRUE" are all accepted for a TRUE/FALSE/NOT GIVEN
    answer, and a fill-in-the-blank answer typed in the wrong case is
    still correct) - spelling must still be exact. A plain `==` used to
    mark a fully correct answer wrong for nothing more than a different
    case or a trailing space.

    `correct_ans` may be a single acceptable answer OR a list/tuple/set of
    several acceptable answers for the same question (e.g. British vs
    American spelling, or a few accepted phrasings) - the user's answer is
    correct if it matches ANY one of them, each compared with the same
    case/whitespace-insensitive rule."""
    if user_ans is None or correct_ans is None:
        return False
    user_norm = str(user_ans).strip().casefold()
    if not user_norm:
        return False

    if isinstance(correct_ans, (list, tuple, set)):
        return any(
            alt is not None and user_norm == str(alt).strip().casefold()
            for alt in correct_ans
        )

    return user_norm == str(correct_ans).strip().casefold()


def evaluate_reading(data: dict):
    # =========================
    # SAFE EXTRACTION
    # =========================
    questions = data.get("questions", [])
    user_answers = data.get("user_answers", {})
    test_type = _normalize_reading_test_type(data.get("test_type"))

    if not questions or not user_answers:
        raise ValueError("Invalid reading input format")

    correct = 0
    wrong_question_types = set()

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

        if _answers_match(user_ans, answer_key):
            correct += 1
        else:
            wrong_question_types.add(qtype)

    # =========================
    # BAND CALCULATION - test type determined FIRST (above), then route
    # to ONLY the corresponding mapping. Explicit if/elif with no
    # implicit else-fallback between the two maps - _normalize_reading_
    # test_type() already guarantees test_type is exactly one of these
    # two values, so this is exhaustive by construction.
    # =========================
    if test_type == "academic":
        band = band_from_correct(correct)
    elif test_type == "general_training":
        band = general_reading_band(correct)
    else:  # pragma: no cover - unreachable, see _normalize_reading_test_type()
        raise ValueError(f"Unhandled Reading test_type: {test_type!r}")

    # =========================
    # IMPROVEMENTS
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
    # FEEDBACK
    # =========================
    examiner_feedback = (
        f"This is a Band {band} reading performance. "
        "Errors were observed in specific question types, indicating areas "
        "where targeted practice is required."
        if wrong_question_types else
        f"This is a Band {band} reading performance with a high level of accuracy."
    )

    # =========================
    # FINAL RESPONSE
    # =========================
    result = {
        "module": "reading",
        "test_type": test_type,  # 👈 NEW (optional but useful)
        "overall_band": band,
        "accuracy": f"{correct}/{len(questions)}",
        "improvements": improvements,
        "examiner_feedback": examiner_feedback
    }

    # Band skill level + description - COMMON to Academic and General
    # Training (the official qualitative descriptions are the same
    # regardless of test type; only the raw-score -> band MAPPING
    # differs, which was already resolved above). Deterministic lookup
    # via get_band_skill_description() - never decided by an LLM.
    skill_level, description = get_band_skill_description(band)
    result["raw_score"] = correct
    result["band_score"] = band
    result["skill_level"] = skill_level
    result["description"] = description

    return result