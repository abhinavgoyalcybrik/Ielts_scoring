from utils.band import academic_listening_band, general_training_listening_band, get_band_skill_description


# Academic and General Training Listening use two COMPLETELY SEPARATE
# raw-score -> band mappings (academic_listening_band() /
# general_training_listening_band(), each backed by its own named
# constant in utils/band.py) that must never be mixed, even though their
# current values happen to match. The test type is resolved into exactly
# one of these two canonical values HERE, before any band lookup happens,
# so the routing below is a plain, exhaustive check with no fallback
# between the two maps. Independent copy of evaluators/reading.py's
# equivalent normalization, kept separate to keep the two evaluator
# modules decoupled.
_GENERAL_TRAINING_TEST_TYPE_ALIASES = {"general", "general_training", "generaltraining", "gt"}


def _normalize_listening_test_type(raw_test_type) -> str:
    key = str(raw_test_type or "").strip().casefold().replace(" ", "_").replace("-", "_")
    if key in _GENERAL_TRAINING_TEST_TYPE_ALIASES:
        return "general_training"
    return "academic"


def _answers_match(user_ans, correct_ans) -> bool:
    """Case- and whitespace-insensitive comparison, per real IELTS
    Listening marking convention: capitalisation is not penalised -
    spelling must still be exact. Same rule/reasoning as
    evaluators/reading.py's _answers_match() (kept as a separate local
    copy to keep the two evaluator modules decoupled).

    `correct_ans` may be a single acceptable answer OR a list/tuple/set of
    several acceptable answers for the same question (e.g. British vs
    American spelling, or a few accepted phrasings) - the user's answer is
    correct if it matches ANY one of them."""
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


def evaluate_listening(data):
    # -----------------------------
    # Input validation
    # -----------------------------
    if "user_answers" not in data or "answer_key" not in data:
        raise ValueError("user_answers and answer_key are required")

    user = data["user_answers"]
    key = data["answer_key"]
    test_type = _normalize_listening_test_type(data.get("test_type"))

    total = len(key)
    correct = 0
    error_types = set()   # 👈 store error categories only

    # -----------------------------
    # Answer checking
    # -----------------------------
    for qid, ans in key.items():
        user_ans = user.get(qid)

        if _answers_match(user_ans, ans):
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
    # Band calculation - test type determined FIRST (above), then route
    # to ONLY the corresponding mapping. Explicit if/elif with no
    # implicit else-fallback between the two maps -
    # _normalize_listening_test_type() already guarantees test_type is
    # exactly one of these two values, so this is exhaustive by
    # construction.
    # -----------------------------
    if test_type == "academic":
        band = academic_listening_band(correct)
    elif test_type == "general_training":
        band = general_training_listening_band(correct)
    else:  # pragma: no cover - unreachable, see _normalize_listening_test_type()
        raise ValueError(f"Unhandled Listening test_type: {test_type!r}")

    # -----------------------------
    # Examiner feedback
    # -----------------------------
    examiner_feedback = (
        f"This is a Band {band} listening performance. "
        "Errors suggest difficulty with identifying precise details and managing distractors."
        if error_types else
        f"This is a Band {band} listening performance with a high level of accuracy."
    )

    # -----------------------------
    # Final response (FLAT FORMAT)
    # -----------------------------
    result = {
        "module": "listening",
        "test_type": test_type,
        "overall_band": band,
        "accuracy": f"{correct}/{total}",
        "improvements": improvements,
        "examiner_feedback": examiner_feedback
    }

    # Band skill level + description - COMMON to Academic and General
    # Training (the official qualitative descriptions are the same
    # regardless of test type; only the raw-score -> band MAPPING
    # differs, which was already resolved above). Deterministic lookup
    # via get_band_skill_description() - never decided by an LLM. Same
    # shared skill-description table already used by Reading.
    skill_level, description = get_band_skill_description(band)
    result["raw_score"] = correct
    result["band_score"] = band
    result["skill_level"] = skill_level
    result["description"] = description

    return result
