import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.band import (
    ACADEMIC_LISTENING_BAND_MAP,
    GENERAL_TRAINING_LISTENING_BAND_MAP,
    academic_listening_band,
    general_training_listening_band,
    get_band_skill_description,
)
from evaluators.listening import evaluate_listening, _normalize_listening_test_type


# ---------------------------------------------------------------------------
# Academic and General Training Listening use two COMPLETELY SEPARATE raw-
# score -> band mappings, each its own named constant
# (ACADEMIC_LISTENING_BAND_MAP / GENERAL_TRAINING_LISTENING_BAND_MAP),
# never merged or falling back to each other - even though their current
# values are identical, per explicit instruction to keep them
# independently configurable rather than one shared table. band_score ->
# skill_level/description is the same shared lookup already used by
# Reading (get_band_skill_description()).
# ---------------------------------------------------------------------------

_SPEC_TABLE = {
    40: 9.0, 39: 9.0,
    38: 8.5, 37: 8.5,
    36: 8.0, 35: 8.0,
    34: 7.5, 33: 7.5, 32: 7.5,
    31: 7.0, 30: 7.0,
    29: 6.5, 28: 6.5, 27: 6.5, 26: 6.5,
    25: 6.0, 24: 6.0, 23: 6.0,
    22: 5.5, 21: 5.5, 20: 5.5, 19: 5.5, 18: 5.5,
    17: 5.0, 16: 5.0,
    15: 4.5, 14: 4.5, 13: 4.5,
    12: 4.0, 11: 4.0,
}


def test_academic_listening_raw_score_to_band_mapping():
    for raw, expected_band in _SPEC_TABLE.items():
        assert academic_listening_band(raw) == expected_band, f"raw={raw}"


def test_general_training_listening_raw_score_to_band_mapping():
    for raw, expected_band in _SPEC_TABLE.items():
        assert general_training_listening_band(raw) == expected_band, f"raw={raw}"


def test_academic_and_general_training_listening_maps_are_separate_objects():
    # Structural guarantee: two distinct named constants, never one
    # aliasing/copying the other - even though their current values match.
    assert ACADEMIC_LISTENING_BAND_MAP is not GENERAL_TRAINING_LISTENING_BAND_MAP
    assert isinstance(ACADEMIC_LISTENING_BAND_MAP, dict)
    assert isinstance(GENERAL_TRAINING_LISTENING_BAND_MAP, dict)


def test_listening_maps_currently_hold_identical_values():
    # The spec states both tables currently use the same numbers - confirm
    # that (this is expected, NOT a sign the two got merged into one).
    for raw in _SPEC_TABLE:
        assert academic_listening_band(raw) == general_training_listening_band(raw)


def _answer_key_and_answers(correct_count, total=40):
    answer_key = {f"q{i}": "correct" for i in range(total)}
    user_answers = {f"q{i}": ("correct" if i < correct_count else "wrong") for i in range(total)}
    return answer_key, user_answers


def test_evaluate_listening_academic_matches_spec_example():
    answer_key, user_answers = _answer_key_and_answers(32)
    result = evaluate_listening({"answer_key": answer_key, "user_answers": user_answers, "test_type": "academic"})

    assert result["test_type"] == "academic"
    assert result["raw_score"] == 32
    assert result["band_score"] == 7.5
    assert result["skill_level"] == "Good user"
    assert result["description"].startswith("Has operational command of the language")
    assert result["overall_band"] == 7.5
    assert result["accuracy"] == "32/40"
    assert result["module"] == "listening"


def test_evaluate_listening_general_training_matches_spec_example():
    answer_key, user_answers = _answer_key_and_answers(32)
    result = evaluate_listening({"answer_key": answer_key, "user_answers": user_answers, "test_type": "general_training"})

    assert result["test_type"] == "general_training"
    assert result["raw_score"] == 32
    assert result["band_score"] == 7.5
    assert result["skill_level"] == "Good user"
    assert result["overall_band"] == 7.5


def test_evaluate_listening_academic_is_the_default_test_type():
    answer_key, user_answers = _answer_key_and_answers(40)
    result = evaluate_listening({"answer_key": answer_key, "user_answers": user_answers})
    assert result["test_type"] == "academic"
    assert "skill_level" in result


# ---------------------------------------------------------------------------
# test_type routing - resolved to exactly one canonical value BEFORE any
# band lookup, never an implicit fallback between the two maps.
# ---------------------------------------------------------------------------

def test_normalize_listening_test_type_general_training_aliases():
    for variant in ("general", "General", "GENERAL", "  general  ", "general_training", "General Training", "GT"):
        assert _normalize_listening_test_type(variant) == "general_training", variant


def test_normalize_listening_test_type_academic_and_invalid_values_default_correctly():
    for variant in ("academic", "Academic", "ACADEMIC", None, "", "  ", "bogus_value"):
        assert _normalize_listening_test_type(variant) == "academic", variant


def test_academic_and_general_training_listening_never_cross_contaminate():
    # Run BOTH test types back-to-back with the SAME raw score, using the
    # separate maps directly (bypassing the fact their current values
    # happen to match) to confirm each routes to its own function.
    assert academic_listening_band(26) == general_training_listening_band(26) == 6.5

    answer_key, user_answers = _answer_key_and_answers(26)
    academic_result = evaluate_listening({"answer_key": answer_key, "user_answers": user_answers, "test_type": "academic"})
    gt_result = evaluate_listening({"answer_key": answer_key, "user_answers": user_answers, "test_type": "general_training"})
    assert academic_result["band_score"] == gt_result["band_score"] == 6.5
