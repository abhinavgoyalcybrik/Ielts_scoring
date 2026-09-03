import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.band import (
    ACADEMIC_READING_BAND_MAP,
    GENERAL_TRAINING_READING_BAND_MAP,
    band_from_correct,
    general_reading_band,
    get_band_skill_description,
)
from evaluators.reading import evaluate_reading, _normalize_reading_test_type


# ---------------------------------------------------------------------------
# Academic and General Training Reading use two COMPLETELY SEPARATE raw-
# score -> band mappings, each its own named constant
# (ACADEMIC_READING_BAND_MAP / GENERAL_TRAINING_READING_BAND_MAP), never
# merged or falling back to each other. band_score -> skill_level/
# description is a deterministic, rule-based lookup (never LLM-decided),
# and is COMMON to both test types (only the raw-score -> band mapping
# differs between them, not the qualitative descriptions).
# ---------------------------------------------------------------------------

def test_academic_reading_raw_score_to_band_mapping():
    cases = {
        40: 9.0, 39: 9.0,
        38: 8.5,
        36: 8.0,
        34: 7.5,
        32: 7.0,
        29: 6.5,
        26: 6.0,
        22: 5.5,
        18: 5.0,
        14: 4.5,
        12: 4.0,
        9: 3.5,
        7: 3.0,
        5: 2.5,
    }
    for raw, expected_band in cases.items():
        assert band_from_correct(raw) == expected_band, f"raw={raw}"


def test_general_training_reading_raw_score_to_band_mapping():
    cases = {
        40: 9.0,
        39: 8.5,
        38: 8.0,
        37: 8.0,
        36: 7.5,
        35: 7.0,
        34: 7.0,
        33: 6.5,
        32: 6.5,
        31: 6.0,
        30: 6.0,
        29: 5.5,
        26: 5.0,
        22: 4.5,
        18: 4.0,
        14: 3.5,
        11: 3.0,
    }
    for raw, expected_band in cases.items():
        assert general_reading_band(raw) == expected_band, f"raw={raw}"


def test_academic_and_general_training_maps_are_separate_objects():
    # Structural guarantee: two distinct named constants, not one merged
    # dict or one derived from the other.
    assert ACADEMIC_READING_BAND_MAP is not GENERAL_TRAINING_READING_BAND_MAP
    assert isinstance(ACADEMIC_READING_BAND_MAP, dict)
    assert isinstance(GENERAL_TRAINING_READING_BAND_MAP, dict)


def test_critical_academic_vs_general_training_differences():
    # The exact cases called out as proof the mappings must stay separate.
    assert band_from_correct(32) == 7.0
    assert general_reading_band(32) == 6.5

    assert band_from_correct(26) == 6.0
    assert general_reading_band(26) == 5.0

    assert band_from_correct(22) == 5.5
    assert general_reading_band(22) == 4.5


def test_get_band_skill_description_matches_spec_example():
    skill_level, description = get_band_skill_description(7.0)
    assert skill_level == "Good user"
    assert description == (
        "Has operational command of the language, though with occasional "
        "inaccuracies, inappropriacies and misunderstandings in some "
        "situations. Generally handles complex language well and "
        "understands detailed reasoning"
    )


def test_get_band_skill_description_covers_every_whole_band():
    expected_skill_levels = {
        0: "Did not attempt test",
        1: "Non-user",
        2: "Intermittent user",
        3: "Extremely limited user",
        4: "Limited user",
        5: "Modest user",
        6: "Competent user",
        7: "Good user",
        8: "Very good user",
        9: "Expert user",
    }
    for band, expected_skill in expected_skill_levels.items():
        skill_level, description = get_band_skill_description(band)
        assert skill_level == expected_skill
        assert description  # non-empty


def test_get_band_skill_description_floors_half_bands():
    # No official IELTS descriptor exists for half bands - floors to the
    # whole-band tier below (e.g. 8.5 -> Band 8's descriptor, not Band 9's).
    skill_level, _ = get_band_skill_description(8.5)
    assert skill_level == "Very good user"

    skill_level, _ = get_band_skill_description(6.5)
    assert skill_level == "Competent user"


def test_get_band_skill_description_clamps_out_of_range_input():
    # Defensive: must never raise on malformed/out-of-range input.
    assert get_band_skill_description(-5)[0] == "Did not attempt test"
    assert get_band_skill_description(15)[0] == "Expert user"
    assert get_band_skill_description(None)[0] == "Did not attempt test"


def _questions_and_answers(correct_count, total=40):
    questions = [{"question_id": f"q{i}", "answer_key": "A", "type": "MCQ"} for i in range(total)]
    user_answers = {f"q{i}": ("A" if i < correct_count else "Z") for i in range(total)}
    return questions, user_answers


def test_evaluate_reading_academic_matches_spec_example():
    questions, user_answers = _questions_and_answers(32)
    result = evaluate_reading({"questions": questions, "user_answers": user_answers, "test_type": "academic"})

    assert result["test_type"] == "academic"
    assert result["raw_score"] == 32
    assert result["band_score"] == 7.0
    assert result["skill_level"] == "Good user"
    assert result["description"].startswith("Has operational command of the language")
    # Existing fields must still be present/unchanged.
    assert result["overall_band"] == 7.0
    assert result["accuracy"] == "32/40"
    assert result["module"] == "reading"


def test_evaluate_reading_general_training_matches_spec_example():
    # The exact worked example from the spec: same raw score 32 as the
    # Academic example above, but General Training's own mapping (via
    # GENERAL_TRAINING_READING_BAND_MAP) gives a genuinely different band.
    questions, user_answers = _questions_and_answers(32)
    result = evaluate_reading({"questions": questions, "user_answers": user_answers, "test_type": "general_training"})

    assert result["test_type"] == "general_training"
    assert result["raw_score"] == 32
    assert result["band_score"] == 6.5
    assert result["skill_level"] == "Competent user"
    assert result["description"].startswith("Has generally effective command of the language")
    assert result["overall_band"] == 6.5


def test_evaluate_reading_academic_is_the_default_test_type():
    # No test_type key at all -> defaults to academic.
    questions, user_answers = _questions_and_answers(40)
    result = evaluate_reading({"questions": questions, "user_answers": user_answers})
    assert result["test_type"] == "academic"
    assert "skill_level" in result


def test_general_reading_band_table_matches_new_mapping_not_academic():
    # Values that genuinely differ between the corrected General Training
    # table and the Academic table, confirming the fix actually took.
    assert general_reading_band(39) == 8.5   # was grouped with 40 at 9.0 before
    assert general_reading_band(33) == 6.5   # was 7.0 before
    assert general_reading_band(31) == 6.0   # was 6.5 before
    assert general_reading_band(30) == 6.0   # unchanged
    # Below raw score 9 was explicitly preserved, not part of the new
    # reference table - spot-check it stayed intact.
    assert general_reading_band(8) == 2.5
    assert general_reading_band(5) == 2.0


# ---------------------------------------------------------------------------
# test_type routing must be case/whitespace-insensitive, and must resolve
# to exactly one of the two canonical values ("academic" or
# "general_training") BEFORE any band lookup happens - never an implicit
# fallback between the two maps. "general" (this project's previous
# canonical spelling) is still accepted as an alias for
# "general_training", so an existing caller sending "general" is never
# silently rescored as Academic.
# ---------------------------------------------------------------------------

def test_normalize_reading_test_type_general_training_aliases():
    for variant in ("general", "General", "GENERAL", "  general  ", "general_training", "General Training", "GT"):
        assert _normalize_reading_test_type(variant) == "general_training", variant


def test_normalize_reading_test_type_academic_and_invalid_values_default_correctly():
    for variant in ("academic", "Academic", "ACADEMIC", None, "", "  ", "bogus_value"):
        assert _normalize_reading_test_type(variant) == "academic", variant


def test_evaluate_reading_general_training_case_and_whitespace_variants():
    questions, user_answers = _questions_and_answers(32)
    for variant in ("general", "General", "GENERAL", "  general  ", "general_training", "GeNeRaL"):
        result = evaluate_reading({"questions": questions, "user_answers": user_answers, "test_type": variant})
        assert result["overall_band"] == general_reading_band(32), variant
        assert result["test_type"] == "general_training"
        assert result["skill_level"] == "Competent user"


def test_evaluate_reading_academic_case_variants_and_invalid_values_default_correctly():
    questions, user_answers = _questions_and_answers(32)
    for variant in ("academic", "Academic", "ACADEMIC", "", "  ", "bogus_value"):
        result = evaluate_reading({"questions": questions, "user_answers": user_answers, "test_type": variant})
        assert result["overall_band"] == band_from_correct(32), variant
        assert result["test_type"] == "academic"
        assert result["skill_level"] == "Good user"


def test_academic_and_general_training_never_cross_contaminate():
    # Run BOTH test types back-to-back with the SAME raw score and confirm
    # each used only its own map - guards against any shared mutable
    # state or accidental fallback between the two.
    questions, user_answers = _questions_and_answers(26)

    academic_result = evaluate_reading({"questions": questions, "user_answers": user_answers, "test_type": "academic"})
    general_result = evaluate_reading({"questions": questions, "user_answers": user_answers, "test_type": "general_training"})

    assert academic_result["band_score"] == 6.0
    assert general_result["band_score"] == 5.0
    assert academic_result["band_score"] != general_result["band_score"]
