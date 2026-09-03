# Item 5 - dedupe mistakes covering the exact same normalized span, keep
# the higher-severity copy. Runs last in the mistake pipeline, after
# severity normalization/escalation, so "higher severity" reflects the
# final value. Only ever removes an object - never invents or edits one.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing

_dedupe = writing._dedupe_mistakes_by_normalized_span


def test_dedupe_keeps_single_occurrence_untouched():
    mistakes = [{"type": "grammar", "original": "she go to school", "severity": "minor"}]
    assert _dedupe(mistakes) == mistakes


def test_dedupe_removes_exact_duplicate_span_keeps_first_on_tie():
    m1 = {"type": "grammar", "category": "Subject-Verb Agreement Errors", "original": "she go to school", "severity": "minor"}
    m2 = {"type": "lexical", "category": "Incorrect Word Choice", "original": "she go to school", "severity": "minor"}
    result = _dedupe([m1, m2])
    assert len(result) == 1
    assert result[0] is m1  # first-seen wins on a severity tie


def test_dedupe_keeps_the_higher_severity_copy_regardless_of_order():
    minor_first = {"original": "she go to school", "severity": "minor", "id": "minor"}
    significant_second = {"original": "she go to school", "severity": "significant", "id": "significant"}
    result = _dedupe([minor_first, significant_second])
    assert len(result) == 1
    assert result[0]["id"] == "significant"

    # Same pair, reversed order - result must be the same (order-independent).
    result_reversed = _dedupe([significant_second, minor_first])
    assert len(result_reversed) == 1
    assert result_reversed[0]["id"] == "significant"


def test_dedupe_span_matching_is_whitespace_and_case_insensitive():
    m1 = {"original": "She   Go to School", "severity": "minor"}
    m2 = {"original": "she go to school", "severity": "significant"}
    result = _dedupe([m1, m2])
    assert len(result) == 1
    assert result[0]["severity"] == "significant"


def test_dedupe_strips_wrapping_quotes_before_comparing():
    m1 = {"original": '"she go to school"', "severity": "minor"}
    m2 = {"original": "she go to school", "severity": "significant"}
    result = _dedupe([m1, m2])
    assert len(result) == 1


def test_dedupe_preserves_first_occurrence_position():
    a = {"original": "first span", "severity": "minor"}
    b = {"original": "second span", "severity": "minor"}
    a_dup = {"original": "first span", "severity": "significant"}
    result = _dedupe([a, b, a_dup])
    # "first span"'s surviving (higher-severity) copy stays at position 0,
    # not moved to where the duplicate appeared.
    assert [m["original"] for m in result] == ["first span", "second span"]
    assert result[0]["severity"] == "significant"


def test_dedupe_leaves_distinct_spans_alone():
    mistakes = [
        {"original": "the first error here", "severity": "minor"},
        {"original": "a completely different error", "severity": "significant"},
    ]
    assert _dedupe(mistakes) == mistakes


def test_dedupe_never_drops_a_mistake_with_no_usable_span():
    # Nothing to dedupe an empty/missing "original" against - both pass
    # through untouched, even though they're "duplicates" of each other by
    # having nothing in common.
    m1 = {"original": "", "severity": "minor"}
    m2 = {"original": "", "severity": "significant"}
    result = _dedupe([m1, m2])
    assert len(result) == 2


def test_dedupe_handles_three_way_duplicate_keeps_only_the_best():
    low = {"original": "he don't know", "severity": "minor", "id": "low"}
    also_low = {"original": "he don't know", "severity": "minor", "id": "also_low"}
    high = {"original": "he don't know", "severity": "significant", "id": "high"}
    result = _dedupe([low, also_low, high])
    assert len(result) == 1
    assert result[0]["id"] == "high"


def test_dedupe_empty_list():
    assert _dedupe([]) == []


# ---------------------------------------------------------------------------
# End-to-end wiring: confirms _dedupe_mistakes_by_normalized_span actually
# runs as part of evaluate_writing()'s mistake pipeline, after severity
# escalation.

_MAKE_BANDS = lambda band: {str(n): (n == band) for n in range(1, 10)}


def _complete_task2_bands(band=7):
    flags = _MAKE_BANDS(band)
    return {
        "task_response_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
    }


def test_evaluate_writing_dedupes_identical_span_reported_under_two_categories(monkeypatch):
    essay = (
        "Governments ought to spend significant public funds on public "
        "health awareness programs since it saves money down the line and "
        "helps citizens stay healthy for many years to come in general."
    )
    mistakes = [
        {
            "type": "grammar", "category": "Article Errors", "severity": "minor",
            "meaning_impact": "low", "original": "helps citizens stay healthy",
            "corrected": "helps the citizens stay healthy", "explanation": "missing article",
        },
        {
            "type": "lexical", "category": "Incorrect Word Choice", "severity": "significant",
            "meaning_impact": "medium", "original": "helps citizens stay healthy",
            "corrected": "helps residents stay healthy", "explanation": "word choice",
        },
    ]
    ai_response = {
        **_complete_task2_bands(),
        "mistakes": mistakes,
        "strengths": "x", "improvement": "y", "topic_relevance": "on_topic",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "refined text here")
    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": essay},
    })
    matching = [m for m in result["mistakes"] if m.get("original", "").strip().lower() == "helps citizens stay healthy"]
    assert len(matching) == 1
    assert matching[0]["severity"] == "significant"
