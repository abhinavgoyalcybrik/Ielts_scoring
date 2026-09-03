# Item 3 - widen _NO_GENUINE_ERROR_PHRASES. Live output contains mistakes
# whose own explanation concedes there is no error, and they passed the
# old (narrower) filter. Tests the 13 newly-added phrases directly (one
# case per phrase, matching the exact wording given), plus a genuine
# explanation containing none of them, plus the 3 real production
# examples verbatim.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing


def _make_bands(true_band):
    return {str(n): (n == true_band) for n in range(1, 10)}


def _complete_task2_bands(band=6):
    flags = _make_bands(band)
    return {
        "task_response_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
    }


def _evaluate_with_mistake(monkeypatch, explanation, original="attended the meeting yesterday"):
    essay = f"During the trip, we attended the meeting yesterday and discussed the results."
    ai_response = {
        **_complete_task2_bands(),
        "mistakes": [{
            "type": "grammar", "category": "Word Choice", "severity": "minor",
            "original": original, "corrected": "went to the meeting yesterday",
            "explanation": explanation,
        }],
        "strengths": "x", "improvement": "y", "topic_relevance": "on_topic",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined text. " * 60)
    return writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": essay},
    })


_NEW_PHRASE_EXPLANATIONS = [
    "This is a minor stylistic point and does not affect meaning.",
    "The original phrasing is acceptable here.",
    "Either wording is also acceptable in this context.",
    "Using the past tense here is also possible.",
    "The candidate's version is also correct.",
    "This can be written either way.",
    "This is really just a minor stylistic point, not an error.",
    "Choosing this word is more of a stylistic preference.",
    "'I would like to' is less direct than the alternative.",
    "The sentence could be improved but is not wrong.",
    "A different word choice would be clearer, though this is not wrong.",
    "This phrasing is not incorrect.",
    "The phrase 'with different cultures' is correct, but could be varied.",
]


def test_each_new_phrase_drops_the_mistake(monkeypatch):
    dropped = []
    for explanation in _NEW_PHRASE_EXPLANATIONS:
        result = _evaluate_with_mistake(monkeypatch, explanation)
        if len(result["mistakes"]) == 0:
            dropped.append(explanation)
    still_present = [e for e in _NEW_PHRASE_EXPLANATIONS if e not in dropped]
    assert not still_present, f"these explanations were NOT dropped: {still_present}"


def test_genuine_explanation_with_none_of_the_new_phrases_survives(monkeypatch):
    result = _evaluate_with_mistake(
        monkeypatch,
        "The verb tense is inconsistent with the rest of the paragraph, which uses past tense throughout.",
    )
    assert len(result["mistakes"]) == 1


# ---------------------------------------------------------------------------
# The 3 real production explanations named in the request, verbatim.
# ---------------------------------------------------------------------------

def test_drops_real_production_case_minor_stylistic_point(monkeypatch):
    result = _evaluate_with_mistake(
        monkeypatch,
        "'Worry-free' is a compound adjective and should be hyphenated when used before a noun, but here it "
        "is used predicatively and can be written without a hyphen. This is a minor stylistic point and does "
        "not affect meaning.",
    )
    assert len(result["mistakes"]) == 0


def test_drops_real_production_case_is_less_direct(monkeypatch):
    result = _evaluate_with_mistake(
        monkeypatch,
        "The phrase 'I would like to' is less direct and can weaken the statement; using 'I will' is clearer "
        "and more appropriate for an academic essay.",
    )
    assert len(result["mistakes"]) == 0


def test_drops_real_production_case_is_correct_but(monkeypatch):
    result = _evaluate_with_mistake(
        monkeypatch,
        "The phrase 'with different cultures' is correct, but the sentence structure could be improved by "
        "adding 'and' to clarify the two different objects of 'behave'.",
    )
    assert len(result["mistakes"]) == 0


def test_widened_list_contains_every_requested_phrase():
    expected = [
        "does not affect meaning", "is acceptable", "is also acceptable",
        "is also possible", "is also correct", "can be written",
        "minor stylistic point", "stylistic preference", "is less direct",
        "could be improved", "would be clearer", "is not incorrect",
        "is correct, but",
    ]
    for phrase in expected:
        assert phrase in writing._NO_GENUINE_ERROR_PHRASES, f"missing: {phrase!r}"
