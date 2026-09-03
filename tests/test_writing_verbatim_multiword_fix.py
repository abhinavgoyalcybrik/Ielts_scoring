# Item 2 - original must be a unique multi-word span. Live output has
# flagged mistakes with "original" set to a bare word ("features",
# "consumed", "includes", "sourrounded") that occurs many times in the
# essay - the candidate can't tell which occurrence is meant, and the
# plain substring check passed trivially. This tests the extended
# _mistake_original_is_verbatim(..., require_unique_multiword=True) call
# directly, and confirms every OTHER call site of the same function
# (default require_unique_multiword=False) is untouched.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing

_FEATURES_ESSAY = (
    "The main features are the hospital and the car park. Both features "
    "remained unchanged. Several new features appeared in 2010, and these "
    "features include a bus station and two roundabouts, which are useful "
    "features for visitors."
)


def test_bare_word_occurring_many_times_is_dropped():
    assert writing._mistake_original_is_verbatim("features", _FEATURES_ESSAY, require_unique_multiword=True) is False


def test_bare_word_occurring_exactly_once_is_still_dropped_for_being_too_short():
    essay = "The candidate wrote about sourrounded incorrectly in this one place only."
    assert writing._mistake_original_is_verbatim("sourrounded", essay, require_unique_multiword=True) is False


def test_three_word_span_occurring_once_survives():
    essay = "Dinner contains the most sodium at forty three percent of the total."
    assert writing._mistake_original_is_verbatim("contains the most", essay, require_unique_multiword=True) is True


def test_three_word_span_occurring_twice_is_dropped():
    essay = "The show's aim is to test skills. Later, the show's aim is repeated again for emphasis."
    assert writing._mistake_original_is_verbatim("the show's aim is", essay, require_unique_multiword=True) is False


def test_two_word_span_occurring_once_is_still_dropped_for_being_under_three_words():
    essay = "The candidate described an unsatisfying job in the second paragraph of the essay."
    assert writing._mistake_original_is_verbatim("unsatisfying job", essay, require_unique_multiword=True) is False


# ---------------------------------------------------------------------------
# Backward compatibility: every OTHER call site (vocabulary suggestions,
# refined-answer leak check) uses the DEFAULT require_unique_multiword=False
# and must be byte-identical to before this item - a bare word or a
# multiply-occurring span must still pass there, since those checks exist
# for a different purpose (basic presence, not locatability).
# ---------------------------------------------------------------------------

def test_default_call_still_accepts_bare_word_occurring_many_times():
    assert writing._mistake_original_is_verbatim("features", _FEATURES_ESSAY) is True


def test_default_call_still_accepts_short_span_occurring_once():
    essay = "The candidate described an unsatisfying job in the essay."
    assert writing._mistake_original_is_verbatim("unsatisfying job", essay) is True


def test_default_call_still_rejects_genuinely_absent_text():
    assert writing._mistake_original_is_verbatim("a phrase never written", "some unrelated essay text here") is False


# ---------------------------------------------------------------------------
# End-to-end: evaluate_writing() must actually drop these mistakes from
# the returned "mistakes" list, not just at the unit level.
# ---------------------------------------------------------------------------

def _make_bands(true_band):
    return {str(n): (n == true_band) for n in range(1, 10)}


def _complete_task2_bands(band=7):
    flags = _make_bands(band)
    return {
        "task_response_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
    }


def test_evaluate_writing_drops_bare_word_mistake_end_to_end(monkeypatch):
    essay = (
        "The main features are the hospital and the car park. Both features "
        "remained unchanged since the earlier period under review recently."
    )
    ai_response = {
        **_complete_task2_bands(),
        "mistakes": [
            {"type": "lexical", "category": "Incorrect Word Choice", "severity": "minor",
             "original": "features", "corrected": "elements", "explanation": "vaguer word choice"},
            {"type": "grammar", "category": "Article Errors", "severity": "minor",
             "original": "the main features are the hospital", "corrected": "the main feature is the hospital",
             "explanation": "subject-verb agreement"},
        ],
        "strengths": "x", "improvement": "y", "topic_relevance": "on_topic",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined text. " * 60)

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": essay},
    })

    originals = [m["original"] for m in result["mistakes"]]
    assert "features" not in originals
    assert "the main features are the hospital" in originals
