# Item 8 - a mistake's "corrected" field sometimes only ADDS to "original"
# (or is identical to it) - "living in a foreign country has its own
# benefits and drawbacks" -> "...benefits and drawbacks TO CONSIDER". The
# original sentence was already complete and correct; the "correction" is
# an optional stylistic preference, not a fix for an actual error. A
# genuine correction changes or removes something. Calibrated against 241
# real Writing mistakes (Speaking scratch data excluded - this fix never
# runs on Speaking's separate pipeline): 7/241 (2.9%) dropped, every one
# confirmed a genuine non-fix (the named bug case, five degenerate
# original==corrected duplicates, one paragraph-break-only reformatting
# with zero word-level change). Zero genuine corrections lost.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing

_MAKE_BANDS = lambda band: {str(n): (n == band) for n in range(1, 10)}


def _complete_bands(task_type, band=7):
    flags = _MAKE_BANDS(band)
    task_key = "task_achievement_bands" if task_type == "task_1" else "task_response_bands"
    return {
        task_key: dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
    }


def _evaluate(monkeypatch, mistakes, refined_text, essay, task_type="task_2"):
    ai_response = {
        **_complete_bands(task_type),
        "mistakes": mistakes,
        "strengths": "x", "improvement": "y", "topic_relevance": "on_topic",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: refined_text)
    return writing.evaluate_writing({
        "metadata": {"task_type": task_type, "question": "Some question?"},
        "user_answers": {"text": essay},
    })


# ---------------------------------------------------------------------------
# _correction_is_pure_addition - the predicate itself.
# ---------------------------------------------------------------------------

def test_trailing_multiword_addition_is_pure_addition():
    # The real, named bug case.
    assert writing._correction_is_pure_addition(
        "living in a foreign country has its own benefits and drawbacks",
        "living in a foreign country has its own benefits and drawbacks to consider",
    ) is True


def test_identical_text_is_pure_addition():
    assert writing._correction_is_pure_addition("same text here", "same text here") is True


def test_single_word_midspan_insertion_is_not_pure_addition():
    # The required exception: a genuinely missing preposition/article is
    # ALSO an insertion - must survive.
    assert writing._correction_is_pure_addition("29% sodium", "29% of sodium") is False


def test_single_word_midspan_article_insertion_is_not_pure_addition():
    assert writing._correction_is_pure_addition(
        "An individual who works in foreign country",
        "An individual who works in a foreign country",
    ) is False


def test_genuine_change_is_not_pure_addition():
    assert writing._correction_is_pure_addition("who come into", "who comes into") is False


def test_substitution_after_a_qualifying_insert_is_not_pure_addition():
    # Regression test for a real bug caught during calibration: an
    # earlier version returned True on the FIRST qualifying insert without
    # checking for a later replace/delete opcode, wrongly dropping a
    # correction that both inserted a 2-word phrase AND genuinely replaced
    # a word further along ("technology" -> "it"). Live example.
    assert writing._correction_is_pure_addition(
        "In my opinion, the benefits outweigh the drawbacks because technology saves time and increases access to information.",
        "In my opinion, the benefits of technology outweigh the drawbacks because it saves time and increases access to information.",
    ) is False


def test_full_rewrite_is_not_pure_addition():
    assert writing._correction_is_pure_addition(
        "Overall, the fourth category clearly dominates spending while the third category represents only a small proportion by comparison.",
        "Overall, it is evident that one category significantly surpasses the others in terms of expenditure, while another category accounts for a relatively minor share.",
    ) is False


def test_empty_original_is_not_pure_addition():
    assert writing._correction_is_pure_addition("", "some new text") is False


# ---------------------------------------------------------------------------
# Integration through evaluate_writing().
# ---------------------------------------------------------------------------

def test_over_correction_mistake_is_dropped_from_result(monkeypatch):
    essay = (
        "It is clear that living in a foreign country has its own benefits and drawbacks to "
        "consider. An individual who come into the country might offend others with their "
        "behaviour or language, so anyone moving abroad should learn local customs first."
    )
    mistakes = [
        {
            "type": "grammar", "category": "Preposition Errors", "severity": "minor",
            "meaning_impact": "low",
            "original": "living in a foreign country has its own benefits and drawbacks",
            "corrected": "living in a foreign country has its own benefits and drawbacks to consider",
            "explanation": "Adds a clarifying phrase.",
        },
        {
            "type": "grammar", "category": "Subject-Verb Agreement Errors", "severity": "significant",
            "meaning_impact": "medium", "original": "An individual who come into the country",
            "corrected": "An individual who comes into the country", "explanation": "subject-verb agreement",
        },
    ]
    result = _evaluate(monkeypatch, mistakes, "A refined Band 9 answer text here for this test.", essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Preposition Errors" not in categories
    assert "Subject-Verb Agreement Errors" in categories


def test_genuine_single_word_insertion_mistake_survives(monkeypatch):
    essay = "Through eating lunch, 29% sodium is consumed by most people in this survey."
    mistakes = [
        {
            "type": "grammar", "category": "Preposition Errors", "severity": "minor",
            "meaning_impact": "low", "original": "29% sodium is consumed",
            "corrected": "29% of sodium is consumed", "explanation": "missing preposition",
        },
    ]
    result = _evaluate(monkeypatch, mistakes, "A refined Band 9 answer text here.", essay, task_type="task_1")

    categories = [m["category"] for m in result["mistakes"]]
    assert "Preposition Errors" in categories
