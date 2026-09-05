# Item 9 - a "Paragraphing Errors" mistake on an essay that genuinely has
# real paragraph breaks is a mislabel, not a real finding. Distinct from
# the confirmed-and-fixed /n/n literal-escaping bug (that essay genuinely
# had ZERO real newlines, so the "lacks paragraph breaks" claim was true
# at the time) - this is the case where the essay DOES have real
# newlines, and the mistake is wrong regardless of what the explanation
# argues. Live QA case: the quoted "original" was a single long SENTENCE,
# and the explanation itself said so ("the sentence is too long and lacks
# clear breaks... breaking it into two sentences improves clarity") - a
# sentence-length observation filed under a paragraph-structure category.
#
# Calibrated against every "Paragraphing Errors" mistake in every saved
# real run this session has (17 total): 10/17 dropped (essay genuinely
# has real newlines), all confirmed genuine mislabels. Separately, of all
# 17, 8 (47%) have a single-sentence "original" - systematic, not a
# one-off, per the explicit instruction to measure that before deciding
# whether it's worth a dedicated fix (not built here - report only).

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


def test_paragraphing_errors_dropped_when_essay_has_real_paragraph_breaks(monkeypatch):
    # The real live case: a single-sentence "original", an explanation
    # that argues sentence length (not paragraphing) - dropped regardless,
    # since the category's own premise is false on this essay.
    essay = (
        "Technology has changed daily life in many ways for people everywhere in the world today.\n\n"
        "The social problems would be language barrier, which means that a person coming from "
        "another country might not be able to speak and understand the language.\n\n"
        "In conclusion, both benefits and drawbacks exist for anyone who moves to a new country."
    )
    mistakes = [
        {
            "type": "coherence", "category": "Paragraphing Errors", "severity": "significant",
            "meaning_impact": "medium",
            "original": (
                "The social problems would be language barrier, which means that a person coming "
                "from another country might not be able to speak and understand the language."
            ),
            "corrected": "The social problems include a language barrier: newcomers may struggle to communicate.",
            "explanation": "The sentence is too long and lacks clear breaks. Breaking it into two sentences improves clarity.",
        },
        {
            "type": "grammar", "category": "Article Errors", "severity": "minor",
            "meaning_impact": "low", "original": "person coming from another country",
            "corrected": "a person coming from another country", "explanation": "missing article",
        },
    ]
    result = _evaluate(monkeypatch, mistakes, "A refined Band 9 answer text here for this test.", essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Paragraphing Errors" not in categories
    assert "Article Errors" in categories


def test_paragraphing_errors_kept_when_essay_genuinely_has_no_paragraph_breaks(monkeypatch):
    # The confirmed-fixed /n/n case: essay genuinely has zero real
    # newlines (this test uses a plain single-block essay, standing in
    # for that same shape) - the claim is TRUE here, must survive. Uses a
    # short, genuinely-corrected span (not the whole essay, and not an
    # addition-only fix) so this test isolates Item 9 from Items 7/8.
    essay = (
        "Technology has changed daily life in many ways. The social problems would be language "
        "barrier, which means that a person coming from another country might not be able to "
        "speak and understand the language. In conclusion, both benefits and drawbacks exist."
    )
    mistakes = [
        {
            "type": "coherence", "category": "Paragraphing Errors", "severity": "significant",
            "meaning_impact": "medium",
            "original": "person coming from another country might not be able to speak",
            "corrected": "a person coming from another country might not be able to communicate",
            "explanation": "The essay lacks clear paragraph breaks entirely.",
        },
    ]
    result = _evaluate(monkeypatch, mistakes, "A refined Band 9 answer text here for this test.", essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Paragraphing Errors" in categories


def test_paragraphing_errors_dropped_even_when_explanation_argues_a_genuine_issue(monkeypatch):
    # Explicit instruction: dropped "regardless of what the explanation
    # argues" - the essay-has-real-breaks premise being false overrides a
    # plausible-sounding explanation.
    essay = (
        "Technology has changed daily life in many ways for people everywhere in the world.\n\n"
        "The social problems would be language barrier, which means that a person coming from "
        "another country might not be able to speak and understand the language.\n\n"
        "In conclusion, both benefits and drawbacks exist for anyone who moves to a new country."
    )
    mistakes = [
        {
            "type": "coherence", "category": "Paragraphing Errors", "severity": "significant",
            "meaning_impact": "medium", "original": "The social problems would be language barrier",
            "corrected": "The social problems include a language barrier",
            "explanation": "Each main idea should start its own paragraph for clearer logical progression.",
        },
    ]
    result = _evaluate(monkeypatch, mistakes, "A refined Band 9 answer text here for this test.", essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Paragraphing Errors" not in categories


def test_category_match_is_case_insensitive(monkeypatch):
    essay = "First point here today.\n\nSecond point here today.\n\nThird point here today."
    mistakes = [
        {
            "type": "coherence", "category": "paragraphing errors", "severity": "minor",
            "meaning_impact": "low", "original": "First point here today.",
            "corrected": "The first point here today.", "explanation": "lacks paragraph breaks",
        },
    ]
    result = _evaluate(monkeypatch, mistakes, "A refined Band 9 answer text here for this test.", essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "paragraphing errors" not in [c.lower() for c in categories]


def test_other_categories_unaffected_by_this_check(monkeypatch):
    essay = "First point here today.\n\nSecond point here today.\n\nThird point here today."
    mistakes = [
        {
            "type": "coherence", "category": "Paragraph Unity Errors", "severity": "minor",
            "meaning_impact": "low", "original": "Second point here today.",
            "corrected": "The second point is here today.", "explanation": "unity issue",
        },
    ]
    result = _evaluate(monkeypatch, mistakes, "A refined Band 9 answer text here for this test.", essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Paragraph Unity Errors" in categories
