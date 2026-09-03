# Item 7 - a mistake whose "original" spans a large fraction of the essay
# isn't a per-span error, it's essay-level feedback (in every calibration
# case, a "needs paragraphing" complaint) wearing a mistake object's shape.
# Distinct from the existing correction-relatedness check (Item 4): that
# check can only fire when "corrected" is long AND unrelated to "original" -
# it can't catch a shorter, or genuinely essay-derived, "corrected" that
# still isn't a real per-span fix. This guard is independent of what
# "corrected" contains; see _mistake_spans_whole_essay()'s docstring and
# _WHOLE_ESSAY_MISTAKE_FRACTION's definition in evaluators/writing.py for
# the calibration numbers (206 real mistakes from every saved run this
# session produced).

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
# Unit tests on the predicate itself
# ---------------------------------------------------------------------------

def test_predicate_true_at_the_calibrated_confirmed_bug_case():
    # The real confirmed case: 227-word original on a 310-word essay = 73%.
    essay = " ".join(f"word{i}" for i in range(310))
    original = " ".join(f"word{i}" for i in range(227))
    assert writing._mistake_spans_whole_essay(original, essay) is True


def test_predicate_false_at_the_calibrated_genuine_ceiling():
    # The real confirmed-genuine anchor: a 40-word task_response mistake on
    # a 58-word essay = 69%, a real per-criterion critique with a real,
    # different, specific correction - must survive.
    essay = " ".join(f"word{i}" for i in range(58))
    original = " ".join(f"word{i}" for i in range(40))
    assert writing._mistake_spans_whole_essay(original, essay) is False


def test_predicate_boundary_is_inclusive_at_70_percent():
    essay = " ".join(f"word{i}" for i in range(100))
    original_70 = " ".join(f"word{i}" for i in range(70))
    original_69 = " ".join(f"word{i}" for i in range(69))
    assert writing._mistake_spans_whole_essay(original_70, essay) is True
    assert writing._mistake_spans_whole_essay(original_69, essay) is False


def test_predicate_false_on_empty_essay_no_division_by_zero():
    assert writing._mistake_spans_whole_essay("some text", "") is False


def test_predicate_true_when_original_is_the_entire_essay():
    essay = "This is the entire short essay body with no paragraphing at all here."
    assert writing._mistake_spans_whole_essay(essay, essay) is True


# ---------------------------------------------------------------------------
# Integration through evaluate_writing()
# ---------------------------------------------------------------------------

def test_whole_essay_mistake_is_dropped_even_when_correction_is_genuinely_related(monkeypatch):
    # Reproduces the confirmed bug shape the existing correction-relatedness
    # check CANNOT catch: "corrected" is essay-scale too (not short, not
    # generic), and shares plenty of real content with "original" (since
    # both describe the same essay) - so its low-overlap path never fires.
    essay = (
        "In this essay I will outline the social and practical problems that "
        "arise from urban overcrowding in major cities today. The social "
        "problems include isolation and reduced community trust among "
        "residents who rarely interact with their neighbours anymore. The "
        "practical problems include strained public transport and housing "
        "shortages that push prices beyond what ordinary workers can afford "
        "to pay each month without significant hardship for their families."
    )
    whole_essay_original = essay
    whole_essay_corrected = (
        "In this essay, I will outline the social and practical problems. "
        "The social problems include isolation and reduced community trust. "
        "The practical problems include strained transport and housing costs."
    )
    mistakes = [
        {
            "type": "coherence", "category": "Paragraphing Errors", "severity": "minor",
            "meaning_impact": "low", "original": whole_essay_original,
            "corrected": whole_essay_corrected,
            "explanation": "The essay lacks clear paragraph breaks between ideas.",
        },
        {
            "type": "grammar", "category": "Subject-Verb Agreement Errors", "severity": "minor",
            "meaning_impact": "low", "original": "arise from urban overcrowding",
            "corrected": "arises from urban overcrowding", "explanation": "subject-verb agreement",
        },
    ]
    # Sanity check: the existing correction-relatedness check alone would
    # NOT catch this (corrected is short of the 10-word floor's opposite
    # problem - it's long, but high-overlap since it's the same content).
    assert writing._mistake_correction_relates_to_original(
        whole_essay_original, whole_essay_corrected, "irrelevant refined answer text"
    ) is True

    result = _evaluate(monkeypatch, mistakes, whole_essay_corrected, essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Paragraphing Errors" not in categories
    assert "Subject-Verb Agreement Errors" in categories


def test_genuine_mistake_on_a_short_essay_survives_despite_high_fraction(monkeypatch):
    # The real calibration anchor: a genuine, specific task_response
    # critique spanning 69% of a short 58-word essay must NOT be dropped -
    # this is exactly the "normal multi-sentence span" the guard must leave
    # alone.
    original = (
        "The bar chart shows four categories of household spending. The "
        "first category has the second highest value, the second category "
        "is lower, the third category is the lowest of all four, and the "
        "fourth category has the highest value overall."
    )
    # The essay has one more sentence than "original" quotes (the overview
    # sentence at the end) - original/essay = 40/58 words = 69%, the real
    # calibration anchor, not 100%.
    essay = original + (
        " Overall, the fourth category clearly dominates spending while the "
        "third category represents only a small proportion by comparison."
    )
    mistakes = [
        {
            "type": "task_response", "category": "Missing Data", "severity": "significant",
            "meaning_impact": "high", "original": original,
            "corrected": "Turning to the remaining categories, the third category registers the lowest expenditure among all four.",
            "explanation": "The candidate fails to provide any actual data or percentages from the chart.",
        },
    ]
    result = _evaluate(monkeypatch, mistakes, "A refined Band 9 answer with real percentages throughout.", essay, task_type="task_1")

    categories = [m["category"] for m in result["mistakes"]]
    assert "Missing Data" in categories


def test_short_normal_mistake_well_under_threshold_survives(monkeypatch):
    essay = (
        "The line graph illustrates changes in coffee consumption across "
        "three countries between 1990 and 2020, with notable growth in two "
        "of them and a slight decline in the third over the same period."
    )
    mistakes = [
        {
            "type": "grammar", "category": "Article Errors", "severity": "minor",
            "meaning_impact": "low", "original": "changes in coffee consumption",
            "corrected": "the changes in coffee consumption", "explanation": "missing article",
        },
    ]
    result = _evaluate(monkeypatch, mistakes, "A refined Band 9 answer text here.", essay, task_type="task_1")

    categories = [m["category"] for m in result["mistakes"]]
    assert "Article Errors" in categories
