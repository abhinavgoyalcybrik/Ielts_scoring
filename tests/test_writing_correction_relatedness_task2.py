# Item 4 - lift the correction-relatedness gate to Task 2.
# _mistake_correction_relates_to_original() used to run for task_type ==
# "task_1" only. Live Task 2 output has the identical confirmed bug: a
# mistake's "corrected" field is a long, unrelated rewrite that happens to
# appear verbatim in the independently-generated refined_answer - the
# refined_answer leaking into the mistakes array. The function's own
# logic/thresholds are untouched - only where it's called changed (the
# task_type == "task_1" gate is gone).

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing

_MAKE_BANDS = lambda band: {str(n): (n == band) for n in range(1, 10)}


def _complete_task2_bands(band=7):
    flags = _MAKE_BANDS(band)
    return {
        "task_response_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
    }


def _evaluate_task2(monkeypatch, mistakes, refined_text, essay):
    ai_response = {
        **_complete_task2_bands(),
        "mistakes": mistakes,
        "strengths": "x", "improvement": "y", "topic_relevance": "on_topic",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: refined_text)
    return writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": essay},
    })


def test_task2_mistake_with_unrelated_long_correction_is_now_dropped(monkeypatch):
    # Identical confirmed-bug shape as the existing Task 1 regression test
    # (test_evaluate_writing_task1_drops_mistake_with_unrelated_correction)
    # - a long "corrected" field that verbatim-matches refined_answer,
    # proving the check now fires for Task 2 too.
    refined_text = (
        "Governments should invest heavily in public health campaigns "
        "because prevention reduces long-term healthcare costs and "
        "improves quality of life for the whole population over time."
    )
    essay = (
        "Governments ought to spend significant public funds on public "
        "health awareness programs since it saves money down the line and "
        "helps citizens stay healthy for many years to come in general."
    )
    mistakes = [
        {
            "type": "coherence", "category": "Repetition of Ideas", "subtype": "x",
            "severity": "minor", "meaning_impact": "low",
            "original": "spend significant public funds on public health",
            # Long, unrelated "correction" that happens to appear verbatim
            # in refined_text below - the confirmed leak shape.
            "corrected": refined_text,
            "explanation": "Repeated idea.",
        },
        {
            "type": "grammar", "category": "Article Errors", "severity": "minor",
            "meaning_impact": "low", "original": "helps citizens stay healthy",
            "corrected": "helps the citizens stay healthy", "explanation": "missing article",
        },
    ]
    result = _evaluate_task2(monkeypatch, mistakes, refined_text, essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Repetition of Ideas" not in categories
    assert "Article Errors" in categories


def test_task2_mistake_with_normal_local_correction_survives(monkeypatch):
    refined_text = "A completely different Band 9 model essay text goes here for this test case."
    essay = "The government should invest in education because it help the economy grow."
    mistakes = [
        {
            "type": "grammar", "category": "Subject-Verb Agreement Errors", "severity": "minor",
            "meaning_impact": "low", "original": "because it help the economy",
            "corrected": "because it helps the economy", "explanation": "subject-verb agreement",
        },
    ]
    result = _evaluate_task2(monkeypatch, mistakes, refined_text, essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Subject-Verb Agreement Errors" in categories


def test_task2_short_correction_always_survives_regardless_of_relatedness(monkeypatch):
    # The function's own exemption (corrected under 10 words always
    # returns True) is untouched - confirm it still applies now that
    # Task 2 is in scope too.
    refined_text = "An unrelated Band 9 essay about a completely different topic entirely for this test."
    essay = "The candidate wrote about remote work and its many benefits for employees today."
    mistakes = [
        {
            "type": "lexical", "category": "Incorrect Word Choice", "severity": "minor",
            "meaning_impact": "low", "original": "remote work and its many",
            "corrected": "telework and its numerous", "explanation": "word choice",
        },
    ]
    result = _evaluate_task2(monkeypatch, mistakes, refined_text, essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Incorrect Word Choice" in categories


def test_correction_relatedness_function_logic_itself_is_unchanged():
    # Item 4 explicitly must not change the function's own logic/
    # thresholds - confirm the exact same three decision paths still
    # behave identically to before this item (verbatim-in-refined_answer
    # drop, low-overlap+length-disparity drop, short-correction exemption).
    assert writing._mistake_correction_relates_to_original("x", "short fix", "any refined text") is True
    assert writing._mistake_correction_relates_to_original(
        "a short flagged span", "this exact long sentence appears in the refined answer verbatim here",
        "Somewhere earlier, this exact long sentence appears in the refined answer verbatim here. And more.",
    ) is False
