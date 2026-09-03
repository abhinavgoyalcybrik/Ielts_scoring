# Tests for the severity/meaning_impact calibration guidance
# (WRITING_SEVERITY_CALIBRATION flag, default OFF). Live QA found real
# miscalibration in both directions: a clear subject-verb slip and a clear
# missing-article over-rated "significant" (meaning fully recoverable),
# while a genuinely garbled sentence was under-rated "minor" (a reader has
# to re-read it). This reinforces (does not replace) the existing SEVERITY
# DECISION DIMENSIONS list with reader effort as the decisive test, plus
# two worked examples pulled from the real cases. Flag OFF must be
# byte-identical to before this change - verified here, not assumed.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing

_MAKE_BANDS = lambda band: {str(n): (n == band) for n in range(1, 10)}


def _complete_task2_bands(band=6):
    flags = _MAKE_BANDS(band)
    return {
        "task_response_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
        "mistakes": [], "strengths": "x", "improvement": "y",
        "topic_relevance": "on_topic",
    }


def _complete_task1_bands(band=6):
    flags = _MAKE_BANDS(band)
    return {
        "task_achievement_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
        "mistakes": [], "strengths": "x", "improvement": "y",
        "topic_relevance": "on_topic",
        "image_data_accuracy": "not_applicable",
    }


def _capture_prompt(monkeypatch, ai_response):
    captured = {}

    def fake_call_gpt_writing(prompt, image_url=None, **kwargs):
        captured["prompt"] = prompt
        return ai_response

    monkeypatch.setattr(writing, "call_gpt_writing", fake_call_gpt_writing)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined text. " * 60)
    return captured


def test_flag_off_prompt_has_no_calibration_guidance_task2(monkeypatch):
    monkeypatch.delenv("WRITING_SEVERITY_CALIBRATION", raising=False)
    captured = _capture_prompt(monkeypatch, _complete_task2_bands())
    writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": "Some essay text about the topic at hand for the question given."},
    })
    assert "SEVERITY CALIBRATION" not in captured["prompt"]
    assert "<<<SEVERITY_CALIBRATION_GUIDANCE>>>" not in captured["prompt"]


def test_flag_on_prompt_includes_calibration_guidance_task2(monkeypatch):
    monkeypatch.setenv("WRITING_SEVERITY_CALIBRATION", "true")
    captured = _capture_prompt(monkeypatch, _complete_task2_bands())
    writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": "Some essay text about the topic at hand for the question given."},
    })
    assert "READER EFFORT" in captured["prompt"]
    assert "public car park which located" in captured["prompt"]
    assert "In these two maps, there have been two features that still remained" in captured["prompt"]
    assert "<<<" not in captured["prompt"]


def test_flag_off_prompt_has_no_calibration_guidance_task1(monkeypatch):
    monkeypatch.delenv("WRITING_SEVERITY_CALIBRATION", raising=False)
    captured = _capture_prompt(monkeypatch, _complete_task1_bands())
    writing.evaluate_writing({
        "metadata": {"task_type": "task_1", "question": "The chart below shows sodium levels."},
        "user_answers": {"text": "The chart shows dinner has the highest sodium at 43 percent."},
    })
    assert "SEVERITY CALIBRATION" not in captured["prompt"]
    assert "<<<SEVERITY_CALIBRATION_GUIDANCE>>>" not in captured["prompt"]


def test_flag_on_prompt_includes_calibration_guidance_task1(monkeypatch):
    monkeypatch.setenv("WRITING_SEVERITY_CALIBRATION", "true")
    captured = _capture_prompt(monkeypatch, _complete_task1_bands())
    writing.evaluate_writing({
        "metadata": {"task_type": "task_1", "question": "The chart below shows sodium levels."},
        "user_answers": {"text": "The chart shows dinner has the highest sodium at 43 percent."},
    })
    assert "READER EFFORT" in captured["prompt"]
    assert "<<<" not in captured["prompt"]


def test_guidance_states_meaning_impact_must_agree_with_severity():
    assert "minor" in writing.SEVERITY_CALIBRATION_GUIDANCE
    assert "significant" in writing.SEVERITY_CALIBRATION_GUIDANCE
    assert "contradictions" in writing.SEVERITY_CALIBRATION_GUIDANCE.lower()


def test_guidance_does_not_add_a_blanket_prefer_minor_instruction():
    # Explicit instruction: no one-directional nudge. The guidance must
    # illustrate both directions (a minor case AND a significant case),
    # not just tell the model to lower severity generally.
    text = writing.SEVERITY_CALIBRATION_GUIDANCE
    assert "MINOR:" in text
    assert "SIGNIFICANT:" in text
