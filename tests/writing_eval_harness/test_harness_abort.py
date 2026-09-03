# Regression tests for the harness's own abort-on-degraded-result behaviour
# (see HarnessAbortError in run_eval_harness.py). These test the HARNESS's
# safety net, not evaluate_writing() itself - evaluate_writing() is mocked
# throughout so these run instantly, with no real API calls.

import io
import contextlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import run_eval_harness as h


def _healthy_result(text, mistakes=None):
    return {
        "overall_band": 6.0,
        "cefr_level": "B2",
        "criteria_scores": {
            "task_response": 6.0, "coherence_cohesion": 6.0,
            "lexical_resource": 6.0, "grammar_accuracy": 6.0,
        },
        "mistakes": mistakes or [],
        "answer_text": text,
        "refined_answer": "x",
        "word_count": len(text.split()),
        "newline_diagnostics": {"newline_count": 0, "paragraph_block_count": 1},
        "ai_evaluation_failed": False,
    }


def _degraded_result(text):
    # Exactly what evaluate_writing() returns when the real GPT call fails
    # on every retry and falls back to its neutral default - see
    # evaluators/writing.py's default_ai / ai_evaluation_failed handling.
    neutral_bands = {str(n): (n == 5) for n in range(1, 10)}
    return {
        "overall_band": 5.0,
        "cefr_level": "B1",
        "criteria_scores": {
            "task_response": 5.0, "coherence_cohesion": 5.0,
            "lexical_resource": 5.0, "grammar_accuracy": 5.0,
        },
        "mistakes": [],
        "answer_text": text,
        "refined_answer": text,
        "word_count": len(text.split()),
        "newline_diagnostics": {"newline_count": 0, "paragraph_block_count": 1},
        "ai_evaluation_failed": True,
    }


def test_run_eval_raises_on_degraded_result(monkeypatch):
    monkeypatch.setattr(h, "evaluate_writing", lambda data: _degraded_result(data["user_answers"]["text"]))
    with pytest.raises(h.HarnessAbortError):
        h.run_eval("task_2", "question", "some essay text here")


def test_run_eval_passes_through_healthy_result(monkeypatch):
    monkeypatch.setattr(h, "evaluate_writing", lambda data: _healthy_result(data["user_answers"]["text"]))
    result = h.run_eval("task_2", "question", "some essay text here")
    assert result["ai_evaluation_failed"] is False
    assert result["overall_band"] == 6.0


def test_run_eval_error_message_includes_context(monkeypatch):
    monkeypatch.setattr(h, "evaluate_writing", lambda data: _degraded_result(data["user_answers"]["text"]))
    with pytest.raises(h.HarnessAbortError, match="during: my custom context"):
        h.run_eval("task_2", "question", "text", _context="my custom context")


def test_print_report_aborts_without_printing_results_table_on_degraded_result(monkeypatch):
    # A realistic partial-failure scenario: the FIRST clean-corpus call
    # (inside check_1_false_positives, the very first check run) comes back
    # degraded. The whole report must abort right there - no results table
    # for check 1 or any later check, even though later checks were never
    # even reached.
    monkeypatch.setattr(h, "evaluate_writing", lambda data: _degraded_result(data["user_answers"]["text"]))

    captured = io.StringIO()
    with pytest.raises(h.HarnessAbortError):
        with contextlib.redirect_stdout(captured):
            h.print_report()

    output = captured.getvalue()
    assert "HARNESS RUN ABORTED" in output
    assert "END OF REPORT" not in output
    assert "CHECK 1" not in output  # the per-check results header must never print
    assert "TOTAL false-positive mistakes" not in output  # nor any computed number


def test_print_report_completes_normally_when_every_result_is_healthy(monkeypatch):
    monkeypatch.setattr(h, "evaluate_writing", lambda data: _healthy_result(data["user_answers"]["text"]))

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        h.print_report()  # must not raise

    output = captured.getvalue()
    assert "HARNESS RUN ABORTED" not in output
    assert "END OF REPORT" in output
    assert "CHECK 1" in output


def test_partial_failure_mid_run_still_aborts(monkeypatch):
    # A more realistic failure shape: most calls succeed, but the
    # underlying API happens to fail on exactly one call partway through
    # the run (e.g. a transient 429 that exhausts safe_gpt_call's own
    # retries). The abort must still fire - a run is only as trustworthy
    # as its worst call, not its average.
    call_count = {"n": 0}

    def flaky(data):
        call_count["n"] += 1
        if call_count["n"] == 5:
            return _degraded_result(data["user_answers"]["text"])
        return _healthy_result(data["user_answers"]["text"])

    monkeypatch.setattr(h, "evaluate_writing", flaky)

    captured = io.StringIO()
    with pytest.raises(h.HarnessAbortError):
        with contextlib.redirect_stdout(captured):
            h.print_report()

    assert call_count["n"] == 5  # aborted immediately on the degraded call, not after finishing the run
    assert "END OF REPORT" not in captured.getvalue()
