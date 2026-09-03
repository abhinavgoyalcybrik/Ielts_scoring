# Negative-control test for the coverage matrix's relational assertions
# (coverage_matrix.py). The existing dry run (run_coverage_matrix() against
# a fully mocked evaluate_writing() that returns uniform 6.0 bands
# everywhere) proves the matrix's wiring runs end to end, but it cannot
# prove the relational assertions are capable of catching a genuine
# violation - a uniform mock trivially satisfies every ">=" check by
# equality, so a broken assertion (e.g. one with an inverted comparison)
# would look identical to a working one in that dry run.
#
# This file builds one results_by_profile dict that deliberately violates
# EVERY relational assertion's failure condition, then asserts each
# assertion actually returns violations against it. An assertion that
# comes back empty here is broken - it would stay green against a
# genuinely broken evaluator too, and must be reported, not silently
# trusted.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coverage_matrix import (
    ALL_RELATIONAL_ASSERTIONS,
    assert_p6_format_mismatch_scores_lower,
    P6_FORMAT_MATCHED_ROWS,
    P6_FORMAT_MISMATCHED_ROWS,
    assert_profile_ordering,
    assert_grammar_vocab_independence,
    assert_off_topic_cap,
    assert_paragraphing_caps,
    assert_underlength_penalty,
    assert_template_answer_not_rewarded,
)


def _scores(tr, cc, lr, gr, mistake_count=1):
    return {
        "criteria_scores": {
            "task_response": tr, "coherence_cohesion": cc,
            "lexical_resource": lr, "grammar_accuracy": gr,
        },
        "mistake_count": mistake_count,
    }


# Every entry is deliberately WRONG relative to what a genuine evaluator
# should show - see the inline comment on each profile for exactly which
# assertion(s) it's designed to violate. p1/p2 are a plausible, unremarkable
# baseline; every profile from p3 onward encodes one specific wrong
# relationship named in the user's request (P3 above P1, P4 with a zero
# grammar/lexical gap, off-topic P7 scoring high, P10 with lexical capped)
# plus the remaining assertions' failure conditions for full coverage.
NEGATIVE_CONTROL_RESULTS_BY_PROFILE = {
    "p1_strong": _scores(6.0, 6.0, 6.0, 6.0),
    "p2_competent_occasional_errors": _scores(6.0, 6.0, 6.0, 6.0),
    # WRONG: weak (p3) scoring ABOVE strong (p1) on every criterion.
    "p3_weak": _scores(8.0, 8.0, 8.0, 8.0),
    # WRONG: strong-grammar/weak-vocab with a zero grammar/lexical gap.
    "p4_strong_grammar_weak_vocab": _scores(6.0, 6.0, 6.0, 6.0),
    # WRONG: weak-grammar/strong-vocab with a zero grammar/lexical gap.
    "p5_weak_grammar_strong_vocab": _scores(6.0, 6.0, 6.0, 6.0),
    # WRONG: memorised template scoring task_response AT OR ABOVE p1's.
    "p6_memorised_template": _scores(7.0, 6.0, 6.0, 6.0),
    # WRONG: fully off-topic scoring task_response above the 5.0 cap.
    "p7_off_topic": _scores(8.0, 6.0, 6.0, 6.0),
    # WRONG: partial off-topic drift scoring task_response AT OR ABOVE p1's.
    "p8_partially_off_topic": _scores(7.0, 6.0, 6.0, 6.0),
    # WRONG: underlength scoring task_response ABOVE the full-length p1.
    "p9_underlength": _scores(7.0, 6.0, 6.0, 6.0),
    # WRONG (both halves): coherence NOT capped below p1's (equal, not
    # less), AND lexical resource DROPPED even though paragraphing must
    # never affect it.
    "p10_no_paragraphing": _scores(6.0, 6.0, 4.0, 6.0),
}


def test_every_relational_assertion_fires_against_deliberately_wrong_data():
    """If any assertion in ALL_RELATIONAL_ASSERTIONS comes back empty
    against this deliberately-wrong mock, that assertion is broken - it
    cannot tell a genuinely broken evaluator from a healthy one. Fails
    with the name of every assertion that let a violation slip through,
    so a regression here names the culprit directly."""
    silent = []
    for name, fn in ALL_RELATIONAL_ASSERTIONS:
        violations = fn(NEGATIVE_CONTROL_RESULTS_BY_PROFILE)
        if not violations:
            silent.append(name)
    assert not silent, (
        f"These relational assertions stayed green against deliberately "
        f"wrong data - they cannot detect real regressions: {silent}"
    )


def test_assert_profile_ordering_fires_on_p3_above_p1():
    violations = assert_profile_ordering(NEGATIVE_CONTROL_RESULTS_BY_PROFILE)
    assert violations
    assert len(violations) == 4  # all four criteria are inverted in the mock


def test_assert_grammar_vocab_independence_fires_on_zero_gap():
    violations = assert_grammar_vocab_independence(NEGATIVE_CONTROL_RESULTS_BY_PROFILE)
    assert violations
    assert any("p4" in v for v in violations)
    assert any("p5" in v for v in violations)


def test_assert_off_topic_cap_fires_on_high_off_topic_score():
    violations = assert_off_topic_cap(NEGATIVE_CONTROL_RESULTS_BY_PROFILE)
    assert violations
    assert any("p7" in v for v in violations)
    assert any("p8" in v for v in violations)


def test_assert_paragraphing_caps_fires_on_lexical_capped():
    violations = assert_paragraphing_caps(NEGATIVE_CONTROL_RESULTS_BY_PROFILE)
    assert violations
    assert any("lexical resource dropped" in v for v in violations)
    assert any("coherence to be capped" in v for v in violations)


def test_assert_underlength_penalty_fires_on_p9_above_p1():
    violations = assert_underlength_penalty(NEGATIVE_CONTROL_RESULTS_BY_PROFILE)
    assert violations


def test_assert_template_answer_not_rewarded_fires_on_p6_at_or_above_p1():
    violations = assert_template_answer_not_rewarded(NEGATIVE_CONTROL_RESULTS_BY_PROFILE)
    assert violations


def test_positive_control_still_passes_with_correct_relationships():
    """Sanity check on the negative-control mock itself: a results set
    with all the CORRECT relationships (p3 below p1, p4/p5 with a real
    gap, off-topic capped, paragraphing not touching lexical, ...)
    produces NO violations. Without this, a negative-control test that
    always fires (e.g. because of a typo unrelated to the intended
    violation) would look identical to one that correctly detects the
    deliberately wrong data."""
    correct_results_by_profile = {
        "p1_strong": _scores(7.0, 7.0, 7.0, 7.0),
        "p2_competent_occasional_errors": _scores(6.5, 6.5, 6.5, 6.5),
        "p3_weak": _scores(5.0, 5.0, 5.0, 5.0),
        "p4_strong_grammar_weak_vocab": _scores(6.0, 6.0, 5.0, 7.0),
        "p5_weak_grammar_strong_vocab": _scores(6.0, 6.0, 7.0, 5.0),
        "p6_memorised_template": _scores(5.5, 6.0, 6.0, 6.0),
        "p7_off_topic": _scores(4.0, 6.0, 6.0, 6.0),
        "p8_partially_off_topic": _scores(6.0, 6.0, 6.0, 6.0),
        "p9_underlength": _scores(6.0, 6.0, 6.0, 6.0),
        "p10_no_paragraphing": _scores(6.0, 5.0, 7.0, 6.0),
    }
    for name, fn in ALL_RELATIONAL_ASSERTIONS:
        violations = fn(correct_results_by_profile)
        assert not violations, f"{name} raised a false positive on correct data: {violations}"


# ---------------------------------------------------------------------------
# assert_p6_format_mismatch_scores_lower is a cross-row assertion (a
# {row_label: task_response_score} dict, not one row's results_by_profile),
# so it's intentionally not in ALL_RELATIONAL_ASSERTIONS and needs its own
# negative/positive controls rather than being covered by the loop test
# above.
# ---------------------------------------------------------------------------
def _p6_scores_by_row(matched_score, mismatched_score):
    scores = {row: matched_score for row in P6_FORMAT_MATCHED_ROWS}
    scores.update({row: mismatched_score for row in P6_FORMAT_MISMATCHED_ROWS})
    return scores


def test_p6_format_mismatch_fires_when_scores_are_equal():
    """WRONG: mismatched-format rows score identically to matched-format
    rows - the evaluator is not reacting to the inappropriate format at
    all, which is exactly the failure this assertion exists to catch."""
    violations = assert_p6_format_mismatch_scores_lower(_p6_scores_by_row(6.0, 6.0))
    assert violations
    assert "NOT detected" in violations[0]


def test_p6_format_mismatch_fires_when_mismatched_scores_higher():
    """WRONG: mismatched-format rows score HIGHER than matched-format
    rows - even more clearly wrong than equal scores."""
    violations = assert_p6_format_mismatch_scores_lower(_p6_scores_by_row(5.0, 7.0))
    assert violations
    assert "NOT detected" in violations[0]


def test_p6_format_mismatch_passes_when_mismatched_scores_genuinely_lower():
    """Positive control: mismatched-format rows score below matched-format
    rows, as they should when the evaluator correctly penalises
    inappropriate format/register - no violation."""
    violations = assert_p6_format_mismatch_scores_lower(_p6_scores_by_row(7.0, 5.0))
    assert not violations


def test_p6_format_mismatch_reports_missing_data_not_a_silent_pass():
    """A smoke-subset run only exercises P6 on the 3 representative rows
    (none of which are in P6_FORMAT_MISMATCHED_ROWS), so this assertion
    has no mismatched-row data to compare against. That must come back as
    an explicit 'missing data' violation - never as a silent, misleading
    pass that looks like the format-mismatch check actually ran and
    succeeded."""
    matched_only = {row: 6.0 for row in P6_FORMAT_MATCHED_ROWS}
    violations = assert_p6_format_mismatch_scores_lower(matched_only)
    assert violations
    assert "missing" in violations[0].lower()
