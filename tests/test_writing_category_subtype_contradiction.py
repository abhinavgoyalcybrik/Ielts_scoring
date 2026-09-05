# _mistake_category_subtype_contradiction() - a live QA finding: category
# "Preposition Errors" paired with subtype "missing article" names two
# different error types, and since nothing was actually missing in the
# flagged span, neither one was describing what was really wrong.
#
# NOT wired into evaluate_writing() or any filtering pipeline - measurement
# only, per explicit instruction ("build the check, report the rate...
# do not act on it yet"). These tests cover the predicate itself, including
# two false positives caught and fixed during calibration against real data:
# bare "singular"/"plural" is standard, correct vocabulary for describing a
# genuine Subject-Verb Agreement error and must NOT be treated as evidence
# the category should have been Noun Number Errors; bare "comma" is
# standard vocabulary for a comma splice (a Sentence Boundary/run-on issue)
# and must NOT be treated as evidence the category should have been
# Punctuation Errors.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing


def test_real_case_preposition_category_with_article_subtype():
    assert writing._mistake_category_subtype_contradiction(
        "Preposition Errors", "missing article"
    ) == "article"


def test_matching_category_and_subtype_is_not_a_contradiction():
    assert writing._mistake_category_subtype_contradiction(
        "Article Errors", "missing article"
    ) is None


def test_subject_verb_agreement_with_singular_plural_wording_is_not_flagged():
    # The calibration false positive: this is standard, correct language
    # for describing subject-verb agreement, not evidence of a noun-number
    # mislabel.
    assert writing._mistake_category_subtype_contradiction(
        "Subject-Verb Agreement Errors", "singular subject with plural verb"
    ) is None
    assert writing._mistake_category_subtype_contradiction(
        "Subject-Verb Agreement Errors", "singular/plural mismatch"
    ) is None


def test_genuine_noun_number_subtype_still_detected_against_wrong_category():
    assert writing._mistake_category_subtype_contradiction(
        "Article Errors", "noun number error - plural noun used where singular required"
    ) == "noun_number"


def test_sentence_boundary_with_comma_splice_wording_is_not_flagged():
    # The other calibration false positive: "comma splice" is standard
    # terminology for a run-on/sentence-boundary error, not evidence the
    # category should have been Punctuation Errors.
    assert writing._mistake_category_subtype_contradiction(
        "Sentence Boundary Errors", "comma splice/run-on sentence"
    ) is None


def test_genuine_punctuation_subtype_still_detected_against_wrong_category():
    assert writing._mistake_category_subtype_contradiction(
        "Article Errors", "missing punctuation at the end of the sentence"
    ) == "punctuation"


def test_verb_form_subtype_against_subject_verb_agreement_category():
    # A real, if borderline, case from calibration - the subtype names a
    # distinct type from its own category.
    assert writing._mistake_category_subtype_contradiction(
        "Subject-Verb Agreement Errors", "incorrect verb form 'not all bad situations needs'"
    ) == "verb_form"


def test_unknown_category_returns_none_rather_than_guessing():
    assert writing._mistake_category_subtype_contradiction(
        "Underdeveloped Main Idea", "missing article"
    ) is None


def test_missing_subtype_returns_none():
    assert writing._mistake_category_subtype_contradiction("Preposition Errors", "") is None
    assert writing._mistake_category_subtype_contradiction("Preposition Errors", None) is None


def test_not_wired_into_the_mistake_pipeline():
    # Schema/behaviour must be unaffected - this function exists but is
    # never called by evaluate_writing() or any of its filters.
    import inspect
    source = inspect.getsource(writing.evaluate_writing)
    assert "_mistake_category_subtype_contradiction" not in source
