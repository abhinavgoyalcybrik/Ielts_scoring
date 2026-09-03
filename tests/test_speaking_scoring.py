import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluators.speaking import _composite_to_pronunciation_band, compute_pronunciation_score
from evaluators.speaking_audio import (
    _aggregate_acoustic_pronunciation,
    _aggregate_speech_timing,
    _apply_completeness_to_feedback,
    _apply_relevance_to_feedback,
    _attach_refined_answers,
    _band9_word_count,
    _BoundedCache,
    _collapse_immediate_repeats,
    _collect_completeness_notices,
    _content_words,
    _dedupe_part_level_text_against_question_mistakes,
    _estimate_linguistic_floor,
    _flagged_span_contains_self_correction,
    _heuristic_off_topic,
    _ielts_round_half_up,
    _normalize_numbered_band9_answer,
    _pattern_key_terms,
    _quote_appears_in_transcript,
    _quote_occurrence_count,
    _repeated_word_observations,
    _simple_lemma,
    _split_band9_answer_per_question,
    _split_mistakes_by_severity,
    _split_part_feedback_by_severity,
    _topic_words_from_question,
    _validate_question_mistakes,
    _SPEAKING_NO_GENUINE_ERROR_PHRASES,
    calculate_overall_band,
    count_word_repetitions,
    detect_answer_alignment_issues,
    detect_systematic_errors,
)
from evaluators import speaking_audio


# ---------------------------------------------------------------------------
# _BoundedCache: the ASR/feature transcript cache. Must actually return
# fresh (miss) for new keys and cached (hit) only for keys seen before, and
# must evict the oldest entry once past max_size so it can't grow forever.
# ---------------------------------------------------------------------------

def test_bounded_cache_miss_then_hit():
    cache = _BoundedCache(max_size=10)
    value, hit = cache.get("hash-a")
    assert hit is False
    assert value is None

    cache.set("hash-a", "transcript A")
    value, hit = cache.get("hash-a")
    assert hit is True
    assert value == "transcript A"


def test_bounded_cache_different_keys_never_collide():
    cache = _BoundedCache(max_size=10)
    cache.set("hash-a", "transcript A")
    value, hit = cache.get("hash-b")
    assert hit is False
    assert value is None


def test_bounded_cache_evicts_oldest_when_full():
    cache = _BoundedCache(max_size=2)
    cache.set("hash-a", "A")
    cache.set("hash-b", "B")
    cache.set("hash-c", "C")  # should evict hash-a (oldest)

    _, hit_a = cache.get("hash-a")
    _, hit_b = cache.get("hash-b")
    _, hit_c = cache.get("hash-c")
    assert hit_a is False
    assert hit_b is True
    assert hit_c is True


# ---------------------------------------------------------------------------
# _ielts_round_half_up: official IELTS overall-band rounding (.25 and .75
# both round UP), replacing Python's round-half-to-even which silently
# rounded these down roughly half the time.
# ---------------------------------------------------------------------------

def test_ielts_round_half_up_rounds_25_up():
    assert _ielts_round_half_up(6.25) == 6.5


def test_ielts_round_half_up_rounds_75_up():
    assert _ielts_round_half_up(6.75) == 7.0


def test_ielts_round_half_up_rounds_down_below_25():
    assert _ielts_round_half_up(6.1) == 6.0


def test_ielts_round_half_up_rounds_to_nearest_half_normally():
    assert _ielts_round_half_up(6.4) == 6.5
    assert _ielts_round_half_up(6.6) == 6.5
    assert _ielts_round_half_up(6.9) == 7.0


# ---------------------------------------------------------------------------
# calculate_overall_band: equal part weighting (per descriptor sheet Note
# (ii): "rated on their average performance across all parts of the test")
# + correct rounding + graceful handling of missing parts.
# ---------------------------------------------------------------------------

def test_calculate_overall_band_applies_equal_part_weighting():
    p1 = {"fluency": 9, "lexical": 9, "grammar": 9, "pronunciation": 9}
    p2 = {"fluency": 5, "lexical": 5, "grammar": 5, "pronunciation": 5}
    p3 = {"fluency": 5, "lexical": 5, "grammar": 5, "pronunciation": 5}
    # Equal weighting: (9 + 5 + 5) / 3 = 6.333... -> rounds to 6.5.
    # Under the old 25/35/40 tiered weighting this would have been 6.0
    # instead (9*0.25 + 5*0.35 + 5*0.40 = 6.0) - these numbers are chosen
    # specifically so the two schemes diverge, unlike a same-result
    # coincidence that wouldn't actually catch a regression back to tiering.
    assert calculate_overall_band(p1, p2, p3) == 6.5


def test_calculate_overall_band_handles_missing_part():
    p2 = {"fluency": 7, "lexical": 7, "grammar": 7, "pronunciation": 7}
    assert calculate_overall_band(None, p2, None) == 7.0


def test_calculate_overall_band_returns_zero_when_nothing_attempted():
    # Per the descriptor's own Band 0 definition ("Does not attend / Does
    # not complete the test"), a submission with no real content in any
    # part must score 0, not a generous middle-of-the-road guess.
    assert calculate_overall_band(None, None, None) == 0.0


def test_calculate_overall_band_partial_attempt_pulls_score_down():
    # Only Part 2 was attempted - the other two parts count as 0 (not
    # excluded from the average), so the overall band should be well below
    # what Part 2 alone would suggest, not silently ignored.
    p2 = {"fluency": 7, "lexical": 7, "grammar": 7, "pronunciation": 7}
    zero_part = {"fluency": 0.0, "lexical": 0.0, "grammar": 0.0, "pronunciation": 0.0}
    result = calculate_overall_band(zero_part, p2, zero_part)
    assert result < 7.0
    # (0 + 7 + 0) / 3 = 2.333... -> rounds to 2.5 under equal weighting.
    assert result == 2.5


def test_calculate_overall_band_uses_default_score_for_missing_criteria():
    # A part dict missing a criterion key should fall back to 5.0 for it,
    # not crash.
    p1 = {"fluency": 9}
    assert calculate_overall_band(p1, None, None) == calculate_overall_band(
        {"fluency": 9, "lexical": 5.0, "grammar": 5.0, "pronunciation": 5.0}, None, None
    )


# ---------------------------------------------------------------------------
# _content_words / _heuristic_off_topic: the topic-relevance backstop.
# Regression test for the real failure case observed in production - a
# transport-topic answer given to a university-clubs question, which the
# first version of the heuristic missed because of a single stray shared
# word ("most").
# ---------------------------------------------------------------------------

def test_content_words_filters_generic_quantifiers():
    words = _content_words("most universities have many clubs")
    assert "most" not in words
    assert "many" not in words
    assert "clubs" in words
    assert "universities" in words


def test_heuristic_off_topic_detects_real_world_mismatch():
    qas = [
        {
            "question": "What types of clubs are available at most universities?",
            "user_answer": "I think the most popular public transports in my country are bus and train.",
        },
        {
            "question": "How can joining a club benefit a student during their university years?",
            "user_answer": "Very rarely I have my own car so I usually use that to travel.",
        },
        {
            "question": "Are there any costs associated with joining university clubs?",
            "user_answer": "I think punctuality is something where we are lacking a lot.",
        },
        {
            "question": "How often do university clubs typically meet?",
            "user_answer": "We talk about flight travel, yes it is expensive.",
        },
    ]
    assert _heuristic_off_topic(qas) is True


def test_heuristic_off_topic_does_not_flag_relevant_answers():
    qas = [
        {
            "question": "What types of clubs are available at most universities?",
            "user_answer": "Most universities offer academic clubs, sports clubs, and cultural societies.",
        },
        {
            "question": "How can joining a club benefit a student?",
            "user_answer": "Joining a club helps students build friendships and leadership skills.",
        },
    ]
    assert _heuristic_off_topic(qas) is False


def test_heuristic_off_topic_handles_empty_input():
    assert _heuristic_off_topic([]) is False
    assert _heuristic_off_topic(None) is False


# ---------------------------------------------------------------------------
# _aggregate_acoustic_pronunciation
# ---------------------------------------------------------------------------

def test_aggregate_acoustic_pronunciation_averages_available_scores():
    raw = [
        {"audio_metrics": {"pronunciation_score": 6.5}},
        {"audio_metrics": {"pronunciation_score": 7.0}},
        {"audio_metrics": {}},
        {},
    ]
    assert _aggregate_acoustic_pronunciation(raw) == 6.75


def test_aggregate_acoustic_pronunciation_returns_none_when_no_data():
    assert _aggregate_acoustic_pronunciation([]) is None
    assert _aggregate_acoustic_pronunciation([{"audio_metrics": {}}]) is None


# ---------------------------------------------------------------------------
# _aggregate_speech_timing / SPEAKING_VOICED_WPM - the WPM denominator fix.
# Both raw and voiced aggregates must always be present regardless of the
# flag (so the gap stays visible); only "avg_wpm"/"wpm_basis" (the fields
# generate_scores() actually reads) should move with the flag.

_TIMING_RAW_PART = [
    {"audio_metrics": {
        "speech_rate_wpm_raw": 100.0, "speech_rate_wpm_voiced": 150.0,
        "duration_sec": 60.0, "voiced_duration_sec": 40.0,
    }},
    {"audio_metrics": {
        "speech_rate_wpm_raw": 120.0, "speech_rate_wpm_voiced": 160.0,
        "duration_sec": 30.0, "voiced_duration_sec": 22.5,
    }},
]


def test_aggregate_speech_timing_defaults_to_raw_when_flag_off(monkeypatch):
    monkeypatch.setattr(speaking_audio, "SPEAKING_VOICED_WPM", False)
    result = _aggregate_speech_timing(_TIMING_RAW_PART)
    assert result["wpm_basis"] == "raw"
    assert result["avg_wpm"] == result["avg_wpm_raw"] == 110.0
    assert result["avg_wpm_voiced"] == 155.0
    assert result["total_duration_sec"] == 90.0
    assert result["total_voiced_duration_sec"] == 62.5
    assert result["silence_fraction"] == round(1 - 62.5 / 90.0, 4)


def test_aggregate_speech_timing_switches_to_voiced_when_flag_on(monkeypatch):
    monkeypatch.setattr(speaking_audio, "SPEAKING_VOICED_WPM", True)
    result = _aggregate_speech_timing(_TIMING_RAW_PART)
    assert result["wpm_basis"] == "voiced"
    assert result["avg_wpm"] == result["avg_wpm_voiced"] == 155.0
    # Raw stays present and correct even though it's no longer "active".
    assert result["avg_wpm_raw"] == 110.0


def test_aggregate_speech_timing_flag_on_falls_back_to_raw_if_voiced_missing(monkeypatch):
    monkeypatch.setattr(speaking_audio, "SPEAKING_VOICED_WPM", True)
    raw_only = [{"audio_metrics": {"speech_rate_wpm_raw": 100.0, "duration_sec": 60.0}}]
    result = _aggregate_speech_timing(raw_only)
    assert result["wpm_basis"] == "raw"
    assert result["avg_wpm"] == 100.0


def test_aggregate_speech_timing_returns_none_when_no_data():
    assert _aggregate_speech_timing([]) is None
    assert _aggregate_speech_timing([{"audio_metrics": {}}]) is None


# ---------------------------------------------------------------------------
# compute_pronunciation_score: must span the genuine 1.0-9.0 range using the
# richer acoustic signals, not be structurally capped near the middle.
# ---------------------------------------------------------------------------

GREAT_ACOUSTICS = {
    "phoneme_accuracy": 0.95, "stress_accuracy": 0.95, "intonation_score": 0.8,
    "audio_quality_score": 8.0, "mispronunciation_rate": 0.05,
    "pause_count": 1, "avg_pause_duration": 0.2, "speech_rate": 135,
}

POOR_ACOUSTICS = {
    "phoneme_accuracy": 0.3, "stress_accuracy": 0.3, "intonation_score": 0.4,
    "audio_quality_score": 4.0, "mispronunciation_rate": 0.7,
    "pause_count": 14, "avg_pause_duration": 1.8, "speech_rate": 250,
}


def test_compute_pronunciation_score_rewards_excellent_acoustic_evidence():
    score = compute_pronunciation_score(GREAT_ACOUSTICS, asr_confidence=0.95)
    assert score > 8.0


def test_compute_pronunciation_score_penalizes_poor_acoustic_evidence():
    score = compute_pronunciation_score(POOR_ACOUSTICS, asr_confidence=0.9)
    assert score < 3.5


def test_compute_pronunciation_score_dampens_toward_middle_on_low_confidence():
    high_conf = compute_pronunciation_score(GREAT_ACOUSTICS, asr_confidence=0.95)
    low_conf = compute_pronunciation_score(GREAT_ACOUSTICS, asr_confidence=0.5)
    assert low_conf < high_conf


def test_compute_pronunciation_score_penalizes_excessive_pauses():
    baseline = dict(GREAT_ACOUSTICS)
    lots_of_pauses = dict(GREAT_ACOUSTICS, pause_count=14, avg_pause_duration=1.8)
    assert compute_pronunciation_score(lots_of_pauses) < compute_pronunciation_score(baseline)


def test_compute_pronunciation_score_penalizes_bad_speech_rate():
    baseline = dict(GREAT_ACOUSTICS)
    too_fast = dict(GREAT_ACOUSTICS, speech_rate=280)
    too_slow = dict(GREAT_ACOUSTICS, speech_rate=40)
    assert compute_pronunciation_score(too_fast) < compute_pronunciation_score(baseline)
    assert compute_pronunciation_score(too_slow) < compute_pronunciation_score(baseline)


# ---------------------------------------------------------------------------
# _composite_to_pronunciation_band: explicit descriptor-anchored mapping
# from the 0-1 acoustic composite onto the IELTS 1-9 pronunciation scale
# (replaces a bare linear formula with boundaries justified against each
# band's actual descriptor language - e.g. band 8/9 requiring "accent has
# minimal/no effect on intelligibility", not just "a high number").
# ---------------------------------------------------------------------------

def test_composite_to_pronunciation_band_endpoints():
    assert _composite_to_pronunciation_band(0.0) == 1.0
    assert _composite_to_pronunciation_band(1.0) == 9.0


def test_composite_to_pronunciation_band_is_monotonic():
    samples = [i / 20 for i in range(21)]
    bands = [_composite_to_pronunciation_band(c) for c in samples]
    assert bands == sorted(bands)


def test_composite_to_pronunciation_band_low_composite_stays_low():
    # A genuinely poor composite (weak articulation/stress/intonation
    # together) must land in the 2-3 range the descriptors describe for
    # that quality of evidence, not an overly generous middle band.
    assert _composite_to_pronunciation_band(0.215) < 3.5


def test_composite_to_pronunciation_band_high_composite_stays_high():
    assert _composite_to_pronunciation_band(0.95) > 8.0


def test_composite_to_pronunciation_band_clamps_out_of_range_input():
    assert _composite_to_pronunciation_band(-0.5) == 1.0
    assert _composite_to_pronunciation_band(1.5) == 9.0


def test_compute_pronunciation_score_handles_empty_features():
    assert compute_pronunciation_score({}) == 5.5
    assert compute_pronunciation_score(None) == 5.5


# ---------------------------------------------------------------------------
# _band9_word_count / _normalize_numbered_band9_answer
# ---------------------------------------------------------------------------

def test_band9_word_count():
    assert _band9_word_count("this has four words") == 4
    assert _band9_word_count("") == 0
    assert _band9_word_count(None) == 0


def test_normalize_numbered_band9_answer_extracts_labeled_answers():
    raw = "Answer 1: First response here.\n\nAnswer 2: Second response here."
    result = _normalize_numbered_band9_answer(raw, 2)
    assert "Answer 1: First response here." in result
    assert "Answer 2: Second response here." in result


def test_normalize_numbered_band9_answer_falls_back_to_line_split():
    raw = "First response line.\nSecond response line."
    result = _normalize_numbered_band9_answer(raw, 2)
    assert "Answer 1: First response line." in result
    assert "Answer 2: Second response line." in result


# ---------------------------------------------------------------------------
# _validate_question_mistakes: enforces the {type, original, corrected,
# explanation} schema shared with the Writing module's error-correction UI.
# ---------------------------------------------------------------------------

def test_validate_question_mistakes_accepts_well_formed_items():
    items = [
        {"type": "grammar", "original": "he go", "corrected": "he goes", "explanation": "subject-verb agreement"},
        {"type": "vocabulary", "original": "big", "corrected": "substantial", "explanation": "more precise"},
    ]
    valid = _validate_question_mistakes(items)
    assert len(valid) == 2
    assert valid[0]["type"] == "grammar"


def test_validate_question_mistakes_rejects_present_to_past_tense_false_positive():
    # Regression test for a real, RECURRING false positive (observed twice
    # on live test data, even after a prompt-only fix attempt): present
    # tense describing a show's ongoing/general nature after past-tense
    # narration of watching it is standard English, not a tense error.
    items = [
        {
            "type": "grammar",
            "original": "the show's aim is to test the skills of the participants",
            "corrected": "the show's aim was to test the skills of the participants",
            "explanation": "The verb tense should remain consistent when discussing a specific past event.",
            "severity": "significant",
        },
    ]
    assert _validate_question_mistakes(items) == []


def test_validate_question_mistakes_rejects_tense_false_positive_regardless_of_explanation_wording():
    # Regression test for a THIRD occurrence of the same false positive,
    # which slipped past the original keyword-based backstop because GPT
    # phrased the justification a new way ("The tense should be past tense
    # to match 'recently watched'") that didn't match any tracked keyword
    # phrase - proving explanation-text keyword matching was too fragile.
    # The fix must catch this on the structural edit alone, independent of
    # how the explanation is worded.
    items = [
        {
            "type": "grammar",
            "original": "the show's aim is to test the skills of the participants",
            "corrected": "the show's aim was to test the skills of the participants",
            "explanation": "The tense should be past tense to match 'recently watched'.",
            "severity": "significant",
        },
    ]
    assert _validate_question_mistakes(items) == []


def test_validate_question_mistakes_accepts_genuinely_time_anchored_tense_fix():
    # A present-tense verb anchored to one specific past moment (e.g.
    # "yesterday") is a real candidate for a genuine tense error, unlike
    # the false-positive pattern (a general/ongoing truth with no such
    # anchor) - the backstop must not blanket-suppress every is/was swap.
    items = [
        {
            "type": "grammar",
            "original": "the weather is bad yesterday",
            "corrected": "the weather was bad yesterday",
            "explanation": "Past tense is required since 'yesterday' anchors this to a specific past day.",
            "severity": "significant",
        },
    ]
    assert len(_validate_question_mistakes(items)) == 1


def test_validate_question_mistakes_accepts_genuine_tense_error():
    # Sanity check the tense backstop doesn't over-trigger on an unrelated
    # genuine grammar issue that happens to share no tense-consistency
    # language and no present-to-past verb swap at all.
    items = [
        {
            "type": "grammar",
            "original": "he go to school every day",
            "corrected": "he goes to school every day",
            "explanation": "Subject-verb agreement error: third person singular requires 'goes'.",
            "severity": "significant",
        },
    ]
    assert len(_validate_question_mistakes(items)) == 1


def test_validate_question_mistakes_strips_hyphens_from_corrected_text():
    # Regression test: a real correction changed "out of the box plots" to
    # "out-of-the-box plots" - hyphenation is a written typesetting
    # convention with no spoken equivalent, so it must never survive into
    # "corrected" text, even when the item also fixes a genuine issue
    # (here, dropping the redundant "they").
    items = [
        {
            "type": "grammar",
            "original": "the kind of shows we watch they are based on out of the box plots",
            "corrected": "the kind of shows we watch are based on out-of-the-box plots",
            "explanation": "Redundant subject pronoun and structure affects clarity.",
            "severity": "significant",
        },
    ]
    valid = _validate_question_mistakes(items)
    assert len(valid) == 1
    assert "-" not in valid[0]["corrected"]
    assert valid[0]["corrected"] == "the kind of shows we watch are based on out of the box plots"


def test_validate_question_mistakes_forces_single_missing_article_to_minor():
    # Regression test for a real case: "It is important to take at least
    # career break" -> "...at least a career break" was tagged
    # "significant", even though "a single dropped article" is this
    # file's OWN named example of a "minor" issue. This must be forced to
    # minor regardless of what severity GPT assigned.
    items = [
        {
            "type": "grammar",
            "original": "It is important to take at least career break",
            "corrected": "It is important to take at least a career break",
            "explanation": "The article 'a' is missing before 'career break', which is necessary for grammatical accuracy.",
            "severity": "significant",
        },
    ]
    valid = _validate_question_mistakes(items)
    assert len(valid) == 1
    assert valid[0]["severity"] == "minor"


def test_validate_question_mistakes_does_not_force_minor_on_unrelated_insertion():
    # Sanity check: inserting a single word that ISN'T an article (so not
    # the "dropped article" pattern at all) must not be force-downgraded -
    # only a/an/the insertions qualify.
    items = [
        {
            "type": "grammar",
            "original": "I go school every day",
            "corrected": "I go to school every day",
            "explanation": "Missing preposition 'to' before 'school'.",
            "severity": "significant",
        },
    ]
    valid = _validate_question_mistakes(items)
    assert len(valid) == 1
    assert valid[0]["severity"] == "significant"


def test_validate_question_mistakes_rejects_because_of_due_to_swap():
    # Regression test: a real case flagged "because of a shortage of time"
    # as a grammar error and "corrected" it to "due to a shortage of
    # time", claiming "due to" is "more grammatically appropriate" - both
    # phrases are fully correct and interchangeable; nothing was wrong.
    items = [
        {
            "type": "grammar",
            "original": "because of a shortage of time",
            "corrected": "due to a shortage of time",
            "explanation": "Using 'due to' is more grammatically appropriate in this context.",
            "severity": "minor",
        },
    ]
    assert _validate_question_mistakes(items) == []


def test_validate_question_mistakes_accepts_genuine_issue_alongside_due_to():
    # Sanity check: a genuine issue elsewhere in the same sentence must
    # still be accepted - the backstop only rejects when the swap is the
    # ONLY difference.
    items = [
        {
            "type": "grammar",
            "original": "because of a shortage of time i don't went there",
            "corrected": "due to a shortage of time I didn't go there",
            "explanation": "Past tense verb form was incorrect.",
            "severity": "significant",
        },
    ]
    assert len(_validate_question_mistakes(items)) == 1


def test_validate_question_mistakes_rejects_shortage_of_time_lack_of_time_swap():
    # Regression test: a real case flagged "shortage of time" as a
    # grammar error and "corrected" it to "lack of time" - even though
    # "shortage of time" is explicitly named as a protected example in the
    # CONFIDENCE BAR prompt text itself, proving the prompt instruction
    # alone isn't reliable enough here either.
    items = [
        {
            "type": "grammar",
            "original": "shortage of time",
            "corrected": "lack of time",
            "explanation": "The phrase 'shortage of time' is less idiomatic; 'lack of time' is more commonly used.",
            "severity": "minor",
        },
    ]
    assert _validate_question_mistakes(items) == []


def test_validate_question_mistakes_rejects_self_correction_flagged_as_mistake():
    # Regression test: the official descriptors allow self-correction even
    # at Band 9 ("only rare repetition or self-correction"), so flagging a
    # candidate's self-correction as a fluency mistake would penalize
    # exactly the behavior the descriptors reward.
    items = [
        {
            "type": "fluency",
            "original": "I go there every day well actually not every day",
            "corrected": "I go there most days",
            "explanation": "The candidate self-corrected mid-sentence, which disrupts fluency.",
            "severity": "significant",
        },
    ]
    assert _validate_question_mistakes(items) == []


def test_validate_question_mistakes_accepts_genuine_fluency_issue():
    # Sanity check the self-correction backstop doesn't over-trigger on an
    # unrelated fluency issue that happens to share no self-correction
    # language at all.
    items = [
        {
            "type": "fluency",
            "original": "so uh it doesn't affect the uh the results",
            "corrected": "so it doesn't affect the results",
            "explanation": "Repetitive filler words disrupt the flow of speech.",
            "severity": "significant",
        },
    ]
    assert len(_validate_question_mistakes(items)) == 1


def test_validate_question_mistakes_rejects_invalid_type():
    items = [{"type": "spelling", "original": "x", "corrected": "y", "explanation": "z"}]
    assert _validate_question_mistakes(items) == []


def test_validate_question_mistakes_rejects_missing_fields():
    items = [{"type": "grammar", "original": "x"}]
    assert _validate_question_mistakes(items) == []


def test_validate_question_mistakes_handles_non_list_input():
    assert _validate_question_mistakes(None) == []
    assert _validate_question_mistakes("not a list") == []


# ---------------------------------------------------------------------------
# _quote_occurrence_count: counts every verbatim occurrence, not just the
# first - the basis for the new "original must match exactly once" check
# below, and a superset of what _quote_appears_in_transcript needs.
# ---------------------------------------------------------------------------

def test_quote_occurrence_count_finds_single_match():
    assert _quote_occurrence_count("went to the market", "yesterday I went to the market with my friend") == 1


def test_quote_occurrence_count_finds_zero_when_absent():
    assert _quote_occurrence_count("bought a new car", "yesterday I went to the market with my friend") == 0


def test_quote_occurrence_count_finds_multiple_matches():
    transcript = "I like coffee. My brother also likes coffee, but my sister does not like coffee at all."
    assert _quote_occurrence_count("like coffee", transcript) == 2


def test_quote_occurrence_count_ignores_punctuation_and_case():
    assert _quote_occurrence_count("Went To The Market!", "yesterday i went to the market with my friend") == 1


def test_quote_appears_in_transcript_still_works_as_a_yes_no_wrapper():
    # Existing caller (detect_systematic_errors) only needs a boolean -
    # confirm the refactor into a thin wrapper over
    # _quote_occurrence_count preserves that contract exactly.
    transcript = "I like coffee. My brother also likes coffee."
    assert _quote_appears_in_transcript("like coffee", transcript) is True
    assert _quote_appears_in_transcript("hate tea", transcript) is False


# ---------------------------------------------------------------------------
# _validate_question_mistakes's new transcript-aware checks (ported from
# evaluators/writing.py's proven verbatim/self-admission filters, plus the
# request's own additional "multi-word, exactly-once" requirement). All
# gated on `transcript` being supplied - every test above this point calls
# _validate_question_mistakes(items) with no transcript, and must keep
# passing unaffected (confirmed by the full suite staying green).
# ---------------------------------------------------------------------------

def test_validate_question_mistakes_skips_verbatim_check_without_transcript():
    # Backward compatibility: no transcript supplied (the default) means
    # the new verbatim/exact-once check never runs at all, even for a
    # quote that would fail it - this is what keeps every pre-existing
    # test in this file (all called without a transcript) passing
    # unchanged.
    items = [{"type": "grammar", "original": "go", "corrected": "goes", "explanation": "agreement"}]
    assert len(_validate_question_mistakes(items)) == 1


def test_validate_question_mistakes_rejects_hallucinated_quote_not_in_transcript():
    transcript = "yesterday I went to the market with my friend and bought some fruit"
    items = [{
        "type": "grammar", "original": "he go to school every day",
        "corrected": "he goes to school every day", "explanation": "subject-verb agreement",
    }]
    assert _validate_question_mistakes(items, transcript=transcript) == []


def test_validate_question_mistakes_accepts_quote_present_exactly_once():
    transcript = "yesterday I went to the market with my friend and bought some fruit"
    items = [{
        "type": "grammar", "original": "went to the market",
        "corrected": "was going to the market", "explanation": "tense clarity",
    }]
    assert len(_validate_question_mistakes(items, transcript=transcript)) == 1


def test_validate_question_mistakes_rejects_bare_single_word_quote():
    # "consumed" (or any bare word) matches everywhere and tells the
    # candidate nothing about which specific instance is meant.
    transcript = "we consumed a lot of coffee and later consumed some tea as well"
    items = [{
        "type": "vocabulary", "original": "consumed",
        "corrected": "drank", "explanation": "more natural word choice",
    }]
    assert _validate_question_mistakes(items, transcript=transcript) == []


def test_validate_question_mistakes_rejects_ambiguous_quote_matching_more_than_once():
    transcript = "my brother likes coffee and my sister also likes coffee every morning"
    items = [{
        "type": "vocabulary", "original": "likes coffee",
        "corrected": "enjoys coffee", "explanation": "more natural word choice",
    }]
    assert _validate_question_mistakes(items, transcript=transcript) == []


def test_validate_question_mistakes_rejects_self_admission_phrases():
    transcript = "the weather was quite nice for the whole trip we took last summer"
    admission_explanations = [
        "This does not affect meaning and is acceptable as written.",
        "This is a minor stylistic point, not a real error.",
        "Either version can be written this way, it is also possible.",
    ]
    for explanation in admission_explanations:
        items = [{
            "type": "grammar", "original": "the weather was quite nice",
            "corrected": "the weather was very nice", "explanation": explanation,
        }]
        assert _validate_question_mistakes(items, transcript=transcript) == [], explanation


def test_speaking_no_genuine_error_phrases_includes_all_requested_additions():
    for phrase in ("does not affect meaning", "is acceptable", "minor stylistic point", "can be written", "is also possible"):
        assert phrase in _SPEAKING_NO_GENUINE_ERROR_PHRASES


def test_validate_question_mistakes_dedupes_overlapping_spans():
    # Fixtures deliberately avoid any present/past tense-word swap
    # (is/are/am/does/has -> was/were/was/did/had), which/that, "as far as
    # ... concerned", a registered synonym-phrase pair, or an article-only
    # insertion - so the only thing under test is the NEW dedup logic, not
    # an interaction with one of the pre-existing structural backstops.
    transcript = "many people believe that technology has changed the way we communicate today"
    items = [
        {
            "type": "grammar", "original": "technology has changed the way we communicate",
            "corrected": "technology has transformed the way we communicate", "explanation": "word choice",
        },
        {
            # A shorter span fully contained within the first item's span -
            # same underlying issue, reported twice.
            "type": "vocabulary", "original": "changed the way we communicate",
            "corrected": "transformed the way we communicate", "explanation": "word choice",
        },
    ]
    valid = _validate_question_mistakes(items, transcript=transcript)
    assert len(valid) == 1


def test_validate_question_mistakes_keeps_distinct_non_overlapping_spans():
    transcript = "technology has changed the way we communicate and rising costs affect many families today"
    items = [
        {
            "type": "grammar", "original": "technology has changed the way we communicate",
            "corrected": "technology has transformed the way we communicate", "explanation": "word choice",
        },
        {
            # Genuinely different, non-overlapping span - must NOT be
            # deduped away by a naive set-overlap check just because both
            # items are grammatically plain sentences.
            "type": "vocabulary", "original": "rising costs affect many families",
            "corrected": "increasing costs affect many families", "explanation": "word choice",
        },
    ]
    valid = _validate_question_mistakes(items, transcript=transcript)
    assert len(valid) == 2


# ---------------------------------------------------------------------------
# _apply_relevance_to_feedback: written feedback must not contradict a
# topic-relevance score cap.
# ---------------------------------------------------------------------------

def test_apply_relevance_to_feedback_prefixes_off_topic_note():
    feedback = {"improvement": "Use more varied vocabulary."}
    result = _apply_relevance_to_feedback(dict(feedback), "completely_off_topic")
    assert "did not address the question" in result["improvement"]
    assert "Use more varied vocabulary." in result["improvement"]


def test_apply_relevance_to_feedback_prefixes_partial_note():
    feedback = {"improvement": "Use more varied vocabulary."}
    result = _apply_relevance_to_feedback(dict(feedback), "partially_off_topic")
    assert "drifted away from the question" in result["improvement"]


def test_apply_relevance_to_feedback_leaves_on_topic_untouched():
    feedback = {"improvement": "Use more varied vocabulary."}
    result = _apply_relevance_to_feedback(dict(feedback), "on_topic")
    assert result["improvement"] == "Use more varied vocabulary."


def test_apply_relevance_to_feedback_handles_non_dict_input():
    result = _apply_relevance_to_feedback(None, "completely_off_topic")
    assert "did not address the question" in result["improvement"]


# ---------------------------------------------------------------------------
# _collect_completeness_notices / _apply_completeness_to_feedback: surfaces
# it explicitly when a candidate only answers part of a multi-part question
# (e.g. "Which jobs pay the most? Why?" answered with only the "which"
# half), so the written feedback explains why the score is affected instead
# of just showing a lower number with generic grammar/vocabulary notes.
# ---------------------------------------------------------------------------

def test_collect_completeness_notices_gathers_non_empty_only():
    qas_clean = [
        {"question": "q1", "user_answer": "a1", "completeness_notice": "You didn't explain why."},
        {"question": "q2", "user_answer": "a2", "completeness_notice": ""},
        {"question": "q3", "user_answer": "a3", "completeness_notice": "You didn't say when."},
    ]
    notices = _collect_completeness_notices(qas_clean)
    assert notices == ["You didn't explain why.", "You didn't say when."]


def test_collect_completeness_notices_handles_empty_input():
    assert _collect_completeness_notices([]) == []
    assert _collect_completeness_notices(None) == []


def test_apply_completeness_to_feedback_prefixes_when_notices_present():
    feedback = {"improvement": "Use more varied vocabulary."}
    result = _apply_completeness_to_feedback(dict(feedback), ["You didn't explain why."])
    assert "wasn't fully answered" in result["improvement"]
    assert "You didn't explain why." in result["improvement"]
    assert "Use more varied vocabulary." in result["improvement"]


def test_apply_completeness_to_feedback_leaves_feedback_untouched_when_no_notices():
    feedback = {"improvement": "Use more varied vocabulary."}
    result = _apply_completeness_to_feedback(dict(feedback), [])
    assert result["improvement"] == "Use more varied vocabulary."


def test_apply_completeness_to_feedback_handles_non_dict_input():
    result = _apply_completeness_to_feedback(None, ["You didn't explain why."])
    assert "wasn't fully answered" in result["improvement"]


# ---------------------------------------------------------------------------
# _estimate_linguistic_floor: code-level backstop against GPT crashing an
# off-topic-but-linguistically-substantial answer down to Band 1, despite
# the prompt explicitly instructing it not to. Regression test uses the
# exact real transcript that produced a 1/1/1 score in production even with
# that instruction in place.
# ---------------------------------------------------------------------------

_DIVORCE_RATE_ANSWER = (
    "I remember a time where I was given a questionnaire and the questionnaire was "
    "about that why these days but in your opinion if the why that the most rate has "
    "been increasing I was being given a questionnaire where I have to give my opinion "
    "that why it is increasing or why not For me, the opinion was that the divorce rate "
    "in India was increasing because earlier women were unemployed and were only "
    "dependent on the financial capacity of her husband and she was been told that you "
    "have to adjust in the future family no matter whatever people say is it to But "
    "these days women are being educated at a level where they might get a chance to "
    "own more than Amanda's. And she is not ready to get adjusted in any of the things. "
    "She wanted to feel empowered and she wanted to feel heard. heard and if in any "
    "case she feels like that she has been ignored she walks away from that relationship."
)


def test_estimate_linguistic_floor_regression_real_case():
    qas = [{"question": "Describe someone you know who does something well.", "user_answer": _DIVORCE_RATE_ANSWER}]
    floor = _estimate_linguistic_floor(qas)
    # This exact transcript was observed scoring 1/1/1 in production despite
    # being fluent, complex, coherent (off-topic) English - the floor must
    # sit comfortably above Band 1-2 territory.
    assert floor >= 4.0


def test_estimate_linguistic_floor_trivial_answer_stays_low():
    qas = [{"question": "q", "user_answer": "I like it."}]
    assert _estimate_linguistic_floor(qas) < 2.0


def test_estimate_linguistic_floor_empty_input():
    assert _estimate_linguistic_floor([]) == 1.0
    assert _estimate_linguistic_floor(None) == 1.0


def test_estimate_linguistic_floor_never_exceeds_four_point_five():
    huge_complex_answer = (
        "Although this is a very long and complex answer, which contains many "
        "subordinate clauses, because I want to test the ceiling, however the score "
        "should never exceed the maximum floor, since that is reserved for genuinely "
        "excellent scores that the model itself should assign, whereas this floor is "
        "only meant to prevent an unjustified collapse toward the bottom of the scale. "
    ) * 5
    qas = [{"question": "q", "user_answer": huge_complex_answer}]
    assert _estimate_linguistic_floor(qas) <= 4.5


# ---------------------------------------------------------------------------
# _check_rate_limit: sliding time-window limiter (replaces the old lifetime
# counter that, once past threshold, would reject every request forever for
# the rest of the process's life instead of resetting over time).
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def _reset_rate_limit_state():
    original = list(speaking_audio._rate_limit_timestamps)
    speaking_audio._rate_limit_timestamps.clear()
    yield
    speaking_audio._rate_limit_timestamps.clear()
    speaking_audio._rate_limit_timestamps.extend(original)


def test_check_rate_limit_allows_up_to_threshold(_reset_rate_limit_state):
    for _ in range(speaking_audio._RATE_LIMIT_THRESHOLD):
        assert speaking_audio._check_rate_limit() is True


def test_check_rate_limit_blocks_once_over_threshold(_reset_rate_limit_state):
    for _ in range(speaking_audio._RATE_LIMIT_THRESHOLD):
        speaking_audio._check_rate_limit()
    assert speaking_audio._check_rate_limit() is False


def test_check_rate_limit_recovers_after_window_expires(_reset_rate_limit_state):
    now = 1_000_000.0
    # Fill the window with timestamps that are already outside the window.
    old_timestamp = now - speaking_audio._RATE_LIMIT_WINDOW_SECONDS - 1
    for _ in range(speaking_audio._RATE_LIMIT_THRESHOLD):
        speaking_audio._rate_limit_timestamps.append(old_timestamp)

    real_time = speaking_audio.time.time
    speaking_audio.time.time = lambda: now
    try:
        # All stale entries should be evicted, allowing a fresh request through.
        assert speaking_audio._check_rate_limit() is True
    finally:
        speaking_audio.time.time = real_time


# ---------------------------------------------------------------------------
# detect_answer_alignment_issues: diagnostic-only check for a client-side
# bug class (an answer's audio gets uploaded under the wrong question's
# label). These tests monkeypatch speaking_audio.safe_gpt_call so no real
# GPT call happens - they exercise the deterministic validation/filtering
# of whatever the model reports, the same way _validate_question_mistakes
# is tested above.
# ---------------------------------------------------------------------------

def _qas(*pairs):
    return [{"question": q, "user_answer": a} for q, a in pairs]


def test_detect_answer_alignment_issues_skips_gpt_call_when_no_part_has_two_questions(monkeypatch):
    called = False

    def _fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(speaking_audio, "safe_gpt_call", _fail_if_called)

    result = detect_answer_alignment_issues(_qas(("q1", "a1")), [], [])

    assert result == []
    assert called is False


def test_detect_answer_alignment_issues_accepts_well_formed_warning(monkeypatch):
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "alignment_warnings": [
                {
                    "part": 3,
                    "question_index": 1,
                    "likely_matches_question_index": 0,
                    "fails_own_question": True,
                    "reason": "Answer discusses young vs old viewers, matching question 0.",
                }
            ]
        },
    )

    part_3 = _qas(
        ("What are the differences between young and old viewers?", "They are like chalk and cheese..."),
        ("What makes a TV show popular?", "There's no particular parameter..."),
    )
    result = detect_answer_alignment_issues([], [], part_3)

    assert len(result) == 1
    warning = result[0]
    assert warning["part"] == 3
    assert warning["question"] == "What makes a TV show popular?"
    assert warning["likely_actual_question"] == "What are the differences between young and old viewers?"
    assert "reason" in warning and warning["reason"]


def test_detect_answer_alignment_issues_rejects_warning_missing_fails_own_question(monkeypatch):
    # Regression test for a real false positive: an answer that fully and
    # directly addressed its own labeled question was flagged anyway,
    # purely because it was topically adjacent to a neighboring question.
    # The model must now explicitly commit to "fails_own_question": true -
    # a soft "reason" string alone is not enough to accept the warning.
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "alignment_warnings": [
                {
                    "part": 3,
                    "question_index": 1,
                    "likely_matches_question_index": 0,
                    "reason": "Topically related to the other question.",
                }
            ]
        },
    )

    part_3 = _qas(("q0", "a0"), ("q1", "a1"))
    result = detect_answer_alignment_issues([], [], part_3)

    assert result == []


def test_detect_answer_alignment_issues_rejects_out_of_range_indices(monkeypatch):
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "alignment_warnings": [
                {"part": 3, "question_index": 5, "likely_matches_question_index": 0, "reason": "bogus"},
            ]
        },
    )

    part_3 = _qas(("q0", "a0"), ("q1", "a1"))
    result = detect_answer_alignment_issues([], [], part_3)

    assert result == []


def test_detect_answer_alignment_issues_rejects_self_match(monkeypatch):
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "alignment_warnings": [
                {"part": 3, "question_index": 0, "likely_matches_question_index": 0, "reason": "bogus"},
            ]
        },
    )

    part_3 = _qas(("q0", "a0"), ("q1", "a1"))
    result = detect_answer_alignment_issues([], [], part_3)

    assert result == []


def test_detect_answer_alignment_issues_rejects_unknown_part(monkeypatch):
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "alignment_warnings": [
                {"part": 7, "question_index": 0, "likely_matches_question_index": 1, "reason": "bogus"},
            ]
        },
    )

    part_3 = _qas(("q0", "a0"), ("q1", "a1"))
    result = detect_answer_alignment_issues([], [], part_3)

    assert result == []


def test_detect_answer_alignment_issues_handles_non_dict_gpt_response(monkeypatch):
    monkeypatch.setattr(speaking_audio, "safe_gpt_call", lambda *a, **k: None)

    part_3 = _qas(("q0", "a0"), ("q1", "a1"))
    result = detect_answer_alignment_issues([], [], part_3)

    assert result == []


def test_detect_answer_alignment_issues_empty_warnings_is_valid(monkeypatch):
    monkeypatch.setattr(speaking_audio, "safe_gpt_call", lambda *a, **k: {"alignment_warnings": []})

    part_3 = _qas(("q0", "a0"), ("q1", "a1"))
    result = detect_answer_alignment_issues([], [], part_3)

    assert result == []


# ---------------------------------------------------------------------------
# _quote_appears_in_transcript / detect_systematic_errors: a "systematic
# error" claim is only real evidence if the quoted occurrences backing it
# are genuinely verbatim from the transcript, not GPT's paraphrase of a
# nearby clause (e.g. swapping in a different noun). These tests monkeypatch
# speaking_audio.safe_gpt_call so no real GPT call happens.
# ---------------------------------------------------------------------------

def test_quote_appears_in_transcript_matches_modulo_case_and_punctuation():
    transcript = "Shows which are having fast-paced stories, and so on."
    assert _quote_appears_in_transcript("which are having fast-paced stories", transcript) is True
    assert _quote_appears_in_transcript("WHICH ARE HAVING FAST-PACED STORIES!", transcript) is True


def test_quote_appears_in_transcript_rejects_paraphrase():
    transcript = "which are having fast-paced stories which are having such kind of dramas"
    assert _quote_appears_in_transcript("which are having such kind of shows", transcript) is False


def test_quote_appears_in_transcript_rejects_empty_quote():
    assert _quote_appears_in_transcript("", "some transcript text") is False


def test_detect_systematic_errors_drops_pattern_when_quotes_dont_verify(monkeypatch):
    transcript = "which are having fast-paced stories which are having such kind of dramas and so on"
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "systematic_errors": [
                {
                    "pattern": "which are having instead of which have",
                    "criterion": "grammar",
                    "occurrences": [
                        "which are having fast-paced stories",
                        "which are having such kind of dramas",
                        "which are having such kind of shows",  # not verbatim - "shows" not in transcript
                    ],
                    "explanation": "recurring misuse",
                }
            ]
        },
    )

    result = detect_systematic_errors(transcript)

    # Only 2 of the 3 claimed occurrences are genuinely verbatim, below the
    # 3+ evidentiary bar, so the whole pattern must be dropped rather than
    # kept on the strength of a partly-fabricated quote.
    assert result == []


def test_detect_systematic_errors_keeps_pattern_when_all_quotes_verify(monkeypatch):
    transcript = (
        "which are having fast-paced stories which are having such kind of dramas "
        "and everything which are having everything already decided"
    )
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "systematic_errors": [
                {
                    "pattern": "which are having instead of which have",
                    "criterion": "grammar",
                    "occurrences": [
                        "which are having fast-paced stories",
                        "which are having such kind of dramas",
                        "which are having everything already decided",
                    ],
                    "explanation": "recurring misuse",
                }
            ]
        },
    )

    result = detect_systematic_errors(transcript)

    assert len(result) == 1
    assert result[0]["criterion"] == "grammar"
    assert len(result[0]["occurrences"]) == 3


def test_pattern_key_terms_extracts_quoted_terms():
    assert _pattern_key_terms("use of 'having' in incorrect contexts") == ["having"]
    assert _pattern_key_terms('repeated use of "such kind of" phrasing') == ["such kind of"]


def test_pattern_key_terms_returns_empty_when_no_quotes():
    assert _pattern_key_terms("which are having instead of which have") == []
    assert _pattern_key_terms("") == []
    assert _pattern_key_terms(None) == []


def test_detect_systematic_errors_drops_occurrence_that_is_real_quote_but_different_error(monkeypatch):
    # Regression test for a real case: a pattern claiming recurring misuse
    # of "having" included a third occurrence, "they're not getting
    # gaining", which IS a genuine verbatim quote from the transcript but
    # is a completely different error (a redundant double verb) with no
    # "having" in it at all - verbatim-ness alone isn't enough to prove an
    # occurrence actually supports the claimed pattern.
    transcript = (
        "which are having fast-paced stories which are having such kind of dramas "
        "so that's why they're not getting gaining more audience compared to televisions"
    )
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "systematic_errors": [
                {
                    "pattern": "use of 'having' in incorrect contexts",
                    "criterion": "grammar",
                    "occurrences": [
                        "which are having fast-paced stories",
                        "which are having such kind of dramas",
                        "they're not getting gaining",
                    ],
                    "explanation": "recurring misuse of having",
                }
            ]
        },
    )

    result = detect_systematic_errors(transcript)

    # Only 2 of the 3 occurrences actually contain "having" - the pattern's
    # own claimed mechanism - so it drops below the 3+ evidentiary bar and
    # the whole pattern must be discarded.
    assert result == []


def test_detect_systematic_errors_keeps_pattern_when_occurrences_match_quoted_key_term(monkeypatch):
    transcript = (
        "which are having fast-paced stories which are having such kind of dramas "
        "and everything which are having everything already decided"
    )
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "systematic_errors": [
                {
                    "pattern": "use of 'having' in incorrect contexts",
                    "criterion": "grammar",
                    "occurrences": [
                        "which are having fast-paced stories",
                        "which are having such kind of dramas",
                        "which are having everything already decided",
                    ],
                    "explanation": "recurring misuse of having",
                }
            ]
        },
    )

    result = detect_systematic_errors(transcript)

    assert len(result) == 1
    assert len(result[0]["occurrences"]) == 3


def test_detect_systematic_errors_rejects_capitalization_pattern(monkeypatch):
    # Regression test for a real case: a pattern titled "use of 'i' in
    # lowercase" was reported as a recurring VOCABULARY error, capping
    # Lexical Resource on a pure transcription artifact - the candidate
    # never "said" a lowercase letter; capitalization is added by
    # automatic speech-to-text, not something a speaker can get wrong.
    transcript = "i live in a house i would like to talk i remember i guess i have two different opinions"
    monkeypatch.setattr(
        speaking_audio,
        "safe_gpt_call",
        lambda *a, **k: {
            "systematic_errors": [
                {
                    "pattern": "use of 'i' in lowercase",
                    "criterion": "vocabulary",
                    "occurrences": [
                        "i live in a house",
                        "i would like to talk",
                        "i remember",
                    ],
                    "explanation": "This pattern shows a lack of capitalization for the first-person pronoun 'I'.",
                }
            ]
        },
    )

    result = detect_systematic_errors(transcript)

    assert result == []


# ---------------------------------------------------------------------------
# _split_band9_answer_per_question / _attach_refined_answers: the portal
# needs a refined/model answer attached to EACH question, not only the
# combined band9_answer block for the whole part.
# ---------------------------------------------------------------------------

def test_split_band9_answer_per_question_splits_cleanly():
    band9 = (
        "Answer 1: I loved visiting parks as a child.\n\n"
        "Answer 2: Not much these days, sadly.\n\n"
        "Answer 3: Yes, my city really needs more green space."
    )
    result = _split_band9_answer_per_question(band9, 3)
    assert result == [
        "I loved visiting parks as a child.",
        "Not much these days, sadly.",
        "Yes, my city really needs more green space.",
    ]


def test_split_band9_answer_per_question_handles_single_long_turn():
    # Part 2's cue-card format has exactly one question and one continuous
    # long-turn answer.
    band9 = "Answer 1: Well, I recently watched an online game show that really captivated me..."
    result = _split_band9_answer_per_question(band9, 1)
    assert result == ["Well, I recently watched an online game show that really captivated me..."]


def test_split_band9_answer_per_question_returns_none_on_mismatch():
    # Never guess when the count doesn't match - attaching a refined answer
    # to the wrong question is worse than showing none at all.
    band9 = "Answer 1: only one answer here"
    result = _split_band9_answer_per_question(band9, 3)
    assert result == [None, None, None]


def test_split_band9_answer_per_question_handles_empty_input():
    assert _split_band9_answer_per_question("", 2) == [None, None]
    assert _split_band9_answer_per_question(None, 0) == []


def test_attach_refined_answers_mutates_qas_in_place():
    qas = [{"question": "Q1", "user_answer": "A1"}, {"question": "Q2", "user_answer": "A2"}]
    _attach_refined_answers(qas, "Answer 1: refined one\n\nAnswer 2: refined two")
    assert qas[0]["refined_answer"] == "refined one"
    assert qas[1]["refined_answer"] == "refined two"


def test_attach_refined_answers_sets_none_on_mismatch_rather_than_guessing():
    qas = [{"question": "Q1", "user_answer": "A1"}, {"question": "Q2", "user_answer": "A2"}]
    _attach_refined_answers(qas, "Answer 1: only one")
    assert qas[0]["refined_answer"] is None
    assert qas[1]["refined_answer"] is None


# ---------------------------------------------------------------------------
# "JUDGE IT AS SPEECH, NOT AS WRITING" - section C's two deterministic
# filters. Synthetic test cases (no saved real transcript was found in the
# project/scratchpad to verify against - see the report), built directly
# from the false-positive patterns named in the request.
# ---------------------------------------------------------------------------

def test_flagged_span_with_sorry_marker_is_self_correction():
    assert _flagged_span_contains_self_correction("people like to enjoy, sorry, people like to visit") is True


def test_flagged_span_with_i_mean_marker_is_self_correction():
    assert _flagged_span_contains_self_correction("we went there, I mean, we visited last year") is True


def test_flagged_span_with_repeated_phrase_no_marker_is_self_correction():
    # Same 3-word run repeated with no marker word at all.
    assert _flagged_span_contains_self_correction("I go there every day I go there every week actually") is True


def test_flagged_span_without_self_correction_pattern_is_not_flagged():
    assert _flagged_span_contains_self_correction("the advantages of this approach is clear") is False


def test_validate_question_mistakes_drops_self_correction_span_even_with_neutral_explanation():
    # The explanation deliberately does NOT admit self-correction (unlike
    # _SELF_CORRECTION_EXPLANATION_KEYWORDS's existing target) - only the
    # flagged SPAN itself reveals it, which is exactly what the new C.1
    # filter checks that the older explanation-text filter does not.
    items = [{
        "type": "fluency", "original": "people like to enjoy, sorry, people like to visit",
        "corrected": "people like to visit", "explanation": "Awkward phrasing in this part of the answer.",
    }]
    assert _validate_question_mistakes(items) == []


def test_dedupe_part_level_text_strips_sentence_matching_a_per_question_mistake():
    question_mistakes = [{"original": "a lot many things", "corrected": "a lot of things"}]
    part_text = (
        "The candidate generally uses accurate grammar. However, the candidate said "
        "a lot many things which is a non-standard construction. Overall this is a minor issue."
    )
    result = _dedupe_part_level_text_against_question_mistakes(part_text, question_mistakes)
    assert "a lot many things" not in result.lower()
    assert "generally uses accurate grammar" in result
    assert "Overall this is a minor issue" in result


def test_dedupe_part_level_text_leaves_genuinely_distinct_sentences_alone():
    question_mistakes = [{"original": "a lot many things", "corrected": "a lot of things"}]
    part_text = "The candidate shows good range with varied vocabulary throughout the answer."
    result = _dedupe_part_level_text_against_question_mistakes(part_text, question_mistakes)
    assert result == part_text


def test_dedupe_part_level_text_ignores_short_flagged_phrases():
    # Only 3+-word flagged phrases are used as dedup signals - a 2-word
    # "original" is too generic to safely use for matching part-level
    # prose (would risk stripping unrelated sentences).
    question_mistakes = [{"original": "is clear", "corrected": "is evident"}]
    part_text = "The main point here is clear and well developed throughout."
    result = _dedupe_part_level_text_against_question_mistakes(part_text, question_mistakes)
    assert result == part_text


def test_dedupe_part_level_text_handles_no_mistakes_or_empty_text():
    assert _dedupe_part_level_text_against_question_mistakes("some text", []) == "some text"


# ---------------------------------------------------------------------------
# Minor/significant split - _split_mistakes_by_severity (per-question list)
# and _split_part_feedback_by_severity (per-part 4-criterion dict). Neither
# assigns severity - both only route an already-resolved severity tag.

def test_split_mistakes_by_severity_routes_each_item_to_exactly_one_list():
    mistakes = [
        {"type": "grammar", "severity": "significant", "original": "a"},
        {"type": "vocabulary", "severity": "minor", "original": "b"},
        {"type": "fluency", "severity": "minor", "original": "c"},
    ]
    significant, minor = _split_mistakes_by_severity(mistakes)
    assert significant == [mistakes[0]]
    assert minor == [mistakes[1], mistakes[2]]


def test_split_mistakes_by_severity_defaults_missing_or_unrecognized_to_significant():
    # Matches _validate_question_mistakes' own default - an unlabeled issue
    # must not be silently downgraded to minor by the split.
    mistakes = [{"type": "grammar", "original": "a"}, {"type": "grammar", "severity": "weird", "original": "b"}]
    significant, minor = _split_mistakes_by_severity(mistakes)
    assert len(significant) == 2
    assert minor == []


def test_split_mistakes_by_severity_empty_input():
    assert _split_mistakes_by_severity([]) == ([], [])
    assert _split_mistakes_by_severity(None) == ([], [])


def test_split_part_feedback_by_severity_routes_each_criterion_independently():
    feedback = {
        "fluency": "Hesitates occasionally.", "fluency_severity": "minor",
        "grammar": "Frequent tense errors.", "grammar_severity": "significant",
        "vocabulary": "Limited range.", "vocabulary_severity": "significant",
        "pronunciation": "Clear throughout.", "pronunciation_severity": "minor",
    }
    significant, minor, praise = _split_part_feedback_by_severity(feedback)
    assert significant["fluency"] == "" and significant["fluency_severity"] is None
    assert significant["grammar"] == "Frequent tense errors." and significant["grammar_severity"] == "significant"
    assert significant["vocabulary"] == "Limited range." and significant["vocabulary_severity"] == "significant"
    assert significant["pronunciation"] == "" and significant["pronunciation_severity"] is None
    assert minor["fluency"] == "Hesitates occasionally." and minor["fluency_severity"] == "minor"
    assert minor["grammar"] == "" and minor["grammar_severity"] is None
    assert minor["vocabulary"] == "" and minor["vocabulary_severity"] is None
    assert minor["pronunciation"] == "Clear throughout." and minor["pronunciation_severity"] == "minor"
    assert praise == []


def test_split_part_feedback_by_severity_pulls_praise_null_severity_out_of_both_dicts():
    # severity=None means praise, not a criticism (generate_mistakes' own
    # docstring). Once mistakes/minor_observations both mean "something to
    # fix", praise belongs in neither - it's returned separately for the
    # caller to route into feedback_summary.strengths instead.
    feedback = {
        "fluency": "Speaks fluently with natural pausing.", "fluency_severity": None,
        "grammar": "", "grammar_severity": None,
        "vocabulary": "", "vocabulary_severity": None,
        "pronunciation": "", "pronunciation_severity": None,
    }
    significant, minor, praise = _split_part_feedback_by_severity(feedback)
    assert significant["fluency"] == "" and significant["fluency_severity"] is None
    assert minor["fluency"] == "" and minor["fluency_severity"] is None
    assert praise == ["Speaks fluently with natural pausing."]


def test_split_part_feedback_by_severity_collects_praise_across_multiple_criteria():
    feedback = {
        "fluency": "Very fluent delivery.", "fluency_severity": None,
        "grammar": "Wide range of structures used accurately.", "grammar_severity": None,
        "vocabulary": "", "vocabulary_severity": "significant",
        "pronunciation": "", "pronunciation_severity": "significant",
    }
    significant, minor, praise = _split_part_feedback_by_severity(feedback)
    assert praise == ["Very fluent delivery.", "Wide range of structures used accurately."]
    assert significant["fluency"] == "" and significant["grammar"] == ""


# ---------------------------------------------------------------------------
# Deterministic word-repetition counter - count_word_repetitions /
# _repeated_word_observations. Must never ask GPT to count; must exclude
# stopwords, fillers, and the question's own topic words; must not flag
# repetition collapsed out of an immediate stumble/restart.

_REPETITION_FILLER_PADDING = (
    " We stayed there for a whole week and visited several different museums "
    "and a beautiful old castle nearby with our cousins."
)


def test_count_word_repetitions_flags_a_genuinely_overused_word():
    text = (
        "It was nice. The food was nice. My friend was nice. "
        "Everyone there was nice and the weather was nice too, which was nice."
    ) + _REPETITION_FILLER_PADDING  # pooled whole-test length must clear the min-words floor
    result = count_word_repetitions(text, "Describe a trip you enjoyed.")
    words = {r["word"]: r for r in result}
    assert "nice" in words
    assert words["nice"]["count"] == 6


def test_count_word_repetitions_below_min_total_words_floor_is_skipped_entirely():
    # A short pooled test (e.g. an early-terminated attempt) is skipped
    # outright, not just unlikely to flag - a single repeat in <40 words
    # produces an unstable rate_per_100 that isn't meaningful evidence
    # either way.
    text = "It was nice. It was nice. It was nice."
    assert len(text.split()) < 40
    assert count_word_repetitions(text, "Describe a trip.") == []


def test_count_word_repetitions_never_flags_the_question_topic_word():
    text = (
        "I really like hiking because hiking is relaxing. Hiking is relaxing and "
        "I like hiking a lot. Hiking is my favourite hobby of all the hobbies I have."
    ) + _REPETITION_FILLER_PADDING
    result = count_word_repetitions(text, "Do you like hiking as a hobby?")
    flagged = {r["word"] for r in result}
    assert "hik" not in flagged  # "hiking"/"hike" lemma (4 occurrences - would clear the threshold otherwise)
    assert "hobby" not in flagged


def test_count_word_repetitions_excludes_given_stopwords_and_fillers_even_at_high_frequency():
    text = " ".join(["I", "think", "so"] * 10) + " and really very like to know because it is the a"
    result = count_word_repetitions(text, "")
    flagged = {r["word"] for r in result}
    assert flagged == set()


def test_count_word_repetitions_below_threshold_is_not_flagged():
    # "okay" appears only twice - below the absolute-count floor (3) - kept
    # under that regardless of the padding, so this tests the per-word
    # threshold specifically, not the whole-text min-words floor above.
    text = "It was okay. The food was okay. Nothing else stood out about the trip really." + _REPETITION_FILLER_PADDING
    result = count_word_repetitions(text, "Describe a trip.")
    assert result == []


def test_count_word_repetitions_empty_text():
    assert count_word_repetitions("", "some question") == []
    assert count_word_repetitions(None, "some question") == []


def test_simple_lemma_folds_common_regular_forms():
    assert _simple_lemma("walking") == "walk"
    assert _simple_lemma("walked") == "walk"
    assert _simple_lemma("walks") == "walk"
    assert _simple_lemma("hobbies") == "hobby"
    assert _simple_lemma("running") == "run"


def test_topic_words_from_question_lemma_folds_and_drops_short_words():
    topic = _topic_words_from_question("Do you like hiking as a hobby?")
    assert "hik" in topic  # "hiking" folded
    assert "hobby" in topic
    assert "do" not in topic  # len <= 2, filtered
    assert "as" not in topic


def test_collapse_immediate_repeats_handles_stumble_and_restart():
    assert _collapse_immediate_repeats("I I think the the beach was really really nice") == \
        "I think the beach was really nice"
    assert _collapse_immediate_repeats("no repeats here at all") == "no repeats here at all"


def test_repeated_word_observations_shape_has_word_count_and_alternatives():
    text = (
        "It was nice. The food was nice. My friend was nice. "
        "Everyone there was nice and the weather was nice too, which was nice."
    ) + _REPETITION_FILLER_PADDING
    observations = _repeated_word_observations(text, "Describe a trip you enjoyed.")
    assert len(observations) == 1
    obs = observations[0]
    assert obs["word"] == "nice"
    assert obs["count"] == 6
    assert isinstance(obs["alternatives"], list) and 2 <= len(obs["alternatives"]) <= 3
    assert "nice" in obs["note"] and "6" in obs["note"]


def test_repeated_word_observations_unknown_word_gets_generic_fallback_alternatives():
    # "gizmo" spread across distinct sentences (not adjacent) - a word not
    # in the static alternatives table, not in the given stopword list.
    text = (
        "I bought a gizmo yesterday. The gizmo was expensive. My brother also has a gizmo. "
        "Everyone at work has a gizmo now, and I think a gizmo is genuinely useful."
    ) + _REPETITION_FILLER_PADDING
    observations = _repeated_word_observations(text, "")
    assert len(observations) == 1
    assert observations[0]["word"] == "gizmo"
    assert observations[0]["alternatives"]  # generic fallback list, non-empty
    assert _dedupe_part_level_text_against_question_mistakes("", [{"original": "a lot many things"}]) == ""
