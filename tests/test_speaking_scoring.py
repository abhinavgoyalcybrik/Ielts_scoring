import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluators.speaking import _composite_to_pronunciation_band, compute_pronunciation_score
from evaluators.speaking_audio import (
    _aggregate_acoustic_pronunciation,
    _apply_completeness_to_feedback,
    _apply_relevance_to_feedback,
    _band9_word_count,
    _BoundedCache,
    _collect_completeness_notices,
    _content_words,
    _estimate_linguistic_floor,
    _heuristic_off_topic,
    _ielts_round_half_up,
    _normalize_numbered_band9_answer,
    _validate_question_mistakes,
    calculate_overall_band,
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
# calculate_overall_band: 25/35/40 part weighting + correct rounding +
# graceful handling of missing parts.
# ---------------------------------------------------------------------------

def test_calculate_overall_band_applies_part_weighting():
    p1 = {"fluency": 5, "lexical": 5, "grammar": 5, "pronunciation": 5.4}
    p2 = {"fluency": 8, "lexical": 8, "grammar": 7, "pronunciation": 5.8}
    p3 = {"fluency": 5, "lexical": 5, "grammar": 5, "pronunciation": 4.8}
    # p1 avg=5.1 (25%), p2 avg=7.2 (35%), p3 avg=4.95 (40%) -> weighted 5.775 -> rounds to 6.0
    assert calculate_overall_band(p1, p2, p3) == 6.0


def test_calculate_overall_band_handles_missing_part():
    p2 = {"fluency": 7, "lexical": 7, "grammar": 7, "pronunciation": 7}
    assert calculate_overall_band(None, p2, None) == 7.0


def test_calculate_overall_band_returns_zero_when_nothing_attempted():
    # Per the descriptor's own Band 0 definition ("Does not attend / Does
    # not complete the test"), a submission with no real content in any
    # part must score 0, not a generous middle-of-the-road guess.
    assert calculate_overall_band(None, None, None) == 0.0


def test_calculate_overall_band_partial_attempt_pulls_score_down():
    # Only Part 2 was attempted (weighted 35%) - the other two parts count
    # as 0, so the overall band should be well below what Part 2 alone
    # would suggest, not silently ignored.
    p2 = {"fluency": 7, "lexical": 7, "grammar": 7, "pronunciation": 7}
    zero_part = {"fluency": 0.0, "lexical": 0.0, "grammar": 0.0, "pronunciation": 0.0}
    result = calculate_overall_band(zero_part, p2, zero_part)
    assert result < 7.0
    assert result == pytest.approx(0.35 * 7.0, abs=0.5)


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
