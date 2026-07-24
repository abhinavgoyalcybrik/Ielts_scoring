import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.audio_transcriber import _build_initial_prompt, _clean_transcript, _estimate_confidence


def test_build_initial_prompt_strips_newlines():
    question = "What is your favorite place in your hometown?\nWhy do you like it?"
    prompt = _build_initial_prompt(question)
    assert "\n" not in prompt
    assert "favorite place" in prompt.lower()


def test_clean_transcript_collapses_whitespace():
    text = "  hello   there   friend  "
    cleaned = _clean_transcript(text)
    assert cleaned == "hello there friend"


def test_clean_transcript_removes_repeated_words_and_punctuation_spacing():
    text = "I I I want to go there  ,  and and and I said yes ."
    cleaned = _clean_transcript(text)
    assert cleaned == "I want to go there, and I said yes."


# ---------------------------------------------------------------------------
# _estimate_confidence: derives genuine ASR confidence from Whisper's own
# avg_logprob/no_speech_prob per segment, replacing what used to be a
# hardcoded 0.9 whenever the transcript was merely non-empty (which meant
# low-confidence dampening logic downstream could never actually trigger).
# ---------------------------------------------------------------------------

def test_estimate_confidence_high_for_clear_audio():
    result = {"segments": [{"avg_logprob": -0.1, "no_speech_prob": 0.02}, {"avg_logprob": -0.15, "no_speech_prob": 0.01}]}
    confidence = _estimate_confidence(result, has_text=True)
    assert confidence > 0.7


def test_estimate_confidence_low_for_unclear_audio():
    result = {"segments": [{"avg_logprob": -0.9, "no_speech_prob": 0.3}]}
    confidence = _estimate_confidence(result, has_text=True)
    assert confidence < 0.3


def test_estimate_confidence_falls_back_without_segments():
    assert _estimate_confidence({}, has_text=True) == 0.5
    assert _estimate_confidence({}, has_text=False) == 0.0


def test_estimate_confidence_bounded_between_zero_and_one():
    extreme = {"segments": [{"avg_logprob": 5.0, "no_speech_prob": -5.0}]}
    confidence = _estimate_confidence(extreme, has_text=True)
    assert 0.0 <= confidence <= 1.0
