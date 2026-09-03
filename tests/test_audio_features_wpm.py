# Prepared (not yet shipped) fix for the legacy pipeline's WPM-denominator
# bug - the same one already fixed behind SPEAKING_VOICED_WPM in
# evaluators/speaking_audio.py. compute_speech_rate_wpm() must always
# return both raw and voiced numbers regardless of the flag (so the gap
# stays visible), and "active" must only move with
# SPEAKING_LEGACY_VOICED_WPM - the flag defaults off, so the two live
# callers (evaluators/api/speaking.py, evaluators/api/speaking_text.py)
# must be unaffected until it's deliberately turned on.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import audio_features


def test_compute_speech_rate_wpm_returns_both_raw_and_voiced_always():
    # 60 words over 60s raw duration but only 30s voiced -> raw=60wpm,
    # voiced=120wpm - a clear, checkable gap.
    audio_metrics = {"duration_sec": 60.0, "voiced_duration_sec": 30.0}
    transcript = " ".join(["word"] * 60)
    result = audio_features.compute_speech_rate_wpm(transcript, audio_metrics)
    assert result["raw"] == 60
    assert result["voiced"] == 120


def test_compute_speech_rate_wpm_active_is_raw_when_flag_off(monkeypatch):
    monkeypatch.setattr(audio_features, "SPEAKING_LEGACY_VOICED_WPM", False)
    audio_metrics = {"duration_sec": 60.0, "voiced_duration_sec": 30.0}
    transcript = " ".join(["word"] * 60)
    result = audio_features.compute_speech_rate_wpm(transcript, audio_metrics)
    assert result["active"] == result["raw"] == 60


def test_compute_speech_rate_wpm_active_is_voiced_when_flag_on(monkeypatch):
    monkeypatch.setattr(audio_features, "SPEAKING_LEGACY_VOICED_WPM", True)
    audio_metrics = {"duration_sec": 60.0, "voiced_duration_sec": 30.0}
    transcript = " ".join(["word"] * 60)
    result = audio_features.compute_speech_rate_wpm(transcript, audio_metrics)
    assert result["active"] == result["voiced"] == 120


def test_compute_speech_rate_wpm_flag_on_falls_back_to_raw_if_voiced_missing(monkeypatch):
    monkeypatch.setattr(audio_features, "SPEAKING_LEGACY_VOICED_WPM", True)
    audio_metrics = {"duration_sec": 60.0}
    transcript = " ".join(["word"] * 60)
    result = audio_features.compute_speech_rate_wpm(transcript, audio_metrics)
    assert result["voiced"] == 0
    assert result["active"] == result["raw"] == 60


def test_compute_speech_rate_wpm_zero_duration_is_zero_not_error():
    result = audio_features.compute_speech_rate_wpm("some words here", {"duration_sec": 0, "voiced_duration_sec": 0})
    assert result == {"raw": 0, "voiced": 0, "active": 0}


def test_compute_speech_rate_wpm_empty_transcript():
    result = audio_features.compute_speech_rate_wpm("", {"duration_sec": 60.0, "voiced_duration_sec": 30.0})
    assert result == {"raw": 0, "voiced": 0, "active": 0}


def test_speaking_legacy_voiced_wpm_flag_defaults_off():
    assert audio_features.SPEAKING_LEGACY_VOICED_WPM is False
