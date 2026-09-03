import os

import librosa

# Prepared fix for the same WPM-denominator bug already fixed (behind
# SPEAKING_VOICED_WPM) in evaluators/speaking_audio.py - default OFF, not
# yet wired to change any candidate-facing score. See
# compute_speech_rate_wpm() below and its two callers
# (evaluators/api/speaking.py, evaluators/api/speaking_text.py). Ready to
# flip the moment real traffic data (utils/eval_log.py) confirms this
# pipeline is actually live - not before.
SPEAKING_LEGACY_VOICED_WPM = os.getenv("SPEAKING_LEGACY_VOICED_WPM", "false").strip().lower() == "true"


def compute_speech_rate_wpm(transcript: str, audio_metrics: dict) -> dict:
    """Both raw-duration and voiced-duration WPM, always both computed and
    returned (so the gap stays visible regardless of the flag) - "active"
    is whichever one SPEAKING_LEGACY_VOICED_WPM says should drive scoring.
    Mirrors evaluators/speaking_audio.py's per-clip speech_rate_wpm_raw/
    speech_rate_wpm_voiced/speech_rate_wpm split exactly.

    NOT covered by this: evaluate_speaking_part()'s own separate internal
    WPM recomputation from a caller-supplied `time_seconds` scalar
    (evaluators/speaking.py ~line 1204) - that path has no waveform
    available at all (just a bare number of seconds), so there is no
    voiced-time to derive the same way. That path stays raw-duration-based
    regardless of this flag - a real, separate gap, not silently covered
    by this fix."""
    words = len((transcript or "").split())
    duration = audio_metrics.get("duration_sec") or 0
    voiced_duration = audio_metrics.get("voiced_duration_sec") or 0
    raw_wpm = round((words / duration) * 60) if duration > 0 else 0
    voiced_wpm = round((words / voiced_duration) * 60) if voiced_duration > 0 else 0
    active_wpm = voiced_wpm if (SPEAKING_LEGACY_VOICED_WPM and voiced_wpm) else raw_wpm
    return {"raw": raw_wpm, "voiced": voiced_wpm, "active": active_wpm}


def extract_audio_features(wav_path: str):
    y, sr = librosa.load(wav_path, sr=16000)

    duration = librosa.get_duration(y=y, sr=sr)
    intervals = librosa.effects.split(y, top_db=25)
    pause_count = max(0, len(intervals) - 1)

    # Additive diagnostic only, mirroring evaluators/speaking_audio.py's
    # extract_acoustic_features() fix - reuses `intervals` (already
    # computed above for pause_count), no new processing. NOT wired into
    # any WPM calculation here - the two callers of this function
    # (evaluators/api/speaking.py, evaluators/api/speaking_text.py) still
    # compute speech_rate_wpm from raw duration_sec exactly as before.
    # Behind SPEAKING_LEGACY_VOICED_WPM (default off, see those callers) -
    # prepared, not shipped, pending real traffic data on whether this
    # pipeline is even in live use.
    voiced_time = sum((end - start) / sr for start, end in intervals) if len(intervals) > 0 else 0.0

    return {
        "duration_sec": round(duration, 2),
        "voiced_duration_sec": round(voiced_time, 2),
        "pause_count": pause_count
    }
