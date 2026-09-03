from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from collections import OrderedDict, deque

import threading

from utils.audio_normalizer import normalize_to_wav

from utils.audio_transcriber import transcribe_audio

from utils.gpt_client import call_gpt, call_gpt_strong, record_token_usage

from utils.eval_log import log_evaluation

from utils.safety import safe_gpt_call, normalize_feedback

try:

    import librosa  # type: ignore

except ImportError:

    librosa = None

import numpy as np

import math

import tempfile

from pathlib import Path

import io

import json

import time

import uuid

import asyncio

import logging

logger = logging.getLogger(__name__)

import random

import hashlib

import re

from difflib import SequenceMatcher

import faster_whisper

import os

from openai import OpenAI


# Relocated verbatim from evaluators/speaking.py (the legacy engine, since
# retired) - this was the one function Engine A genuinely depended on from
# that file, so it moved here rather than being deleted with the rest.
# Identical code, identical behaviour, only the location changed.
def _normalize_speech_rate_for_pronunciation(rate: float) -> float:
    """0-1 goodness score for speech rate as a pronunciation signal."""
    if not rate or rate <= 0:
        return 0.5
    if 110 <= rate <= 160:
        return 1.0
    if rate < 110:
        return max(0.0, rate / 110)
    return max(0.0, 1.0 - (rate - 160) / 120)


_PRONUNCIATION_BAND_ANCHORS = [
    (0.00, 1.0),
    (0.08, 2.0),
    (0.20, 3.0),
    (0.35, 4.0),
    (0.48, 5.0),
    (0.60, 6.0),
    (0.72, 7.0),
    (0.85, 8.0),
    (0.97, 9.0),
    (1.00, 9.0),
]


def _composite_to_pronunciation_band(composite: float) -> float:
    composite = max(0.0, min(1.0, composite))
    for (low_x, low_band), (high_x, high_band) in zip(_PRONUNCIATION_BAND_ANCHORS, _PRONUNCIATION_BAND_ANCHORS[1:]):
        if low_x <= composite <= high_x:
            if high_x == low_x:
                return high_band
            fraction = (composite - low_x) / (high_x - low_x)
            return low_band + fraction * (high_band - low_band)
    return 9.0


def compute_pronunciation_score(acoustic_features: dict, asr_confidence: float = 1.0) -> float:
    """
    Pronunciation score derived from acoustic evidence: pause patterns,
    stress accuracy, intonation, speech rate, articulation, and audio quality.
    """
    if not acoustic_features:
        return 5.5

    phoneme_accuracy = acoustic_features.get("phoneme_accuracy", 0.7)
    phoneme_accuracy = 0.7 if phoneme_accuracy is None else phoneme_accuracy

    stress_accuracy = acoustic_features.get("stress_accuracy", 0.7)
    stress_accuracy = 0.7 if stress_accuracy is None else stress_accuracy

    intonation = acoustic_features.get("intonation_score", 0.6)
    intonation = 0.6 if intonation is None else intonation

    audio_quality = acoustic_features.get("audio_quality_score", 6.0)
    audio_quality = 6.0 if audio_quality is None else audio_quality

    mispronunciation_rate = acoustic_features.get("mispronunciation_rate", 0.3)
    mispronunciation_rate = 0.3 if mispronunciation_rate is None else mispronunciation_rate

    pause_count = acoustic_features.get("pause_count", 3)
    pause_count = 3 if pause_count is None else pause_count

    avg_pause = acoustic_features.get("avg_pause_duration", 0.6)
    avg_pause = 0.6 if avg_pause is None else avg_pause

    speech_rate = acoustic_features.get("speech_rate") or acoustic_features.get("speech_rate_wpm") or 0.0

    articulation_norm = max(0.0, min(1.0, (phoneme_accuracy + (1.0 - mispronunciation_rate)) / 2))
    stress_norm = max(0.0, min(1.0, stress_accuracy))
    intonation_norm = max(0.0, min(1.0, (intonation - 0.3) / 0.5))
    speech_rate_norm = _normalize_speech_rate_for_pronunciation(speech_rate)
    pause_count_norm = max(0.0, min(1.0, 1.0 - (pause_count / 15)))
    avg_pause_norm = max(0.0, min(1.0, 1.0 - (avg_pause / 2.0)))
    pause_norm = (pause_count_norm + avg_pause_norm) / 2
    audio_quality_norm = max(0.0, min(1.0, (audio_quality - 4.0) / 4.0))

    composite = (
        articulation_norm * 0.25
        + stress_norm * 0.20
        + intonation_norm * 0.15
        + speech_rate_norm * 0.15
        + pause_norm * 0.15
        + audio_quality_norm * 0.10
    )

    score = _composite_to_pronunciation_band(composite)

    if asr_confidence < 0.7:
        score = (score * 0.6) + (5.5 * 0.4)

    return round(max(1.0, min(9.0, score)), 1)


VOCAB_FALLBACK_PART1 = [
    {"word": "beneficial", "meaning": "having a positive effect"},
    {"word": "enroll", "meaning": "officially join a group or course"},
    {"word": "facilitate", "meaning": "make something easier"},
    {"word": "interact", "meaning": "communicate and engage with others"},
    {"word": "diverse", "meaning": "showing variety and difference"},
    {"word": "convenient", "meaning": "suiting your needs or plans well"},
    {"word": "occasionally", "meaning": "sometimes, but not often"},
    {"word": "routine", "meaning": "a regular way or order of doing things"},
]

VOCAB_FALLBACK_PART2 = [
    {"word": "elaborate", "meaning": "explain in more detail"},
    {"word": "furthermore", "meaning": "in addition to what was said"},
    {"word": "highlight", "meaning": "draw attention to something important"},
    {"word": "contribute", "meaning": "give or add to something"},
    {"word": "worthwhile", "meaning": "worth the time or effort spent"},
    {"word": "vivid", "meaning": "producing a clear, sharp impression in the mind"},
    {"word": "memorable", "meaning": "worth remembering or easy to remember"},
    {"word": "eventually", "meaning": "after some time; in the end"},
]

VOCAB_FALLBACK_PART3 = [
    {"word": "implication", "meaning": "a possible consequence or effect"},
    {"word": "mitigate", "meaning": "reduce the severity of something"},
    {"word": "postulate", "meaning": "suggest as a theory or idea"},
    {"word": "perspective", "meaning": "a particular way of viewing something"},
    {"word": "integration", "meaning": "combining parts into a whole"},
    {"word": "significant", "meaning": "important enough to be worth noting"},
    {"word": "consequently", "meaning": "as a result of something"},
    {"word": "controversial", "meaning": "causing disagreement or differing opinions"},
]


# Explicit timeout on every direct OpenAI API call in this module - without
# one, a hung request to OpenAI would hang the whole request indefinitely
# (bounded only by whatever default the SDK/httpx happens to use).
OPENAI_TIMEOUT_SECONDS = 60

# Reject audio clips larger than this to prevent a single oversized/
# malicious upload from consuming excessive memory or processing time. A
# real IELTS speaking answer is at most a couple of minutes of compressed
# audio - 25MB is already generous headroom (matches OpenAI's own Whisper
# API upload limit as a reference point).
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Lightweight in-process rate limiting (best-effort, per-worker)

_RATE_LIMIT_THRESHOLD = 30

_RATE_LIMIT_WINDOW_SECONDS = 60





_rate_limit_lock = threading.Lock()

_rate_limit_timestamps = deque()


def _check_rate_limit() -> bool:
    """Sliding time-window rate limit: at most _RATE_LIMIT_THRESHOLD calls
    per _RATE_LIMIT_WINDOW_SECONDS, per worker process. Previously this was
    a lifetime request counter that, once past the threshold, rejected
    every request forever for the rest of the process's life - not a real
    rate limit. This is in-memory and per-process only; it does not
    coordinate across multiple server instances (a shared store like Redis
    would be needed for that)."""
    now = time.time()
    with _rate_limit_lock:
        while _rate_limit_timestamps and now - _rate_limit_timestamps[0] > _RATE_LIMIT_WINDOW_SECONDS:
            _rate_limit_timestamps.popleft()
        if len(_rate_limit_timestamps) >= _RATE_LIMIT_THRESHOLD:
            return False
        _rate_limit_timestamps.append(now)
        return True





# In-memory caches to avoid repeated ASR/feature work within a process.
# Bounded (LRU eviction) and observable - unbounded dicts would accumulate
# every transcript ever processed for the lifetime of the server, and gave
# zero visibility into whether a "recycled" transcript came from a genuine
# re-submission of the same audio bytes (a real cache hit) vs a coincidence
# of similar wording. Every hit/miss is now logged with the audio hash so
# that's provable from server logs instead of guessed at.

class _BoundedCache:
    """Thread-safe: _evaluate_speaking_part_audio() now runs concurrently
    across a real thread pool (via asyncio.to_thread), not just cooperative
    asyncio tasks, so concurrent get()/set() from multiple audio clips in
    the same request is an expected, normal case - not an edge case."""

    def __init__(self, max_size: int = 200):
        self._store = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._store:
                return None, False
            self._store.move_to_end(key)
            return self._store[key], True

    def set(self, key, value):
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)


_ASR_CACHE = _BoundedCache(max_size=200)

_FEATURE_CACHE = _BoundedCache(max_size=200)

_USED_VOCAB = set()



# The `import whisper` above (module-level) already guarantees the package
# is importable here - this used to ALSO load its own separate "small"
# model instance just to double as an availability check, which doubled
# memory/startup cost for a model that was never actually used for
# transcription (utils.audio_transcriber._load_model() loads the one model
# that's actually used, once, and reuses it).
WHISPER_AVAILABLE = True



router = APIRouter(prefix="/speaking", tags=["Speaking"])





def validate_part_duration(part: int, duration: float):

    rules = {

        1: (10, 30),

        2: (60, 120),

        3: (30, 60)

    }

    min_d, max_d = rules.get(part, (0, 999))

    return min_d <= duration <= max_d





def _safe_log10(x):

    return math.log10(x) if x > 0 else -5





def _wav_duration_seconds(path: str) -> float:

    import wave

    with wave.open(path, "rb") as wf:

        frames = wf.getnframes()

        rate = wf.getframerate()

        return frames / float(rate) if rate else 0.0





def _trim_wav(path: str, max_seconds: float = 300.0):

    """Trim WAV file to max_seconds in-place."""

    import wave

    if not os.path.exists(path):

        return

    with wave.open(path, "rb") as wf:

        params = wf.getparams()

        frames = wf.getnframes()

        rate = wf.getframerate()

        duration = frames / float(rate) if rate else 0

        if duration <= max_seconds or rate == 0:

            return

        max_frames = int(max_seconds * rate)

        audio_data = wf.readframes(max_frames)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")

    os.close(tmp_fd)

    with wave.open(tmp_path, "wb") as wf_out:

        wf_out.setparams(params)

        wf_out.writeframes(audio_data)

    os.replace(tmp_path, path)





def refine_pronunciation_with_word_confidence(words):

    confidences = [w.get("confidence", 0.8) for w in words]



    if not confidences:

        return 0.6, 0.4, 0.6



    avg_conf = sum(confidences) / len(confidences)



    phoneme_accuracy = avg_conf

    mispronunciation_rate = max(0, 1 - avg_conf)

    stress_accuracy = min(1.0, avg_conf + 0.1)



    return phoneme_accuracy, mispronunciation_rate, stress_accuracy





def compute_intonation_score(signal):

    variation = np.std(signal)



    if variation < 0.01:

        return 0.4  # monotone

    elif variation < 0.05:

        return 0.6

    else:

        return 0.8





def compute_micro_timing(word_timestamps):

    durations = [w["end"] - w["start"] for w in word_timestamps]



    if not durations:

        return 0.5



    avg = sum(durations) / len(durations)

    variance = sum((d - avg) ** 2 for d in durations) / len(durations)



    if variance < 0.01:

        return 0.8

    elif variance < 0.03:

        return 0.6

    else:

        return 0.4





def extract_acoustic_features(audio_path, transcript: str = ""):

    """

    Lightweight acoustic feature extraction for real-time scoring.

    Returns dict with:

        - duration_sec

        - pause_count

        - avg_pause_duration

        - speech_rate (voiced WPM proxy)

        - speech_variability

        - energy_variation

        - pause_distribution_score

        - speech_rhythm_score

        - hesitation_score

        - sentence_flow_score

        - phoneme_accuracy

        - mispronunciation_rate

        - stress_accuracy

        - audio_quality_score

        - pronunciation_confidence

        - intonation_score

        - micro_timing_score

    """

    if librosa is None:

        raise RuntimeError("Audio feature extraction requires librosa. Please install with `pip install librosa`.")

    y, sr = librosa.load(audio_path, sr=16000)



    duration = librosa.get_duration(y=y, sr=sr)

    intervals = librosa.effects.split(y, top_db=25)

    pause_count = max(0, len(intervals) - 1)



    # Pause durations and positions

    pause_durations = []

    pause_positions = []

    for i in range(len(intervals) - 1):

        gap_frames = (intervals[i + 1][0] - intervals[i][1])

        gap = gap_frames / sr

        if gap > 0:

            pause_durations.append(gap)

            mid = (intervals[i][1] + gap_frames // 2) / len(y)

            pause_positions.append(mid)

    avg_pause_duration = round(float(np.mean(pause_durations)), 2) if pause_durations else 0.0



    # Voiced time to estimate speech rate (WPM proxy assuming 150 wpm fully voiced)

    voiced_time = sum((end - start) / sr for start, end in intervals) if len(intervals) > 0 else 0.0

    speech_rate = round((voiced_time / duration) * 150, 2) if duration > 0 else 0.0



    # Pause distribution scoring (natural vs mid-sentence)

    punctuation_marks = [",", ".", "?", "!", ";", ":"]

    total_words = len(transcript.split())

    punct_positions = []

    if total_words > 0:

        words = transcript.split()

        cumulative = 0

        for idx, w in enumerate(words):

            cumulative += 1

            if any(w.endswith(p) for p in punctuation_marks):

                punct_positions.append(cumulative / total_words)

    def _natural_pause_ratio():

        if not pause_positions:

            return 1.0

        if not punct_positions:

            return 0.7  # neutral when transcript lacks punctuation cues

        naturals = 0

        for p in pause_positions:

            if any(abs(p - pp) < 0.05 for pp in punct_positions):

                naturals += 1

        return naturals / len(pause_positions)

    natural_ratio = _natural_pause_ratio()

    pause_distribution_score = round(8 - (1 - natural_ratio) * 4, 2) if pause_positions else 7.5

    pause_distribution_score = max(4.0, min(8.0, pause_distribution_score))



    # Rhythm consistency (lower CV of voiced segment lengths => better)

    voiced_durations = [ (end - start) / sr for start, end in intervals ] if len(intervals) > 0 else []

    if voiced_durations:

        mean_v = np.mean(voiced_durations)

        cv = np.std(voiced_durations) / mean_v if mean_v > 0 else 1.0

    else:

        cv = 1.0

    if cv <= 0.25:

        speech_rhythm_score = 8.0

    elif cv <= 0.4:

        speech_rhythm_score = 7.0

    elif cv <= 0.55:

        speech_rhythm_score = 6.0

    else:

        speech_rhythm_score = 5.0



    # Hesitation & sentence flow

    short_pauses = len([p for p in pause_durations if p < 0.25])

    hesitation_score = max(4.0, min(8.0, 8 - (short_pauses * 0.2 + pause_count * 0.1)))

    sentence_flow_score = round((pause_distribution_score + speech_rhythm_score) / 2, 2)



    # Variability metrics

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

    speech_variability = round(float(np.std(rms)), 4)

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    energy_variation = round(float(np.std(zcr)), 4)



    # Audio quality detection (noise, clipping, low volume)

    peak = np.max(np.abs(y)) if len(y) else 0

    rms_mean = float(np.mean(rms)) if len(rms) else 0.0

    noise_floor = float(np.percentile(np.abs(y), 10)) if len(y) else 0.0

    snr_proxy = peak / (noise_floor + 1e-4)

    clipping_ratio = float(np.mean(np.abs(y) > 0.98)) if len(y) else 0.0

    low_volume_penalty = 1.0 if rms_mean < 0.01 else 0.0

    audio_quality_score = 8 - (clipping_ratio * 4) - low_volume_penalty

    if snr_proxy < 5:

        audio_quality_score -= 1.0

    audio_quality_score = max(4.0, min(8.0, round(audio_quality_score, 2)))



    # Pronunciation scoring (heuristic phoneme proxy)

    # Use clarity + variability + ASR confidence

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    centroid_var = float(np.std(spectral_centroid)) if len(spectral_centroid) else 0.0

    clarity_factor = max(0.0, min(1.0, (snr_proxy / 20)))

    phoneme_accuracy_auto = max(0.0, min(1.0, (clarity_factor * 0.5) + (1 - cv) * 0.3 + (1 - centroid_var / 1000) * 0.2))



    # Word-level confidence refinement (if available)

    word_timestamps = []

    # Placeholder: derive simple timestamps from intervals as proxy

    for start, end in intervals:

        word_timestamps.append({"start": start / sr, "end": end / sr, "confidence": 0.8})

    phoneme_accuracy_conf, mispronunciation_rate_conf, stress_accuracy_conf = refine_pronunciation_with_word_confidence(word_timestamps)

    phoneme_accuracy = max(phoneme_accuracy_auto, phoneme_accuracy_conf)

    mispronunciation_rate = round(max(0.0, 1 - phoneme_accuracy), 3)

    stress_accuracy = round(min(1.0, max(stress_accuracy_conf, phoneme_accuracy * 0.8 + (speech_rhythm_score - 5) / 10)), 3)



    intonation_score = compute_intonation_score(y) if len(y) else 0.6

    micro_timing_score = compute_micro_timing(word_timestamps)



    # Confidence for pronunciation

    pronunciation_confidence = max(0.0, min(1.0, (audio_quality_score / 8) * 0.5 + 0.5 * clarity_factor))



    return {

        "duration_sec": round(duration, 2),

        # Always present regardless of SPEAKING_VOICED_WPM (diagnostic, not
        # behaviour) - voiced_time was already being computed above for the
        # (unused-elsewhere) speech_rate proxy; exposing it directly is the
        # actual raw/voiced-duration gap, in seconds, so both numbers and
        # the silence fraction are visible side by side rather than only
        # one or the other.
        "voiced_duration_sec": round(voiced_time, 2),

        "silence_fraction": round(1 - (voiced_time / duration), 4) if duration > 0 else None,

        "pause_count": pause_count,

        "avg_pause_duration": avg_pause_duration,

        "speech_rate": speech_rate,

        "speech_variability": speech_variability,

        "energy_variation": energy_variation,

        "pause_distribution_score": pause_distribution_score,

        "speech_rhythm_score": speech_rhythm_score,

        "hesitation_score": round(hesitation_score, 2),

        "sentence_flow_score": sentence_flow_score,

        "phoneme_accuracy": round(phoneme_accuracy, 3),

        "mispronunciation_rate": mispronunciation_rate,

        "stress_accuracy": stress_accuracy,

        "audio_quality_score": audio_quality_score,

        "pronunciation_confidence": round(pronunciation_confidence, 3),

        "intonation_score": intonation_score,

        "micro_timing_score": micro_timing_score,

    }





def split_transcript_with_gpt(transcript: str, questions: list, usage_log=None):

    """

    Use GPT to semantically split a transcript into answers aligned to questions.

    Returns list of answers with length == len(questions).

    """

    prompt = f"""
You are given an IELTS speaking response that covers multiple questions in
one continuous transcript. The candidate was asked these questions IN ORDER
and answered them one after another in a single take - the transcript is
one unbroken timeline, not a jumbled bag of topics.

Your task:
Split the transcript into separate answers for EACH question, using ONLY
the candidate's EXACT original words for each segment.

CRITICAL - SPLIT BY SPEAKING ORDER, NOT BY TOPIC SIMILARITY:
- The N-th answer segment must be the part of the transcript the candidate
  spoke in response to the N-th question, in the SAME chronological order
  the questions were asked - never reorder or search the transcript for
  whichever segment merely sounds most topically related.
- Two different questions can be about closely related topics (e.g. "what
  makes a show popular" and "do people watch too much TV") - do NOT assign
  a segment to a question just because its topic seems relevant; assign it
  based on WHERE in the timeline it was actually spoken, immediately after
  the previous question's segment ended.
- The segments, concatenated in order, should reconstruct the transcript
  from start to end with no gaps and no overlaps - every word belongs to
  exactly one answer.

CRITICAL - DO NOT MODIFY THE CANDIDATE'S WORDS:
- Copy each answer segment VERBATIM from the transcript - do not paraphrase,
  summarize, shorten, reword, or "clean up" anything. Every word, filler,
  and grammar mistake must be preserved exactly as spoken.
- This is a splitting task only, not a rewriting task. You are drawing
  boundaries between existing text, not producing new text.

IMPORTANT:
- Number of answers MUST equal number of questions
- Do NOT merge answers
- Do NOT skip any question

Questions (in the order asked):
{questions}

Transcript (one continuous timeline, in speaking order):
{transcript}

Return a JSON array of answers in order (no extra text).
"""

    fallback = [transcript] * len(questions) if questions else [transcript]

    response = safe_gpt_call(
        prompt,
        fallback=fallback,
        caller=lambda p: call_gpt(p, usage_log=usage_log)
    )

    if isinstance(response, list):
        answers = response
    elif isinstance(response, str):
        try:
            answers = json.loads(response)
        except Exception:
            answers = [a.strip() for a in response.split("\n") if a.strip()]
    else:
        answers = []

    if not isinstance(answers, list) or len(answers) != len(fallback):
        answers = fallback

    return answers


def sanitize_result(result):
    if not isinstance(result, dict):
        return {}

    return {
        "fluency": result.get("fluency", 5),
        "lexical": result.get("lexical", 5),
        "grammar": result.get("grammar", 5),
        "pronunciation": result.get("pronunciation", 5),
        "feedback": result.get("feedback", {}),
        "vocabulary_feedback": result.get("vocabulary_feedback", {}),
        "relevance_score": result.get("relevance_score", 0.5),
    }

def is_invalid_band9(original: str, new: str) -> bool:
    """Relaxed validation: basic length check + light similarity guard."""
    if not new or len(new.split()) < 25:
        return True
    similarity = original[:50].lower() in new.lower() if original else False
    return similarity


def extract_keywords(text: str, top_n: int = 5) -> list:
    """Simple keyword extractor: keeps frequent, meaningful words."""
    if not text:
        return []
    stop = {
        "the", "and", "for", "with", "that", "this", "those", "these", "have",
        "has", "had", "was", "were", "are", "is", "am", "i", "you", "he", "she",
        "it", "we", "they", "of", "to", "in", "on", "at", "a", "an", "as", "be",
        "but", "or", "so", "very", "really", "just", "my", "your", "our", "their"
    }
    words = re.findall(r"[a-zA-Z']{4,}", text.lower())
    freq = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def _count_band9_answers(combined_with_context: str, answers_only: str | None = None) -> int:
    if answers_only:
        answers = [a.strip() for a in answers_only.split("\n\n") if a.strip()]
        if answers:
            return len(answers)

    if combined_with_context:
        return max(1, len(re.findall(r"^Question:\s*", combined_with_context, re.M)))

    return 1


def _normalize_numbered_band9_answer(result: str, answer_count: int) -> str:
    result = (result or "").strip()
    if not result:
        return result

    matches = re.findall(r"(?im)^(answer\s*\d+)\s*[:\-]\s*(.*?)(?=^answer\s*\d+\s*[:\-]|\Z)", result, re.S | re.M)
    if matches:
        normalized = []
        for i, (_, content) in enumerate(matches[:answer_count]):
            normalized.append(f"Answer {i+1}: {content.strip()}")
        return "\n\n".join(normalized)

    lines = [line.strip() for line in re.split(r"[\r\n]+", result) if line.strip()]
    if len(lines) >= answer_count:
        return "\n\n".join([f"Answer {i+1}: {lines[i]}" for i in range(answer_count)])

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", result) if s.strip()]
    if len(sentences) >= answer_count:
        return "\n\n".join([f"Answer {i+1}: {sentences[i]}" for i in range(answer_count)])

    return result


def _split_band9_answer_per_question(normalized_band9_answer: str, expected_count: int) -> list:
    """Split an already-normalized band9_answer block (guaranteed to be
    "Answer 1: ...\\n\\nAnswer 2: ..." by _normalize_numbered_band9_answer,
    which always emits sequential "Answer N:" labels regardless of which
    internal parsing path produced it) back into one string per question,
    so each question can carry its OWN refined/model answer instead of
    only the combined block at the end of the part. Returns a list of
    length expected_count; every item is None (not a guess) if the split
    didn't produce exactly that many labeled answers - attaching a refined
    answer to the wrong question would be worse than showing none."""
    text = (normalized_band9_answer or "").strip()
    if not text or expected_count <= 0:
        return [None] * max(expected_count, 0)

    matches = re.findall(r"(?im)^Answer\s*\d+:\s*(.*?)(?=\n\nAnswer\s*\d+:|\Z)", text, re.S)
    if len(matches) != expected_count:
        return [None] * expected_count

    return [m.strip() for m in matches]


def _attach_refined_answers(qas_clean: list, band9_answer: str) -> None:
    """Mutates qas_clean in place, adding a "refined_answer" field to each
    question - the portal needs a corrected/polished version attached to
    each individual Q&A, not only the combined band9_answer block for the
    whole part."""
    pieces = _split_band9_answer_per_question(band9_answer, len(qas_clean))
    for qa, piece in zip(qas_clean, pieces):
        qa["refined_answer"] = piece


BAND9_QUALITY_BAR = """
A genuine Band 9 response, per the official IELTS Speaking Band Descriptors, must:
- Fully develop the topic with relevant, coherent content - never a single clipped
  summary sentence, however grammatically correct, since that fails "fully develops
  topics" regardless of accuracy.
- Use cohesive devices and connectives naturally and flexibly to link ideas together.
- Use a wide range of vocabulary with precision, including natural idiomatic language
  and effective paraphrase.
- Use a full range of grammatical structures, naturally and accurately.
- Read like something a fluent, articulate speaker would actually SAY out loud, not
  like a formal written essay - natural spoken rhythm and phrasing, not stiff prose.
"""

BAND9_PART_INSTRUCTIONS = {
    1: (
        "Write ONE Band 9 answer per question. Each answer must be 2-3 full sentences "
        "(never a single clipped phrase) so it shows real idea development - a direct "
        "answer plus a reason, preference, or brief example - while still sounding like "
        "natural, concise conversational speech appropriate for Part 1's personal-topic format."
    ),
    2: (
        "Write ONE continuous Band 9 long-turn response of roughly 200-250 words - about "
        "what a candidate would say speaking for 1-2 minutes without stopping. It MUST "
        "explicitly cover every bullet point listed in the cue card question, as a single "
        "flowing, well-organized talk using natural spoken discourse markers (e.g. 'to "
        "begin with', 'what's more', 'overall') - NOT a short summary and NOT a list of "
        "disconnected short answers."
    ),
    3: (
        "Write ONE Band 9 answer per question. Each answer must be 3-5 sentences showing "
        "real analytical depth - a clear position, supporting reasoning, and a specific "
        "example or illustration - reflecting the sophistication expected in a Part 3 discussion."
    ),
}


def _band9_word_count(text: str) -> int:
    return len((text or "").split())


def generate_band9_answer(part_number: int, combined_with_context: str, answers_only: str | None = None, usage_log=None) -> str:
    overlap_text = answers_only or combined_with_context
    answer_count = _count_band9_answers(combined_with_context, answers_only)

    numbered_lines = "\n".join([f"Answer {i+1}: <Band 9 response>" for i in range(answer_count)])
    prompt = f"""You are an IELTS Band 9 speaking examiner.

A student gave these answers in IELTS Speaking Part {part_number}:

{combined_with_context}

Rewrite these as genuine Band 9-quality responses to the SAME questions, keeping
the student's original ideas/topic but expressed at Band 9 level.
{BAND9_QUALITY_BAR}

Output format:
{numbered_lines}

RULES:
- Fix ALL grammar errors
- Remove fillers: yeah, you know, kind of, sort of
- Replace basic words: help->facilitate, a lot of->a wide array of, get over->overcome, big->substantial
- Use different sentence structures
- Do not repeat "In my opinion" more than once
- Stay directly tied to each question, but FULLY DEVELOP the answer - do not
  just compress it into the shortest possible correct sentence
- {BAND9_PART_INSTRUCTIONS.get(part_number, BAND9_PART_INSTRUCTIONS[1])}

TOPIC AWARENESS RULES:
- Read the student's answer carefully and identify
  the specific topic (transport, festivals, education,
  technology, environment, health, etc.)
- Use vocabulary appropriate to that topic throughout
  your rewrite
- Transport topic: use words like commute, infrastructure,
  congestion, accessibility, punctuality
- Festival/culture topic: use words like commemorate,
  illuminate, festivities, traditions, heritage
- Education topic: use words like facilitate, curriculum,
  pedagogy, academic, extracurricular
- Economics topic: use words like revenue, expenditure,
  stimulate, fiscal, socioeconomic
- Environment topic: use words like sustainable,
  emissions, conservation, ecological, renewable
- Health topic: use words like well-being, holistic,
  preventive, ailment, therapeutic
- If topic does not match above, use vocabulary
  naturally relevant to what the student discussed
- The rewritten answer must sound like it was written
  by an expert on that specific topic

Output only the numbered answers. No extra labels. No explanation."""

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"), timeout=OPENAI_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_OVERRIDE", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )
        record_token_usage(response, usage_log)
        result = response.choices[0].message.content.strip()
        logging.warning(f"[BAND9 SUCCESS] part={part_number} words={len(result.split())}")

        orig_words = set(overlap_text.lower().split())
        new_words = set(result.lower().split())
        overlap = len(orig_words & new_words) / max(len(orig_words), 1)
        logging.warning(f"[BAND9 OVERLAP] {overlap:.0%}")

        if overlap > 0.60:
            logging.warning("[BAND9] Too similar, retrying...")
            retry_prompt = prompt + "\n\nREJECTED: Too similar to student answer. Use completely different words and sentence structures."
            # Deliberately NOT 0.0 like every other Speaking call - this
            # retry's whole purpose is to sample something DIFFERENT from
            # the first (identically-prompted-minus-one-line) attempt.
            # temperature=0 would make it deterministically reproduce the
            # same "too similar" text again, defeating the retry entirely.
            # Kept low (0.3, not the previous 0.9) so it's still far more
            # deterministic than before, without being pointless.
            retry_response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL_OVERRIDE", "gpt-4o"),
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=0.3,
                max_tokens=800
            )
            record_token_usage(retry_response, usage_log)
            result = retry_response.choices[0].message.content.strip()

        # Part 2 must be a full long-turn talk (~200-250 words), not a
        # one-line summary. If the model under-delivers on length, force
        # one retry with an explicit word-count demand before accepting it.
        if part_number == 2 and _band9_word_count(result) < 120:
            logging.warning(f"[BAND9] Part 2 answer too short ({_band9_word_count(result)} words), retrying with length enforcement...")
            length_retry_prompt = prompt + (
                "\n\nREJECTED: Your previous answer was far too short for a Band 9 "
                "Part 2 long-turn response. Write AT LEAST 200 words as ONE continuous "
                "flowing talk that explicitly covers every bullet point in the cue card. "
                "Do not summarize - develop each point with detail."
            )
            length_retry_response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL_OVERRIDE", "gpt-4o"),
                messages=[{"role": "user", "content": length_retry_prompt}],
                temperature=0.0,
                max_tokens=800
            )
            record_token_usage(length_retry_response, usage_log)
            candidate = length_retry_response.choices[0].message.content.strip()
            if _band9_word_count(candidate) > _band9_word_count(result):
                result = candidate

        return _normalize_numbered_band9_answer(result, answer_count)

    except Exception as e:
        logging.error(f"[BAND9 DIRECT CALL FAILED] {e}")
        fallback = answers_only if answers_only else combined_with_context
        if answer_count > 1:
            parts = [p.strip() for p in (answers_only or combined_with_context).split("\n\n") if p.strip()]
            if len(parts) >= answer_count:
                return "\n".join([f"Answer {i+1}: {parts[i]}" for i in range(answer_count)])
        return fallback


RELEVANCE_NOTICE_MESSAGES = {
    "completely_off_topic": "Your answer is not relevant to the topic of the question asked.",
    "partially_off_topic": "Your answer only partly addresses the topic of the question asked.",
}

# Static, non-GPT explanatory text for the "severity" tag attached to
# mistakes throughout this report. Mistakes are shown so the candidate can
# polish further, but the tag itself is the only signal for whether an
# issue actually affected the band - without this, a candidate seeing
# several "minor" tags has no way to know those did NOT lower their score.
SEVERITY_LEGEND = {
    "minor": (
        "Occasional, non-systematic issues where meaning stayed clear and a listener would "
        "understand effortlessly. These do NOT lower your band score - the official IELTS "
        "descriptors allow this kind of imperfection even at the top bands (Band 9 explicitly "
        "permits \"mistakes characteristic of native speaker speech\"; Band 8 allows "
        "\"occasional inappropriacies/non-systematic errors\")."
    ),
    "significant": (
        "The same type of issue recurring across the answer (systematic), or an issue that "
        "actually changed or obscured meaning, or required real listener effort to understand. "
        "These are the issues that genuinely affect your band score."
    ),
}

_RELEVANCE_FEEDBACK_PREFIX = {
    "completely_off_topic": (
        "Your answer did not address the question that was asked - this is the main reason "
        "your score is capped here, more than any grammar or vocabulary issue below. Focus "
        "first on directly answering what's asked. "
    ),
    "partially_off_topic": (
        "Part of your answer drifted away from the question asked, which limited your score - "
        "make sure every part of your response stays focused on what's actually being asked. "
    ),
}

# Config flag for the minor/significant mistake split - default OFF. This is
# a real, non-additive break to the "mistakes" array's contents (today it
# holds every issue regardless of severity; flag ON narrows it to
# significant/band-affecting only), so unlike the purely-additive fields
# elsewhere in this file it must not change behaviour for any existing
# caller until explicitly opted in. Does NOT affect how severity itself is
# assigned - see _validate_question_mistakes / generate_mistakes for that;
# this only routes an already-assigned severity tag to one of two places.
SPEAKING_MISTAKE_SEVERITY_SPLIT = os.getenv("SPEAKING_MISTAKE_SEVERITY_SPLIT", "false").strip().lower() == "true"

# Config flag for the WPM denominator fix - default OFF, same reasoning as
# above: this changes what evidence generate_scores() sees for Fluency and
# Coherence (see the pacing guidance text below), so it's a real behaviour
# change, not a purely additive field. Flag OFF: "speech_rate_wpm" is
# words / total clip duration INCLUDING silence, exactly as before - a
# measured speaker's thinking pauses inflate their denominator and read as
# slow, a nervous fast-talker's lack of pauses reads as fluent, entirely as
# an artifact of the denominator rather than of anything the model judged.
# Flag ON: "speech_rate_wpm" becomes words / voiced-only duration (from
# librosa.effects.split, already computed for the pause-count/avg-pause-
# duration features above - reused here, not newly computed). Either way,
# both speech_rate_wpm_raw and speech_rate_wpm_voiced stay present on every
# response (see extract_acoustic_features' voiced_duration_sec/
# silence_fraction and the per-clip block below) so the gap is visible
# regardless of which one is currently driving scoring evidence.
SPEAKING_VOICED_WPM = os.getenv("SPEAKING_VOICED_WPM", "false").strip().lower() == "true"


def _split_mistakes_by_severity(mistakes: list) -> tuple:
    """Partition a per-question mistakes list (each item already carries a
    resolved "minor"/"significant" severity - see _validate_question_
    mistakes) into (significant, minor). Every item lands in exactly one of
    the two lists; neither list's item shape changes."""
    significant, minor = [], []
    for item in mistakes or []:
        if str(item.get("severity", "")).strip().lower() == "minor":
            minor.append(item)
        else:
            significant.append(item)
    return significant, minor


def _split_part_feedback_by_severity(feedback: dict) -> tuple:
    """Partition a part-level mistakes dict (4 criteria, each with its own
    "{criterion}_severity" tag - see generate_mistakes) into (significant,
    minor, praise_texts). significant/minor are dicts of the identical
    4-criteria shape. A criterion whose severity is null is praise, not a
    criticism (generate_mistakes' own docstring: "severity does not apply
    to praise") - once mistakes/minor_observations both mean "something to
    fix", a positive statement belongs in neither, so it is pulled out into
    praise_texts (a list of strings) instead and both dicts get "" for that
    criterion. The caller is responsible for routing praise_texts into
    feedback_summary.strengths - the natural existing home for genuine,
    specific positive callouts (see generate_feedback_summary)."""
    significant, minor, praise_texts = {}, {}, []
    for criterion in ("fluency", "grammar", "vocabulary", "pronunciation"):
        text = feedback.get(criterion, "")
        severity = feedback.get(f"{criterion}_severity")
        sev_key = f"{criterion}_severity"
        if str(severity).strip().lower() == "minor":
            minor[criterion] = text
            minor[sev_key] = severity
            significant[criterion] = ""
            significant[sev_key] = None
        elif severity is None:
            if str(text).strip():
                praise_texts.append(str(text).strip())
            significant[criterion] = ""
            significant[sev_key] = None
            minor[criterion] = ""
            minor[sev_key] = None
        else:
            significant[criterion] = text
            significant[sev_key] = severity
            minor[criterion] = ""
            minor[sev_key] = None
    return significant, minor, praise_texts


# --- Deterministic (non-GPT) word-repetition detection -------------------
#
# "Models count badly" - occurrence counting and rate thresholding are done
# entirely in Python. GPT is used (see _REPETITION_ALTERNATIVES / its
# generic fallback below) only for the free-text "alternatives" suggestion
# text, never for deciding what counts as repeated.

_REPETITION_STOPWORDS = {
    "the", "a", "is", "are", "was", "i", "you", "it", "and", "but", "so", "to",
    "of", "in", "that", "this", "have", "do", "go", "get", "very", "really",
    "like", "think", "know", "because", "would", "will", "can",
}

# Fillers/discourse markers - explicitly out of scope per the spec's own
# DO-NOT list ("never flag fillers/discourse markers here"), but not all
# covered by the stopword list above (which was given verbatim). Added as a
# judgment call, flagged for confirmation rather than silently assumed.
_REPETITION_FILLER_WORDS = {
    "um", "uh", "erm", "actually", "basically", "well", "okay", "yeah", "right",
}

_REPETITION_MIN_WORD_LEN = 4

# PROVISIONAL - not yet calibrated against real data (no persisted
# transcripts exist anywhere in this system to calibrate against - see the
# no-persistence finding reported alongside this feature). These numbers
# are a placeholder only so the mechanism is complete and testable - they
# must not be treated as confirmed/final, and the flag stays OFF until they
# are. Because this whole feature sits behind SPEAKING_MISTAKE_SEVERITY_
# SPLIT (default OFF), shipping with a provisional threshold has zero
# effect on any live output until the flag is deliberately switched on.
_REPETITION_MIN_RATE_PER_100 = 2.5
_REPETITION_MIN_ABSOLUTE_COUNT = 3

# Backstop for a genuinely short whole test (e.g. an early-terminated
# attempt) - below this many total words, any rate_per_100 is too unstable
# to mean anything (a single repeat in a 20-word test is already 5/100),
# so repetition detection is skipped entirely rather than risk a noisy
# flag. Also provisional - same reasoning as the two constants above.
_REPETITION_MIN_TOTAL_WORDS = 40

# A small static table for the most commonly overused words in IELTS
# speaking answers - deliberately not a GPT call (this task's own two
# additions are both meant to be cheap/deterministic; a per-word GPT call
# for alternatives would be a third, unrequested addition). Anything not in
# this table gets a generic fallback line instead of no suggestion at all.
_REPETITION_ALTERNATIVES = {
    "nice": ["pleasant", "enjoyable", "appealing"],
    "good": ["beneficial", "valuable", "worthwhile"],
    "bad": ["poor", "unpleasant", "problematic"],
    "big": ["large", "significant", "considerable"],
    "small": ["minor", "modest", "limited"],
    "happy": ["pleased", "delighted", "content"],
    "interesting": ["engaging", "fascinating", "compelling"],
    "important": ["significant", "crucial", "essential"],
    "thing": ["aspect", "factor", "element"],
    "things": ["aspects", "factors", "elements"],
    "people": ["individuals", "others", "the public"],
    "said": ["mentioned", "explained", "noted"],
    "went": ["headed", "travelled", "made my way"],
    "lot": ["a great deal", "plenty", "a considerable amount"],
    "stuff": ["items", "material", "belongings"],
    "beautiful": ["stunning", "picturesque", "attractive"],
    "difficult": ["challenging", "demanding", "tough"],
    "easy": ["straightforward", "manageable", "simple"],
    "helpful": ["useful", "supportive", "beneficial"],
    "amazing": ["remarkable", "impressive", "outstanding"],
    "place": ["location", "spot", "destination"],
}
_REPETITION_GENERIC_ALTERNATIVES = ["a synonym", "a more specific word", "a rephrased sentence"]


def _simple_lemma(word: str) -> str:
    """Crude plural/-ing/-ed folding, deliberately not a full lemmatiser
    (per spec). Good enough to fold regular forms (walks/walking/walked ->
    walk) for repetition counting; irregular forms are left as-is."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            return stem[:-1]
        return stem
    if len(w) > 4 and w.endswith("ed"):
        stem = w[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            return stem[:-1]
        return stem
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _topic_words_from_question(question_text: str) -> set:
    """Content words drawn from the question itself, lemma-folded the same
    way as the repetition counter, so a candidate reusing the question's
    own topic word (e.g. "hobby" in a question about hobbies) is never
    flagged for it - that's staying on topic, not a limited vocabulary."""
    words = re.findall(r"[a-zA-Z']+", (question_text or "").lower())
    return {_simple_lemma(w) for w in words if len(w) > 2}


def _collapse_immediate_repeats(text: str) -> str:
    """Collapse an exact word immediately repeated (optionally with one
    filler word between, e.g. "I I think" or "the the beach") down to a
    single occurrence before counting. This is the mechanical, deterministic
    slice of "never flag repetition inside a self-correction/repair": a
    stumble-and-restart repeating the same word back-to-back should not
    inflate that word's count. Non-adjacent repairs (a correction phrase in
    between) are not caught here - a full self-correction-span analysis is
    out of scope for a pure counting pass."""
    return re.sub(r"\b(\w+)\b(?:\s+\w+\b)?\s+\1\b", r"\1", text, flags=re.IGNORECASE)


def count_word_repetitions(answer_text: str, question_text: str = "") -> list:
    """Deterministic repetition counter, run once across the WHOLE test's
    pooled answers (not per part - see _REPETITION_MIN_TOTAL_WORDS below
    and the module-level note on why: a single part, especially Part 1,
    can be too short for a per-100-words rate to mean anything, and
    over-reliance on a word is a candidate-wide trait, not a per-part one).
    Returns [{word, count, rate_per_100}], sorted by count descending, for
    words clearing BOTH the per-100-words rate threshold and the absolute
    floor - or [] outright if the pooled text is below
    _REPETITION_MIN_TOTAL_WORDS (too short for any rate to be meaningful).
    Excludes the given stopword list, filler/discourse markers, and the
    question's own topic words. Does not compute severity or touch scoring
    - purely a counting/thresholding pass; the caller attaches alternatives
    and decides where the result is shown."""
    collapsed = _collapse_immediate_repeats(answer_text or "")
    words = re.findall(r"[a-zA-Z']+", collapsed.lower())
    total = len(words)
    if total == 0 or total < _REPETITION_MIN_TOTAL_WORDS:
        return []
    topic_words = _topic_words_from_question(question_text)
    counts = {}
    for w in words:
        if len(w) < _REPETITION_MIN_WORD_LEN:
            continue
        if w in _REPETITION_STOPWORDS or w in _REPETITION_FILLER_WORDS:
            continue
        lemma = _simple_lemma(w)
        if lemma in _REPETITION_STOPWORDS or lemma in _REPETITION_FILLER_WORDS:
            continue
        if lemma in topic_words or w in topic_words:
            continue
        counts[lemma] = counts.get(lemma, 0) + 1
    results = []
    for lemma, count in counts.items():
        rate = (count / total) * 100
        if count >= _REPETITION_MIN_ABSOLUTE_COUNT and rate >= _REPETITION_MIN_RATE_PER_100:
            results.append({"word": lemma, "count": count, "rate_per_100": round(rate, 2)})
    results.sort(key=lambda r: r["count"], reverse=True)
    return results


def _repeated_word_observations(answer_text: str, question_text: str = "") -> list:
    """count_word_repetitions() plus 2-3 context-fit alternatives per word,
    phrased usefully rather than as a scolding notice - shaped for direct
    use as a minor_observations entry."""
    observations = []
    for item in count_word_repetitions(answer_text, question_text):
        alternatives = _REPETITION_ALTERNATIVES.get(item["word"], _REPETITION_GENERIC_ALTERNATIVES)
        observations.append({
            "word": item["word"],
            "count": item["count"],
            "rate_per_100": item["rate_per_100"],
            "alternatives": alternatives,
            "note": (
                f"You used \"{item['word']}\" {item['count']} times - try "
                f"{', '.join(alternatives[:-1])} or {alternatives[-1]} for variety."
            ) if len(alternatives) > 1 else (
                f"You used \"{item['word']}\" {item['count']} times - try {alternatives[0]} for variety."
            ),
        })
    return observations


def _apply_relevance_to_feedback(feedback: dict, topic_relevance: str) -> dict:
    """Ensure the written feedback doesn't contradict a topic-relevance score
    cap. Without this, generate_mistakes() can hand back ordinary grammar/
    vocabulary notes with no mention of why the score was actually capped,
    making the written feedback and the number inconsistent with each other."""
    if not isinstance(feedback, dict):
        feedback = {}
    prefix = _RELEVANCE_FEEDBACK_PREFIX.get(topic_relevance)
    if prefix:
        feedback["improvement"] = (prefix + str(feedback.get("improvement", "")).strip()).strip()
    return feedback


def _collect_completeness_notices(qas_clean: list) -> list:
    """Pull the non-empty per-question completeness_notice values out of a
    part's question list, in order, for surfacing at the part level."""
    return [
        str(qa.get("completeness_notice", "")).strip()
        for qa in (qas_clean or [])
        if isinstance(qa, dict) and str(qa.get("completeness_notice", "")).strip()
    ]


def _apply_completeness_to_feedback(feedback: dict, completeness_notices: list) -> dict:
    """Ensure the written feedback explicitly explains it when a candidate
    only answered part of a multi-part question (e.g. "Which jobs pay the
    most? Why?" answered with only the "which" half). Without this, the
    candidate sees a lower band and generic grammar/vocabulary notes with
    no indication their answer was actually incomplete relative to the task."""
    if not isinstance(feedback, dict):
        feedback = {}
    if completeness_notices:
        prefix = (
            "Part of the question wasn't fully answered, which affects your score "
            "alongside any language issues below: " + " ".join(completeness_notices) + " "
        )
        feedback["improvement"] = (prefix + str(feedback.get("improvement", "")).strip()).strip()
    return feedback


def generate_ideal_band9_answer(part_number: int, questions: list, usage_log=None) -> str:
    """Generate a Band 9 model answer directly from the question(s), ignoring
    the student's actual answer entirely. Used when the student's answer was
    off-topic - generate_band9_answer() "polishes" the student's own words,
    which is useless when those words were about the wrong subject. This
    answers the question itself instead."""
    questions = [str(q).strip() for q in (questions or []) if q and str(q).strip()]
    if not questions:
        return "Model answer unavailable."

    answer_count = len(questions)
    numbered_questions = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])
    numbered_lines = "\n".join([f"Answer {i + 1}: <Band 9 response>" for i in range(answer_count)])

    prompt = f"""You are an IELTS Band 9 speaking examiner.

The candidate's actual answer for IELTS Speaking Part {part_number} did not
address the question asked (it was off-topic), so there is nothing usable
in it to upgrade. Instead, write your OWN ideal Band 9 answer(s) directly
to the question(s) below, as if a Band 9 candidate were answering them for
the first time. Do NOT reference, reuse, or paraphrase the candidate's
original (off-topic) answer.
{BAND9_QUALITY_BAR}

Question(s):
{numbered_questions}

Output format:
{numbered_lines}

RULES:
- Answer the question(s) directly and specifically - do not go off-topic.
- Use natural, idiomatic Band 9 vocabulary and varied, accurate grammar.
- {BAND9_PART_INSTRUCTIONS.get(part_number, BAND9_PART_INSTRUCTIONS[1])}

Output only the numbered answers. No extra labels. No explanation."""

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"), timeout=OPENAI_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_OVERRIDE", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )
        record_token_usage(response, usage_log)
        result = response.choices[0].message.content.strip()
        logging.warning(f"[IDEAL BAND9] part={part_number} words={len(result.split())}")

        if part_number == 2 and _band9_word_count(result) < 120:
            logging.warning(f"[IDEAL BAND9] Part 2 answer too short ({_band9_word_count(result)} words), retrying with length enforcement...")
            length_retry_prompt = prompt + (
                "\n\nREJECTED: Your previous answer was far too short for a Band 9 "
                "Part 2 long-turn response. Write AT LEAST 200 words as ONE continuous "
                "flowing talk that explicitly covers every bullet point in the cue card. "
                "Do not summarize - develop each point with detail."
            )
            retry_response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL_OVERRIDE", "gpt-4o"),
                messages=[{"role": "user", "content": length_retry_prompt}],
                temperature=0.0,
                max_tokens=800
            )
            record_token_usage(retry_response, usage_log)
            candidate = retry_response.choices[0].message.content.strip()
            if _band9_word_count(candidate) > _band9_word_count(result):
                result = candidate

        return _normalize_numbered_band9_answer(result, answer_count)
    except Exception as e:
        logging.error(f"[IDEAL BAND9 FAILED] {e}")
        return "Model answer unavailable."


def generate_improvement(summary):

    prompt = f"""

You are an IELTS examiner.



Give 2-3 short improvement tips.



Rules:

- Keep it concise

- Actionable suggestions

- No long sentences

- No rewriting answers



Feedback:

{summary.get("feedback", {}).get("improvements", "") if isinstance(summary, dict) else ""}



Return only 2-3 short bullet-style suggestions.

"""



    res = safe_gpt_call(prompt, fallback="Focus on answering more directly and using clearer examples.", caller=call_gpt)



    if res and len(str(res).strip()) < 200:

        return str(res).strip()



    return "Focus on answering more directly and using clearer examples."





def generate_feedback_summary(part_number: int, combined_transcripts: str, usage_log=None) -> dict:
    """Scannable, bulleted feedback for a part: genuine strengths, specific
    areas to improve, and actionable tips - a report-card summary format,
    distinct from (and additive to) the detailed per-criterion "mistakes"
    text and the per-question mistake list elsewhere in this module."""
    prompt = f"""You are an IELTS Speaking examiner writing a concise,
scannable summary for a student, covering Part {part_number}.

The student's answers in this part:

{combined_transcripts}

Write three short bulleted lists:
- "strengths": 2-4 genuine, SPECIFIC things this candidate actually did
  well, grounded in their actual words - not generic praise like "good
  effort". If something is a real strength, name it and quote or reference
  the specific part of their answer that shows it.
- "areas_to_improve": 2-5 specific, evidence-based issues - each a short,
  self-contained observation a student could act on (not a full paragraph).
  Only include a genuine issue you can point to in their actual answer - it
  is completely fine to return fewer items for a strong answer.
- "tips": 3-5 short, actionable, practical pieces of advice for this
  student specifically, based on what you saw in THIS answer - not generic
  IELTS advice that could apply to anyone.

DO NOT FLAG (CRITICAL - this is SPOKEN language, not writing): never
mention punctuation, capitalization, or spelling - that is a transcription
artifact, not something the candidate did. Never comment on whether an
opinion is correct or well-reasoned - only the language is assessed. Never
mention accent (you only have text, not audio). A natural, correctly-used
idiom or informal expression is a STRENGTH (evidence for higher Lexical
Resource), never something to list as an area to improve. A single
self-correction or restart mid-sentence is normal, expected speech (the
descriptors allow it even at Band 9) - never list it as an area to
improve unless it happens so often it disrupts the whole answer. Talking
around a word the candidate doesn't know (paraphrasing instead of using
it) is a communication STRATEGY worth naming as a strength, not a
vocabulary weakness. Contractions ("I'm", "don't", "it's") and trailing-
off/ellipsis are normal features of natural spoken English, never an
area to improve. Hedges and stance markers ("I believe", "I think",
"I guess", "of course") are interchangeable and never need replacing by
another hedge - do not list swapping one for another as an area to
improve. Vague, informal connector phrases ("or anything", "or
something", "and stuff") are normal casual speech, not vagueness to
correct. A short, elliptical answer that directly answers the question
(e.g. "I believe dogs.") is a complete, natural spoken answer, not an
incomplete sentence - never list it as a fragment (a genuinely
too-short answer is a completeness issue, already tracked separately via
completeness_notice, not a language area to improve). Long, connected
speech with no clear sentence breaks is a transcription artifact, not
something to describe as "run-on sentences" or missing structure - if
pacing/organization is genuinely weak, describe the actual disorganization
in the ideas themselves, never the absence of written sentence
boundaries.

CONFIDENCE BAR: only name a strength or an area to improve if you are
genuinely confident it's real, evidenced by their actual words - an empty
or shorter list is better than a padded, generic-sounding one.

Return ONLY this JSON object, no explanation, no markdown:
{{
  "strengths": ["specific strength 1", "specific strength 2"],
  "areas_to_improve": ["specific issue 1", "specific issue 2"],
  "tips": ["actionable tip 1", "actionable tip 2", "actionable tip 3"]
}}"""

    result = safe_gpt_call(prompt, fallback=None, caller=lambda p: call_gpt(p, usage_log=usage_log))

    parsed = result if isinstance(result, dict) else None
    if parsed is None and isinstance(result, str) and result.strip():
        try:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            parsed = None

    def _clean_list(value, min_items):
        if not isinstance(value, list):
            return []
        items = [str(v).strip() for v in value if str(v).strip()]
        return items if len(items) >= min_items else items

    if isinstance(parsed, dict):
        return {
            "strengths": _clean_list(parsed.get("strengths"), 1),
            "areas_to_improve": _clean_list(parsed.get("areas_to_improve"), 0),
            "tips": _clean_list(parsed.get("tips"), 1),
        }

    return {"strengths": [], "areas_to_improve": [], "tips": []}


def generate_mistakes(
    part_number: int,
    combined_transcripts: str,
    usage_log=None
) -> dict:
    prompt = f"""You are a strict IELTS Speaking examiner.

A student gave these answers in Part {part_number}:

{combined_transcripts}

Analyze the student's actual spoken language and give
specific feedback for each of these 4 criteria.

For each criterion, mention a SPECIFIC example from
their actual answer — quote their exact words, then
explain what is wrong and how to fix it.

Return ONLY this JSON object, no explanation, no markdown:
{{
  "fluency": "specific feedback with example from their answer",
  "fluency_severity": "minor" or "significant" or null,
  "grammar": "specific feedback with example from their answer",
  "grammar_severity": "minor" or "significant" or null,
  "vocabulary": "specific feedback with example from their answer",
  "vocabulary_severity": "minor" or "significant" or null,
  "pronunciation": "one general, honest, text-grounded observation - not a fabricated claim about specific stress/intonation you didn't actually hear",
  "pronunciation_severity": "minor" or "significant" or null,
  "improvement": "one specific actionable tip for THIS student based on their actual answers"
}}

SEVERITY CLASSIFICATION (per criterion, alongside its feedback text):
Judge severity by FREQUENCY + SYSTEMATICITY + COMMUNICATION IMPACT, never by
counting errors or applying a fixed deduction:
- "minor": the issue is occasional and non-systematic, meaning stayed
  completely clear, and a listener would understand effortlessly - e.g. one
  tense slip, one article omission, one slightly awkward phrase used only
  once. This is exactly the kind of imperfection the official descriptors
  allow even at the top bands (Band 9 explicitly permits "mistakes
  characteristic of native speaker speech"; Band 8 allows "occasional
  inappropriacies/non-systematic errors").
- "significant": the SAME type of issue recurs across the answer
  (systematic), or it actually changes/obscures meaning, or the listener
  would need real effort to understand what was meant.
- null: the criterion's text above is praise, not a criticism (per the
  CRITICAL RULES below, a genuinely strong criterion should get brief
  praise, not a forced issue) - severity does not apply to praise.
Do not default everything to "significant" to seem thorough, and do not
default everything to "minor" to seem lenient.

DO NOT FLAG (CRITICAL - this is SPOKEN language, not writing):
This is a transcript of something the student SAID out loud, not something
they wrote. Never flag, correct, or mention punctuation, capitalization, or
spelling - any of that in this text is a transcription artifact added by
automatic speech-to-text, not something the candidate did wrong. Never
flag contractions ("I'm", "don't", "it's") or trailing-off/ellipsis - both
are normal, correct features of natural spoken English, not errors. Never
describe the answer as containing "run-on sentences" or lacking sentence
structure - speech has no sentences; the transcriber inserted whatever
breaks appear in this text, so their absence is never a grammar or
punctuation issue. A short, elliptical answer that directly answers the
question (e.g. "I believe dogs.") is complete and natural, not a
fragment. Hedges and stance markers ("I believe", "I think", "I guess",
"of course") never need replacing by a different hedge - that is a
preference, not an error. Vague, informal connector phrases ("or
anything", "or something", "and stuff") are normal casual speech, not
something to correct into a more formal phrase.

WHAT NOT TO CRITIQUE (CRITICAL): never comment on whether an opinion is
correct, sensible, or well-reasoned - there is no "correct" opinion in
IELTS Speaking, only the language used to express it. Never penalize
British vs American vocabulary/word choice. You only have text, never
audio, so never claim to know or comment on the candidate's accent. Never
treat the candidate briefly asking the examiner to repeat or clarify a
question as a fluency mistake - it's normal, permitted exam behavior.

CONFIDENCE BAR (CRITICAL): a wrong "correction" is worse than a missed
one. Before naming an error, ask whether a Band 9 native speaker would
genuinely avoid it, or whether it just differs from your own phrasing
preference. Restrictive "which" (standard British English), fixed idioms
like "as far as X is concerned" (always takes "is", even with an "or"-
joined subject), and regionally-common dictionary words (e.g. "telecast")
are NOT errors - do not invent a correction for any of them. If not fully
confident something is a genuine error, leave it out.

IDIOMS AND INFORMAL EXPRESSIONS ARE NOT MISTAKES (CRITICAL): a natural,
correctly-used idiom or informal expression (e.g. "chalk and cheese",
"out of the box") is POSITIVE evidence for Lexical Resource - the
descriptors name "less common and idiomatic items" as a real gate for
Band 7+. Never "correct" one into a plainer/more formal phrase (e.g.
"chalk and cheese" -> "very different", "out of the box" -> "unique") -
that punishes the candidate for exactly the kind of language the higher
bands reward. Only flag an idiom if it is genuinely misused, means
something different from what the candidate intended, or is wrong-register
for the context - never merely because a plainer alternative exists.

SELF-CORRECTION AND CIRCUMLOCUTION ARE NOT MISTAKES (CRITICAL): a
candidate self-correcting or restarting mid-sentence (e.g. "I go there
every day... well, actually not every day") is normal, expected speech -
the descriptors explicitly allow it even at Band 9. Never flag it as a
fluency problem unless it happens so often it visibly disrupts the whole
answer (a frequency judgment, not a one-off). Likewise, if the candidate
talks around a word they don't know instead of using it (e.g. "keep
putting things off" instead of "procrastinate"), that is a communication
STRATEGY the descriptors credit, not something to flag under vocabulary.

RULES:
- fluency: only comment on filler use (yeah, you know, kind of) if it is
  frequent/repeated enough to disrupt the flow - one or two natural
  fillers in an answer is normal speech, not a fluency issue worth
  flagging. Comment on sentence linking and hesitation patterns instead
  when fillers themselves are not genuinely excessive.
- grammar: quote a specific grammatical error they made
  and show the correction
  Example: 'You said "One of them are" — correct form
  is "One of them is" as the subject is singular.'
  Do not flag mixing past tense (for a one-time past action, e.g. "I
  recently watched...") with present tense (for the enduring/general
  nature of what was watched, e.g. "it's a reality show") - that split is
  standard, correct English, not a tense error.
- vocabulary: quote a specific IMPRECISE or WRONG word choice and suggest
  a better alternative - never a word that is already correct just
  because a fancier synonym exists, and never a deliberate paraphrase
  used to work around a harder word (that is a strength, not a weakness).
  Example: 'Instead of "a lot of clubs" consider
  using "a wide array of clubs"'
- pronunciation: this is TEXT ONLY - you have no audio, so do NOT claim to
  diagnose specific stress/intonation patterns as if you heard them. Give
  one general, honest, text-grounded observation instead (e.g. that longer
  or more complex sentences in their answer are worth rehearsing for pacing
  and clarity). Do not invent a claim about how a specific word was
  actually pronounced.
- Keep each feedback to 2 sentences maximum
- Be specific, not generic
- Do NOT return template sentences
CRITICAL RULES FOR EVERY CRITERION:
- Only include a criterion's feedback if there is a genuine, specific issue
  to point to - do NOT invent or manufacture an issue just to have
  something to say. If a criterion is genuinely strong with nothing
  specific to flag, it is fine to give brief genuine, specific praise
  instead of forcing a fabricated criticism.
- Avoid generic, unsubstantiated praise ("performed well", "did a great
  job") with no specific evidence behind it - but a specific, accurate
  positive observation (e.g. "correctly used the third conditional in
  'if I had...'") is fine and should not be avoided just to sound critical.
- Format: quote their exact words → explain the issue
  → show the correct version or better alternative
- Keep each feedback to 2 sentences maximum
Return only the JSON object. No markdown.
No ```json fence. No explanation before or after.
Start your response with {{ and end with }}"""

    try:
        logging.warning(f"[MISTAKES CALLED] part={part_number}")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"), timeout=OPENAI_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_OVERRIDE", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600
        )
        record_token_usage(response, usage_log)
        result = response.choices[0].message.content.strip()
        logging.warning(f"[MISTAKES RESULT] {result[:100]}")
        if result:
            try:
                clean = result.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)
                if all(k in parsed for k in [
                    "fluency", "grammar", "vocabulary", "pronunciation"
                ]):
                    # Deterministic backstop: don't trust GPT's severity
                    # value on faith - a criterion with real critique text
                    # must carry a valid "minor"/"significant" severity
                    # (default to "significant", the safer direction, if
                    # missing or garbled); a criterion GPT left empty/falsy
                    # (no genuine issue - see CRITICAL RULES in the prompt)
                    # correctly has no severity to apply at all.
                    for criterion in ("fluency", "grammar", "vocabulary", "pronunciation"):
                        severity_key = f"{criterion}_severity"
                        has_real_text = bool(str(parsed.get(criterion, "")).strip())
                        severity = parsed.get(severity_key)
                        if not has_real_text:
                            parsed[severity_key] = None
                        elif severity not in ("minor", "significant"):
                            parsed[severity_key] = "significant"
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
    except Exception as e:
        logging.error(f"[MISTAKES/SCORES FAIL] {e}")

    return {
        "fluency": "Work on reducing filler words and improving idea linkage.",
        "fluency_severity": "significant",
        "grammar": "Review subject-verb agreement and sentence variety.",
        "grammar_severity": "significant",
        "vocabulary": "Replace basic words with more precise academic vocabulary.",
        "vocabulary_severity": "significant",
        "pronunciation": "Focus on natural stress patterns and rhythm.",
        "pronunciation_severity": "significant",
        "improvement": "Focus on expanding your answers with specific examples."
    }


QUESTION_MISTAKES_FALLBACK = [
    {"type": "fluency", "original": "", "corrected": "", "explanation": "Reduce hesitation and link ideas more clearly.", "severity": "significant"},
    {"type": "grammar", "original": "", "corrected": "", "explanation": "Review subject-verb agreement and sentence variety.", "severity": "significant"},
    {"type": "vocabulary", "original": "", "corrected": "", "explanation": "Replace basic words with more precise academic vocabulary.", "severity": "significant"},
    {"type": "pronunciation", "original": "", "corrected": "", "explanation": "Work on natural stress and rhythm in connected speech.", "severity": "significant"},
]


_COMPLETENESS_SIGNAL_PATTERN = re.compile(
    r"\b(why|because|explain|reason|when|how often|how long|how many|"
    r"how much|where|who|which|what about)\b",
    re.IGNORECASE,
)


def _question_expects_additional_content(question: str) -> bool:
    """Heuristic backstop: does the question text itself actually ask for
    more than one thing (a reason, a specific extra detail, or multiple
    cue-card bullets)? Used to forcibly clear a completeness_notice GPT
    returned when the question doesn't support it - the prompt already
    instructs GPT not to invent unstated sub-parts, but that instruction
    isn't reliably followed (observed real cases: a plain "Do you think
    people spend too much time watching TV?" flagged for not covering
    unrelated content nobody asked about). Uses \\b word boundaries, not
    plain substring matching - "where" must not match inside "somewhere",
    same class of bug as the "clear"-inside-"unclear" issue found earlier.
    """
    text = (question or "").strip()
    if not text:
        return False
    if "|" in text:
        return True  # explicit cue-card bullet separator
    if text.count("?") > 1:
        return True  # multiple distinct questions
    return bool(_COMPLETENESS_SIGNAL_PATTERN.search(text))


def _normalize_for_mistake_comparison(text: str) -> list:
    """Strip punctuation and case, so a purely punctuation/capitalization
    'correction' can be detected and discarded - not left to rely on the
    GPT prompt alone."""
    return re.sub(r"[^\w\s]", "", text).lower().split()


def _dedupe_part_level_text_against_question_mistakes(text: str, question_mistakes: list) -> str:
    """Item C.2 - live output has reported the same error twice: once as a
    specific, locatable per-question mistake, and again inside the
    free-text part-level criterion summary produced by generate_mistakes()
    (e.g. "a lot many things" and the "advantages ... is" agreement error
    both appeared in both places). Keeps the per-question version, which
    is specific and locatable; strips any SENTENCE from the part-level
    prose whose normalized text contains a 3+-word flagged phrase
    (original or corrected) already reported per-question, so the same
    error isn't double-counted. Pure filter - never changes any score."""
    if not text or not question_mistakes:
        return text
    flagged_phrases = []
    for m in question_mistakes:
        if not isinstance(m, dict):
            continue
        for key in ("original", "corrected"):
            tokens = _normalize_for_mistake_comparison(m.get(key) or "")
            if len(tokens) >= 3:
                flagged_phrases.append(" ".join(tokens))
    if not flagged_phrases:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    kept = [
        s for s in sentences
        if not any(phrase in " ".join(_normalize_for_mistake_comparison(s)) for phrase in flagged_phrases)
    ]
    return " ".join(kept).strip()


# Explanation text citing any of these as the reason for a "mistake" means
# the underlying justification is about punctuation/capitalization - a
# transcription artifact, not a speaking error - even if the "original"/
# "corrected" pair also happens to differ in real wording elsewhere. A pure
# text-equality check alone misses this: GPT can bundle a genuine wording
# fix together with an unwarranted punctuation nitpick in the same item, and
# the explanation is the only reliable signal that punctuation was (at least
# partly) the stated reason.
# Covers punctuation AND capitalization/spelling - both are transcription
# artifacts (added by speech-to-text, not something the candidate did),
# never a genuine speaking error. A real observed case: systematic_errors
# flagged "use of 'i' in lowercase" as a recurring VOCABULARY pattern
# (occurrences: "i live in a house", "i would like to talk", ...) - this
# keyword list already existed for per-question mistakes but was never
# applied to detect_systematic_errors()'s validation at all, and didn't
# cover "lowercase"/"uppercase" wording either.
_PUNCTUATION_EXPLANATION_KEYWORDS = (
    "comma", "commas", "full stop", "period", "punctuation",
    "capital letter", "capitalization", "capitalisation",
    "capitalize", "capitalise", "lowercase", "uppercase",
    "spelling", "misspell", "misspelled", "misspelling",
)


# Explanation text citing any of these as the reason for a "mistake" means
# the underlying justification is that the candidate self-corrected or
# restarted mid-sentence. The official band descriptors explicitly list
# this as NORMAL, EXPECTED speech from Band 6 up ("occasional repetition,
# self-correction..."), and it is Band 9's OWN allowance ("only rare
# repetition or self-correction") - not an error at all unless it happens
# so often it becomes disruptive (a frequency judgment the checklist in
# generate_scores() already makes correctly; a single flagged instance in
# a per-question mistakes list is never that). Flagging a single instance
# as a "mistake" would penalize a candidate for demonstrating exactly the
# self-monitoring awareness the descriptors reward.
_SELF_CORRECTION_EXPLANATION_KEYWORDS = (
    "self-correct", "self correct", "corrected themselves", "correcting themselves",
    "corrects themselves", "restarted the sentence", "restart the sentence",
    "restarting the sentence", "changed their answer", "changed their mind mid",
    "backtrack", "false start", "interrupting themselves", "interrupted themselves",
    "rephrased mid-sentence", "rephrasing mid-sentence", "correct themselves",
)

# Speech-level self-correction markers - a candidate audibly catching and
# fixing themselves mid-answer ("People like to enjoy, sorry, people like
# to visit..."). Different from _SELF_CORRECTION_EXPLANATION_KEYWORDS
# above, which scans GPT's own EXPLANATION text for an admission of this;
# this checks the flagged "original" SPAN itself, catching the case where
# GPT flags the self-correction as the "mistake" without ever admitting
# that's what it is in the explanation.
_SELF_CORRECTION_SPAN_MARKERS = ("sorry", "i mean", "no wait", "my mistake", "i meant", "my bad")


def _flagged_span_contains_self_correction(original: str) -> bool:
    """True if the flagged span itself is a self-correction: either it
    contains an audible self-correction marker, or the same short phrase
    appears twice in a row (the candidate restating themselves without a
    marker word) - self-correction is a fluency POSITIVE per the official
    descriptors (allowed even at Band 9), never a mistake to flag."""
    tokens = _normalize_for_mistake_comparison(original)
    text_norm = " ".join(tokens)
    if any(marker in text_norm for marker in _SELF_CORRECTION_SPAN_MARKERS):
        return True
    for n in (3, 2):
        if len(tokens) < n * 2:
            continue
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if len(ngrams) != len(set(ngrams)):
            return True
    return False

# Explanation text conceding there's no genuine error means the "mistake"
# shouldn't exist at all - GPT flagged something, then talked itself out
# of it in its own justification, but still returned the item anyway.
# Ported from evaluators/writing.py's _NO_GENUINE_ERROR_PHRASES (same
# proven fix, same reasoning - "mistakes" is an array slot GPT fills
# whether or not a real error exists, so a self-admission in the
# explanation is a reliable signal to drop the item regardless of what
# "type"/"severity" it was given), extended with additional phrasing
# specific to how this file's own prompts describe borderline
# non-errors ("does not affect meaning", "is acceptable", "minor
# stylistic point", "can be written", "is also possible").
_SPEAKING_NO_GENUINE_ERROR_PHRASES = (
    "no error",
    "not an error",
    "not a genuine error",
    "no correction needed",
    "no correction is needed",
    "does not need correction",
    "doesn't need correction",
    "not incorrect",
    "original is acceptable",
    "borderline but not",
    "not necessary to correct",
    "no need to correct",
    "acceptable as is",
    "acceptable and no correction",
    "this is correct",
    "is actually correct",
    "is grammatically correct",
    "already correct",
    "does not affect meaning",
    "is acceptable",
    "minor stylistic point",
    "can be written",
    "is also possible",
)


# A demonstrated, RECURRING false positive on real test data (the exact
# same sentence, three separate times, surviving a prompt-only fix AND a
# first backstop attempt): a candidate correctly uses present tense to
# describe the enduring/general nature of something they experienced in
# the past ("I recently watched a show... the show's aim IS to test...")
# and GPT flags this as a tense error, "correcting" it to past tense -
# which is actively WRONG. "I watched a great show yesterday. It's about
# a detective in Paris." is standard, correct English (past tense for the
# one-time viewing, present tense for the show's ongoing nature).
#
# The first backstop attempt matched keywords in GPT's free-form
# "explanation" text ("verb tense should remain consistent") - but GPT
# phrased the SAME misjudgment a third way ("The tense should be past
# tense to match 'recently watched'") that didn't match any keyword,
# proving explanation-text matching is too fragile for this. This version
# is structural instead: it looks at the actual token-level edit, not the
# wording used to justify it, so it's immune to how GPT phrases the
# explanation.
_PRESENT_TO_PAST_TENSE_SWAPS = {
    "is": "was", "are": "were", "am": "was", "does": "did", "has": "had",
}

# If the original clause is ALREADY anchored to one specific past moment
# (e.g. "the weather is bad yesterday"), a present-tense verb there could
# be a genuine error worth fixing - only suppress the false-positive
# pattern when there's no such anchor, which is the demonstrated real case
# (a general/ongoing truth about something, not tied to one past instant).
_PAST_TIME_ANCHOR_PATTERN = re.compile(
    r"\b(yesterday|last (night|week|month|year|time)|\bago\b|back then|"
    r"at that (time|point|moment)|previously|that (day|time|year))\b",
    re.IGNORECASE,
)


def _is_present_to_past_tense_consistency_correction(original: str, corrected: str) -> bool:
    if _PAST_TIME_ANCHOR_PATTERN.search(original):
        return False
    orig_tokens = _normalize_for_mistake_comparison(original)
    corr_tokens = _normalize_for_mistake_comparison(corrected)
    if len(orig_tokens) != len(corr_tokens):
        return False
    diffs = [(o, c) for o, c in zip(orig_tokens, corr_tokens) if o != c]
    if len(diffs) != 1:
        return False
    present, past = diffs[0]
    return _PRESENT_TO_PAST_TENSE_SWAPS.get(present) == past


def _is_only_which_that_swap(original: str, corrected: str) -> bool:
    """Restrictive 'which' is standard, correct British English (IELTS
    uses British conventions), not an error - it recurs across nearly
    every test as a false-positive "mistake". If the ONLY difference
    between original and corrected is swapping 'which' for 'that' (or vice
    versa), this is a style preference being mislabeled as a mistake."""
    def _placeholder(text):
        tokens = _normalize_for_mistake_comparison(text)
        return ["RELPRON" if t in ("which", "that") else t for t in tokens]
    return _placeholder(original) == _placeholder(corrected)


# Phrase pairs that are fully interchangeable in standard English - neither
# side is more "correct" than the other, so a "correction" that only swaps
# one for the other fixes nothing real. This is the same abstract bug
# recurring under different specific words each time (which/that and "as
# far as X is concerned" above are the same failure class with their own
# dedicated checks, since they need structural handling that isn't just a
# phrase pair): a real case flagged "because of a shortage of time" ->
# "due to a shortage of time" as a grammar fix, then a SEPARATE real case
# flagged "shortage of time" -> "lack of time" - even though "shortage of
# time" is already named as a protected example in the CONFIDENCE BAR
# prompt text, proving prompt instructions alone don't reliably prevent
# this. Each new instance of this failure class gets added here as one
# more pair, rather than a new one-off function, since the underlying
# check is identical regardless of which specific words are involved.
_INTERCHANGEABLE_PHRASE_PAIRS = (
    ("because of", "due to"),
    ("shortage of time", "lack of time"),
)


def _is_only_known_synonym_phrase_swap(original: str, corrected: str) -> bool:
    def _placeholder(text):
        replaced = text
        for i, (a, b) in enumerate(_INTERCHANGEABLE_PHRASE_PAIRS):
            token = f"SYNPHRASE{i}"
            replaced = re.sub(rf"\b{re.escape(a)}\b", token, replaced, flags=re.IGNORECASE)
            replaced = re.sub(rf"\b{re.escape(b)}\b", token, replaced, flags=re.IGNORECASE)
        return _normalize_for_mistake_comparison(replaced)
    return _placeholder(original) == _placeholder(corrected)


_ARTICLES = ("a", "an", "the")


def _is_single_missing_article_insertion(original: str, corrected: str) -> bool:
    """A real observed case: a missing article ('a' before "career break")
    was tagged "significant" - despite "a single dropped article" being
    THIS FILE'S OWN named example of what "minor" means (see STEP 3 in
    generate_question_mistakes() below). This isn't a case where the
    minor/significant definition is unclear - it's the model failing to
    apply its own rule. Rather than trust severity classification alone
    for this one narrow, structurally-checkable pattern, detect it
    directly: "corrected" has exactly one extra token versus "original",
    and that one extra token is a/an/the, with every other token
    unchanged and in the same order."""
    orig_tokens = _normalize_for_mistake_comparison(original)
    corr_tokens = _normalize_for_mistake_comparison(corrected)
    if len(corr_tokens) != len(orig_tokens) + 1:
        return False
    for i in range(len(corr_tokens)):
        if corr_tokens[i] not in _ARTICLES:
            continue
        if corr_tokens[:i] + corr_tokens[i + 1:] == orig_tokens:
            return True
    return False


_AS_FAR_AS_CONCERNED_PATTERN = re.compile(r"\bas far as\b.+\b(is|are)\s+concerned\b", re.IGNORECASE)


def _is_only_concerned_verb_swap(original: str, corrected: str) -> bool:
    """'As far as X is concerned' is a fixed idiom that always takes 'is',
    regardless of whether X is singular, plural, or an 'or'-joined
    compound (which itself takes agreement with the nearer noun under
    standard English rules) - 'are concerned' is not a valid alternative
    here. A real observed failure: the model "corrected" a genuinely
    correct 'is concerned' to 'are concerned', actively teaching wrong
    grammar rather than a harmless nitpick."""
    if not (_AS_FAR_AS_CONCERNED_PATTERN.search(original) and _AS_FAR_AS_CONCERNED_PATTERN.search(corrected)):
        return False
    def _placeholder(text):
        tokens = _normalize_for_mistake_comparison(text)
        return ["VERB" if t in ("is", "are") else t for t in tokens]
    return _placeholder(original) == _placeholder(corrected)


def _validate_question_mistakes(items, transcript: str = "") -> list:
    """`transcript` is optional and defaults to "" - when omitted (as every
    existing caller/test that validates hand-constructed mistake dicts
    without a real transcript already does), the verbatim/exact-once check
    below is simply skipped, so this stays fully backward compatible.
    generate_question_mistakes() (the only production caller) always
    passes the real transcript."""
    if not isinstance(items, list):
        return []
    valid = []
    for item in items:
        if (
            isinstance(item, dict)
            and item.get("type") in ("fluency", "grammar", "vocabulary", "pronunciation")
            and "original" in item
            and "corrected" in item
        ):
            original = str(item.get("original", ""))
            corrected = str(item.get("corrected", ""))
            # A hyphen has no acoustic reality in speech - "out-of-the-box"
            # and "out of the box" sound identical spoken aloud, so any
            # hyphen joining two words here is a written-typesetting
            # artifact GPT defaulted to, never a genuine spoken-language
            # fix. Strip it before any other check, regardless of what
            # else the correction legitimately fixes, since "corrected"
            # must model how something should be SAID, not typed.
            corrected = re.sub(r"(?<=\w)-(?=\w)", " ", corrected)
            corrected = re.sub(r"\s+", " ", corrected).strip()
            explanation = str(item.get("explanation", ""))
            # This is a transcript of speech, not writing - punctuation and
            # capitalization are transcription artifacts, not something the
            # candidate did wrong. The prompt already instructs GPT not to
            # generate these, but instruction-following isn't guaranteed, so
            # this is a hard backstop: if the only difference between
            # "original" and "corrected" is punctuation/case, discard it.
            if _normalize_for_mistake_comparison(original) == _normalize_for_mistake_comparison(corrected):
                continue
            # Also discard if punctuation/capitalization is cited as (part
            # of) the reason, even when other genuine wording differs too -
            # e.g. "adding a comma before 'but' improves the flow" bundled
            # alongside an unrelated tense fix in the same item.
            explanation_lower = explanation.lower()
            if any(kw in explanation_lower for kw in _PUNCTUATION_EXPLANATION_KEYWORDS):
                continue
            # Self-correction/restarting mid-sentence is normal, expected
            # speech per the official descriptors (even Band 9 allows "rare
            # self-correction") - discard any item whose stated reason is
            # that the candidate self-corrected, regardless of what the
            # prompt-level instructions say, matching the punctuation
            # backstop above.
            if any(kw in explanation_lower for kw in _SELF_CORRECTION_EXPLANATION_KEYWORDS):
                continue
            # Item C.1 - same principle, checked on the flagged SPAN itself
            # rather than GPT's explanation text: catches GPT flagging a
            # self-correction as if it were the error, without ever
            # admitting that's what it is in "explanation".
            if _flagged_span_contains_self_correction(original):
                continue
            # A demonstrated, recurring false positive: present tense used
            # to describe the ongoing/general nature of something
            # experienced in the past (e.g. "the show's aim IS to test...")
            # gets flagged as a tense-consistency error and "corrected" to
            # past tense - actively wrong, since past-tense narration +
            # present-tense description of the thing itself is standard
            # English. Prompt-level guidance alone did not hold up against
            # this on real test data, so it's discarded here regardless.
            if _is_present_to_past_tense_consistency_correction(original, corrected):
                continue
            # Restrictive "which" is standard British English, not an
            # error - and "as far as X is concerned" is a fixed idiom that
            # always takes "is". Both were observed as confidently WRONG
            # "corrections" (the second one actively teaches incorrect
            # grammar), not just unhelpful nitpicks - discard regardless of
            # what the prompt-level instructions say, since instruction-
            # following on this class of subtle grammar judgment isn't
            # reliable enough to trust alone.
            if _is_only_which_that_swap(original, corrected):
                continue
            if _is_only_concerned_verb_swap(original, corrected):
                continue
            if _is_only_known_synonym_phrase_swap(original, corrected):
                continue
            # Per the official descriptors, a one-off slip (allowed even at
            # Band 9: "mistakes characteristic of native speaker speech")
            # doesn't cap the band the way a recurring/systematic issue
            # does - default to "significant" if GPT omits or returns an
            # unrecognized value, since that's the safer default (a
            # candidate isn't shortchanged by an unlabeled issue being
            # under-weighted).
            severity = str(item.get("severity", "")).strip().lower()
            if severity not in ("minor", "significant"):
                severity = "significant"
            # Deterministic override, not just a default: a single missing
            # article is THIS FILE'S OWN canonical example of "minor" (see
            # STEP 3 below) - force it regardless of what GPT's severity
            # value says, the same way the backstops above override GPT's
            # judgment on other narrow, structurally-checkable patterns.
            if _is_single_missing_article_insertion(original, corrected):
                severity = "minor"
            # Verbatim + exact-once, when a real transcript is available -
            # ported from evaluators/writing.py's
            # _mistake_original_is_verbatim (the verbatim check: "original"
            # must be an actual substring of what the candidate really
            # said, or the mistake is a hallucination/paraphrase), plus an
            # additional requirement this file didn't have before: "original"
            # must be a multi-word span matching EXACTLY ONCE - a bare
            # single word matches everywhere and tells the candidate
            # nothing about which specific moment is meant, and a span that
            # matches more than once is ambiguous for the same reason.
            # Gated on `transcript` being supplied (see the docstring above)
            # so this never changes behaviour for a caller that isn't
            # validating against real speech.
            if transcript:
                if len(_normalize_for_mistake_comparison(original)) < 2:
                    continue
                if _quote_occurrence_count(original, transcript) != 1:
                    continue
            # Self-admission: the explanation itself concedes there's no
            # genuine error, even though GPT still returned the item -
            # ported from Writing's _NO_GENUINE_ERROR_PHRASES/
            # _SPEAKING_NO_GENUINE_ERROR_PHRASES, same reasoning ("mistakes"
            # is an array slot GPT fills whether or not a real error
            # exists, so an explanation that talks itself out of the
            # mistake is a reliable signal to drop it regardless of what
            # type/severity it was given).
            if any(kw in explanation_lower for kw in _SPEAKING_NO_GENUINE_ERROR_PHRASES):
                continue
            valid.append({
                "type": item.get("type"),
                "original": original,
                "corrected": corrected,
                "explanation": explanation,
                "severity": severity,
            })

    # Dedupe mistakes covering the same span - two items whose normalized
    # "original" tokens are identical, or one is a contiguous run inside
    # the other (not just a set-overlap, which would wrongly merge two
    # unrelated mistakes that just happen to share an isolated common
    # word), are the same underlying issue reported twice - occasionally
    # once as e.g. "grammar" and once as "vocabulary". Keep the first
    # (already-validated) occurrence, drop the rest.
    def _is_contiguous_span_subset(shorter: list, longer: list) -> bool:
        if len(shorter) > len(longer):
            return False
        for i in range(len(longer) - len(shorter) + 1):
            if longer[i:i + len(shorter)] == shorter:
                return True
        return False

    deduped = []
    seen_spans = []
    for item in valid:
        span = _normalize_for_mistake_comparison(item["original"])
        if any(_is_contiguous_span_subset(span, s) or _is_contiguous_span_subset(s, span) for s in seen_spans):
            continue
        seen_spans.append(span)
        deduped.append(item)
    return deduped


def generate_question_mistakes(question: str, answer: str, usage_log=None) -> dict:
    prompt = f"""You are a strict IELTS Speaking examiner.

A student answered this question:
Question: {question}
Answer: {answer}

STEP 1 - CHECK TASK COMPLETENESS FIRST:
Many IELTS questions have more than one part - e.g. "Which kinds of jobs
have the highest salaries in your country? Why is this?" asks BOTH which
jobs AND why. Look carefully at the question and identify whether it has
multiple distinct parts. Common patterns that signal multiple parts:
- More than one question mark ("...? Why?", "...? When?", "...? How?")
- A main question followed by a short follow-up word/clause: why, why is
  this, why not, when, how, how often, where, what about, who, which,
  do you agree, would you, and why/how/etc.
- Two questions joined by "and" or "or" ("What...and why...?")
- A request for both a description AND a reason/opinion/example - but ONLY
  when the question TEXT itself asks for the reason (contains words like
  "why", "because", "explain", "reason"). Do NOT treat a plain factual
  question as multi-part just because a fuller answer COULD have included
  more explanation. Counter-example: "What types of TV shows are most
  popular in your country?" is SINGLE-part - it only asks WHAT, not why -
  so an answer that names the types (e.g. "reality shows and daily soaps
  are popular") is complete. Do not flag "didn't explain why they're
  popular" here; that reason was never asked for.
For EACH distinct part you identify, check whether the answer clearly
addresses it, not just one of them.
- If one or more parts are NOT addressed, set "completeness_notice" to a
  short, specific sentence naming exactly what was missed - e.g. "You
  named which jobs pay well, but didn't explain why." or "You didn't say
  when this typically happens."
- If the question is genuinely single-part, or the answer addresses every
  part it has, set "completeness_notice" to an empty string "".
- Do not invent missing parts for a genuinely single-part question just
  because the answer is short - only flag it when the QUESTION itself
  asked for more than one thing.
- CRITICAL: an explicit "there is/are none" answer IS a complete, valid
  answer to that part - do NOT flag it as missing. E.g. if asked "what
  special food is connected with this?" and the candidate says "there is
  no special food for this, but..." they HAVE answered that part (the
  answer is "none") - do not say they "didn't specify" or "didn't
  mention" it.
- If this looks like a Part 2 cue-card prompt (a long prompt with several
  bullet points, e.g. containing "|" or multiple "should say" / "describe"
  clauses), do NOT require literal coverage of every single bullet - a
  natural 1-2 minute spoken answer often runs out of time before covering
  every bullet, and real IELTS scoring does not penalize this. Only set a
  completeness_notice here if the response barely engages with the topic
  at all, not merely because one minor bullet point was skipped.
- CRITICAL - CHECK FOR ECHOING: if the "answer" is mostly just the
  candidate repeating or closely paraphrasing the question itself back
  (a stalling tactic - "parroting the examiner's exact question... instead
  of answering directly") rather than actually providing an answer, this
  is NOT a complete response even if it superficially seems on-topic. Set
  "completeness_notice" to something like "This mostly repeats the
  question rather than answering it - no real answer was given." Judge
  this by whether genuinely NEW content (an actual opinion, fact, or
  detail not present in the question) was added, not by surface topic
  overlap - naturally reusing a few words from the question while
  answering it properly is completely normal and must NOT be flagged.
- CRITICAL - READ THE WHOLE ANSWER FOR BURIED CONTENT: a reason/explanation
  does NOT need to be clearly signposted (e.g. starting with "because") to
  count as answered - spoken answers are often long, run-on, and full of
  fillers, with the actual reason embedded in the middle rather than
  clearly flagged. Read the ENTIRE answer carefully for any causal or
  explanatory content (comparisons, consequences, "you can/don't have to
  X", "that's why", etc.) before concluding a "why" was not addressed. A
  real observed case: an answer explaining that streaming is flexible and
  lets you watch "at any hour... at your convenience" unlike fixed TV
  broadcast times DOES explain why streaming is more popular, even though
  it never uses the word "because" and is delivered as one long rambling
  sentence - flagging that as missing the "why" is wrong. Disorganized
  delivery is a fluency/coherence issue to note separately in STEP 2, not
  evidence that the content itself is missing.

STEP 2 - IDENTIFY LANGUAGE ISSUES:
Identify concrete issues in the student's answer covering these categories:
fluency, grammar, vocabulary, pronunciation.

DO NOT FLAG (CRITICAL - this is SPOKEN language, not writing):
This is a transcript of something the student SAID out loud, not something
they wrote. Never flag, correct, or mention:
- Punctuation (commas, periods, question marks, etc.) - the student never
  "typed" punctuation; any punctuation in this text was added by automatic
  speech-to-text transcription, not the speaker.
- Capitalization - same reason, this is a transcription artifact.
- Spelling of words - the student spoke words, they did not spell them;
  a "misspelling" in this text is a transcription error, not something the
  candidate did wrong.
- Sentence fragments that are only "wrong" in written-text terms (natural
  spoken English is full of run-ons, trailing clauses, and restarts that
  are completely normal in speech and are NOT grammar errors).
- A short, elliptical answer that directly answers the question (e.g.
  "I believe dogs.") - this is a complete, natural spoken answer, not an
  incomplete sentence. A genuinely too-short answer is a completeness
  issue, already tracked separately via completeness_notice, never a
  language mistake here.
- The candidate briefly asking the examiner to repeat or clarify a
  question (e.g. "sorry, could you repeat that?") - this is normal,
  permitted exam behavior in Parts 1 and 3, not a fluency or coherence
  mistake.
- Contractions ("I'm", "don't", "it's") or trailing-off/ellipsis - both
  are normal, correct features of natural spoken English, not errors.
- Hedges and stance markers ("I believe", "I think", "I guess", "of
  course") - never flag one as needing replacement by a different hedge;
  that is a preference, not an error.
- Vague, informal connector phrases ("or anything", "or something", "and
  stuff") - normal casual speech, never something to correct into a more
  formal phrase just because it is vague or informal.
Only flag issues a listener would actually notice when hearing the answer
spoken aloud: wrong verb tense, subject-verb agreement, article/preposition
errors, wrong word choice, awkward word order, filler words, imprecise
vocabulary, and similar genuinely spoken-language issues.

THE "corrected" FIELD MUST ALSO BE SPOKEN-STYLE, NOT WRITTEN-STYLE
(CRITICAL): whatever you put in "corrected" models how the candidate
should have SAID it, not how it should be typed. Never introduce hyphens
to join compound modifiers (e.g. do not "correct" a spoken phrase into
"out-of-the-box plots" - hyphenation is a written typesetting convention
that has no spoken equivalent; "out of the box plots" is exactly how it
would be said). Never add quotation marks, em dashes, semicolons, or any
other punctuation a speaker cannot audibly produce. If the only thing
your "corrected" version changes versus the original is adding this kind
of written formatting, do not include the item at all.

CONFIDENCE BAR (CRITICAL - a wrong "correction" is worse than a missed one):
Before flagging anything, ask: would a Band 9 native speaker genuinely
avoid this, or does it just differ from how YOU would phrase it? Only flag
genuine errors, never a stylistic preference. Specific traps to avoid:
- Restrictive "which" (e.g. "the activities which I enjoy") is standard,
  correct British English - IELTS uses British conventions. Do not
  "correct" it to "that".
- Fixed idioms have fixed grammar that can look "wrong" out of context -
  e.g. "as far as X is concerned" always takes "is", even when X is
  plural or an "or"-joined compound (standard English agreement rule:
  with "or", the verb agrees with the nearer noun). Do not "correct" this
  to "are concerned" - that would be actively teaching wrong grammar, not
  a helpful fix.
- A word being regionally more common (e.g. "telecast", "shortage of
  time") does not make it incorrect - only flag a word if it is genuinely
  wrong or unclear, not because a different word is more common elsewhere.
- Mixing past and present tense is NOT automatically an error: describing
  a one-time past action in past tense ("I recently watched...") while
  describing the enduring/general nature or purpose of what was watched
  in present tense ("it's a reality show", "the show's aim is to test...")
  is standard, correct English - the same pattern as "I read a great book
  last week. It's about a detective in Paris." Only flag a tense issue
  when the SAME event or fact is inconsistently described (e.g. switching
  tense mid-sentence about one specific past occurrence), never for this
  kind of correct past-event/general-truth split.
- A word that is already correct and clear (e.g. "different", "good") is
  NOT a vocabulary error just because a fancier synonym exists (e.g.
  "distinct", "excellent") - only flag a word choice if it is actually
  imprecise, unclear, or wrong for the context, never merely to suggest a
  more advanced-sounding alternative to something that already works.
- Never remove or change a word in a way that alters the candidate's
  actual meaning (e.g. dropping "only" from "I only enjoy X" changes what
  they said) - a correction must preserve their intended meaning exactly.
- A natural, correctly-used idiom or informal expression (e.g. "chalk and
  cheese", "out of the box") is POSITIVE evidence for Lexical Resource,
  not an error - the descriptors name "less common and idiomatic items"
  as a real gate for Band 7+. Never "correct" one into a plainer/more
  formal phrase (e.g. "chalk and cheese" -> "very different"). Only flag
  an idiom if it is genuinely misused or wrong-register for the context,
  never merely because a plainer alternative exists.
- Self-correcting or restarting mid-sentence (e.g. "I go there every
  day... well, actually not every day, maybe three or four times a week")
  is normal, expected speech - the descriptors explicitly allow it even at
  Band 9 ("only rare repetition or self-correction"). Never flag a single
  instance of this as a fluency mistake; it demonstrates self-monitoring,
  not a language gap.
- If the candidate visibly talks around a word they don't know instead of
  using it (e.g. saying "keep putting things off" instead of
  "procrastinate"), that is a genuine communication STRATEGY the
  descriptors credit, not a vocabulary weakness - never flag it as
  "should have used a more advanced word". Only flag vocabulary as an
  issue when the word actually chosen is imprecise or wrong, not because
  a harder word could theoretically have been used instead.
If you are not fully confident something is a genuine error, do not
include it - an empty or shorter mistakes array is always better than a
confident-sounding but wrong correction.

DO NOT OVER-APPLY THIS AND GO SILENT (CRITICAL): the confidence bar above
exists to stop you from INVENTING a correction or flagging something that
isn't actually wrong - it is NOT a reason to skip a genuine, real,
correctly-identified minor issue just because it's small. A dropped
article, a slightly informal phrase, one awkward word order - if it is
GENUINELY there in the answer, report it and tag it "minor" (per STEP 3
below); do not silently return an empty mistakes array just because
nothing you found felt serious enough to mention. The candidate benefits
from seeing real minor issues to polish, precisely BECAUSE tagging them
"minor" already guarantees they won't be penalized for it - there is no
reason to hide something real out of caution. Only leave "mistakes" empty
when the answer genuinely has nothing worth pointing out at all.

STEP 3 - CLASSIFY SEVERITY (CRITICAL):
Per the official band descriptors, NOT every issue you flag actually caps
the candidate's band. Band 9 itself explicitly allows "mistakes
characteristic of native speaker speech" - real speakers make small slips
without it affecting fluent, accurate speech. Judge severity by FREQUENCY
+ SYSTEMATICITY + COMMUNICATION IMPACT - never by counting errors or
applying any fixed deduction. For EACH mistake, set "severity" to exactly
one of:
- "minor": a one-off slip that does not repeat elsewhere in the answer,
  does not impede understanding, and is the kind of small imperfection
  natural even in strong speech (e.g. a single dropped article, one
  informal filler, one slightly awkward phrasing used only once). Minor
  issues are worth mentioning so the candidate can polish further, but
  they do NOT reflect a gap in the candidate's actual range or control,
  and do NOT justify a lower band.
- "significant": a recurring or systematic issue (the same type of error
  happens more than once), an error that actually impedes understanding or
  requires the listener to reinterpret what was meant, or one that reflects
  a genuine gap in the candidate's grammatical/lexical range rather than a
  slip. These are the issues that genuinely justify a lower band.
Do not default everything to "significant" to seem thorough, and do not
default everything to "minor" to seem lenient - classify each on its own
merits using the definitions above.

RESPONSE FORMAT (STRICT JSON, NO MARKDOWN):
- Return ONLY a valid JSON object (no prose, no code fences).
- "mistakes": an array of 0-6 REAL issues covering fluency, grammar,
  vocabulary, and pronunciation where relevant. Only include something
  here if it is a genuine issue you can point to in the answer - do NOT
  invent a "correction" for a category just to have something to say
  about it. A phrase that is already natural and correct (e.g. "daily
  trivia") must NOT be "corrected" into something else just to fill a
  quota - if a category has no real issue, leave it out of the array
  entirely. It is completely fine, and expected for a strong answer, to
  return fewer items or even an empty array. Each
  object must have: "type" (fluency|grammar|vocabulary|pronunciation),
  "original" (exact text from the student's answer), "corrected" (the
  improved version), "explanation" (why it matters, 1 sentence),
  "severity" ("minor" or "significant", per STEP 3 above). Use
  exact candidate wording in "original"; keep spacing/punctuation. For
  pronunciation, "original" should be the word/phrase likely
  mispronounced based on word choice and sentence complexity, and
  "corrected" should be brief stress/rhythm guidance.
- "completeness_notice": string, per STEP 1 above.

Return ONLY this structure:
{{
  "mistakes": [
    {{"type": "grammar", "original": "", "corrected": "", "explanation": "", "severity": "significant"}},
    {{"type": "vocabulary", "original": "", "corrected": "", "explanation": "", "severity": "minor"}}
  ],
  "completeness_notice": ""
}}
"""
    try:
        logging.warning(f"[QUESTION_MISTAKES] question={question[:50]} answer={answer[:50]}")
        response = safe_gpt_call(
            prompt,
            fallback=None,
            caller=lambda p: call_gpt(p, usage_log=usage_log),
        )
        parsed = None
        if isinstance(response, dict):
            parsed = response
        elif isinstance(response, str):
            clean = response.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

        if isinstance(parsed, dict):
            mistakes = _validate_question_mistakes(parsed.get("mistakes"), transcript=answer)
            notice = str(parsed.get("completeness_notice", "") or "").strip()
            # Deterministic backstop: if the question itself doesn't
            # actually ask for a reason/extra detail/multiple parts, force
            # the notice to empty regardless of what GPT returned - see
            # _question_expects_additional_content for why this can't be
            # left to the prompt instruction alone.
            if notice and not _question_expects_additional_content(question):
                notice = ""
            # A successfully parsed response is authoritative even if it
            # found zero genuine issues - a strong answer can legitimately
            # have no real mistakes, and silently swapping that in for the
            # generic QUESTION_MISTAKES_FALLBACK below would fabricate
            # criticism (e.g. "reduce hesitation") that wasn't actually
            # true of this specific answer. Only the fallback path (GPT
            # call failed, or returned something unparseable) should use
            # the generic placeholder.
            return {"mistakes": mistakes, "completeness_notice": notice}
    except Exception as e:
        logging.error(f"[QUESTION_MISTAKES FAIL] {e}")

    return {"mistakes": QUESTION_MISTAKES_FALLBACK, "completeness_notice": ""}


_RELEVANCE_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "am", "i", "you", "he", "she", "it", "we", "they",
    "of", "to", "in", "on", "at", "as", "by", "for", "with", "and", "or", "but", "so", "because",
    "that", "this", "these", "those", "my", "your", "our", "their", "his", "her", "its",
    "do", "does", "did", "have", "has", "had", "be", "been", "being", "will", "would", "can", "could",
    "think", "really", "very", "just", "like", "about", "also", "there", "which", "what", "how",
    "when", "where", "why", "who", "if", "not", "no", "yes",
    # Generic quantifiers/fillers that overlap across unrelated topics without
    # signalling real topical connection (e.g. "most" in "most universities"
    # vs "most popular" is not evidence the answer is on-topic).
    "most", "some", "many", "much", "more", "less", "other", "every", "all",
    "any", "each", "one", "two", "three", "get", "got", "going", "go",
    "something", "anything", "everything", "usually", "often", "sometimes",
    "typically", "generally", "quite", "pretty", "type", "types", "kind",
}


def _content_words(text: str) -> set:
    words = str(text or "").lower().split()
    return {
        w.strip(".,!?;:\"'()") for w in words
        if w.strip(".,!?;:\"'()") and w.strip(".,!?;:\"'()") not in _RELEVANCE_STOPWORDS
    }


def _heuristic_off_topic(qas_clean: list) -> bool:
    """Backstop: flags a part as off-topic when at most one in four answers
    shares any meaningful content word with its question. Acts as a floor
    under the GPT self-reported topic_relevance so an off-topic submission
    can't slip through as "on_topic" (e.g. via one incidental shared word)
    and keep an inflated band."""
    if not qas_clean:
        return False
    scored = 0
    on_topic = 0
    for qa in qas_clean:
        q_words = _content_words(qa.get("question", ""))
        a_words = _content_words(qa.get("user_answer", ""))
        if not q_words or not a_words:
            continue
        scored += 1
        if q_words & a_words:
            on_topic += 1
    if scored == 0:
        return False
    return (on_topic / scored) <= 0.25


_SUBORDINATE_MARKERS = re.compile(
    r"\b(because|although|though|which|that|while|since|unless|whereas|"
    r"however|moreover|therefore|so that|in order to|if)\b",
    re.IGNORECASE,
)


def _estimate_linguistic_floor(qas_clean: list) -> float:
    """Even when an answer is off-topic, if the language actually used is
    clearly substantial and complex (long, multi-clause, varied
    vocabulary), fluency/lexical/grammar must not be allowed to crash to
    Band 1-2 - those bands specifically require "no rateable language" /
    "totally incoherent" speech, which a lengthy, grammatically complex
    response plainly is not, regardless of whether it's on-topic.

    This exists because the prompt-level instruction telling GPT not to
    over-penalize below the topic-relevance cap was NOT reliably followed
    in testing (observed real cases scoring 1/1/1 on fluent, complex,
    off-topic answers) - same lesson as topic-relevance detection itself:
    an instruction alone isn't enough, it needs a code-level backstop.

    Returns a WHOLE-NUMBER floor from 1.0 (trivial/no real answer) up to
    5.0 (clearly substantial, complex language). This is a MINIMUM, not a
    target - the model's own score is still used whenever it's already at
    or above this. Must be a whole number: fluency/lexical/grammar are
    whole-band-only values (per the conjunctive checklist in
    generate_scores()), and this floor gets assigned directly into those
    fields - a continuous float here (this used to return e.g. round(x, 1),
    producing values like 3.1) would inject an invalid, non-band value
    into an otherwise whole-number system. Rounds to the nearest whole
    number (half-up, not Python's banker's-rounding default - same reason
    as _ielts_round_half_up elsewhere in this file) rather than always
    rounding up, so a genuinely trivial answer still floors near 1.0
    rather than every off-topic answer being bumped an extra full band.
    """
    combined = " ".join(
        str(qa.get("user_answer", "")) for qa in (qas_clean or []) if qa.get("user_answer")
    ).strip()
    if not combined:
        return 1.0

    words = combined.split()
    word_count = len(words)
    if word_count == 0:
        return 1.0

    sentences = [s for s in re.split(r"[.!?]+", combined) if s.strip()]
    sentence_count = max(1, len(sentences))
    avg_sentence_len = word_count / sentence_count
    subordinate_hits = len(_SUBORDINATE_MARKERS.findall(combined))
    unique_words = {w.lower().strip(".,!?;:\"'()") for w in words}
    diversity = len(unique_words) / word_count

    length_signal = min(1.0, word_count / 150)
    complexity_signal = min(1.0, avg_sentence_len / 20)
    subordinate_signal = min(1.0, subordinate_hits / 6)
    # Vocabulary diversity (unique/total) is only a meaningful signal once
    # there are enough words to make repetition possible - a 3-word answer
    # is trivially 100% "unique" and would otherwise look as diverse as a
    # genuinely rich long answer. Scale down the diversity signal for short
    # answers so trivial responses don't get credited with false diversity.
    diversity_confidence = min(1.0, word_count / 20)
    diversity_signal = min(1.0, diversity / 0.55) * diversity_confidence

    substantiality = (
        length_signal * 0.35
        + complexity_signal * 0.25
        + subordinate_signal * 0.25
        + diversity_signal * 0.15
    )

    return float(math.floor(1.0 + substantiality * 3.5 + 0.5))


def _aggregate_acoustic_pronunciation(raw_part_results: list) -> float | None:
    """Average the real acoustic pronunciation_score (from librosa features via
    compute_pronunciation_score) across every audio clip in a part. Returns
    None when no acoustic data is available (e.g. all clips failed to
    transcribe), so callers can fall back to text-only scoring."""
    scores = []
    for r in raw_part_results or []:
        audio_metrics = r.get("audio_metrics") or {}
        val = audio_metrics.get("pronunciation_score")
        if isinstance(val, (int, float)):
            scores.append(float(val))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _aggregate_speech_timing(raw_part_results: list) -> dict | None:
    """Average speech rate and sum duration across every audio clip in a
    part, so generate_scores() can weigh real pacing evidence (too slow /
    too fast) instead of having zero access to timing data - mirrors
    _aggregate_acoustic_pronunciation() above. Returns None when no timing
    data is available at all.

    Both raw-duration and voiced-duration aggregates are always computed
    (avg_wpm_raw/avg_wpm_voiced, total_duration_sec/total_voiced_duration_
    sec, silence_fraction) regardless of SPEAKING_VOICED_WPM - "avg_wpm" is
    the single field generate_scores() actually reads as the active pacing
    number, and which underlying aggregate it points to is the only thing
    the flag controls; wpm_basis tells generate_scores() which one it's
    looking at so it can select the right prompt guidance (see below)."""
    raw_wpm_values = []
    voiced_wpm_values = []
    total_duration = 0.0
    total_voiced_duration = 0.0
    for r in raw_part_results or []:
        audio_metrics = r.get("audio_metrics") or {}
        raw_wpm = audio_metrics.get("speech_rate_wpm_raw")
        if isinstance(raw_wpm, (int, float)) and raw_wpm > 0:
            raw_wpm_values.append(float(raw_wpm))
        voiced_wpm = audio_metrics.get("speech_rate_wpm_voiced")
        if isinstance(voiced_wpm, (int, float)) and voiced_wpm > 0:
            voiced_wpm_values.append(float(voiced_wpm))
        duration = audio_metrics.get("duration_sec")
        if isinstance(duration, (int, float)) and duration > 0:
            total_duration += float(duration)
        voiced_duration = audio_metrics.get("voiced_duration_sec")
        if isinstance(voiced_duration, (int, float)) and voiced_duration > 0:
            total_voiced_duration += float(voiced_duration)
    if not raw_wpm_values and not voiced_wpm_values and total_duration <= 0:
        return None
    avg_wpm_raw = (sum(raw_wpm_values) / len(raw_wpm_values)) if raw_wpm_values else None
    avg_wpm_voiced = (sum(voiced_wpm_values) / len(voiced_wpm_values)) if voiced_wpm_values else None
    active_wpm = avg_wpm_voiced if (SPEAKING_VOICED_WPM and avg_wpm_voiced) else avg_wpm_raw
    return {
        "avg_wpm": active_wpm,
        "wpm_basis": "voiced" if (SPEAKING_VOICED_WPM and avg_wpm_voiced) else "raw",
        "avg_wpm_raw": avg_wpm_raw,
        "avg_wpm_voiced": avg_wpm_voiced,
        "total_duration_sec": round(total_duration, 1) if total_duration > 0 else None,
        "total_voiced_duration_sec": round(total_voiced_duration, 1) if total_voiced_duration > 0 else None,
        "silence_fraction": (
            round(1 - (total_voiced_duration / total_duration), 4) if total_duration > 0 else None
        ),
    }


def generate_scores(
    part_number: int,
    combined_transcripts: str,
    qas_clean: list | None = None,
    acoustic_pronunciation: float | None = None,
    speech_timing: dict | None = None,
    usage_log=None
) -> dict:
    pacing_section = ""
    if speech_timing:
        avg_wpm = speech_timing.get("avg_wpm")
        wpm_basis = speech_timing.get("wpm_basis", "raw")
        # "Measured total spoken duration" intentionally stays tied to raw
        # clip duration regardless of wpm_basis - it's answering "did the
        # candidate fill their allotted time" (a clock-time budget
        # question, e.g. Part 2's ~60s+ expectation), not a pacing-rate
        # question, so it shouldn't move with the WPM denominator fix.
        total_duration = speech_timing.get("total_duration_sec")
        pacing_lines = []
        if avg_wpm:
            pacing_lines.append(f"- Measured average speech rate: {avg_wpm:.0f} words per minute.")
        if total_duration:
            pacing_lines.append(f"- Measured total spoken duration for this part: {total_duration:.0f} seconds.")
        if pacing_lines:
            if wpm_basis == "voiced":
                # SPEAKING_VOICED_WPM path: the old "under 100 / over 200"
                # numbers below were calibrated against raw-clip-duration
                # WPM, which runs systematically LOWER than voiced-only WPM
                # (silence inflates the raw denominator). Reusing those same
                # numbers here would silently mean something different than
                # what they were written for - and there is no real
                # candidate data yet to derive correct voiced-WPM numbers
                # from (see the no-persistence finding this session
                # surfaced). Rather than guess a replacement threshold,
                # this drops the numeric anchors entirely for this path and
                # asks for a qualitative judgment instead, until the eval
                # log has enough real voiced-WPM data to recalibrate them
                # properly.
                pacing_section = "\n\nSPEECH PACING EVIDENCE (real measured data, weigh alongside the transcript):\n" + "\n".join(pacing_lines) + """
- This measured rate excludes silence/pauses (voiced speaking time only) -
  it is NOT directly comparable to generic "words per minute" benchmarks
  you may know, which are usually based on total clock time including
  pauses. Do not apply a specific numeric cutoff to it.
- This is EVIDENCE to consider, not a separate score or override - use it to
  inform Fluency and Coherence, not as a rule you mechanically apply.
- Judge the rate qualitatively from the transcript and this number together:
  a rate that feels unusually slow even for someone speaking without long
  pauses is consistent with the effortful, hesitant pattern described at
  Band 5 and below. A rate that feels unusually fast can indicate rhythm/
  clarity breakdown rather than genuine fluency - do not reward raw speed by
  itself.
- For Part 2 specifically: a full long turn is normally expected to run
  roughly 60 seconds or more. A markedly shorter duration limits how much
  development, structure, and range the candidate had room to demonstrate -
  factor this into the Answer Completeness assessment below (this is about
  what evidence exists to judge, not a separate penalty)."""
            else:
                pacing_section = "\n\nSPEECH PACING EVIDENCE (real measured data, weigh alongside the transcript):\n" + "\n".join(pacing_lines) + """
- This is EVIDENCE to consider, not a separate score or override - use it to
  inform Fluency and Coherence, not as a rule you mechanically apply.
- A clearly slow rate (roughly under 100 WPM) sustained across the answer is
  consistent with the effortful, hesitant pattern described at Band 5 and
  below - weigh it alongside what the transcript itself shows.
- A clearly fast rate (roughly over 200 WPM) can indicate rhythm/clarity
  breakdown rather than genuine fluency - do not reward raw speed by itself.
- For Part 2 specifically: a full long turn is normally expected to run
  roughly 60 seconds or more. A markedly shorter duration limits how much
  development, structure, and range the candidate had room to demonstrate -
  factor this into the Answer Completeness assessment below (this is about
  what evidence exists to judge, not a separate penalty)."""

    prompt = f"""You are a certified IELTS Speaking examiner.

A student gave these answers in Part {part_number}. Each "Q:" is the
question asked and each "A:" is the student's answer to it:

{combined_transcripts}{pacing_section}

Score this student on THREE of the official IELTS criteria - Fluency and
Coherence, Lexical Resource, and Grammatical Range and Accuracy - using the
checklists below. Do NOT score Pronunciation; that is assessed separately
from real audio evidence, not from this text.

HOW SCORING WORKS HERE (CRITICAL - read before scoring):
Per the official descriptor sheet's own note: "A candidate must fully fit
the positive features of the descriptor at a particular level." This is a
conjunctive (AND) rule, not an average or overall impression. For each
band listed below, you must judge whether EVERY feature listed for that
band is met - true only if ALL of them hold, false if even one is missing.
Do not average or "round up" - a candidate meeting 2 of band 7's 3 features
is NOT a 7, they are whatever the highest band is where ALL features hold.
Judge every band from 9 down to 1 independently and honestly - do not
assume higher bands are false just because a lower one is true, and do not
default to the middle of the range. Bands 8-9 should be marked true when
genuinely earned, and bands 1-3 should be marked true when the evidence is
that poor - do not hedge toward the middle out of caution.

FLUENCY AND COHERENCE - assessment guidance (use this to judge each
feature below accurately):
- Hesitation: is it content-related (thinking about ideas) or
  language-related (searching for words/grammar)? Language-related
  hesitation scores lower.
- Self-correction: rare and minor (high band) vs frequent and disruptive (low band).
- Repetition as a stalling tactic: repeating a word or restarting a sentence
  to buy thinking time is a lower-band signal, distinct from repetition used
  as a genuine (if unsophisticated) cohesive device.
- Pause location: a pause between sentences/ideas is far less costly than a
  pause mid-sentence or mid-phrase, which more strongly signals
  language-access difficulty rather than normal thinking time.
- Discourse markers / connectives: range and appropriacy of words like
  "however", "moreover", "in that case" - used flexibly and correctly
  (high band) vs repetitive, absent, or misused (low band). Note both
  failure modes: relying only on "and/but/because/so" caps around Band 4-5,
  but mechanically overusing formal connectives ("furthermore", "moreover")
  without natural need is itself a Band 6-ish sign of un-natural, rehearsed
  usage rather than genuine flexible range.
- Logical progression: do ideas build on each other coherently, or jump
  around / lose the thread?
- Willingness to produce long turns: from Band 6 up this is required -
  a one-line or minimal-effort answer to a question that invites
  development caps the score regardless of how clean the little language
  produced is, simply because there isn't enough evidence of sustained
  fluency to justify a higher band.
- Answer completeness: is the response fully developed with detail and
  examples, or does it stay superficial/underdeveloped?

FLUENCY AND COHERENCE - band checklist (fluency_bands["N"] = true only if
ALL features listed for band N hold):
- Band 9: speaks fluently with only rare repetition or self-correction AND
  any hesitation is content-related rather than to find words or grammar
  AND speaks coherently with fully appropriate cohesive features AND
  develops topics fully and appropriately.
- Band 8: fluent with only occasional repetition or self-correction AND
  hesitation is usually content-related and only rarely to search for
  language AND topic development is coherent, appropriate and relevant.
- Band 7: speaks at length without noticeable effort or loss of coherence
  AND may demonstrate language-related hesitation at times, or some
  repetition and/or self-correction AND uses a range of connectives and
  discourse markers with some flexibility.
- Band 6: is willing to speak at length, though may lose coherence at
  times due to occasional repetition, self-correction or hesitation AND
  uses a range of connectives and discourse markers but not always
  appropriately.
- Band 5: usually maintains flow of speech but uses repetition, self-
  correction and/or slow speech to keep going AND may over-use certain
  connectives and discourse markers AND produces simple speech fluently,
  but more complex communication causes fluency problems.
- Band 4: cannot respond without noticeable pauses and may speak slowly,
  with frequent repetition and self-correction AND links basic sentences
  but with repetitious use of simple connectives and some breakdowns in
  coherence.
- Band 3: speaks with long pauses AND has limited ability to link simple
  sentences AND gives only simple responses and is frequently unable to
  convey basic message.
- Band 2: pauses lengthily before most words AND little communication
  possible.
- Band 1: no communication possible AND no rateable language.

LEXICAL RESOURCE - explicitly assess ALL of these before scoring:
- Repetition ratio: how often the same basic words/phrases are reused
  instead of varying vocabulary - high repetition of simple words caps the band.
- Repeating the question's own vocabulary back verbatim instead of
  rephrasing it in the candidate's own words is a lower-range signal, even
  when the answer is otherwise coherent.
- Topical range: can the candidate discuss only familiar, personal topics
  (Band 5 territory), or do they sustain vocabulary on less familiar/more
  abstract topics too (Band 6+)? This is specifically about range across
  topic types, not just quantity of words used.
- Collocation: are word pairings natural ("make a decision") or unnatural/
  incorrect ("do a decision")?
- Idiomatic usage: any natural idiomatic language, and is it used
  correctly and appropriately (not forced)? Note: idiomatic/less-common
  vocabulary is a real gate for Band 7 - competent but entirely common,
  everyday vocabulary caps around Band 6 even if used accurately throughout.
- Paraphrasing: when a word is unknown or avoided, does the speaker
  successfully rephrase, or do they get stuck / oversimplify? This is a
  second, separate gate for Band 7 - accurate vocabulary without any
  paraphrasing ability still caps around Band 6.
- Precision: are words chosen with exact, specific meaning, or are they
  vague/generic ("thing", "stuff", "good", "nice")?
- Register/style awareness: does the vocabulary suit the context - e.g.
  noticeably slangy/casual language in an analytical Part 3 answer, or
  stiff, over-formal "book" language in a casual Part 1 answer, both signal
  weaker style awareness than vocabulary that's appropriately pitched to
  the part.

LEXICAL RESOURCE - band checklist (lexical_bands["N"] = true only if ALL
features listed for band N hold):
- Band 9: uses vocabulary with full flexibility and precision in all
  topics AND uses idiomatic language naturally and accurately.
- Band 8: uses a wide vocabulary resource readily and flexibly to convey
  precise meaning AND uses less common and idiomatic vocabulary skilfully,
  with occasional inaccuracies AND uses paraphrase effectively as required.
- Band 7: uses vocabulary resource flexibly to discuss a variety of topics
  AND uses some less common and idiomatic vocabulary and shows some
  awareness of style and collocation, with some inappropriate choices AND
  uses paraphrase effectively.
- Band 6: has a wide enough vocabulary to discuss topics at length and
  make meaning clear in spite of inappropriacies AND generally paraphrases
  successfully.
- Band 5: manages to talk about familiar and unfamiliar topics but uses
  vocabulary with limited flexibility AND attempts to use paraphrase but
  with mixed success.
- Band 4: is able to talk about familiar topics but can only convey basic
  meaning on unfamiliar topics and makes frequent errors in word choice
  AND rarely attempts paraphrase.
- Band 3: uses simple vocabulary to convey personal information AND has
  insufficient vocabulary for less familiar topics.
- Band 2: only produces isolated words or memorised utterances.
- Band 1: no communication possible AND no rateable language.

GRAMMATICAL RANGE AND ACCURACY - this has TWO separate dimensions, both
of which must be assessed together (a high score needs both range AND
accuracy, not just a low error count):
- Range: variety of structures attempted - simple sentences only, vs a mix
  of simple/complex, vs a genuinely wide range (relative clauses,
  conditionals, passive voice, varied tenses) used naturally.
- Accuracy: how often those structures are grammatically correct, and
  whether errors (when present) impede communication or are minor slips.
- A speaker who uses only simple, error-free sentences has HIGH accuracy
  but LOW range, and should NOT score as high as one who uses a wide range
  of complex structures with only occasional errors.
- Error type matters, not just error count: a systematic error repeated
  every time a structure is attempted (e.g. always getting past tense
  wrong, or always misusing an article) signals a real gap in control and
  should weigh more heavily than a single one-off slip that isn't repeated.
- Tense control deserves particular attention where the task naturally
  demands it - past tense narration in Part 2 long turns, and conditionals
  in Part 3 hypothetical/opinion discussion - since consistent errors there
  are strong evidence either way.
- Articles, prepositions, and subject-verb agreement are the most common
  persistent errors that keep an otherwise fluent, coherent candidate capped
  around Band 6 - weigh their frequency specifically, not just whether an
  error happened to occur somewhere.

GRAMMATICAL RANGE AND ACCURACY - band checklist (grammar_bands["N"] = true
only if ALL features listed for band N hold):
- Band 9: uses a full range of structures naturally and appropriately AND
  produces consistently accurate structures apart from "slips".
- Band 8: uses a wide range of structures flexibly AND produces a majority
  of error-free sentences with only very occasional inappropriacies or
  basic/non-systematic errors.
- Band 7: uses a range of complex structures with some flexibility AND
  frequently produces error-free sentences, though some grammatical
  mistakes persist.
- Band 6: uses a mix of simple and complex structures, but with limited
  flexibility AND may make frequent mistakes with complex structures,
  though these rarely cause comprehension problems.
- Band 5: produces basic sentence forms with reasonable accuracy AND uses
  a limited range of more complex structures, but these usually contain
  errors and may cause some comprehension problems.
- Band 4: produces basic sentence forms and some correct simple sentences
  but subordinate structures are rare AND errors are frequent and may lead
  to misunderstanding.
- Band 3: attempts basic sentence forms but with limited success, or
  relies on apparently memorised utterances AND makes numerous errors
  except in memorised expressions.
- Band 2: cannot produce basic sentence forms.
- Band 1: no communication possible AND no rateable language.

Part-specific notes:
- Part 1 answers are typically shorter - mark accordingly
- Part 2 long-turn answers should reward structure and development with
  higher fluency/lexical scores if demonstrated
- Part 3 answers should reward analytical depth

TOPIC RELEVANCE RULE (CRITICAL - CHECK THIS FIRST):
- Compare each "A:" against its own "Q:" and judge whether the answer
  actually addresses the subject the question asked about, not just
  whether it is fluent, grammatical speech on some other subject.
- Per the IELTS descriptors, Band 6+ Fluency and Coherence requires
  "topic development is relevant, appropriate and coherent". An answer
  that discusses a completely different topic than the question CANNOT
  reach Band 6 or higher on ANY criterion, however well-formed the
  language is, because it fails to address the task.
- Set "topic_relevance" to exactly one of:
  - "on_topic": most answers directly address their questions
  - "partially_off_topic": answers drift from the question but keep some connection
  - "completely_off_topic": most answers discuss an unrelated subject with no real connection to what was asked
- If "topic_relevance" is "completely_off_topic", every score
  (fluency, lexical, grammar) MUST be capped at 5.0 maximum.
- If "topic_relevance" is "partially_off_topic", every score MUST be
  capped at 6.0 maximum.
- CRITICAL - DO NOT OVER-PENALIZE BELOW THE CAP: the cap above is a
  MAXIMUM, not a target you should crash toward. Being off-topic affects
  Coherence (task achievement), NOT the candidate's actual demonstrated
  fluency, vocabulary range, or grammatical accuracy/range, which are
  independent linguistic skills the candidate is still showing you, just
  applied to the wrong subject. Score each criterion at whatever the
  language quality genuinely supports, THEN apply the cap as a ceiling.
  A fluent, grammatically complex, coherent answer on the wrong topic
  should land AT the cap (e.g. 5.0), not fall to Band 1-3. Bands 1-3
  require "no rateable language" / "totally incoherent" / "basic sentence
  forms attempted but numerous errors" - reserve those ONLY for responses
  that are ALSO linguistically weak on their own terms, independent of topic.

POSSIBLY MEMORISED / REHEARSED ANSWER:
- If an answer sounds noticeably rehearsed or templated - a register/style
  that doesn't match the rest of the candidate's speech, generic content
  that doesn't specifically respond to the actual question asked, or
  suspiciously polished phrasing inconsistent with the candidate's
  demonstrated level elsewhere - this is a real IELTS concern (the
  descriptors themselves note "no rateable language unless memorised" at
  the lowest bands). Judge the language as actually demonstrated, not the
  apparent rehearsed fluency, and weigh whether it genuinely responds to
  what was asked.

WHAT NOT TO SCORE (CRITICAL - these must NEVER influence any score):
- Do not judge whether an opinion is correct, sensible, or well-reasoned as
  a matter of factual/logical content - there is no "correct" opinion in
  IELTS Speaking. Only the LANGUAGE used to express it is assessed.
- Do not penalize a candidate for a topic being unfamiliar to them
  (real-world knowledge) - you are judging their language use, not their
  general knowledge, as long as they communicate SOMETHING responsive.
- Do not penalize British vs American vocabulary or spelling-adjacent word
  choices (e.g. "flat" vs "apartment", "holiday" vs "vacation") - both are
  acceptable; only flag it if usage is inconsistent/mixed in a way that
  itself signals uncertainty rather than natural bidialectal fluency.
- You are only given a text transcript here, never audio - so you have no
  way to judge accent, and must not infer or penalize one from spelling or
  word choice in the transcript.
- If the transcript shows the candidate briefly asking the examiner to
  repeat or clarify a question (e.g. "sorry, could you repeat that?"),
  this is normal, permitted exam behavior in Parts 1 and 3 - do not treat
  it as a hesitation, coherence, or fluency problem. Judge the candidate's
  actual answer once given, not the fact that they asked for clarification.

Return ONLY this JSON object, no explanation, no markdown. Each band key
below is a STRING "1" through "9", and each value is a boolean - true only
if EVERY feature listed for that band in the checklists above is met:
{{
  "fluency_bands": {{"9": false, "8": false, "7": false, "6": false, "5": false, "4": false, "3": false, "2": false, "1": true}},
  "lexical_bands": {{"9": false, "8": false, "7": false, "6": false, "5": false, "4": false, "3": false, "2": false, "1": true}},
  "grammar_bands": {{"9": false, "8": false, "7": false, "6": false, "5": false, "4": false, "3": false, "2": false, "1": true}},
  "topic_relevance": "on_topic"
}}

The example above (all false except band 1) is only illustrating the
SHAPE of the object, not a suggested answer - judge every band on its own
merits per the checklists above.
Return only the JSON object. No markdown.
No ```json fence. No explanation before or after.
Start your response with {{ and end with }}"""

    parsed = None
    try:
        logging.warning(f"[SCORES CALLED] part={part_number}")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"), timeout=OPENAI_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_OVERRIDE", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1200
        )
        record_token_usage(response, usage_log)
        result = response.choices[0].message.content.strip()
        logging.warning(f"[SCORES RESULT] {result[:100]}")
        if result:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            candidate = json.loads(clean)
            if all(k in candidate for k in ["fluency_bands", "lexical_bands", "grammar_bands"]):
                parsed = candidate
    except Exception as e:
        logging.error(f"[MISTAKES/SCORES FAIL] {e}")

    if parsed is None:
        # GPT call failed or returned something unparseable - this is a
        # system/infrastructure failure, not evidence the candidate produced
        # "no rateable language". An empty band grid would floor through
        # _highest_fully_met_band() to 1.0 for all three criteria, which
        # would wrongly crash a real candidate's score to the bottom band
        # just because of a transient API failure - use a neutral default
        # instead, same as this function's fallback did before this change.
        validated = {"fluency": 5.0, "lexical": 5.0, "grammar": 5.0}
        parsed = {"topic_relevance": "on_topic"}
    else:
        # Deterministic band selection: the highest band where GPT marked
        # EVERY feature of that band's checklist as met, per the descriptor
        # sheet's own conjunctive rule ("must fully fit the positive
        # features... at a particular level"). This is computed here in code
        # rather than trusted as a single float GPT self-reports, for the
        # same reason every other fix this session moved logic out of
        # prompt-only instructions: GPT doesn't reliably self-apply a rule
        # like this consistently on its own.
        validated = {}
        for key, bands_key in (("fluency", "fluency_bands"), ("lexical", "lexical_bands"), ("grammar", "grammar_bands")):
            validated[key] = _highest_fully_met_band(parsed.get(bands_key) or {})

    topic_relevance = str(parsed.get("topic_relevance", "on_topic")).strip().lower()
    if topic_relevance not in ("on_topic", "partially_off_topic", "completely_off_topic"):
        topic_relevance = "on_topic"

    # Code-level safety net: don't rely solely on the model obeying the cap
    # instruction above. If our own content-word overlap check finds ZERO
    # connection between any answer and its question, force off-topic status.
    if _heuristic_off_topic(qas_clean or []):
        topic_relevance = "completely_off_topic"
        logging.warning(f"[TOPIC RELEVANCE] Part {part_number}: heuristic overlap check forced completely_off_topic")

    # The topic-relevance cap applies only to fluency/lexical/grammar - these
    # are judged from what was SAID, so an off-topic answer legitimately
    # caps them (Band 6+ Fluency and Coherence requires relevant topic
    # development). Pronunciation is judged from how it SOUNDS, entirely
    # independent of content, so it is deliberately excluded from this cap -
    # a candidate can articulate clearly regardless of whether they answered
    # the right question.
    cap = {"completely_off_topic": 5.0, "partially_off_topic": 6.0}.get(topic_relevance)
    if cap is not None:
        logging.warning(f"[TOPIC RELEVANCE] Part {part_number}: {topic_relevance} -> capping scores at {cap}")
        validated = {k: min(v, cap) for k, v in validated.items()}

        # Code-level floor: the cap above is a ceiling, but GPT has been
        # observed crashing scores all the way to Band 1 for off-topic
        # answers even when the prompt explicitly says not to - despite the
        # language itself being fluent, complex, and coherent. If the
        # actual text is substantial, don't let it sit below what that
        # substance deserves, capped at the same topic-relevance ceiling.
        floor = min(cap, _estimate_linguistic_floor(qas_clean or []))
        if floor > 1.0:
            for k in ("fluency", "lexical", "grammar"):
                if validated.get(k, 0) < floor:
                    logging.warning(f"[LINGUISTIC FLOOR] Part {part_number}: {k} {validated[k]} -> {floor} (substantial language despite off-topic)")
                    validated[k] = floor

    # Pronunciation per the descriptors is fundamentally about how the
    # answer SOUNDS (stress, intonation, pausing, articulation, pacing) -
    # not about word choice or grammar. It is derived ENTIRELY from real
    # acoustic evidence, with no text-based GPT guess involved at all, and
    # is added AFTER the topic-relevance cap above so it is never affected
    # by it.
    if acoustic_pronunciation is not None:
        validated["pronunciation"] = round(max(1.0, min(9.0, acoustic_pronunciation)) * 2) / 2
        pronunciation_source = "audio_only"
    else:
        # No usable audio evidence at all (e.g. transcription/audio pipeline
        # failure upstream) - neutral default rather than a text-based guess.
        validated["pronunciation"] = 5.5
        pronunciation_source = "unavailable"

    validated["topic_relevance"] = topic_relevance
    validated["pronunciation_source"] = pronunciation_source
    return validated


def _highest_fully_met_band(band_flags: dict) -> float:
    """Deterministically pick the highest band (9 down to 1) where GPT
    marked EVERY feature of that band's checklist as true. Implements the
    descriptor sheet's own conjunctive rule ("a candidate must fully fit
    the positive features of the descriptor at a particular level") in
    code, rather than trusting GPT to self-apply that rule while also
    self-reporting a single float - the same reasoning behind every other
    deterministic backstop added this session."""
    for band in (9, 8, 7, 6, 5, 4, 3, 2, 1):
        if band_flags.get(str(band)) is True:
            return float(band)
    return 1.0


def _ielts_round_half_up(x: float) -> float:
    """Official IELTS overall-band rounding: an average ending in .25 rounds
    UP to the next half band, and .75 rounds UP to the next whole band -
    i.e. round-half-up on the doubled value. Python's built-in round() uses
    round-half-to-even, which silently rounds .25/.75 averages DOWN roughly
    half the time (e.g. round(6.25*2)/2 = 6.0, not the correct 6.5)."""
    return math.floor(x * 2 + 0.5) / 2


def _quote_occurrence_count(quote: str, transcript: str) -> int:
    """Counts every verbatim occurrence of `quote` in `transcript` (modulo
    whitespace/punctuation/case differences that don't change the actual
    words - full punctuation stripping, so quote/dash style never matters
    either). Used two ways: >0 means genuinely verbatim (not a paraphrase
    or hallucination), and the exact count also lets a caller reject an
    AMBIGUOUS quote that matches more than once - a candidate can't tell
    which occurrence a mistake refers to if the quoted span isn't unique
    in their own transcript. _quote_appears_in_transcript() below is a
    thin backward-compatible wrapper around this for its one existing
    caller (detect_systematic_errors()), which only ever needed a
    yes/no answer."""
    def _tokens(text):
        return re.sub(r"[^\w\s]", "", (text or "")).lower().split()

    quote_tokens = _tokens(quote)
    transcript_tokens = _tokens(transcript)
    if not quote_tokens:
        return 0
    span = len(quote_tokens)
    count = 0
    for i in range(len(transcript_tokens) - span + 1):
        if transcript_tokens[i:i + span] == quote_tokens:
            count += 1
    return count


def _quote_appears_in_transcript(quote: str, transcript: str) -> bool:
    """Verify a claimed quoted occurrence is genuinely verbatim from the
    transcript - the 3+-occurrence threshold in detect_systematic_errors()
    is only a real evidentiary bar if the quotes it counts are real
    quotes, not GPT's approximation of what the candidate said."""
    return _quote_occurrence_count(quote, transcript) > 0


def _pattern_key_terms(pattern: str) -> list:
    """Extract the specific recurring word/phrase a systematic-error
    pattern claims to be about, e.g. "use of 'having' in incorrect
    contexts" -> ["having"]. The prompt requires this quoting so it can be
    checked: a "pattern" is only as trustworthy as the shared mechanism it
    names, and verbatim-quote-checking alone (see
    _quote_appears_in_transcript) can't catch a real-but-unrelated quote
    being counted toward a DIFFERENT pattern's occurrence count."""
    return [q.lower().strip() for q in re.findall(r"['\"‘’]([^'\"‘’]+)['\"‘’]", pattern or "") if q.strip()]


def detect_systematic_errors(whole_test_transcript: str, usage_log=None) -> list:
    """Find error PATTERNS that recur across the WHOLE test (all 3 parts
    together), not isolated per-part instances.

    This exists because scoring runs as 3 independent per-part GPT calls,
    each seeing only ~30-40% of the total evidence - none of them can ever
    notice that the same underlying error (e.g. progressive aspect on a
    stative verb: "which are having", "we were not having", "in case you
    are having") recurs across Part 2 AND Part 3. That distinction matters
    directly: Band 8 Grammatical Range and Accuracy explicitly requires
    "only occasional inappropriacies/non-systematic errors" - a genuinely
    systematic pattern is real, descriptor-grounded evidence capping GRA
    (and, for repeated inappropriate word/collocation choices, Lexical
    Resource) at Band 7, regardless of what each isolated per-part call
    concluded on its own.
    """
    prompt = f"""You are an IELTS examiner reviewing a candidate's COMPLETE
speaking test transcript (all three parts combined) for recurring error
PATTERNS - not isolated one-off mistakes, but the same underlying error
happening multiple times across the whole test.

Full transcript (all parts):
{whole_test_transcript}

TASK: Identify error patterns that occur 3 OR MORE times across this
transcript, where each occurrence is genuinely the SAME underlying error
(the same grammatical structure misused, or the same word/phrase
overused/misused), not merely superficially similar sentences.

Examples of genuine systematic patterns (illustrative only, not the only
valid kinds):
- The same grammatical structure gets misused repeatedly, e.g. using
  progressive aspect with a stative verb multiple times ("which are
  having", "we were not having", "you are having" instead of "which
  have", "we did not have", "you have").
- The same word or phrase is used as an inappropriate filler/hedge
  repeatedly in a way that becomes a collocation error or lexical
  repetition pattern (e.g. "such kind of X" used many times instead of
  "such X" or "this kind of X").

Do NOT report something as systematic if:
- It only happens 1-2 times (that's a one-off slip, not a pattern).
- The occurrences are only superficially similar but are actually
  different errors (e.g. two different tense mistakes that happen to both
  be "wrong tense" is too vague - be specific about the actual shared
  mechanism).
- It's a stylistic preference (e.g. always using "very" as an intensifier)
  rather than a genuine error.

This is a transcript of SPOKEN language - do not flag punctuation,
capitalization, or spelling as part of any pattern; those are
transcription artifacts, not something the candidate did. This includes
patterns like "the pronoun 'I' appears in lowercase" - a speaker cannot
"say" a lowercase letter; any casing in this text was added by automatic
speech-to-text, never something to report as a recurring error.

Return ONLY this JSON object, no explanation, no markdown:
{{
  "systematic_errors": [
    {{
      "pattern": "short description of the specific recurring error mechanism, with the exact recurring word or phrase itself in single quotes, e.g. use of 'having' in incorrect contexts",
      "criterion": "grammar",
      "occurrences": ["exact quote 1 from the transcript", "exact quote 2", "exact quote 3"],
      "explanation": "why this is a genuine recurring pattern, in 1 sentence"
    }}
  ]
}}

"criterion" must be exactly "grammar" or "vocabulary". The "pattern" field
MUST name the specific recurring word or phrase in single quotes - every
occurrence you list must actually contain that word/phrase (or an
inflected form of it), not just be a different error that happens to feel
similarly wrong. If there are genuinely no patterns meeting the 3+
occurrence bar, return {{"systematic_errors": []}}. Do not invent patterns
to fill the array - an empty array is a completely valid and expected
answer for a candidate without recurring errors."""

    result = safe_gpt_call(prompt, fallback=None, caller=lambda p: call_gpt(p, usage_log=usage_log))

    parsed = result if isinstance(result, dict) else None
    if parsed is None and isinstance(result, str) and result.strip():
        try:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            parsed = None

    if not isinstance(parsed, dict):
        return []

    raw_errors = parsed.get("systematic_errors")
    if not isinstance(raw_errors, list):
        return []

    validated = []
    for item in raw_errors:
        if not isinstance(item, dict):
            continue
        criterion = str(item.get("criterion", "")).strip().lower()
        if criterion not in ("grammar", "vocabulary"):
            continue
        occurrences = item.get("occurrences")
        if not isinstance(occurrences, list):
            continue
        pattern = str(item.get("pattern", "")).strip()
        if not pattern:
            continue
        explanation_text = str(item.get("explanation", "")).strip()
        # This is a transcript of SPOKEN language - the prompt already says
        # not to flag punctuation/capitalization/spelling as a pattern, but
        # a real case slipped through anyway ("use of 'i' in lowercase"
        # flagged as a recurring VOCABULARY pattern, capping Lexical
        # Resource on a pure transcription artifact from automatic
        # speech-to-text, not anything the candidate did). Check both
        # "pattern" and "explanation" text, since either can carry the
        # disqualifying signal.
        pattern_and_explanation = f"{pattern} {explanation_text}".lower()
        if any(kw in pattern_and_explanation for kw in _PUNCTUATION_EXPLANATION_KEYWORDS):
            continue
        # Deterministic backstop: don't trust a self-reported count/pattern
        # claim on faith - require at least 3 occurrences that are actually
        # verbatim (modulo punctuation/case) substrings of the real
        # transcript, matching the 3+ threshold the prompt was given. GPT
        # was asked to quote "exact quote[s]" but sometimes paraphrases a
        # near-miss instead (e.g. swapping in a different noun from a
        # nearby clause) - counting those toward the 3+ bar would let a
        # pattern claim ride on partly-fabricated evidence.
        occurrences = [str(o).strip() for o in occurrences if str(o).strip()]
        occurrences = [o for o in occurrences if _quote_appears_in_transcript(o, whole_test_transcript)]
        # Being a real quote isn't enough on its own - a real quote can
        # still be a DIFFERENT error miscounted toward this pattern (a real
        # observed case: a "having" misuse pattern included "they're not
        # getting gaining", a genuine transcript quote but a completely
        # unrelated double-verb error with no "having" in it at all). The
        # prompt now requires "pattern" to name the specific recurring
        # word/phrase in quotes - require every surviving occurrence to
        # actually contain it (or leave the check off if the model didn't
        # quote a key term, rather than reject everything on a formatting
        # miss).
        key_terms = _pattern_key_terms(pattern)
        if key_terms:
            occurrences = [
                o for o in occurrences
                if any(term in o.lower() for term in key_terms)
            ]
        if len(occurrences) < 3:
            continue
        validated.append({
            "pattern": pattern,
            "criterion": criterion,
            "occurrences": occurrences,
            "explanation": str(item.get("explanation", "")).strip(),
        })
    return validated


def detect_answer_alignment_issues(part_1_qas: list, part_2_qas: list, part_3_qas: list) -> list:
    """Diagnostic-only check for a bug class that lives outside this
    backend: the audio-recording client can attach the wrong question's
    label to a clip (e.g. a UI race where the displayed/stored question
    advances before the previous answer's recording is finalized), so the
    text this backend receives as "answer to question N" is actually what
    the candidate said in response to a different question. This backend
    has no way to detect that from the audio alone - it only ever sees
    already-labeled (audio, question) pairs, and scoring, feedback, and
    mistakes are all generated per that label - but when an answer's
    content clearly, obviously matches a DIFFERENT question in the SAME
    part, that's a strong same-part signal worth surfacing to whoever
    reviews the report. This never rewrites the question/answer pairing or
    touches scoring - there is no ground truth for the TRUE pairing here,
    only a same-part candidate to point at.
    """
    parts = {1: part_1_qas or [], 2: part_2_qas or [], 3: part_3_qas or []}
    entries_by_part = {}
    sections = []
    for part_no, qas in parts.items():
        entries = [qa for qa in qas if qa.get("question") or qa.get("user_answer")]
        entries_by_part[part_no] = entries
        if len(entries) < 2:
            continue
        lines = [f"Part {part_no}:"]
        for idx, qa in enumerate(entries):
            lines.append(f"  [{idx}] Question: {qa.get('question', '')}")
            lines.append(f"      Answer: {qa.get('user_answer', '')}")
        sections.append("\n".join(lines))

    if not sections:
        return []

    prompt = f"""You are reviewing an IELTS speaking test transcript for a
DATA QUALITY issue, not a language issue: each answer below is labeled
with the question it was recorded in response to, but a bug in the
recording app can sometimes attach the WRONG question's label to an
answer's audio clip.

Two same-part questions are often about ADJACENT topics on purpose (e.g.
"what makes a show popular" and "differences between what young and old
people watch" both concern TV shows). An answer that merely happens to
touch on a related topic is NOT a mismatch - most answers you see will be
correctly labeled, including ones that share vocabulary or themes with a
neighboring question.

For EACH answer, apply this two-step test before ever flagging it:
STEP 1 - Does this answer directly and reasonably respond to its OWN
labeled question, even briefly or imperfectly? If yes, STOP - it is
correctly labeled. Do NOT flag it, no matter how much its content also
overlaps with another question's topic.
STEP 2 - Only if the answer CLEARLY FAILS to address its own labeled
question at all (e.g. it's a yes/no answer to a question that isn't a
yes/no question, or it directly answers a different question's specific
ask instead), check whether it instead directly answers a different
question in the SAME part. Only flag it if swapping the two answers would
make BOTH of them clearly better, more direct answers to their new
(swapped) question than they currently are to their own labeled question.

Never compare across different parts - Part 1/2/3 intentionally cover
different topics and formats.

{chr(10).join(sections)}

Return ONLY this JSON object, no explanation, no markdown:
{{
  "alignment_warnings": [
    {{"part": 3, "question_index": 1, "likely_matches_question_index": 0, "fails_own_question": true, "reason": "short reason citing specifically what the labeled question asks for that this answer never addresses, and what the other question asks for that it does address"}}
  ]
}}

"fails_own_question" must be true for every item - if an answer does
reasonably address its own question, do not include it at all. If every
answer plausibly matches its own labeled question, return
{{"alignment_warnings": []}} - this is the expected, normal result for
most tests. Do not invent mismatches to fill the array."""

    # gpt-4o-mini (call_gpt's model) was tested on real data and repeatedly,
    # confidently mis-judged the exact same case even after the prompt was
    # tightened with an explicit boolean commitment - this specific
    # same-part semantic comparison needs a more capable model.
    result = safe_gpt_call(prompt, fallback=None, caller=call_gpt_strong)

    parsed = result if isinstance(result, dict) else None
    if parsed is None and isinstance(result, str) and result.strip():
        try:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            parsed = None

    if not isinstance(parsed, dict):
        return []

    raw_warnings = parsed.get("alignment_warnings")
    if not isinstance(raw_warnings, list):
        return []

    validated = []
    for item in raw_warnings:
        if not isinstance(item, dict):
            continue
        try:
            part_no = int(item.get("part"))
            q_idx = int(item.get("question_index"))
            match_idx = int(item.get("likely_matches_question_index"))
        except (TypeError, ValueError):
            continue
        if part_no not in entries_by_part:
            continue
        entries = entries_by_part[part_no]
        if not (0 <= q_idx < len(entries)) or not (0 <= match_idx < len(entries)) or q_idx == match_idx:
            continue
        # Deterministic backstop: a soft "be conservative" instruction alone
        # was not reliable in testing - a real case flagged an answer that
        # fully, directly addressed its own labeled question just because it
        # was topically adjacent to another question. Require the model to
        # have explicitly committed to the stronger claim (its own question
        # was NOT addressed at all) rather than accepting free-form prose
        # that could rest on topical proximity alone.
        if item.get("fails_own_question") is not True:
            continue
        reason = str(item.get("reason", "")).strip()
        if not reason:
            continue
        validated.append({
            "part": part_no,
            "question": entries[q_idx].get("question", ""),
            "labeled_answer_preview": (entries[q_idx].get("user_answer", "") or "")[:160],
            "likely_actual_question": entries[match_idx].get("question", ""),
            "note": (
                "This answer appears to respond to a different question in this part. "
                "Possible audio/question mislabeling during recording - review recommended."
            ),
            "reason": reason,
        })
    return validated


def calculate_overall_band(
    part1_scores: dict,
    part2_scores: dict,
    part3_scores: dict
) -> float:
    """Combine the three parts' criteria averages into one overall band.

    Per the descriptor sheet's own note (ii): "A candidate will be rated
    on their average performance across all parts of the test." Parts are
    therefore weighted equally - not tiered by part, which was a previous
    guess not actually supported by the descriptor sheet.
    """

    def _part_avg(scores):
        if not scores:
            return None
        return (
            scores.get("fluency", 5.0)
            + scores.get("lexical", 5.0)
            + scores.get("grammar", 5.0)
            + scores.get("pronunciation", 5.0)
        ) / 4

    weighted_parts = [
        (_part_avg(part1_scores), 1.0),
        (_part_avg(part2_scores), 1.0),
        (_part_avg(part3_scores), 1.0),
    ]
    present = [(avg, weight) for avg, weight in weighted_parts if avg is not None]

    if not present:
        # Nothing was submitted for any part at all - per the descriptor's
        # own Band 0 definition ("Does not attend / Does not complete the
        # test"), this must be 0, not a generous middle-of-the-road guess.
        return 0.0

    total_weight = sum(weight for _, weight in present)
    weighted_avg = sum(avg * weight for avg, weight in present) / total_weight

    return _ielts_round_half_up(weighted_avg)












_SPELLCHECKER = None
_SPELLCHECKER_LOCK = threading.Lock()


def _get_spellchecker():
    global _SPELLCHECKER
    if _SPELLCHECKER is None:
        with _SPELLCHECKER_LOCK:
            if _SPELLCHECKER is None:
                from spellchecker import SpellChecker
                _SPELLCHECKER = SpellChecker()
    return _SPELLCHECKER


def _filter_misspelled_vocab(vocab_list: list) -> list:
    """Deterministically catch and drop spelling hallucinations in
    GPT-generated vocabulary words (e.g. "envigorate" instead of
    "invigorate") - a real English dictionary check, not tied to any
    specific word list or topic, so this applies to every test, not just
    the one that first exposed the bug.

    Deliberately DROPS unrecognized words rather than auto-replacing them
    with the dictionary's "closest" suggestion: tested against short/
    modern terms that could plausibly come up across different topics
    (wifi, covid, ielts), the suggested corrections were themselves wrong
    and misleading ("wifi" -> "wife", "ielts" -> "belts", "covid" ->
    "bovid") - silently teaching a wrong word would be worse than showing
    one fewer vocabulary item. The existing "< 3 items -> fallback" logic
    in generate_vocabulary() already covers the case where too many get
    filtered out here."""
    try:
        spell = _get_spellchecker()
    except Exception:
        # If the spellchecker fails to load for any reason, don't let that
        # break vocabulary generation entirely - just skip the check.
        return vocab_list

    kept = []
    for item in vocab_list:
        if not isinstance(item, dict) or not item.get("word"):
            continue
        word = str(item["word"])
        tokens = [t for t in re.split(r"[ \-]", word) if t.isalpha()]
        if not tokens or all(t.lower() in spell for t in tokens):
            kept.append(item)
        else:
            logging.warning(f"[VOCAB SPELLING] Dropped unrecognized word: '{word}'")
    return kept


def generate_vocabulary(part: int, combined_transcripts: str, usage_log=None) -> list:
    fallbacks = {
        1: VOCAB_FALLBACK_PART1,
        2: VOCAB_FALLBACK_PART2,
        3: VOCAB_FALLBACK_PART3
    }

    part_focus = {
        1: "everyday conversational English, personal topics, daily life vocabulary",
        2: "narrative language, descriptive adjectives, linking/connector words",
        3: "abstract nouns, academic verbs, analytical and argumentative vocabulary"
    }

    part_number = part
    prompt = f"""You are an IELTS vocabulary coach.

A student answered IELTS Speaking Part {part_number}
questions on this topic:

{combined_transcripts}

Generate exactly 8 vocabulary words following these rules:

RULE 1 — TOPIC SPECIFIC:
Every word must be directly relevant to what the student
talked about. If they talked about Diwali → suggest words
like illuminate, revelry, commemorate. If they talked
about transport → suggest words like commute, congestion,
infrastructure. If they talked about economics → suggest
words like revenue, expenditure, stimulate.
Never suggest generic words like "community", "engage",
"diverse" unless the topic is specifically about those.

RULE 2 — PART SPECIFIC:
- Part 1 (personal questions): suggest conversational
  but precise words the student can use in daily speech
  Examples: punctual, efficient, affordable, accessible
- Part 2 (long turn/cue card): suggest descriptive and
  narrative words that help develop a story or description
  Examples: vibrant, commemorate, illuminate, indulge,
  elaborate, reminisce
- Part 3 (abstract discussion): suggest analytical and
  academic words for arguing and discussing ideas
  Examples: socioeconomic, facilitate, implications,
  inclusive, stimulate, alleviate, detrimental,
  foster, substantiate, inevitable

RULE 3 — NOT ALREADY USED WELL:
Do not suggest words the student already used correctly
in their answer. Only suggest words that would improve
or replace weak words they actually used.

RULE 4 — USEFUL FOR IELTS SPEAKING:
Every word must be something a student can realistically
use in a speaking exam. No overly rare or obscure words.

RULE 5 — CORRECT SPELLING (CRITICAL):
Every word MUST be a real, standard, correctly-spelled English word.
Double-check the spelling of each word before including it - do not
invent or guess at a spelling (e.g. "envigorate" is not a word; the
correct word is "invigorate").

Return ONLY this JSON object. No markdown. No explanation:
{{
  "vocabulary": [
    {{"word": "example", "meaning": "clear simple explanation of meaning in 1 sentence"}},
    {{"word": "example2", "meaning": "clear simple explanation of meaning in 1 sentence"}}
  ]
}}"""

    # NOTE: this used to ask for a bare JSON array and call safe_gpt_call()
    # with no explicit `caller`, which defaults to utils.gpt_client.call_gpt
    # - that function forces response_format={"type": "json_object"} on the
    # OpenAI call, which REQUIRES the model to output a JSON object, not a
    # bare array. That mismatch meant the result was never in a shape this
    # function recognized (isinstance(result, list) was never true), so it
    # silently fell through to the hardcoded fallback list on every single
    # call, regardless of topic - confirmed by real test output repeatedly
    # returning the exact same 5 words for every part across unrelated
    # topics (parks, TV shows, etc). Now asks for an object wrapping the
    # array, matching what the enforced response format actually requires.
    result = safe_gpt_call(prompt, fallback=None, caller=lambda p: call_gpt(p, usage_log=usage_log))

    def _extract_vocab_list(value):
        # Don't slice to the final 5 here - the spelling filter below may
        # drop some items, so keep every raw candidate available to filter
        # from first, and slice only after filtering.
        if isinstance(value, dict):
            value = value.get("vocabulary")
        if isinstance(value, list) and len(value) >= 3:
            return value
        return None

    def _finalize(raw_list):
        checked = _filter_misspelled_vocab(raw_list)
        if len(checked) >= 3:
            return checked[:8]
        return None

    extracted = _extract_vocab_list(result)
    if extracted is not None:
        finalized = _finalize(extracted)
        if finalized is not None:
            return finalized

    # Defensive fallback: handle a string response (e.g. if the model still
    # wraps the JSON in prose/fences despite the response_format constraint).
    if isinstance(result, str) and result.strip():
        try:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            extracted = _extract_vocab_list(json.loads(clean))
            if extracted is not None:
                finalized = _finalize(extracted)
                if finalized is not None:
                    return finalized
        except (json.JSONDecodeError, ValueError):
            pass

    return fallbacks.get(part, VOCAB_FALLBACK_PART1)


def grammar_corrections(transcript: str) -> list:
    prompt = f"""You are an IELTS grammar examiner.

Read this spoken answer and identify grammatical errors:
"{transcript}"

Return ONLY a JSON array of objects. Each object must have exactly these two keys:
- "original": the sentence as the student said it (with the error)
- "corrected": the same sentence with the grammar fixed

Example format:
[
  {{"original": "I think clubs can be broadly categorized into three categories.", "corrected": "I think clubs can be broadly categorised into three main categories."}},
  {{"original": "One of them are cultural clubs.", "corrected": "One of them is a cultural club."}}
]

If there are no errors return: []

IMPORTANT:
- Do NOT return plain strings in the array
- Do NOT return the full answer as one item
- Every item MUST have both "original" and "corrected" keys
- Return ONLY the JSON array, no explanation, no markdown fences"""

    result = safe_gpt_call(prompt, fallback=None)

    if not result:
        return []

    if isinstance(result, list):
        valid = [item for item in result if isinstance(item, dict) and "original" in item and "corrected" in item]
        return valid if valid else []

    if isinstance(result, str):
        try:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                valid = [item for item in parsed if isinstance(item, dict) and "original" in item and "corrected" in item]
                return valid if valid else []
        except (json.JSONDecodeError, ValueError):
            pass

    return []


def sentence_improvements(transcript: str) -> list:
    prompt = f"""You are an IELTS speaking coach.

Improve 2-3 sentences from this answer to make them smoother and more natural:
"{transcript}"

Return ONLY a JSON array of objects:
[
  {{"original": "...", "improved": "..."}}
]

Do NOT include explanations or markdown."""

    result = safe_gpt_call(prompt, fallback=None)

    if not result:
        return []

    if isinstance(result, list):
        valid = [item for item in result if isinstance(item, dict) and "original" in item and "improved" in item]
        return valid if valid else []

    if isinstance(result, str):
        try:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                valid = [item for item in parsed if isinstance(item, dict) and "original" in item and "improved" in item]
                return valid if valid else []
        except (json.JSONDecodeError, ValueError):
            pass

    return []




# ------------------------------------------------------------

# Helper to process a single part (reused by both endpoints)

# ------------------------------------------------------------

def _evaluate_speaking_part_audio(audio_bytes: bytes, part: int, question: str = None, questions: str = None, debug: bool = False, usage_log=None):

    part_start = time.time()

    # Validate raw bytes

    if not audio_bytes or len(audio_bytes) < 1000:

        logger.error({"event": "audio_error", "part": part, "reason": "invalid_audio_bytes"})

        return {

            "part": part,

            "error": "invalid_audio",

            "message": "Uploaded audio is empty or too small",

            "transcript": "",

            "audio_metrics": {},

            "result": None,

            "processing_time": round(time.time() - part_start, 3)

        }

    if len(audio_bytes) > MAX_AUDIO_BYTES:

        logger.error({"event": "audio_error", "part": part, "reason": "audio_too_large", "size_bytes": len(audio_bytes)})

        return {

            "part": part,

            "error": "audio_too_large",

            "message": f"Uploaded audio is {len(audio_bytes) / (1024 * 1024):.1f}MB, which exceeds the {MAX_AUDIO_BYTES // (1024 * 1024)}MB limit per clip",

            "transcript": "",

            "audio_metrics": {},

            "result": None,

            "processing_time": round(time.time() - part_start, 3)

        }

    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    # Convert to wav once using in-memory bytes

    dummy_upload = type("DummyUpload", (), {"file": io.BytesIO(audio_bytes)})

    try:
        wav_path = normalize_to_wav(dummy_upload)
    except RuntimeError as exc:
        logger.error({"event": "audio_error", "part": part, "reason": "conversion_exception", "error": str(exc)})
        return {
            "part": part,
            "error": "conversion_failed",
            "message": str(exc),
            "transcript": "",
            "audio_metrics": {},
            "result": None,
            "processing_time": round(time.time() - part_start, 3)
        }



    if (not os.path.exists(wav_path)) or (os.path.getsize(wav_path) < 2000):

        logger.error({"event": "audio_error", "part": part, "reason": "conversion_failed"})

        return {

            "part": part,

            "error": "conversion_failed",

            "message": "Audio conversion failed or produced empty file",

            "transcript": "",

            "audio_metrics": {},

            "result": None,

            "processing_time": round(time.time() - part_start, 3)

        }



    # Cap ABSURDLY long audio (guards against a malformed/runaway upload,
    # not real exam answers - MAX_AUDIO_BYTES above already handles abuse
    # prevention), and reject audio too short/near-empty to contain real
    # speech BEFORE handing it to Whisper. Whisper's mel-spectrogram
    # computation produces a zero-length feature tensor for near-silent/
    # empty input, which crashes deep inside its decoding code with a
    # cryptic tensor-reshape error rather than a clear message - catching
    # it here up front gives a real, actionable error instead ("cannot
    # reshape tensor of 0 elements...").
    #
    # This used to cap at 90 seconds "to keep Whisper fast" - but that
    # silently truncated legitimate answers, not just abuse: official
    # IELTS Part 2 gives candidates up to 2 minutes (120s) for the long
    # turn, and this endpoint also combines multiple Part 3 questions into
    # one recording, which can easily run several minutes total. A real
    # candidate's answer trailing off mid-sentence in the transcript
    # ("...so participants are also from") was this cap silently cutting
    # off real content, not a transcription accuracy problem. Raised to a
    # generous ceiling that only trims genuinely pathological uploads.

    try:

        dur_sec = _wav_duration_seconds(wav_path)

        if dur_sec > 300:

            _trim_wav(wav_path, 300)

        if dur_sec < 0.3:

            logger.error({"event": "audio_error", "part": part, "reason": "audio_too_short", "duration_sec": dur_sec})

            return {

                "part": part,

                "error": "audio_too_short",

                "message": f"Audio is only {dur_sec:.2f}s long - too short or silent to transcribe.",

                "transcript": "",

                "audio_metrics": {},

                "result": None,

                "processing_time": round(time.time() - part_start, 3)

            }

    except Exception as exc:

        logger.warning({"event": "audio_warn", "part": part, "reason": "duration_check_failed", "details": str(exc)})



    cached_entry, asr_hit = _ASR_CACHE.get(audio_hash)

    if asr_hit:

        transcript, real_asr_confidence = cached_entry

        logger.info({"event": "asr_cache_hit", "part": part, "audio_hash": audio_hash[:16], "transcript_preview": (transcript or "")[:60]})

    else:

        if not WHISPER_AVAILABLE:

            raise RuntimeError("Whisper model not available; install dependencies.")

        try:

            logger.info({"event": "transcription_start", "part": part, "audio_hash": audio_hash[:16]})

            question_context = question if question and str(question).strip().lower() != "string" else None

            transcript, real_asr_confidence = transcribe_audio(

                wav_path,

                question=question_context,

                return_confidence=True,

            )

        except Exception as exc:

            logger.error({"event": "audio_error", "part": part, "reason": "transcription_failed", "details": str(exc)})

            return {

                "part": part,

                "error": "transcription_failed",

                "details": str(exc),

                "transcript": "",

                "audio_metrics": {},

                "result": None,

                "processing_time": round(time.time() - part_start, 3)

            }

        _ASR_CACHE.set(audio_hash, (transcript, real_asr_confidence))



    cached_metrics, feature_hit = _FEATURE_CACHE.get(audio_hash)

    if feature_hit:

        audio_metrics = cached_metrics.copy()

    else:

        audio_metrics = extract_acoustic_features(wav_path, transcript)

        _FEATURE_CACHE.set(audio_hash, audio_metrics.copy())



    if not transcript or not transcript.strip():

        logger.error({"event": "audio_error", "part": part, "reason": "no_speech_detected"})

        return {

            "part": part,

            "error": "no_speech_detected",

            "message": "No speech detected in audio",

            "transcript": "",

            "audio_metrics": audio_metrics,

            "result": None,

            "processing_time": round(time.time() - part_start, 3)

        }



    # Speech rate - both denominators always computed and exposed
    # (speech_rate_wpm_raw / speech_rate_wpm_voiced), so the raw-vs-voiced
    # gap stays visible regardless of SPEAKING_VOICED_WPM. speech_rate_wpm
    # itself is the "active" field generate_scores() actually reads as
    # pacing evidence - which denominator that means is controlled by the
    # flag (see its definition above).

    words = len(transcript.split())

    duration = audio_metrics["duration_sec"]

    voiced_duration = audio_metrics.get("voiced_duration_sec") or 0.0

    audio_metrics["speech_rate_wpm_raw"] = round((words / duration) * 60) if duration > 0 else 0

    audio_metrics["speech_rate_wpm_voiced"] = round((words / voiced_duration) * 60) if voiced_duration > 0 else 0

    audio_metrics["speech_rate_wpm"] = (
        audio_metrics["speech_rate_wpm_voiced"] if SPEAKING_VOICED_WPM else audio_metrics["speech_rate_wpm_raw"]
    )



    # Duration validation + ASR confidence + pronunciation proxy

    audio_metrics["duration_valid"] = duration >= 45

    # Genuine confidence from Whisper's own per-segment avg_logprob/
    # no_speech_prob (see _estimate_confidence in audio_transcriber.py) -
    # previously this was a hardcoded 0.9 whenever the transcript was
    # merely non-empty, which meant the "low confidence" dampening logic
    # below (and throughout the pronunciation/fluency fusion code) could
    # essentially never trigger regardless of how unclear the audio was.
    audio_metrics["asr_confidence"] = real_asr_confidence if transcript.strip() else 0.0

    audio_metrics["pronunciation_score"] = compute_pronunciation_score(audio_metrics, audio_metrics["asr_confidence"])

    if "pronunciation_confidence" in audio_metrics:

        base_pc = audio_metrics["pronunciation_confidence"]

        audio_metrics["pronunciation_confidence"] = round(min(1.0, (base_pc * 0.5) + (audio_metrics["asr_confidence"] * 0.5)), 3)



    clean_question = question if question and str(question).strip().lower() != "string" else None



    # Parse questions payload (JSON array) and merge with legacy single question

    question_list = []

    if questions:

        try:

            parsed = json.loads(questions)

            if isinstance(parsed, list):

                question_list = parsed

            elif isinstance(parsed, str):

                question_list = [parsed]

        except Exception as exc:

            logger.warning({"event": "question_parse_failed", "details": str(exc), "raw": questions})



    if not question_list and clean_question:

        question_list = [clean_question]



    # Remove placeholder values like "string"

    question_list = [q.strip().lstrip(". ") for q in question_list if str(q).strip().lower() != "string"]



    # With exactly one question (the normal case for this endpoint - one
    # audio clip per question), there is nothing to split: the whole
    # transcript IS the answer. Use it verbatim rather than routing it
    # through GPT's "split" prompt, which was asked to keep answers
    # "concise but meaningful" - an instruction that invites paraphrasing/
    # condensing, silently altering the candidate's actual spoken words.
    # Only genuinely multi-question audio (multiple questions answered in
    # one recording) needs GPT to divide the transcript at all.
    if len(question_list) <= 1:
        answers = [transcript]
    else:
        answers = split_transcript_with_gpt(transcript, question_list, usage_log=usage_log)

    if len(answers) != len(question_list):

        answers = [transcript] * len(question_list) if question_list else [transcript]

    while len(answers) < len(question_list):

        answers.append("")



    logger.info({

        "part": part,

        "question_used": clean_question if not question_list else "multiple",

        "questions": len(question_list),

        "answers": len(answers),

    })



    evaluated_qas = []

    for idx, q in enumerate(question_list):

        ans = answers[idx] if idx < len(answers) else ""
        if not ans:
            ans = transcript

        # NOTE: this used to also call evaluate_speaking_part() here and
        # store its output as qa_result/"result". That function runs a full
        # GPT call plus heavy regex/audio post-processing per question, but
        # nothing downstream ever reads qa_pairs[i]["result"] - the actual
        # scores shown to the user always come from generate_scores() later
        # in this endpoint, a completely separate code path. Removed as
        # pure wasted work (traced end-to-end with zero consumers) - this
        # was one of the biggest contributors to overall latency.
        qa_result = {}

        question_result = generate_question_mistakes(q, ans, usage_log=usage_log) if ans else {
            "mistakes": [
                {"type": "fluency", "original": "", "corrected": "", "explanation": "No answer to check."},
                {"type": "grammar", "original": "", "corrected": "", "explanation": "No answer to check."},
                {"type": "vocabulary", "original": "", "corrected": "", "explanation": "No answer to check."},
                {"type": "pronunciation", "original": "", "corrected": "", "explanation": "No answer to check."},
            ],
            "completeness_notice": "",
        }

        qa_mistakes_all = question_result.get("mistakes", [])
        qa_entry = {

            "question": q,

            "answer": ans,

            "user_answer": ans,

            "result": qa_result,

            "mistakes": qa_mistakes_all,

            "completeness_notice": question_result.get("completeness_notice", ""),

        }
        if SPEAKING_MISTAKE_SEVERITY_SPLIT:
            qa_significant, qa_minor = _split_mistakes_by_severity(qa_mistakes_all)
            qa_entry["mistakes"] = qa_significant
            qa_entry["minor_observations"] = qa_minor

        evaluated_qas.append(qa_entry)



    # NOTE: this used to call evaluate_speaking_part() again here on the
    # combined transcript. Same as above - part_result["result"] has no
    # consumer downstream (the real scores come from generate_scores()),
    # so this was a second full GPT call per audio clip for nothing.
    if question_list:

        combined_eval_text = "\n\n".join(

            [f"Question: {q}\nAnswer: {a}" for q, a in zip(question_list, answers)]

        )

    else:

        combined_eval_text = f"Question: {clean_question}\nAnswer: {transcript}" if clean_question else transcript

    result = {}



    latency = round(time.time() - part_start, 3)

    warning = None

    if duration < 3:

        warning = "audio_too_short"

    elif duration > 120:

        warning = "audio_too_long"



    return {

        "part": part,

        "transcript": transcript,

        "question": clean_question,

        "questions": question_list,

        "audio_metrics": audio_metrics,

        "result": result,

        "qa_pairs": evaluated_qas,

        "processing_time": latency,

        "warning": warning

    }






# ------------------------------------------------------------

# New endpoint: question-wise audio evaluation (1-15 questions)

# ------------------------------------------------------------

@router.post("/audio/question-wise")

async def evaluate_question_wise_audio(

    audio_1: UploadFile = File(None),

    audio_2: UploadFile = File(None),

    audio_3: UploadFile = File(None),

    audio_4: UploadFile = File(None),

    audio_5: UploadFile = File(None),

    audio_6: UploadFile = File(None),

    audio_7: UploadFile = File(None),

    audio_8: UploadFile = File(None),

    audio_9: UploadFile = File(None),

    audio_10: UploadFile = File(None),

    audio_11: UploadFile = File(None),

    audio_12: UploadFile = File(None),

    audio_13: UploadFile = File(None),

    audio_14: UploadFile = File(None),

    audio_15: UploadFile = File(None),

    question_1: str = Form(None),

    question_2: str = Form(None),

    question_3: str = Form(None),

    question_4: str = Form(None),

    question_5: str = Form(None),

    question_6: str = Form(None),

    question_7: str = Form(None),

    question_8: str = Form(None),

    question_9: str = Form(None),

    question_10: str = Form(None),

    question_11: str = Form(None),

    question_12: str = Form(None),

    question_13: str = Form(None),

    question_14: str = Form(None),

    question_15: str = Form(None),

    part_number_1: str = Form(None),

    part_number_2: str = Form(None),

    part_number_3: str = Form(None),

    part_number_4: str = Form(None),

    part_number_5: str = Form(None),

    part_number_6: str = Form(None),

    part_number_7: str = Form(None),

    part_number_8: str = Form(None),

    part_number_9: str = Form(None),

    part_number_10: str = Form(None),

    part_number_11: str = Form(None),

    part_number_12: str = Form(None),

    part_number_13: str = Form(None),

    part_number_14: str = Form(None),

    part_number_15: str = Form(None),

):

    if not _check_rate_limit():
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {_RATE_LIMIT_THRESHOLD} requests per {_RATE_LIMIT_WINDOW_SECONDS} seconds. Please try again shortly."
        )

    # Accumulates real token usage (from the API's own usage metadata)
    # across every LLM call made during this evaluation, including ones
    # made concurrently across real OS threads via asyncio.to_thread()
    # below - a plain list is used (not a counter) because list.append()
    # is atomic under the GIL, so it's safe to mutate from multiple
    # threads without a lock, unlike a shared running total which would
    # need one. Summed into the response's "usage" field at the end.
    usage_log = []

    audios = [

        audio_1, audio_2, audio_3, audio_4, audio_5,

        audio_6, audio_7, audio_8, audio_9, audio_10,

        audio_11, audio_12, audio_13, audio_14, audio_15

    ]



    questions = [

        question_1, question_2, question_3, question_4, question_5,

        question_6, question_7, question_8, question_9, question_10,

        question_11, question_12, question_13, question_14, question_15

    ]

    part_numbers = [

        part_number_1, part_number_2, part_number_3, part_number_4, part_number_5,

        part_number_6, part_number_7, part_number_8, part_number_9, part_number_10,

        part_number_11, part_number_12, part_number_13, part_number_14, part_number_15

    ]



    filtered = [

        (a, q, p) for a, q, p in zip(audios, questions, part_numbers)

        if a is not None and q is not None

    ]



    if len(filtered) == 0:

        raise HTTPException(status_code=400, detail="no_audio_provided: at least one audio file with a matching question is required")

    if len(filtered) > 15:

        raise HTTPException(status_code=400, detail="max_15_questions_allowed: at most 15 audio/question pairs are supported per request")



    # Read every uploaded file's bytes up front (cheap, no CPU/GPT cost),
    # then run all audio processing CONCURRENTLY instead of one-at-a-time.
    # _evaluate_speaking_part_audio() is genuinely synchronous internally
    # (ffmpeg subprocess, Whisper, librosa, GPT calls - no real await
    # points), so asyncio.gather() alone would NOT parallelize it; it needs
    # to actually run in a thread pool via asyncio.to_thread() to get real
    # concurrency. Each audio clip's transcription/scoring is fully
    # independent of the others, so this turns what used to be N strictly
    # sequential 10-40s pipelines into one N-way concurrent batch - by far
    # the largest lever on overall request latency (up to 15 audio clips).
    audio_byte_pairs = [(await audio_file.read(), question, part_no) for audio_file, question, part_no in filtered]

    part_results = await asyncio.gather(*[
        asyncio.to_thread(_evaluate_speaking_part_audio, audio_bytes=audio_bytes, part=1, question=question, usage_log=usage_log)
        for audio_bytes, question, _ in audio_byte_pairs
    ])

    results = []

    for (_, question, part_no), part_result in zip(audio_byte_pairs, part_results):

        if isinstance(part_result, dict):

            pr = part_result.get("result")

            if pr:

                try:
                    part_result["result"] = sanitize_result(pr)
                except Exception as e:
                    logger.error("sanitize_result error: %s", e)
                    part_result["result"] = {}

            if not part_result.get("result"):
                part_result["result"] = {}



        qa_pairs = part_result.get("qa_pairs", []) or []
        first_answer = ""
        if isinstance(qa_pairs, list) and qa_pairs:
            first_answer = str(qa_pairs[0].get("answer", "")).strip()
        if not first_answer:
            first_answer = str(part_result.get("transcript", "")).strip()

        results.append({

            "question": question,

            "part_no": part_no,

            "transcript": part_result.get("transcript"),

            "result": part_result.get("result"),

            "qa_pairs": qa_pairs,

            "user_answer": first_answer,

            "audio_error": part_result.get("error"),

            "audio_error_message": part_result.get("message") or part_result.get("details"),

            "audio_metrics": part_result.get("audio_metrics")

        })



    results = [r for r in results if r.get("question") and str(r["question"]).strip().lower() != "string"]

    # NOTE: this used to also compute a local "overall_band" here from
    # scored_results/avg_fluency/etc. That value was never actually used -
    # the real overall_band in the final response always comes from
    # calculate_overall_band() further down (using generate_scores()'
    # output), a completely separate computation. Removed as dead work.

    # Build part-wise aggregation using cue-card detection
    def is_cue_card(question_text: str) -> bool:
        text = (question_text or "").strip().lstrip(".,; ")
        if "|" in text:
            return True
        if text.lower().startswith("describe"):
            return True
        if len(text.split()) > 15:
            return True
        return False

    def _extract_answer_from_text(text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return ""
        if "Answer:" in text:
            return text.split("Answer:", 1)[1].strip()
        return text

    def _build_question_items(r):
        qa_pairs = r.get("qa_pairs") or []
        items = []

        if isinstance(qa_pairs, list) and qa_pairs:
            for qa in qa_pairs:
                q_text = str(qa.get("question", "")).strip()
                answer_text = str(qa.get("answer", "")).strip() or str(qa.get("user_answer", "")).strip()
                if not answer_text:
                    answer_text = _extract_answer_from_text(qa.get("transcript", ""))
                if not answer_text:
                    answer_text = str(r.get("user_answer", "")).strip() or str(r.get("transcript", "")).strip()
                if q_text or answer_text:
                    mistakes_list = qa.get("mistakes", []) if isinstance(qa, dict) else []
                    item = {
                        "question": q_text,
                        "user_answer": answer_text,
                        # An empty list means "no mistakes found" - return
                        # null instead of [] so it's an explicit signal
                        # rather than something a consumer might mistake for
                        # "mistakes weren't computed at all".
                        "mistakes": mistakes_list if mistakes_list else None,
                        "completeness_notice": qa.get("completeness_notice", "") if isinstance(qa, dict) else "",
                    }
                    # Only present at all when the severity-split flag was
                    # on upstream (see evaluated_qas.append above) - keeps
                    # this key fully absent, not just empty, when the flag
                    # is off.
                    if isinstance(qa, dict) and "minor_observations" in qa:
                        minor_list = qa.get("minor_observations")
                        item["minor_observations"] = minor_list if minor_list else None
                    if not answer_text and r.get("audio_error"):
                        item["audio_error"] = r.get("audio_error")
                        item["audio_error_message"] = r.get("audio_error_message")
                    items.append(item)
            return items

        q_text = str(r.get("question", "")).strip()
        user_answer = str(r.get("user_answer", "")).strip()
        if not user_answer:
            user_answer = _extract_answer_from_text(r.get("transcript", ""))
        if q_text or user_answer:
            item = {"question": q_text, "user_answer": user_answer, "mistakes": None, "completeness_notice": ""}
            if not user_answer and r.get("audio_error"):
                item["audio_error"] = r.get("audio_error")
                item["audio_error_message"] = r.get("audio_error_message")
            return [item]
        return []

    part_1_qas = []
    part_2_qas = []
    part_3_qas = []

    # Prefer the caller-supplied part number for every question over guessing
    # from question text. The cue-card heuristic below only looks at whether
    # a question *looks like* a Part 2 prompt (contains "|", starts with
    # "describe", or is a long sentence) and otherwise splits purely by
    # position - it has no way to know a Part 3 discussion question wasn't
    # meant to be Part 1, and a Part 2 prompt that doesn't match the pattern
    # silently corrupts the whole split. When every result carries an
    # explicit, valid part_no, trust it completely and skip the heuristic.
    explicit_part_map = {1: part_1_qas, 2: part_2_qas, 3: part_3_qas}

    def _valid_part_no(value) -> int | None:
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return parsed if parsed in (1, 2, 3) else None

    all_parts_explicit = len(results) > 0 and all(_valid_part_no(r.get("part_no")) is not None for r in results)

    if all_parts_explicit:
        for r in results:
            explicit_part_map[_valid_part_no(r.get("part_no"))].append(r)
    else:
        cue_card_found = False

        for r in results:
            question_text = r.get("question") or r.get("question_text", "")
            if not cue_card_found and is_cue_card(question_text):
                part_2_qas.append(r)
                cue_card_found = True
            elif not cue_card_found:
                part_1_qas.append(r)
            else:
                part_3_qas.append(r)

        if not cue_card_found:
            part_1_qas = results[:3]
            part_2_qas = []
            part_3_qas = results[3:]

    part_1_qas_clean = []
    for r in part_1_qas:
        part_1_qas_clean.extend(_build_question_items(r))

    part_2_qas_clean = []
    for r in part_2_qas:
        part_2_qas_clean.extend(_build_question_items(r))

    part_3_qas_clean = []
    for r in part_3_qas:
        part_3_qas_clean.extend(_build_question_items(r))

    # NOTE: this used to also compute part_1_summary/part_2_summary/
    # part_3_summary here via a call to _aggregate_part(), a function that
    # is never defined anywhere in this codebase - every call raised
    # NameError, was silently swallowed by the except block, and logged a
    # spurious error on every single request. The resulting summaries were
    # never used downstream either. Removed entirely as dead, error-logging
    # work.

    def _combine_context(qas_clean):
        return "\n\n".join(
            [
                f"Question: {qa.get('question', '')}\nStudent answer: {qa.get('user_answer', '')}"
                for qa in qas_clean
                if qa.get("question") or qa.get("user_answer")
            ]
        ).strip()

    def _combine_answers_only(qas_clean):
        return "\n\n".join(
            [qa.get("user_answer", "") for qa in qas_clean if qa.get("user_answer")]
        ).strip()

    def _combine_transcripts(qas):
        return " ".join(
            [qa.get("transcript", "") for qa in qas if qa.get("transcript")]
        ).strip()

    p1_combined_context = _combine_context(part_1_qas_clean)
    p2_combined_context = _combine_context(part_2_qas_clean)
    p3_combined_context = _combine_context(part_3_qas_clean)

    p1_answers_only = _combine_answers_only(part_1_qas_clean)
    p2_answers_only = _combine_answers_only(part_2_qas_clean)
    p3_answers_only = _combine_answers_only(part_3_qas_clean)

    p1_combined_transcripts = _combine_transcripts(part_1_qas)
    p2_combined_transcripts = _combine_transcripts(part_2_qas)
    p3_combined_transcripts = _combine_transcripts(part_3_qas)

    def _combined_for_feedback(qas_clean):
        return "\n\n".join(
            [
                f"Q: {qa.get('question', '')}\nA: {qa.get('user_answer', '')}"
                for qa in qas_clean
                if qa.get("question") or qa.get("user_answer")
            ]
        ).strip()

    # NOTE: a second, out-of-date copy of _build_question_items() used to be
    # defined here, redundant with the one above and never called (it was
    # defined AFTER part_1_qas_clean/part_2_qas_clean/part_3_qas_clean were
    # already built using the real one). Removed as dead duplicate code -
    # exactly the kind of trap where someone edits the wrong copy.

    p1_feedback_text = _combined_for_feedback(part_1_qas_clean)
    p2_feedback_text = _combined_for_feedback(part_2_qas_clean)
    p3_feedback_text = _combined_for_feedback(part_3_qas_clean)

    p1_acoustic_pron = _aggregate_acoustic_pronunciation(part_1_qas)
    p2_acoustic_pron = _aggregate_acoustic_pronunciation(part_2_qas)
    p3_acoustic_pron = _aggregate_acoustic_pronunciation(part_3_qas)

    p1_speech_timing = _aggregate_speech_timing(part_1_qas)
    p2_speech_timing = _aggregate_speech_timing(part_2_qas)
    p3_speech_timing = _aggregate_speech_timing(part_3_qas)

    # A part with zero real answer content means the candidate did not
    # attempt it at all - per the descriptor's own Band 0 definition
    # ("Does not attend / Does not complete the test"), that must score 0,
    # not a generous middle-of-the-road default. This previously defaulted
    # to 5.0, silently rewarding a non-attempt with a passing-looking band.
    _default_scores = {"fluency": 0.0, "lexical": 0.0, "grammar": 0.0, "pronunciation": 0.0, "topic_relevance": "on_topic", "pronunciation_source": "unavailable"}
    p1_not_attempted = not p1_feedback_text
    p2_not_attempted = not p2_feedback_text
    p3_not_attempted = not p3_feedback_text

    async def _immediate(value):
        return value

    # Whole-test transcript (all 3 parts together) for systematic-error
    # detection - a pattern spanning multiple parts (e.g. the same
    # grammatical error in both Part 2 and Part 3) is invisible to any of
    # the per-part generate_scores()/generate_mistakes() calls below, since
    # each only ever sees its own part's transcript.
    whole_test_transcript = "\n\n".join(
        f"=== PART {i} ===\n{text}"
        for i, text in ((1, p1_feedback_text), (2, p2_feedback_text), (3, p3_feedback_text))
        if text
    )

    # WAVE 1: vocabulary, scores, mistakes for all 3 parts, and whole-test
    # systematic-error detection are fully independent of each other's
    # results - none of these GPT calls need anything the others produce.
    # Run them all concurrently in a thread pool instead of one after
    # another (this used to be ~9 sequential blocking calls; now it's
    # bounded by the single slowest one).
    # detect_answer_alignment_issues() is deliberately NOT called here -
    # across this session's real test reviews it produced 4 confirmed
    # false positives against 1 confirmed true positive, even after
    # tightening the prompt with a "fails_own_question" structural gate
    # AND switching to gpt-4o via call_gpt_strong. The signal-to-noise
    # ratio proved too poor for a "review recommended" flag to be
    # trustworthy, and the root cause it targets (a client-side recording
    # bug attaching the wrong question label to an audio clip) lives
    # outside this repo anyway. Left defined below, tested, and easy to
    # re-enable if a future model/approach actually fixes the reliability
    # problem - just not spending a GPT call on it while it doesn't.
    vocab_part1, vocab_part2, vocab_part3, p1_scores, p2_scores, p3_scores, p1_feedback, p2_feedback, p3_feedback, systematic_errors, p1_feedback_summary, p2_feedback_summary, p3_feedback_summary = await asyncio.gather(
        asyncio.to_thread(generate_vocabulary, 1, p1_combined_transcripts, usage_log=usage_log) if p1_combined_transcripts else _immediate(VOCAB_FALLBACK_PART1),
        asyncio.to_thread(generate_vocabulary, 2, p2_combined_transcripts, usage_log=usage_log) if p2_combined_transcripts else _immediate(VOCAB_FALLBACK_PART2),
        asyncio.to_thread(generate_vocabulary, 3, p3_combined_transcripts, usage_log=usage_log) if p3_combined_transcripts else _immediate(VOCAB_FALLBACK_PART3),
        asyncio.to_thread(generate_scores, 1, p1_feedback_text, part_1_qas_clean, p1_acoustic_pron, p1_speech_timing, usage_log=usage_log) if p1_feedback_text else _immediate(dict(_default_scores)),
        asyncio.to_thread(generate_scores, 2, p2_feedback_text, part_2_qas_clean, p2_acoustic_pron, p2_speech_timing, usage_log=usage_log) if p2_feedback_text else _immediate(dict(_default_scores)),
        asyncio.to_thread(generate_scores, 3, p3_feedback_text, part_3_qas_clean, p3_acoustic_pron, p3_speech_timing, usage_log=usage_log) if p3_feedback_text else _immediate(dict(_default_scores)),
        asyncio.to_thread(generate_mistakes, 1, p1_feedback_text, usage_log=usage_log) if p1_feedback_text else _immediate({}),
        asyncio.to_thread(generate_mistakes, 2, p2_feedback_text, usage_log=usage_log) if p2_feedback_text else _immediate({}),
        asyncio.to_thread(generate_mistakes, 3, p3_feedback_text, usage_log=usage_log) if p3_feedback_text else _immediate({}),
        asyncio.to_thread(detect_systematic_errors, whole_test_transcript, usage_log=usage_log) if whole_test_transcript else _immediate([]),
        asyncio.to_thread(generate_feedback_summary, 1, p1_feedback_text, usage_log=usage_log) if p1_feedback_text else _immediate({"strengths": [], "areas_to_improve": [], "tips": []}),
        asyncio.to_thread(generate_feedback_summary, 2, p2_feedback_text, usage_log=usage_log) if p2_feedback_text else _immediate({"strengths": [], "areas_to_improve": [], "tips": []}),
        asyncio.to_thread(generate_feedback_summary, 3, p3_feedback_text, usage_log=usage_log) if p3_feedback_text else _immediate({"strengths": [], "areas_to_improve": [], "tips": []}),
    )

    p1_topic_relevance = p1_scores.pop("topic_relevance", "on_topic")
    p2_topic_relevance = p2_scores.pop("topic_relevance", "on_topic")
    p3_topic_relevance = p3_scores.pop("topic_relevance", "on_topic")

    p1_pronunciation_source = p1_scores.pop("pronunciation_source", "unavailable")
    p2_pronunciation_source = p2_scores.pop("pronunciation_source", "unavailable")
    p3_pronunciation_source = p3_scores.pop("pronunciation_source", "unavailable")

    # Per the descriptor sheet: Band 8 Grammatical Range and Accuracy
    # explicitly requires "only occasional inappropriacies/non-systematic
    # errors". A genuinely systematic pattern (occurring 3+ times, verified
    # in detect_systematic_errors) is real evidence that GRA cannot be
    # Band 8+ for this test, regardless of what each per-part checklist
    # concluded in isolation - so cap it here, after all 3 parts' scores
    # are known, at the point where the whole-test evidence is available.
    # The same "Repetition ratio... caps the band" feature already exists
    # in the Lexical Resource checklist, so a systematic vocabulary/
    # collocation pattern caps LR the same way.
    if any(e["criterion"] == "grammar" for e in systematic_errors):
        for scores in (p1_scores, p2_scores, p3_scores):
            if scores.get("grammar", 0) > 7:
                scores["grammar"] = 7.0
    if any(e["criterion"] == "vocabulary" for e in systematic_errors):
        for scores in (p1_scores, p2_scores, p3_scores):
            if scores.get("lexical", 0) > 7:
                scores["lexical"] = 7.0

    def _extract_questions(qas_clean):
        seen = []
        for qa in qas_clean or []:
            q = str(qa.get("question", "")).strip()
            if q and q not in seen:
                seen.append(q)
        return seen

    _OFF_TOPIC_TIERS = ("completely_off_topic", "partially_off_topic")

    # WAVE 2: band9 answers depend on each part's topic_relevance (from
    # Wave 1), so they have to wait for that - but the 3 parts are still
    # independent of EACH OTHER, so run all 3 concurrently rather than
    # sequentially. If the student's answer didn't address the question,
    # "polishing" it via generate_band9_answer() would just produce a
    # fluent version of the WRONG topic; generate_ideal_band9_answer()
    # instead writes a fresh model answer straight from the question.
    band9_part1, band9_part2, band9_part3 = await asyncio.gather(
        asyncio.to_thread(generate_ideal_band9_answer, 1, _extract_questions(part_1_qas_clean), usage_log=usage_log)
        if p1_topic_relevance in _OFF_TOPIC_TIERS
        else (asyncio.to_thread(generate_band9_answer, 1, p1_combined_context, answers_only=p1_answers_only, usage_log=usage_log) if p1_combined_context else _immediate(p1_answers_only)),

        asyncio.to_thread(generate_ideal_band9_answer, 2, _extract_questions(part_2_qas_clean), usage_log=usage_log)
        if p2_topic_relevance in _OFF_TOPIC_TIERS
        else (asyncio.to_thread(generate_band9_answer, 2, p2_combined_context, answers_only=p2_answers_only, usage_log=usage_log) if p2_combined_context else _immediate(p2_answers_only)),

        asyncio.to_thread(generate_ideal_band9_answer, 3, _extract_questions(part_3_qas_clean), usage_log=usage_log)
        if p3_topic_relevance in _OFF_TOPIC_TIERS
        else (asyncio.to_thread(generate_band9_answer, 3, p3_combined_context, answers_only=p3_answers_only, usage_log=usage_log) if p3_combined_context else _immediate(p3_answers_only)),
    )

    # Attach a per-question refined/model answer alongside the combined
    # band9_answer block, so each Q&A card in the report can show its own
    # corrected version instead of only the whole-part text at the end.
    _attach_refined_answers(part_1_qas_clean, band9_part1)
    _attach_refined_answers(part_2_qas_clean, band9_part2)
    _attach_refined_answers(part_3_qas_clean, band9_part3)

    p1_completeness_notices = _collect_completeness_notices(part_1_qas_clean)
    p2_completeness_notices = _collect_completeness_notices(part_2_qas_clean)
    p3_completeness_notices = _collect_completeness_notices(part_3_qas_clean)

    # Item C.2 - drop any part-level criterion sentence that duplicates a
    # specific error already reported per-question (see
    # _dedupe_part_level_text_against_question_mistakes). Runs on the raw
    # generate_mistakes() text, before the relevance/completeness notices
    # below are appended, so it only ever touches GPT's own criterion
    # prose, never text this module added itself.
    def _collect_question_mistakes(qas_clean):
        collected = []
        for qa in qas_clean or []:
            collected.extend(qa.get("mistakes") or [])
        return collected

    p1_question_mistakes = _collect_question_mistakes(part_1_qas_clean)
    p2_question_mistakes = _collect_question_mistakes(part_2_qas_clean)
    p3_question_mistakes = _collect_question_mistakes(part_3_qas_clean)
    for feedback, question_mistakes in (
        (p1_feedback, p1_question_mistakes),
        (p2_feedback, p2_question_mistakes),
        (p3_feedback, p3_question_mistakes),
    ):
        for key in ("fluency", "grammar", "vocabulary", "pronunciation"):
            if feedback.get(key):
                feedback[key] = _dedupe_part_level_text_against_question_mistakes(feedback[key], question_mistakes)

    p1_feedback = _apply_relevance_to_feedback(p1_feedback, p1_topic_relevance)
    p2_feedback = _apply_relevance_to_feedback(p2_feedback, p2_topic_relevance)
    p3_feedback = _apply_relevance_to_feedback(p3_feedback, p3_topic_relevance)

    p1_feedback = _apply_completeness_to_feedback(p1_feedback, p1_completeness_notices)
    p2_feedback = _apply_completeness_to_feedback(p2_feedback, p2_completeness_notices)
    p3_feedback = _apply_completeness_to_feedback(p3_feedback, p3_completeness_notices)

    # Severity split (Part 1 of the minor/significant + repetition task) -
    # behind SPEAKING_MISTAKE_SEVERITY_SPLIT, default OFF. p{N}_mistakes is
    # byte-identical to the dict this file always built here when the flag
    # is off; only the flag-on branch changes what "mistakes" contains and
    # adds the new minor_observations dict (which also carries the
    # deterministic word-repetition results - Part 2 of the same task).
    p1_mistakes = {
        "fluency": p1_feedback.get("fluency", ""),
        "fluency_severity": p1_feedback.get("fluency_severity"),
        "grammar": p1_feedback.get("grammar", ""),
        "grammar_severity": p1_feedback.get("grammar_severity"),
        "vocabulary": p1_feedback.get("vocabulary", ""),
        "vocabulary_severity": p1_feedback.get("vocabulary_severity"),
        "pronunciation": p1_feedback.get("pronunciation", ""),
        "pronunciation_severity": p1_feedback.get("pronunciation_severity"),
    }
    p2_mistakes = {
        "fluency": p2_feedback.get("fluency", ""),
        "fluency_severity": p2_feedback.get("fluency_severity"),
        "grammar": p2_feedback.get("grammar", ""),
        "grammar_severity": p2_feedback.get("grammar_severity"),
        "vocabulary": p2_feedback.get("vocabulary", ""),
        "vocabulary_severity": p2_feedback.get("vocabulary_severity"),
        "pronunciation": p2_feedback.get("pronunciation", ""),
        "pronunciation_severity": p2_feedback.get("pronunciation_severity"),
    }
    p3_mistakes = {
        "fluency": p3_feedback.get("fluency", ""),
        "fluency_severity": p3_feedback.get("fluency_severity"),
        "grammar": p3_feedback.get("grammar", ""),
        "grammar_severity": p3_feedback.get("grammar_severity"),
        "vocabulary": p3_feedback.get("vocabulary", ""),
        "vocabulary_severity": p3_feedback.get("vocabulary_severity"),
        "pronunciation": p3_feedback.get("pronunciation", ""),
        "pronunciation_severity": p3_feedback.get("pronunciation_severity"),
    }
    p1_minor_observations = p2_minor_observations = p3_minor_observations = None
    whole_test_repeated_words = []
    if SPEAKING_MISTAKE_SEVERITY_SPLIT:
        p1_mistakes, p1_minor_observations, p1_praise = _split_part_feedback_by_severity(p1_mistakes)
        p2_mistakes, p2_minor_observations, p2_praise = _split_part_feedback_by_severity(p2_mistakes)
        p3_mistakes, p3_minor_observations, p3_praise = _split_part_feedback_by_severity(p3_mistakes)
        # Praise (severity: null) is neither a band-affecting problem nor a
        # minor one - relocated to feedback_summary.strengths, the existing
        # home for genuine positive callouts, rather than left sitting in
        # either "fix this" array.
        for feedback_summary, praise in (
            (p1_feedback_summary, p1_praise),
            (p2_feedback_summary, p2_praise),
            (p3_feedback_summary, p3_praise),
        ):
            if praise and isinstance(feedback_summary, dict):
                feedback_summary["strengths"] = list(feedback_summary.get("strengths") or []) + praise
        # Repetition is pooled across the WHOLE TEST, not per part (see
        # count_word_repetitions' docstring): Part 1's answers alone are
        # often too short for a per-100-words rate to be stable, and
        # over-reliance on a word is a candidate-wide habit, not something
        # that resets at a part boundary. Topic words are pooled the same
        # way, from all three parts' questions together.
        whole_test_answers = "\n\n".join(filter(None, [p1_answers_only, p2_answers_only, p3_answers_only]))
        whole_test_questions = " ".join(
            _extract_questions(part_1_qas_clean) + _extract_questions(part_2_qas_clean) + _extract_questions(part_3_qas_clean)
        )
        whole_test_repeated_words = _repeated_word_observations(whole_test_answers, whole_test_questions)

    part_1 = {

        "questions": part_1_qas_clean,

        "scores": p1_scores,

        "mistakes": p1_mistakes,

        "improvement": p1_feedback.get("improvement", "Focus on expanding your answers with specific examples."),

        "feedback_summary": p1_feedback_summary,

        "band9_answer": band9_part1,

        "vocabulary_to_learn": vocab_part1,

        "relevance_notice": RELEVANCE_NOTICE_MESSAGES.get(p1_topic_relevance),

        "completeness_notice": " ".join(p1_completeness_notices) or None,

        "topic_relevance": p1_topic_relevance,

        "pronunciation_source": p1_pronunciation_source,

        "not_attempted": p1_not_attempted,

    }
    if SPEAKING_MISTAKE_SEVERITY_SPLIT:
        part_1["minor_observations"] = p1_minor_observations

    part_2 = {

        "questions": part_2_qas_clean,

        "scores": p2_scores,

        "mistakes": p2_mistakes,

        "improvement": p2_feedback.get("improvement", "Focus on expanding your answers with specific examples."),

        "feedback_summary": p2_feedback_summary,

        "band9_answer": band9_part2,

        "vocabulary_to_learn": vocab_part2,

        "relevance_notice": RELEVANCE_NOTICE_MESSAGES.get(p2_topic_relevance),

        "completeness_notice": " ".join(p2_completeness_notices) or None,

        "topic_relevance": p2_topic_relevance,

        "pronunciation_source": p2_pronunciation_source,

        "not_attempted": p2_not_attempted,

    }
    if SPEAKING_MISTAKE_SEVERITY_SPLIT:
        part_2["minor_observations"] = p2_minor_observations

    part_3 = {

        "questions": part_3_qas_clean,

        "scores": p3_scores,

        "mistakes": p3_mistakes,

        "improvement": p3_feedback.get("improvement", "Focus on expanding your answers with specific examples."),

        "feedback_summary": p3_feedback_summary,

        "band9_answer": band9_part3,

        "vocabulary_to_learn": vocab_part3,

        "relevance_notice": RELEVANCE_NOTICE_MESSAGES.get(p3_topic_relevance),

        "completeness_notice": " ".join(p3_completeness_notices) or None,

        "topic_relevance": p3_topic_relevance,

        "pronunciation_source": p3_pronunciation_source,

        "not_attempted": p3_not_attempted,

    }
    if SPEAKING_MISTAKE_SEVERITY_SPLIT:
        part_3["minor_observations"] = p3_minor_observations



    recalculated_overall = calculate_overall_band(
        part_1.get("scores", {}),
        part_2.get("scores", {}),
        part_3.get("scores", {})
    )

    final_response = {

        "overall_band": recalculated_overall,

        "part_1": part_1,

        "part_2": part_2,

        "part_3": part_3,

        "systematic_errors": systematic_errors,

        # Hardcoded empty rather than removed - detect_answer_alignment_
        # issues() is disabled (see the comment above the Wave 1 gather
        # call), but the key stays present so nothing consuming this
        # response shape breaks on a missing field.
        "alignment_warnings": [],

        "severity_legend": SEVERITY_LEGEND,

    }

    # Real LLM token usage for this evaluation, summed from every API
    # call's own usage metadata (never estimated) - see usage_log above.
    # Missing/malformed usage on any individual call was already handled
    # inside record_token_usage() (that call simply contributes nothing),
    # so this sum is always safe to compute even if some calls lacked it.
    final_response["usage"] = {
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage_log),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage_log),
        "total_tokens": sum(u.get("total_tokens", 0) for u in usage_log),
    }

    # Whole-test word-repetition results (see the pooling note above) live
    # at the top level, not inside any one part - absent entirely when the
    # flag is off, same additive contract as part_N["minor_observations"].
    if SPEAKING_MISTAKE_SEVERITY_SPLIT:
        final_response["minor_observations"] = {"repeated_words": whole_test_repeated_words}



    logger.info({

        "mode": "part_wise",

        "questions": len(questions)

    })

    # Per-clip raw AND voiced WPM, so the SPEAKING_VOICED_WPM prompt
    # threshold (currently qualitative-only under the flag - see
    # generate_scores()) can eventually be recalibrated from real usage
    # instead of the synthetic audio this was validated against.
    speech_timing_per_clip = [
        {
            "part": part_no,
            "duration_sec": (r.get("audio_metrics") or {}).get("duration_sec"),
            "voiced_duration_sec": (r.get("audio_metrics") or {}).get("voiced_duration_sec"),
            "speech_rate_wpm_raw": (r.get("audio_metrics") or {}).get("speech_rate_wpm_raw"),
            "speech_rate_wpm_voiced": (r.get("audio_metrics") or {}).get("speech_rate_wpm_voiced"),
        }
        for part_no, part_qas in ((1, part_1_qas), (2, part_2_qas), (3, part_3_qas))
        for r in (part_qas or [])
        if r.get("audio_metrics")
    ]
    log_evaluation({
        "evaluator": "speaking",
        "task_or_part": "full_test",
        "question": _extract_questions(part_1_qas_clean) + _extract_questions(part_2_qas_clean) + _extract_questions(part_3_qas_clean),
        "input_text": whole_test_transcript,
        "response": final_response,
        "model_default": "gpt-4o-mini",
        "model_strong": "gpt-4o",
        "flags": {
            "SPEAKING_MISTAKE_SEVERITY_SPLIT": SPEAKING_MISTAKE_SEVERITY_SPLIT,
            "SPEAKING_VOICED_WPM": SPEAKING_VOICED_WPM,
        },
        "speech_timing_per_clip": speech_timing_per_clip,
    })

    return final_response
