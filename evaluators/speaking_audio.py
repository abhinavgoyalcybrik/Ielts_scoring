from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from collections import OrderedDict, deque

import threading

from utils.audio_normalizer import normalize_to_wav

from utils.audio_transcriber import transcribe_audio

from utils.gpt_client import call_gpt

from utils.safety import safe_gpt_call, normalize_feedback

from evaluators.speaking import (

    compute_pronunciation_score,

)

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


VOCAB_FALLBACK_PART1 = [
    {"word": "beneficial", "meaning": "having a positive effect"},
    {"word": "enroll", "meaning": "officially join a group or course"},
    {"word": "facilitate", "meaning": "make something easier"},
    {"word": "interact", "meaning": "communicate and engage with others"},
    {"word": "diverse", "meaning": "showing variety and difference"},
]

VOCAB_FALLBACK_PART2 = [
    {"word": "elaborate", "meaning": "explain in more detail"},
    {"word": "furthermore", "meaning": "in addition to what was said"},
    {"word": "highlight", "meaning": "draw attention to something important"},
    {"word": "contribute", "meaning": "give or add to something"},
    {"word": "worthwhile", "meaning": "worth the time or effort spent"},
]

VOCAB_FALLBACK_PART3 = [
    {"word": "implication", "meaning": "a possible consequence or effect"},
    {"word": "mitigate", "meaning": "reduce the severity of something"},
    {"word": "postulate", "meaning": "suggest as a theory or idea"},
    {"word": "perspective", "meaning": "a particular way of viewing something"},
    {"word": "integration", "meaning": "combining parts into a whole"},
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





def _trim_wav(path: str, max_seconds: float = 90.0):

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





def split_transcript_with_gpt(transcript: str, questions: list):

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
        caller=call_gpt
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


def generate_band9_answer(part_number: int, combined_with_context: str, answers_only: str | None = None) -> str:
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
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        result = response.choices[0].message.content.strip()
        logging.warning(f"[BAND9 SUCCESS] part={part_number} words={len(result.split())}")

        orig_words = set(overlap_text.lower().split())
        new_words = set(result.lower().split())
        overlap = len(orig_words & new_words) / max(len(orig_words), 1)
        logging.warning(f"[BAND9 OVERLAP] {overlap:.0%}")

        if overlap > 0.60:
            logging.warning("[BAND9] Too similar, retrying...")
            retry_prompt = prompt + "\n\nREJECTED: Too similar to student answer. Use completely different words and sentence structures."
            retry_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=0.9,
                max_tokens=800
            )
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
                model="gpt-4o",
                messages=[{"role": "user", "content": length_retry_prompt}],
                temperature=0.7,
                max_tokens=800
            )
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


def generate_ideal_band9_answer(part_number: int, questions: list) -> str:
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
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
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
                model="gpt-4o",
                messages=[{"role": "user", "content": length_retry_prompt}],
                temperature=0.7,
                max_tokens=800
            )
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





def generate_mistakes(
    part_number: int,
    combined_transcripts: str
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
  "grammar": "specific feedback with example from their answer",
  "vocabulary": "specific feedback with example from their answer",
  "pronunciation": "one general, honest, text-grounded observation - not a fabricated claim about specific stress/intonation you didn't actually hear",
  "improvement": "one specific actionable tip for THIS student based on their actual answers"
}}

DO NOT FLAG (CRITICAL - this is SPOKEN language, not writing):
This is a transcript of something the student SAID out loud, not something
they wrote. Never flag, correct, or mention punctuation, capitalization, or
spelling - any of that in this text is a transcription artifact added by
automatic speech-to-text, not something the candidate did wrong.

WHAT NOT TO CRITIQUE (CRITICAL): never comment on whether an opinion is
correct, sensible, or well-reasoned - there is no "correct" opinion in
IELTS Speaking, only the language used to express it. Never penalize
British vs American vocabulary/word choice. You only have text, never
audio, so never claim to know or comment on the candidate's accent.

RULES:
- fluency: comment on their use of fillers (yeah, you know,
  kind of), sentence linking, hesitation patterns
- grammar: quote a specific grammatical error they made
  and show the correction
  Example: 'You said "One of them are" — correct form
  is "One of them is" as the subject is singular.'
- vocabulary: quote a specific basic word they used and
  suggest a better alternative
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
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )
        result = response.choices[0].message.content.strip()
        logging.warning(f"[MISTAKES RESULT] {result[:100]}")
        if result:
            try:
                clean = result.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)
                if all(k in parsed for k in [
                    "fluency", "grammar", "vocabulary", "pronunciation"
                ]):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
    except Exception as e:
        logging.error(f"[MISTAKES/SCORES FAIL] {e}")

    return {
        "fluency": "Work on reducing filler words and improving idea linkage.",
        "grammar": "Review subject-verb agreement and sentence variety.",
        "vocabulary": "Replace basic words with more precise academic vocabulary.",
        "pronunciation": "Focus on natural stress patterns and rhythm.",
        "improvement": "Focus on expanding your answers with specific examples."
    }


QUESTION_MISTAKES_FALLBACK = [
    {"type": "fluency", "original": "", "corrected": "", "explanation": "Reduce hesitation and link ideas more clearly."},
    {"type": "grammar", "original": "", "corrected": "", "explanation": "Review subject-verb agreement and sentence variety."},
    {"type": "vocabulary", "original": "", "corrected": "", "explanation": "Replace basic words with more precise academic vocabulary."},
    {"type": "pronunciation", "original": "", "corrected": "", "explanation": "Work on natural stress and rhythm in connected speech."},
]


def _normalize_for_mistake_comparison(text: str) -> list:
    """Strip punctuation and case, so a purely punctuation/capitalization
    'correction' can be detected and discarded - not left to rely on the
    GPT prompt alone."""
    return re.sub(r"[^\w\s]", "", text).lower().split()


def _validate_question_mistakes(items) -> list:
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
            # This is a transcript of speech, not writing - punctuation and
            # capitalization are transcription artifacts, not something the
            # candidate did wrong. The prompt already instructs GPT not to
            # generate these, but instruction-following isn't guaranteed, so
            # this is a hard backstop: if the only difference between
            # "original" and "corrected" is punctuation/case, discard it.
            if _normalize_for_mistake_comparison(original) == _normalize_for_mistake_comparison(corrected):
                continue
            valid.append({
                "type": item.get("type"),
                "original": original,
                "corrected": corrected,
                "explanation": str(item.get("explanation", "")),
            })
    return valid


def generate_question_mistakes(question: str, answer: str) -> dict:
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
Only flag issues a listener would actually notice when hearing the answer
spoken aloud: wrong verb tense, subject-verb agreement, article/preposition
errors, wrong word choice, awkward word order, filler words, imprecise
vocabulary, and similar genuinely spoken-language issues.

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
  improved version), "explanation" (why it matters, 1 sentence). Use
  exact candidate wording in "original"; keep spacing/punctuation. For
  pronunciation, "original" should be the word/phrase likely
  mispronounced based on word choice and sentence complexity, and
  "corrected" should be brief stress/rhythm guidance.
- "completeness_notice": string, per STEP 1 above.

Return ONLY this structure:
{{
  "mistakes": [
    {{"type": "grammar", "original": "", "corrected": "", "explanation": ""}},
    {{"type": "vocabulary", "original": "", "corrected": "", "explanation": ""}}
  ],
  "completeness_notice": ""
}}
"""
    try:
        logging.warning(f"[QUESTION_MISTAKES] question={question[:50]} answer={answer[:50]}")
        response = safe_gpt_call(
            prompt,
            fallback=None,
            caller=call_gpt,
        )
        parsed = None
        if isinstance(response, dict):
            parsed = response
        elif isinstance(response, str):
            clean = response.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

        if isinstance(parsed, dict):
            mistakes = _validate_question_mistakes(parsed.get("mistakes"))
            notice = str(parsed.get("completeness_notice", "") or "").strip()
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

    Returns a floor from 1.0 (trivial/no real answer) up to 4.5 (clearly
    substantial, complex language). This is a MINIMUM, not a target - the
    model's own score is still used whenever it's already at or above this.
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
    diversity_signal = min(1.0, diversity / 0.55)

    substantiality = (
        length_signal * 0.35
        + complexity_signal * 0.25
        + subordinate_signal * 0.25
        + diversity_signal * 0.15
    )

    return round(1.0 + substantiality * 3.5, 1)


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
    """Average speech_rate_wpm and sum duration_sec across every audio clip in
    a part, so generate_scores() can weigh real pacing evidence (too slow /
    too fast) instead of having zero access to timing data - mirrors
    _aggregate_acoustic_pronunciation() above. Returns None when no timing
    data is available at all."""
    wpm_values = []
    total_duration = 0.0
    for r in raw_part_results or []:
        audio_metrics = r.get("audio_metrics") or {}
        wpm = audio_metrics.get("speech_rate_wpm")
        if isinstance(wpm, (int, float)) and wpm > 0:
            wpm_values.append(float(wpm))
        duration = audio_metrics.get("duration_sec")
        if isinstance(duration, (int, float)) and duration > 0:
            total_duration += float(duration)
    if not wpm_values and total_duration <= 0:
        return None
    return {
        "avg_wpm": (sum(wpm_values) / len(wpm_values)) if wpm_values else None,
        "total_duration_sec": round(total_duration, 1) if total_duration > 0 else None,
    }


def generate_scores(
    part_number: int,
    combined_transcripts: str,
    qas_clean: list | None = None,
    acoustic_pronunciation: float | None = None,
    speech_timing: dict | None = None
) -> dict:
    pacing_section = ""
    if speech_timing:
        avg_wpm = speech_timing.get("avg_wpm")
        total_duration = speech_timing.get("total_duration_sec")
        pacing_lines = []
        if avg_wpm:
            pacing_lines.append(f"- Measured average speech rate: {avg_wpm:.0f} words per minute.")
        if total_duration:
            pacing_lines.append(f"- Measured total spoken duration for this part: {total_duration:.0f} seconds.")
        if pacing_lines:
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
FULL band scale below. Do NOT score Pronunciation; that is assessed
separately from real audio evidence, not from this text. Do not default to
the middle of the range - use whatever band the evidence actually
supports, including bands 8-9 for excellent performance and bands 1-3 for
very poor performance.

Bands available: 1.0 to 9.0 in 0.5 increments.

FLUENCY AND COHERENCE - explicitly assess ALL of these before scoring:
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
- 9: Fluent with only very occasional repetition/self-correction; any
  hesitation is content-related, not a search for words or grammar; fully
  develops topics coherently with fully appropriate cohesive features.
- 8: Fluent with only occasional repetition/self-correction; hesitation is
  usually content-related, rarely to search for language; topic
  development is relevant, appropriate and coherent.
- 7: Speaks at length without noticeable effort or loss of coherence; some
  hesitation, repetition or self-correction; uses a range of connectives
  and discourse markers with some flexibility.
- 6: Willing to speak at length but may lose coherence at times due to
  repetition, self-correction or hesitation; uses connectives and
  discourse markers but not always appropriately.
- 5: Usually maintains flow but relies on repetition, self-correction or
  slow speech to keep going; may over-use certain connectives; simple
  speech is fluent but more complex communication causes problems.
- 4: Cannot keep going without noticeable pauses; slow speech with
  frequent repetition; often self-corrects; links only simple sentences,
  often repetitively; some breakdowns in coherence.
- 3: Frequent, sometimes long, pauses; limited ability to link simple
  sentences; gives only basic responses, cannot develop beyond that.
- 2: Long pauses before nearly every word; isolated words may be
  recognizable but speech has virtually no communicative significance.
- 1: Speech is totally incoherent; no real communication possible.

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
- 9: Total flexibility and precise, natural use of vocabulary in all contexts.
- 8: Wide vocabulary used fluently/flexibly for precise meaning; skilful
  use of less common/idiomatic items despite occasional inaccuracy;
  paraphrases effectively.
- 7: Flexible vocabulary resource to discuss a variety of topics; some
  less common/idiomatic vocabulary with some awareness of style and
  collocation; paraphrases effectively.
- 6: Vocabulary sufficient to discuss topics at length, meaning generally
  clear despite inappropriacies; generally paraphrases successfully.
- 5: Sufficient vocabulary for familiar and unfamiliar topics but limited
  flexibility; attempts paraphrase, not always successful.
- 4: Vocabulary sufficient for familiar topics only, meaning often
  imprecise; rarely attempts paraphrase.
- 3: Vocabulary resource limited, inadequate for unfamiliar topics.
- 2: Very limited resource, essentially reduced to isolated words.
- 1: No rateable vocabulary produced.

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
- 9: Full range of structures used naturally and appropriately;
  consistently accurate apart from native-speaker-like slips.
- 8: Wide range of structures used flexibly; majority of sentences
  error-free; only occasional inappropriacies/non-systematic errors.
- 7: Range of complex structures used with some flexibility; frequent
  error-free sentences, though some mistakes persist.
- 6: Mix of simple and complex structures but limited flexibility; errors
  occur, especially in complex forms, but rarely impede communication.
- 5: Basic sentence forms fairly well controlled; limited range of more
  complex structures attempted, usually with errors, may need reformulation.
- 4: Produces basic sentence forms and some correct simple sentences but
  subordinate structures are rare; overall turns are short.
- 3: Basic sentence forms attempted but grammatical errors are numerous
  except in memorised utterances.
- 2: No evidence of basic sentence forms.
- 1: No rateable language produced.

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

Return ONLY this JSON object, no explanation, no markdown:
{{
  "fluency": 5.0,
  "lexical": 5.0,
  "grammar": 5.0,
  "topic_relevance": "on_topic"
}}

Return only numeric values for the three scores. No strings there.
Return only the JSON object. No markdown.
No ```json fence. No explanation before or after.
Start your response with {{ and end with }}"""

    parsed = None
    try:
        logging.warning(f"[SCORES CALLED] part={part_number}")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800
        )
        result = response.choices[0].message.content.strip()
        logging.warning(f"[SCORES RESULT] {result[:100]}")
        if result:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            candidate = json.loads(clean)
            if all(k in candidate for k in ["fluency", "lexical", "grammar"]):
                parsed = candidate
    except Exception as e:
        logging.error(f"[MISTAKES/SCORES FAIL] {e}")

    if parsed is None:
        parsed = {"fluency": 5.0, "lexical": 5.0, "grammar": 5.0, "topic_relevance": "on_topic"}

    validated = {}
    for k in ["fluency", "lexical", "grammar"]:
        try:
            score = float(parsed.get(k, 5.0))
            validated[k] = score if 1.0 <= score <= 9.0 else 5.0
        except (TypeError, ValueError):
            validated[k] = 5.0

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


def _ielts_round_half_up(x: float) -> float:
    """Official IELTS overall-band rounding: an average ending in .25 rounds
    UP to the next half band, and .75 rounds UP to the next whole band -
    i.e. round-half-up on the doubled value. Python's built-in round() uses
    round-half-to-even, which silently rounds .25/.75 averages DOWN roughly
    half the time (e.g. round(6.25*2)/2 = 6.0, not the correct 6.5)."""
    return math.floor(x * 2 + 0.5) / 2


def calculate_overall_band(
    part1_scores: dict,
    part2_scores: dict,
    part3_scores: dict
) -> float:
    """Combine the three parts' criteria averages into one overall band.

    The IELTS descriptors themselves don't define this step - a real exam
    assigns one holistic band per criterion for the whole test rather than
    scoring parts separately. Since this system DOES score per part (for
    granular feedback), Part 2 (long turn) and Part 3 (discussion) are
    weighted more heavily than Part 1 (small talk), reflecting standard
    IELTS examiner guidance that later parts are more diagnostic of overall
    ability - not a rule taken from the descriptor sheet itself.
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
        (_part_avg(part1_scores), 0.25),
        (_part_avg(part2_scores), 0.35),
        (_part_avg(part3_scores), 0.40),
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












def generate_vocabulary(part: int, combined_transcripts: str) -> list:
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

Generate exactly 5 vocabulary words following these rules:

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

Return ONLY a JSON array. No markdown. No explanation.
Start with [ and end with ]:
[
  {{"word": "example", "meaning": "clear simple explanation
    of meaning in 1 sentence"}},
  {{"word": "example2", "meaning": "clear simple explanation
    of meaning in 1 sentence"}}
]"""

    result = safe_gpt_call(prompt, fallback=None)

    # handle case where result is already a list
    if isinstance(result, list):
        if len(result) >= 3:
            return result[:5]
        else:
            return fallbacks.get(part, VOCAB_FALLBACK_PART1)

    # handle string result
    if isinstance(result, str) and result.strip():
        try:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            vocab_list = json.loads(clean)
            if isinstance(vocab_list, list) and len(vocab_list) >= 3:
                return vocab_list[:5]
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

def _evaluate_speaking_part_audio(audio_bytes: bytes, part: int, question: str = None, questions: str = None, debug: bool = False):

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



    # Cap overly long audio to keep Whisper fast, and reject audio too
    # short/near-empty to contain real speech BEFORE handing it to Whisper.
    # Whisper's mel-spectrogram computation produces a zero-length feature
    # tensor for near-silent/empty input, which crashes deep inside its
    # decoding code with a cryptic tensor-reshape error rather than a clear
    # message - catching it here up front gives a real, actionable error
    # instead ("cannot reshape tensor of 0 elements...").

    try:

        dur_sec = _wav_duration_seconds(wav_path)

        if dur_sec > 90:

            _trim_wav(wav_path, 90)

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



    # Speech rate

    words = len(transcript.split())

    duration = audio_metrics["duration_sec"]

    audio_metrics["speech_rate_wpm"] = round((words / duration) * 60) if duration > 0 else 0



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
        answers = split_transcript_with_gpt(transcript, question_list)

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

        question_result = generate_question_mistakes(q, ans) if ans else {
            "mistakes": [
                {"type": "fluency", "original": "", "corrected": "", "explanation": "No answer to check."},
                {"type": "grammar", "original": "", "corrected": "", "explanation": "No answer to check."},
                {"type": "vocabulary", "original": "", "corrected": "", "explanation": "No answer to check."},
                {"type": "pronunciation", "original": "", "corrected": "", "explanation": "No answer to check."},
            ],
            "completeness_notice": "",
        }

        evaluated_qas.append({

            "question": q,

            "answer": ans,

            "user_answer": ans,

            "result": qa_result,

            "mistakes": question_result.get("mistakes", []),

            "completeness_notice": question_result.get("completeness_notice", ""),

        })



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
        asyncio.to_thread(_evaluate_speaking_part_audio, audio_bytes=audio_bytes, part=1, question=question)
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
                    item = {
                        "question": q_text,
                        "user_answer": answer_text,
                        "mistakes": qa.get("mistakes", []) if isinstance(qa, dict) else [],
                        "completeness_notice": qa.get("completeness_notice", "") if isinstance(qa, dict) else "",
                    }
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
            item = {"question": q_text, "user_answer": user_answer, "mistakes": [], "completeness_notice": ""}
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

    # WAVE 1: vocabulary, scores, and mistakes for all 3 parts are fully
    # independent of each other's results - none of these 9 GPT calls need
    # anything the others produce. Run them all concurrently in a thread
    # pool instead of one after another (this used to be ~9 sequential
    # blocking calls; now it's bounded by the single slowest one).
    vocab_part1, vocab_part2, vocab_part3, p1_scores, p2_scores, p3_scores, p1_feedback, p2_feedback, p3_feedback = await asyncio.gather(
        asyncio.to_thread(generate_vocabulary, 1, p1_combined_transcripts) if p1_combined_transcripts else _immediate(VOCAB_FALLBACK_PART1),
        asyncio.to_thread(generate_vocabulary, 2, p2_combined_transcripts) if p2_combined_transcripts else _immediate(VOCAB_FALLBACK_PART2),
        asyncio.to_thread(generate_vocabulary, 3, p3_combined_transcripts) if p3_combined_transcripts else _immediate(VOCAB_FALLBACK_PART3),
        asyncio.to_thread(generate_scores, 1, p1_feedback_text, part_1_qas_clean, p1_acoustic_pron, p1_speech_timing) if p1_feedback_text else _immediate(dict(_default_scores)),
        asyncio.to_thread(generate_scores, 2, p2_feedback_text, part_2_qas_clean, p2_acoustic_pron, p2_speech_timing) if p2_feedback_text else _immediate(dict(_default_scores)),
        asyncio.to_thread(generate_scores, 3, p3_feedback_text, part_3_qas_clean, p3_acoustic_pron, p3_speech_timing) if p3_feedback_text else _immediate(dict(_default_scores)),
        asyncio.to_thread(generate_mistakes, 1, p1_feedback_text) if p1_feedback_text else _immediate({}),
        asyncio.to_thread(generate_mistakes, 2, p2_feedback_text) if p2_feedback_text else _immediate({}),
        asyncio.to_thread(generate_mistakes, 3, p3_feedback_text) if p3_feedback_text else _immediate({}),
    )

    p1_topic_relevance = p1_scores.pop("topic_relevance", "on_topic")
    p2_topic_relevance = p2_scores.pop("topic_relevance", "on_topic")
    p3_topic_relevance = p3_scores.pop("topic_relevance", "on_topic")

    p1_pronunciation_source = p1_scores.pop("pronunciation_source", "unavailable")
    p2_pronunciation_source = p2_scores.pop("pronunciation_source", "unavailable")
    p3_pronunciation_source = p3_scores.pop("pronunciation_source", "unavailable")

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
        asyncio.to_thread(generate_ideal_band9_answer, 1, _extract_questions(part_1_qas_clean))
        if p1_topic_relevance in _OFF_TOPIC_TIERS
        else (asyncio.to_thread(generate_band9_answer, 1, p1_combined_context, answers_only=p1_answers_only) if p1_combined_context else _immediate(p1_answers_only)),

        asyncio.to_thread(generate_ideal_band9_answer, 2, _extract_questions(part_2_qas_clean))
        if p2_topic_relevance in _OFF_TOPIC_TIERS
        else (asyncio.to_thread(generate_band9_answer, 2, p2_combined_context, answers_only=p2_answers_only) if p2_combined_context else _immediate(p2_answers_only)),

        asyncio.to_thread(generate_ideal_band9_answer, 3, _extract_questions(part_3_qas_clean))
        if p3_topic_relevance in _OFF_TOPIC_TIERS
        else (asyncio.to_thread(generate_band9_answer, 3, p3_combined_context, answers_only=p3_answers_only) if p3_combined_context else _immediate(p3_answers_only)),
    )

    p1_completeness_notices = _collect_completeness_notices(part_1_qas_clean)
    p2_completeness_notices = _collect_completeness_notices(part_2_qas_clean)
    p3_completeness_notices = _collect_completeness_notices(part_3_qas_clean)

    p1_feedback = _apply_relevance_to_feedback(p1_feedback, p1_topic_relevance)
    p2_feedback = _apply_relevance_to_feedback(p2_feedback, p2_topic_relevance)
    p3_feedback = _apply_relevance_to_feedback(p3_feedback, p3_topic_relevance)

    p1_feedback = _apply_completeness_to_feedback(p1_feedback, p1_completeness_notices)
    p2_feedback = _apply_completeness_to_feedback(p2_feedback, p2_completeness_notices)
    p3_feedback = _apply_completeness_to_feedback(p3_feedback, p3_completeness_notices)

    part_1 = {

        "questions": part_1_qas_clean,

        "scores": p1_scores,

        "mistakes": {
            "fluency": p1_feedback.get("fluency", ""),
            "grammar": p1_feedback.get("grammar", ""),
            "vocabulary": p1_feedback.get("vocabulary", ""),
            "pronunciation": p1_feedback.get("pronunciation", "")
        },

        "improvement": p1_feedback.get("improvement", "Focus on expanding your answers with specific examples."),

        "band9_answer": band9_part1,

        "vocabulary_to_learn": vocab_part1,

        "relevance_notice": RELEVANCE_NOTICE_MESSAGES.get(p1_topic_relevance),

        "completeness_notice": " ".join(p1_completeness_notices) or None,

        "topic_relevance": p1_topic_relevance,

        "pronunciation_source": p1_pronunciation_source,

        "not_attempted": p1_not_attempted,

    }

    part_2 = {

        "questions": part_2_qas_clean,

        "scores": p2_scores,

        "mistakes": {
            "fluency": p2_feedback.get("fluency", ""),
            "grammar": p2_feedback.get("grammar", ""),
            "vocabulary": p2_feedback.get("vocabulary", ""),
            "pronunciation": p2_feedback.get("pronunciation", "")
        },

        "improvement": p2_feedback.get("improvement", "Focus on expanding your answers with specific examples."),

        "band9_answer": band9_part2,

        "vocabulary_to_learn": vocab_part2,

        "relevance_notice": RELEVANCE_NOTICE_MESSAGES.get(p2_topic_relevance),

        "completeness_notice": " ".join(p2_completeness_notices) or None,

        "topic_relevance": p2_topic_relevance,

        "pronunciation_source": p2_pronunciation_source,

        "not_attempted": p2_not_attempted,

    }

    part_3 = {

        "questions": part_3_qas_clean,

        "scores": p3_scores,

        "mistakes": {
            "fluency": p3_feedback.get("fluency", ""),
            "grammar": p3_feedback.get("grammar", ""),
            "vocabulary": p3_feedback.get("vocabulary", ""),
            "pronunciation": p3_feedback.get("pronunciation", "")
        },

        "improvement": p3_feedback.get("improvement", "Focus on expanding your answers with specific examples."),

        "band9_answer": band9_part3,

        "vocabulary_to_learn": vocab_part3,

        "relevance_notice": RELEVANCE_NOTICE_MESSAGES.get(p3_topic_relevance),

        "completeness_notice": " ".join(p3_completeness_notices) or None,

        "topic_relevance": p3_topic_relevance,

        "pronunciation_source": p3_pronunciation_source,

        "not_attempted": p3_not_attempted,

    }



    recalculated_overall = calculate_overall_band(
        part_1.get("scores", {}),
        part_2.get("scores", {}),
        part_3.get("scores", {})
    )

    final_response = {

        "overall_band": recalculated_overall,

        "part_1": part_1,

        "part_2": part_2,

        "part_3": part_3

    }



    logger.info({

        "mode": "part_wise",

        "questions": len(questions)

    })



    return final_response
