from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from utils.audio_normalizer import normalize_to_wav

from utils.audio_transcriber import transcribe_audio

from utils.gpt_client import call_gpt

from utils.safety import safe_gpt_call, normalize_feedback

from evaluators.speaking import (

    evaluate_speaking_part,

    compute_pronunciation_score,

    evaluate_speaking,

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

import random

import hashlib

import re

from difflib import SequenceMatcher

import whisper

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


def round_to_ielts_band(score):

    try:

        if score is None:

            return 5.0

        return round(score * 2) / 2

    except Exception:

        return 5.0


def normalize_summary_bands(summary):

    if not isinstance(summary, dict):

        return {}

    def round_band(x):

        try:

            return round(float(x) * 2) / 2

        except Exception:

            return 5.0

    return {

        "fluency": round_band(summary.get("fluency")),

        "lexical": round_band(summary.get("lexical")),

        "grammar": round_band(summary.get("grammar")),

        "pronunciation": round_band(summary.get("pronunciation")),

        "feedback": summary.get("feedback", {}),

        "vocabulary_feedback": summary.get("vocabulary_feedback", {}),

    }


# Lightweight in-process rate limiting (best-effort, per-worker)

_REQUEST_COUNT = 0

_RATE_LIMIT_THRESHOLD = 500





def _check_rate_limit():

    global _REQUEST_COUNT

    _REQUEST_COUNT += 1

    return _REQUEST_COUNT <= _RATE_LIMIT_THRESHOLD





# In-memory caches to avoid repeated ASR/feature work within a process

_ASR_CACHE = {}

_FEATURE_CACHE = {}

_USED_VOCAB = set()



# Load Whisper once per process (CPU-friendly)

try:

    # Use a faster Whisper variant to reduce latency; switch to "tiny" if extreme speed is needed.

    WHISPER_MODEL = whisper.load_model("small")

except Exception as exc:

    WHISPER_MODEL = None

    print({"event": "whisper_load_failed", "error": str(exc)})



router = APIRouter(prefix="/speaking", tags=["Speaking"])


@router.post("/part/{part}/audio")
async def evaluate_part_audio_compat(
    part: int,
    file: UploadFile = File(...),
    attempt_id: str | None = Form(None),
    question: str | None = Form(None),
):
    """
    Backward-compatible endpoint for existing clients that evaluate speaking part-wise.
    This keeps old request/response shape while running the new audio pipeline.
    """
    if part not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Invalid part number")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio file provided")

    default_question_by_part = {
        1: "Tell me about yourself.",
        2: "Describe an event you remember well.",
        3: "What are your views on this topic and why?",
    }

    part_result = await _evaluate_speaking_part_audio(
        audio_bytes=audio_bytes,
        part=part,
        question=(question or default_question_by_part.get(part)),
    )

    if not isinstance(part_result, dict):
        raise HTTPException(status_code=500, detail="Unexpected evaluator response")

    if part_result.get("error") and not part_result.get("result"):
        raise HTTPException(status_code=400, detail=part_result.get("error"))

    result = part_result.get("result") or {}
    if not isinstance(result, dict):
        result = {}

    result = sanitize_result(result)

    if "overall_band" not in result:
        fluency = float(result.get("fluency", 0) or 0)
        lexical = float(result.get("lexical", 0) or 0)
        grammar = float(result.get("grammar", 0) or 0)
        pronunciation = float(result.get("pronunciation", 0) or 0)
        result["overall_band"] = round_to_ielts_band((fluency + lexical + grammar + pronunciation) / 4)

    return {
        "attempt_id": (attempt_id or uuid.uuid4().hex),
        "part": part,
        "result": result,
        "transcript": part_result.get("transcript", ""),
        "audio_metrics": part_result.get("audio_metrics", {}),
    }





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
You are given an IELTS speaking response.

Your task:
Split the response into separate answers for EACH question.

IMPORTANT:
- Number of answers MUST equal number of questions
- Each answer should correspond to the most relevant part of the transcript
- Keep answers concise but meaningful
- Do NOT merge answers
- Do NOT skip any question

Questions:
{questions}

Transcript:
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


def generate_band9_answer(part_number: int, combined_with_context: str, answers_only: str | None = None) -> str:
    overlap_text = answers_only or combined_with_context

    part_instructions = {
        1: "Write 4-6 concise sentences. Remove all fillers. Be direct and confident.",
        2: "Write 8-10 sentences with clear structure: opening, 3 developed points, conclusion. Use: Furthermore, In addition, To conclude.",
        3: "Write 6-8 analytical sentences. Use: From a broader perspective, One could argue that, It is worth noting that."
    }

    prompt = f"""You are an IELTS Band 9 speaking examiner.

A student gave these answers in IELTS Speaking Part {part_number}:

{combined_with_context}

Rewrite as ONE fluent Band 9 response.

RULES:
- Fix ALL grammar errors
- Remove fillers: yeah, you know, kind of, sort of
- Replace basic words: help->facilitate, a lot of->a wide array of, get over->overcome, big->substantial
- Use different sentence structures
- Do not repeat "In my opinion" more than once
- {part_instructions.get(part_number, part_instructions[1])}

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

Output the rewritten answer only. No labels. No explanation."""

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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

        return result

    except Exception as e:
        logging.error(f"[BAND9 DIRECT CALL FAILED] {e}")
        return answers_only if answers_only else combined_with_context


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
  "pronunciation": "specific feedback based on their word choices and sentence patterns",
  "improvement": "one specific actionable tip for THIS student based on their actual answers"
}}

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
- pronunciation: based on their vocabulary choices and
  sentence complexity, comment on likely stress and
  rhythm patterns they should work on
- Keep each feedback to 2 sentences maximum
- Be specific, not generic
- Do NOT return template sentences
CRITICAL RULES FOR EVERY CRITERION:
- Never say the student "demonstrated good" anything
- Never say the student "performed well" on anything
- Never use words like: good, great, excellent, well done,
  impressive, effectively, successfully
- Every criterion MUST contain one specific improvement point
- If the student did something well, still find ONE small
  thing they can improve and mention that instead
- Format: quote their exact words → explain the issue
  → show the correct version or better alternative
- Keep each feedback to 2 sentences maximum
Return only the JSON object. No markdown.
No ```json fence. No explanation before or after.
Start your response with {{ and end with }}"""

    try:
        logging.warning(f"[MISTAKES CALLED] part={part_number}")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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


def generate_scores(
    part_number: int,
    combined_transcripts: str
) -> dict:
    prompt = f"""You are a certified IELTS Speaking examiner.

A student gave these answers in Part {part_number}:

{combined_transcripts}

Score this student on the official IELTS 4 criteria.
Use IELTS band descriptors strictly.
Bands available: 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0

Scoring guidance:
- fluency: 4-5 if hesitant with fillers, 5.5-6 if mostly
  fluent with some pausing, 6.5-7 if smooth and natural
- lexical: 4-5 if basic everyday words only, 5.5-6 if
  some range with occasional precise words, 6.5-7 if
  varied and mostly precise vocabulary
- grammar: 4-5 if frequent errors, 5.5-6 if errors do
  not impede communication, 6.5-7 if mostly accurate
  with good range of structures
- pronunciation: 4-5 if effort needed to understand,
  5.5-6 if generally clear with some L1 influence,
  6.5-7 if easy to understand throughout

Part-specific notes:
- Part 1 answers are typically shorter - mark accordingly
- Part 2 long-turn answers should reward structure
  and development with higher fluency/lexical scores
  if demonstrated
- Part 3 answers should reward analytical depth

Return ONLY this JSON object, no explanation, no markdown:
{{
  "fluency": 5.0,
  "lexical": 5.0,
  "grammar": 5.0,
  "pronunciation": 5.0
}}

Return only numeric values. No strings. No explanation.
Return only the JSON object. No markdown.
No ```json fence. No explanation before or after.
Start your response with {{ and end with }}"""

    try:
        logging.warning(f"[SCORES CALLED] part={part_number}")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )
        result = response.choices[0].message.content.strip()
        logging.warning(f"[SCORES RESULT] {result[:100]}")
        if result:
            try:
                clean = result.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)
                if all(k in parsed for k in [
                    "fluency", "lexical", "grammar", "pronunciation"
                ]):
                    validated = {}
                    for k, v in parsed.items():
                        try:
                            score = float(v)
                            if 4.0 <= score <= 9.0:
                                validated[k] = score
                            else:
                                validated[k] = 5.0
                        except (TypeError, ValueError):
                            validated[k] = 5.0
                    return validated
            except (json.JSONDecodeError, ValueError):
                pass
    except Exception as e:
        logging.error(f"[MISTAKES/SCORES FAIL] {e}")

    return {
        "fluency": 5.0,
        "lexical": 5.0,
        "grammar": 5.0,
        "pronunciation": 5.0
    }


def calculate_overall_band(
    part1_scores: dict,
    part2_scores: dict,
    part3_scores: dict
) -> float:

    all_scores = []
    for scores in [part1_scores, part2_scores, part3_scores]:
        if scores:
            all_scores.extend([
                scores.get("fluency", 5.0),
                scores.get("lexical", 5.0),
                scores.get("grammar", 5.0),
                scores.get("pronunciation", 5.0)
            ])

    if not all_scores:
        return 5.0

    avg = sum(all_scores) / len(all_scores)

    return round(avg * 2) / 2





def refine_feedback(summary):

    if not summary or not isinstance(summary, dict):

        return summary



    improvements = summary.get("feedback", {}).get("improvements", "")

    prompt = f"""

You are an IELTS examiner.



Improve this feedback to sound natural and professional.



Make it:

- concise

- realistic

- human-like



Feedback:

{improvements}



Return improved version only.

"""



    res = safe_gpt_call(prompt, fallback=improvements, caller=call_gpt)

    if res:

        summary.setdefault("feedback", {})

        summary["feedback"]["improvements"] = normalize_feedback(str(res))



    return summary







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

async def _evaluate_speaking_part_audio(audio_bytes: bytes, part: int, question: str = None, questions: str = None, debug: bool = False):

    part_start = time.time()

    # Validate raw bytes

    if not audio_bytes or len(audio_bytes) < 1000:

        print({"event": "audio_error", "part": part, "reason": "invalid_audio_bytes"})

        return {

            "part": part,

            "error": "invalid_audio",

            "message": "Uploaded audio is empty or too small",

            "transcript": "",

            "audio_metrics": {},

            "result": None,

            "processing_time": round(time.time() - part_start, 3)

        }

    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    # Convert to wav once using in-memory bytes

    dummy_upload = type("DummyUpload", (), {"file": io.BytesIO(audio_bytes)})

    wav_path = normalize_to_wav(dummy_upload)



    if (not os.path.exists(wav_path)) or (os.path.getsize(wav_path) < 2000):

        print({"event": "audio_error", "part": part, "reason": "conversion_failed"})

        return {

            "part": part,

            "error": "conversion_failed",

            "message": "Audio conversion failed or produced empty file",

            "transcript": "",

            "audio_metrics": {},

            "result": None,

            "processing_time": round(time.time() - part_start, 3)

        }



    # Cap overly long audio to keep Whisper fast

    try:

        dur_sec = _wav_duration_seconds(wav_path)

        if dur_sec > 90:

            _trim_wav(wav_path, 90)

    except Exception as exc:

        print({"event": "audio_warn", "part": part, "reason": "duration_check_failed", "details": str(exc)})



    if audio_hash in _ASR_CACHE:

        transcript = _ASR_CACHE[audio_hash]

    else:

        if WHISPER_MODEL is None:

            raise RuntimeError("Whisper model not available; install dependencies.")

        try:

            print({"event": "transcription_start", "part": part})

            transcript = WHISPER_MODEL.transcribe(

                wav_path,

                fp16=False,

                verbose=False

            )["text"]

        except Exception as exc:

            print({"event": "audio_error", "part": part, "reason": "transcription_failed", "details": str(exc)})

            return {

                "part": part,

                "error": "transcription_failed",

                "details": str(exc),

                "transcript": "",

                "audio_metrics": {},

                "result": None,

                "processing_time": round(time.time() - part_start, 3)

            }

        _ASR_CACHE[audio_hash] = transcript



    if audio_hash in _FEATURE_CACHE:

        audio_metrics = _FEATURE_CACHE[audio_hash].copy()

    else:

        audio_metrics = extract_acoustic_features(wav_path, transcript)

        _FEATURE_CACHE[audio_hash] = audio_metrics.copy()



    if not transcript or not transcript.strip():

        print({"event": "audio_error", "part": part, "reason": "no_speech_detected"})

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

    audio_metrics["asr_confidence"] = 0.9 if transcript.strip() else 0.6

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

            print({"event": "question_parse_failed", "details": str(exc), "raw": questions})



    if not question_list and clean_question:

        question_list = [clean_question]



    # Remove placeholder values like "string"

    question_list = [q.strip().lstrip(". ") for q in question_list if str(q).strip().lower() != "string"]



    answers = split_transcript_with_gpt(transcript, question_list) if question_list else [transcript]

    if len(answers) != len(question_list):

        answers = [transcript] * len(question_list) if question_list else [transcript]

    while len(answers) < len(question_list):

        answers.append("")



    print({

        "part": part,

        "question_used": clean_question if not question_list else "multiple",

        "questions": len(question_list),

        "answers": len(answers),

    })



    evaluated_qas = []

    for idx, q in enumerate(question_list):

        ans = answers[idx] if idx < len(answers) else ""

        eval_text = f"Question: {q}\nAnswer: {ans}"

        qa_result = evaluate_speaking_part(
            part=part,
            transcript=eval_text,
            audio_metrics=audio_metrics
        )

        try:
            qa_result = sanitize_result(qa_result)
        except Exception as e:
            print("sanitize_result error:", e)
            qa_result = {}

        if not qa_result:
            qa_result = {}

        evaluated_qas.append({

            "question": q,

            "answer": ans,

            "result": qa_result

        })



    if question_list:

        combined_eval_text = "\n\n".join(

            [f"Question: {q}\nAnswer: {a}" for q, a in zip(question_list, answers)]

        )

        # Reuse single QA evaluation when only one question is present

        if len(evaluated_qas) == 1:

            result = evaluated_qas[0]["result"]

        else:

            result = evaluate_speaking_part(

                part=part,

                transcript=combined_eval_text,

                audio_metrics=audio_metrics

            )

            result = sanitize_result(result)

    else:

        combined_eval_text = f"Question: {clean_question}\nAnswer: {transcript}" if clean_question else transcript

        result = evaluate_speaking_part(

            part=part,

            transcript=combined_eval_text,

            audio_metrics=audio_metrics

        )

        result = sanitize_result(result)



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

):

    test_result = safe_gpt_call("Say the word HELLO only.")
    print(f"[GPT TEST] result={test_result}")

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



    filtered = [

        (a, q) for a, q in zip(audios, questions)

        if a is not None and q is not None

    ]



    if len(filtered) == 0:

        return {"error": "no_audio_provided"}

    if len(filtered) > 15:

        return {"error": "max_15_questions_allowed"}



    results = []

    for audio_file, question in filtered:

        audio_bytes = await audio_file.read()

        part_result = await _evaluate_speaking_part_audio(

            audio_bytes=audio_bytes,

            part=1,

            question=question

        )

        if isinstance(part_result, dict):

            pr = part_result.get("result")

            if pr:

                try:
                    part_result["result"] = sanitize_result(pr)
                except Exception as e:
                    print("sanitize_result error:", e)
                    part_result["result"] = {}

            if not part_result.get("result"):
                part_result["result"] = {}



        results.append({

            "question": question,

            "transcript": part_result.get("transcript"),

            "result": part_result.get("result")

        })



    results = [r for r in results if r.get("question") and str(r["question"]).strip().lower() != "string"]



    scored_results = [r["result"] for r in results if isinstance(r.get("result"), dict)]

    if scored_results:

        avg_fluency = sum(r.get("fluency", 0) for r in scored_results) / len(scored_results)

        avg_lexical = sum(r.get("lexical", 0) for r in scored_results) / len(scored_results)

        avg_grammar = sum(r.get("grammar", 0) for r in scored_results) / len(scored_results)

        avg_pronunciation = sum(r.get("pronunciation", 0) for r in scored_results) / len(scored_results)

        raw_overall = (avg_fluency + avg_lexical + avg_grammar + avg_pronunciation) / 4

        try:
            overall_band = round_to_ielts_band(raw_overall)
        except Exception as e:
            print("band rounding error:", e)
            overall_band = 5.0

    else:

        overall_band = None



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

    def _clean_result(r):
        return {
            "question": str(r.get("question", "")).strip(),
            "user_answer": str(r.get("transcript", "")).strip()
        }

    part_1_qas = []
    part_2_qas = []
    part_3_qas = []
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

    part_1_qas_clean = [_clean_result(r) for r in part_1_qas]
    part_2_qas_clean = [_clean_result(r) for r in part_2_qas]
    part_3_qas_clean = [_clean_result(r) for r in part_3_qas]

    try:
        part_1_summary = normalize_summary_bands(refine_feedback(_aggregate_part(part_1_qas)))
    except Exception as e:
        print("normalize_summary_bands error:", e)
        part_1_summary = {}

    try:
        part_2_summary = normalize_summary_bands(refine_feedback(_aggregate_part(part_2_qas)))
    except Exception as e:
        print("normalize_summary_bands error:", e)
        part_2_summary = {}

    try:
        part_3_summary = normalize_summary_bands(refine_feedback(_aggregate_part(part_3_qas)))
    except Exception as e:
        print("normalize_summary_bands error:", e)
        part_3_summary = {}



    for summary in [part_1_summary, part_2_summary, part_3_summary]:

        if isinstance(summary, dict):

            summary["fluency"] = round_to_ielts_band(summary.get("fluency"))
            summary["lexical"] = round_to_ielts_band(summary.get("lexical"))
            summary["grammar"] = round_to_ielts_band(summary.get("grammar"))
            summary["pronunciation"] = round_to_ielts_band(summary.get("pronunciation"))

            fb = summary.get("feedback", {})

            if isinstance(fb, dict) and fb.get("improvements"):

                fb["improvements"] = clean_feedback(fb.get("improvements", ""))

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

    band9_part1 = generate_band9_answer(1, p1_combined_context, answers_only=p1_answers_only) if p1_combined_context else p1_answers_only
    band9_part2 = generate_band9_answer(2, p2_combined_context, answers_only=p2_answers_only) if p2_combined_context else p2_answers_only
    band9_part3 = generate_band9_answer(3, p3_combined_context, answers_only=p3_answers_only) if p3_combined_context else p3_answers_only

    vocab_part1 = generate_vocabulary(1, p1_combined_transcripts) if p1_combined_transcripts else VOCAB_FALLBACK_PART1
    vocab_part2 = generate_vocabulary(2, p2_combined_transcripts) if p2_combined_transcripts else VOCAB_FALLBACK_PART2
    vocab_part3 = generate_vocabulary(3, p3_combined_transcripts) if p3_combined_transcripts else VOCAB_FALLBACK_PART3

    def _combined_for_feedback(qas_clean):
        return "\n\n".join(
            [
                f"Q: {qa.get('question', '')}\nA: {qa.get('user_answer', '')}"
                for qa in qas_clean
                if qa.get("question") or qa.get("user_answer")
            ]
        ).strip()

    p1_feedback_text = _combined_for_feedback(part_1_qas_clean)
    p2_feedback_text = _combined_for_feedback(part_2_qas_clean)
    p3_feedback_text = _combined_for_feedback(part_3_qas_clean)

    p1_scores = generate_scores(1, p1_feedback_text) if p1_feedback_text else {"fluency": 5.0, "lexical": 5.0, "grammar": 5.0, "pronunciation": 5.0}
    p2_scores = generate_scores(2, p2_feedback_text) if p2_feedback_text else {"fluency": 5.0, "lexical": 5.0, "grammar": 5.0, "pronunciation": 5.0}
    p3_scores = generate_scores(3, p3_feedback_text) if p3_feedback_text else {"fluency": 5.0, "lexical": 5.0, "grammar": 5.0, "pronunciation": 5.0}

    p1_feedback = generate_mistakes(1, p1_feedback_text) if p1_feedback_text else {}
    p2_feedback = generate_mistakes(2, p2_feedback_text) if p2_feedback_text else {}
    p3_feedback = generate_mistakes(3, p3_feedback_text) if p3_feedback_text else {}

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



    print({

        "mode": "part_wise",

        "questions": len(questions)

    })



    return final_response
