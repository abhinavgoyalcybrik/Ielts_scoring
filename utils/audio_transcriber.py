import os
import re
import threading
from pathlib import Path
from typing import Optional

from utils.audio_normalizer import normalize_to_wav

_MODEL = None
_MODEL_LOCK = threading.Lock()

# faster-whisper's CTranslate2 backend is generally more thread-friendly than
# openai-whisper's raw PyTorch decode path (which is what caused production
# errors like "cannot reshape tensor of 0 elements..." under concurrent
# calls on a shared model instance), but there's no documented guarantee that
# a single WhisperModel is safe for concurrent transcribe() calls from
# separate threads - faster-whisper's own docs point toward multiple model
# instances or its internal multi-threading rather than confirming this.
# Keeping the same conservative lock as before: it serializes only the
# actual inference call, while ffmpeg conversion, feature extraction, and
# GPT calls for OTHER clips still proceed concurrently in their own threads.
_TRANSCRIBE_LOCK = threading.Lock()


def _load_model():
    global _MODEL
    if _MODEL is None:
        # Multiple audio clips are transcribed concurrently from a thread
        # pool - without this lock, several threads could all see _MODEL as
        # None simultaneously and each load their own full model instance in
        # parallel (wasteful, and briefly spikes memory well past what's
        # needed).
        with _MODEL_LOCK:
            if _MODEL is None:
                try:
                    from faster_whisper import WhisperModel  # type: ignore
                except ImportError as exc:
                    raise RuntimeError(
                        "faster-whisper is required for transcription. Install faster-whisper in the active environment."
                    ) from exc
                # No CUDA/GPU anywhere in this deployment - int8 is
                # faster-whisper's recommended CPU compute type, trading a
                # small amount of numerical precision for the speed that
                # makes running the full-accuracy large-v3 model (rather than
                # a distilled/smaller one) practical on CPU at all.
                _MODEL = WhisperModel("large-v3", device="cpu", compute_type="int8")
    return _MODEL


def _clean_transcript(text: str) -> str:
    if not text:
        return ""

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Remove repeated words produced by ASR stuttering or recognition artifacts.
    text = re.sub(r"\b([A-Za-z]+)(?:\s+\1)+\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = " ".join(text.split())

    return text.strip()


def _build_initial_prompt(question: Optional[str]) -> str:
    if not question:
        return ""
    cleaned = str(question).strip()
    if not cleaned:
        return ""
    return cleaned.replace("\n", " ").strip()


def _estimate_confidence(segments: list, has_text: bool) -> float:
    """Derive a genuine ASR confidence from faster-whisper's own per-segment
    output (avg_logprob, no_speech_prob) instead of guessing. avg_logprob
    is a mean log-probability per segment (roughly 0 = very confident,
    more negative = less confident); no_speech_prob flags segments the
    model itself thinks are silence/noise rather than real speech."""
    logprobs = [s.avg_logprob for s in segments if getattr(s, "avg_logprob", None) is not None]
    no_speech_probs = [s.no_speech_prob for s in segments if getattr(s, "no_speech_prob", None) is not None]

    if not logprobs:
        return 0.5 if has_text else 0.0

    mean_logprob = sum(logprobs) / len(logprobs)
    mean_no_speech = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0

    confidence = max(0.0, min(1.0, 1.0 + mean_logprob))
    confidence *= (1.0 - mean_no_speech)

    return round(max(0.0, min(1.0, confidence)), 3)


def transcribe_audio(audio_source, question: Optional[str] = None, language: str = "en", return_confidence: bool = False):
    if isinstance(audio_source, (str, os.PathLike)):
        wav_path = str(audio_source)
    else:
        wav_path = normalize_to_wav(audio_source)

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    model = _load_model()

    initial_prompt = _build_initial_prompt(question)

    transcribe_kwargs = {
        "language": language,
        "task": "transcribe",
        "initial_prompt": initial_prompt or None,
        "condition_on_previous_text": False,
        # A single fixed temperature disables the quality fallback: normally,
        # if the temperature=0 decode looks bad (too repetitive per
        # compression_ratio_threshold, or too low-confidence per
        # log_prob_threshold), the model automatically retries at a higher
        # temperature instead of returning the bad first attempt. The tuple
        # keeps that retry behavior active for garbled/noisy/accented audio.
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "beam_size": 5,
        "best_of": 5,
        "no_speech_threshold": 0.5,
        # Detects and skips silent stretches before decoding instead of
        # feeding dead air to the model - reduces the chance of hallucinated
        # text over silence (e.g. a clip where recording started before the
        # candidate began speaking).
        "vad_filter": True,
    }

    with _TRANSCRIBE_LOCK:
        try:
            segments_iter, _info = model.transcribe(wav_path, **transcribe_kwargs)
        except TypeError:
            segments_iter, _info = model.transcribe(
                wav_path,
                language=language,
                task="transcribe",
                initial_prompt=initial_prompt or None,
                condition_on_previous_text=False,
            )
        segments = list(segments_iter)

    text = _clean_transcript("".join(seg.text for seg in segments))

    if not return_confidence:
        return text

    return text, _estimate_confidence(segments, bool(text.strip()))
