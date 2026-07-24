import os
import re
import threading
from pathlib import Path
from typing import Optional

from utils.audio_normalizer import normalize_to_wav

_MODEL = None
_MODEL_LOCK = threading.Lock()

# openai-whisper's model.transcribe() is NOT thread-safe for concurrent
# calls on the same shared model instance - confirmed in production by
# errors like "cannot reshape tensor of 0 elements into shape [5, 0, 12,
# -1]" (and even a raw PyTorch layer repr leaking out as an "error
# message") when multiple audio clips were transcribed at the same time
# from a thread pool. This serializes actual inference calls so only one
# runs at a time, while still letting ffmpeg conversion, librosa feature
# extraction, and GPT calls for OTHER clips proceed concurrently in their
# own threads - most of the parallelization benefit is preserved, just not
# for this one non-thread-safe step.
_TRANSCRIBE_LOCK = threading.Lock()


def _load_model():
    global _MODEL
    if _MODEL is None:
        # Multiple audio clips are now transcribed concurrently from a
        # thread pool - without this lock, several threads could all see
        # _MODEL as None simultaneously and each load their own full
        # Whisper model instance in parallel (wasteful, and briefly spikes
        # memory well past what's needed).
        with _MODEL_LOCK:
            if _MODEL is None:
                try:
                    import whisper  # type: ignore
                except ImportError as exc:
                    raise RuntimeError(
                        "Whisper is required for transcription. Install openai-whisper in the active environment."
                    ) from exc
                _MODEL = whisper.load_model("small")
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


def _estimate_confidence(result: dict, has_text: bool) -> float:
    """Derive a genuine ASR confidence from Whisper's own per-segment
    output (avg_logprob, no_speech_prob) instead of guessing. avg_logprob
    is a mean log-probability per segment (roughly 0 = very confident,
    more negative = less confident); no_speech_prob flags segments Whisper
    itself thinks are silence/noise rather than real speech."""
    segments = result.get("segments") or []
    logprobs = [s.get("avg_logprob") for s in segments if isinstance(s, dict) and s.get("avg_logprob") is not None]
    no_speech_probs = [s.get("no_speech_prob") for s in segments if isinstance(s, dict) and s.get("no_speech_prob") is not None]

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
        "fp16": False,
        "initial_prompt": initial_prompt or None,
        "condition_on_previous_text": False,
        "verbose": False,
        "temperature": 0.0,
        "beam_size": 5,
        "best_of": 5,
        "no_speech_threshold": 0.5,
    }

    with _TRANSCRIBE_LOCK:
        try:
            result = model.transcribe(wav_path, **transcribe_kwargs)
        except TypeError:
            result = model.transcribe(
                wav_path,
                language=language,
                fp16=False,
                initial_prompt=initial_prompt or None,
                condition_on_previous_text=False,
                verbose=False,
            )

    text = _clean_transcript(result.get("text", ""))

    if not return_confidence:
        return text

    return text, _estimate_confidence(result, bool(text.strip()))
