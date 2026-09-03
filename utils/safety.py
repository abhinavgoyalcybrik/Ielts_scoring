import logging
import re
from typing import Any, Callable

from utils.gpt_client import call_gpt as _default_call_gpt


def safe_gpt_call(
    prompt: str,
    fallback: Any = None,
    caller: Callable[[str], Any] | None = None,
    retries: int = 2,
    on_failure: Callable[[Exception], None] | None = None,
):
    """
    Centralized GPT guard rail with retry + logging.
    - Executes the provided caller (defaults to utils.gpt_client.call_gpt)
    - Treats empty/short responses as failures
    - Retries up to `retries` times before returning fallback

    on_failure (optional): called with the last exception right before the
    fallback is returned, ONLY when every retry failed - lets a caller
    record WHY a fallback fired (e.g. to distinguish a refusal from a
    plain parse error in evaluate_writing()'s ai_evaluation_failed_reason)
    without changing this function's return value or any existing
    caller's behaviour. Default None: every existing call site is
    completely unaffected."""
    func = caller or _default_call_gpt
    retries = max(1, int(retries)) if retries is not None else 1
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            res = func(prompt)

            # Validate presence / length
            if res is None:
                raise ValueError("Empty GPT")

            if isinstance(res, str):
                res = res.strip()
                if len(res) < 10:
                    raise ValueError("Empty GPT")
                logging.warning(f"[GPT OK] attempt={attempt} length={len(res.split())} preview={res[:80]}")
                return res

            # For structured outputs, ensure not empty
            if hasattr(res, "__len__") and len(res) == 0:
                raise ValueError("Empty GPT")

            logging.warning(f"[GPT OK] attempt={attempt} type={type(res)} length={len(res) if hasattr(res, '__len__') else 'NA'}")
            return res

        except Exception as e:  # pragma: no cover - defensive logging
            last_error = e
            logging.error(f"[GPT FAIL] attempt={attempt}/{retries} error={e}")

    # All attempts failed
    logging.error(f"[GPT FALLBACK] returning fallback value after error={last_error}")
    if on_failure is not None and last_error is not None:
        try:
            on_failure(last_error)
        except Exception:
            pass  # never let a diagnostic callback break the fallback path itself
    return fallback


def normalize_feedback(text: str) -> str:
    """
    Make feedback concise, de-duplicated, and capped to three sentences.
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Split only where sentence-ending punctuation is followed by whitespace
    # and a capital letter/quote - a bare `[.!?]` split breaks on every
    # period, including ones inside abbreviations like "e.g." or "vs."
    # (real GPT feedback text uses both), which used to chop the sentence
    # apart mid-abbreviation and silently truncate it to a garbled fragment.
    parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text) if s.strip()]
    parts = list(dict.fromkeys(parts))

    if not parts:
        return ""

    normalized = " ".join(parts[:3]).strip()
    if not re.search(r"[.!?]$", normalized):
        normalized += "."
    return normalized


def safe_output(value, fallback):
    """
    Prevent UI-breaking None/empty values.
    """
    return value if value else fallback
