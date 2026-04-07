import logging
import re
from typing import Any, Callable

from utils.gpt_client import call_gpt as _default_call_gpt


def safe_gpt_call(
    prompt: str,
    fallback: Any = None,
    caller: Callable[[str], Any] | None = None,
    retries: int = 2,
):
    """
    Centralized GPT guard rail with retry + logging.
    - Executes the provided caller (defaults to utils.gpt_client.call_gpt)
    - Treats empty/short responses as failures
    - Retries up to `retries` times before returning fallback
    """
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
    return fallback


def normalize_feedback(text: str) -> str:
    """
    Make feedback concise, de-duplicated, and capped to three sentences.
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Split on sentence enders, strip, drop empties, dedupe preserving order
    parts = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    parts = list(dict.fromkeys(parts))

    if not parts:
        return ""

    normalized = ". ".join(parts[:3]).strip()
    if not normalized.endswith("."):
        normalized += "."
    return normalized


def safe_output(value, fallback):
    """
    Prevent UI-breaking None/empty values.
    """
    return value if value else fallback
