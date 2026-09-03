import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

from utils.gpt_client import record_token_usage


# Ensure .env is loaded and get the correct API key
load_dotenv()
API_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()
if not API_KEY:
    # fallback: try to find any env var that looks like an OpenAI key
    for k, v in os.environ.items():
        if k.upper().startswith("OPENAI") and isinstance(v, str) and v.startswith("sk-"):
            API_KEY = v.strip()
            break
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=API_KEY, base_url=os.getenv("OPENAI_BASE_URL"))


def _writing_model() -> str:
    """Writing's model, resolved independently of Speaking's (see the
    session's determinism/cost investigation: the right model differs by
    module - gpt-4o measurably fixed Writing's mistake-detection misses
    (1/15 and 0/15 -> 5/5 on a working parser) but destabilized Speaking's
    equivalent function, so a single shared override can't express both).

    FLIPPED to gpt-4o as the default (confirmed: 5/5 detection on both
    test essays, 0.5-band max spread on one of two, +5.6 cents/candidate,
    no measured latency penalty - spend approved). WRITING_MODEL_OVERRIDE
    takes priority when set - an instant rollback lever (e.g.
    WRITING_MODEL_OVERRIDE=gpt-4.1-mini) with no code deploy needed. Falls
    back to the existing OPENAI_MODEL_OVERRIDE for backward compatibility
    with anyone already using the global override, then to gpt-4o. Not
    read anywhere else - Speaking's model selection in utils/gpt_client.py
    is untouched by this and stays on its current models."""
    return os.getenv("WRITING_MODEL_OVERRIDE") or os.getenv("OPENAI_MODEL_OVERRIDE", "gpt-4o")


def _call_gpt(prompt: str, system_msg: str, image_url: str | None = None, usage_log=None) -> str:
    # Academic Writing Task 1 (chart/graph/diagram) can optionally attach
    # the actual question image, so the model verifies the candidate's
    # stated figures/trends against the real data instead of only judging
    # internal consistency of the candidate's own claims. Vision-capable
    # models (gpt-4o, gpt-4o-mini) accept an image alongside text via a
    # multi-part "content" list; when image_url is None, this is byte-for-
    # byte the same plain-text message as before - every existing caller
    # (call_gpt_refine_answer, call_gpt_text, and text-only Writing calls)
    # is unaffected.
    user_content = prompt
    if image_url:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    response = client.chat.completions.create(
        model=_writing_model(),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
    record_token_usage(response, usage_log)

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty GPT response")

    return content.strip()


def _strip_code_fence(content: str) -> str:
    """Strip ```json ... ``` / ``` ... ``` fences some models wrap JSON in -
    ported from utils/gpt_client.py, which already had this; this module's
    _parse_json didn't, and a fenced response (observed live from gpt-4o -
    its very first response during a determinism measurement came back
    fence-wrapped) failed straight to ValueError with zero tolerance,
    burning a retry (or, in the worst case, exhausting both retries and
    hitting evaluate_writing()'s fallback) for a response that was
    actually valid JSON underneath the fence."""
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _extract_json_object(content: str) -> str:
    """Fall back to the outermost {...} span if the model added stray prose
    - same fallback as utils/gpt_client.py."""
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start:end + 1]
    return content


# Observed live: gpt-4o intermittently refuses outright on an ordinary
# essay (a candidate answer, or the pipeline's own model-generated
# refined_answer fed back through scoring) - "I'm sorry, I can't assist
# with that request." with no discernible trigger, non-reproducible on an
# identical retry. Not a parse error - the response is well-formed text,
# just not JSON and not a score. safe_gpt_call's retry+fallback already
# catches this correctly (ai_evaluation_failed=True fires, no candidate
# gets a fake score) - this only makes the DISTINCT reason visible instead
# of collapsing it into the same generic "invalid JSON" bucket as a
# malformed-but-genuine attempt at scoring would be. Deliberately narrow
# and literal (exact refusal openings actually observed) rather than a
# broad heuristic - a false positive here would mislabel a genuine parse
# failure as a refusal, which is the wrong direction to guess wrong in.
_REFUSAL_OPENINGS = (
    "i'm sorry, but i can't", "i'm sorry, i can't",
    "i cannot assist", "i can't assist",
    "i'm sorry, but i cannot", "i'm sorry, i cannot",
)


def _looks_like_refusal(content: str) -> bool:
    """Accepts either the raw model content directly, or a caller's
    exception string that WRAPS it (e.g. _parse_json's own "Invalid JSON
    from GPT:\\n{content}" - the caller in evaluators/writing.py passes
    str(exception), not the bare content) - checked as a substring within
    the first 200 chars rather than a strict prefix match specifically so
    both cases work without the caller needing to unwrap anything first."""
    lowered = (content or "").strip().lower()[:200]
    return any(p in lowered for p in _REFUSAL_OPENINGS)


def _parse_json(content: str) -> dict:
    candidate = _strip_code_fence(content)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    candidate = _extract_json_object(candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from GPT:\n{content}")


def call_gpt_writing(prompt: str, image_url: str | None = None, usage_log=None) -> dict:
    content = _call_gpt(
        prompt,
        system_msg="You are a certified IELTS Writing examiner. Respond ONLY in valid JSON.",
        image_url=image_url,
        usage_log=usage_log,
    )
    return _parse_json(content)


def _call_gpt_json(prompt: str, system_msg: str, image_url: str | None = None) -> dict:
    # Same multi-part image handling as _call_gpt, plus response_format
    # enforcement - both new v2-refine-pipeline wrappers below (call_gpt_extract,
    # call_gpt_refine) use this instead of _call_gpt directly, since neither
    # has any legacy plain-text caller to stay byte-compatible with (unlike
    # call_gpt_writing/call_gpt_text, left untouched above). response_format
    # cuts malformed-JSON retry churn on these two calls; not added to
    # call_gpt_writing to avoid touching the live scoring call's behavior.
    user_content = prompt
    if image_url:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    response = client.chat.completions.create(
        model=_writing_model(),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty GPT response")

    return _parse_json(content.strip())


def call_gpt_extract(prompt: str, image_url: str | None = None) -> dict:
    """Structured data-extraction call (currently: Task 1 chart data) - given
    ONLY the image + a prompt built from the question, never the candidate's
    essay. Kept as its own function/system-message (distinct from
    call_gpt_writing) so extraction calls are independently monkeypatchable
    in tests without affecting the scoring call."""
    return _call_gpt_json(
        prompt,
        system_msg="You are a precise data-extraction assistant. Respond ONLY in valid JSON.",
        image_url=image_url,
    )


def call_gpt_refine(prompt: str, image_url: str | None = None) -> dict:
    """JSON-returning Band 9 model-answer generator (the v2 refine pipeline -
    see evaluators/writing.py's _generate_refined_answer_v2). Distinct from
    the legacy call_gpt_text refine path, which stays plain-text and
    unchanged for the flag-OFF path."""
    return _call_gpt_json(
        prompt,
        system_msg="You are an IELTS Writing tutor. Respond ONLY in valid JSON.",
        image_url=image_url,
    )


def call_gpt_refine_answer(question: str, answer: str, target_band: int = 8) -> str:
    prompt = (
        f"Improve the following IELTS Writing answer to Band {target_band}.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Return ONLY the improved answer."
    )

    return _call_gpt(
        prompt,
        system_msg="You are an IELTS Writing tutor."
    )


def call_gpt_text(prompt: str, system_msg: str = "You are an IELTS assistant.") -> str:
    """
    Lightweight text-only GPT helper (no JSON parsing).
    """
    return _call_gpt(prompt, system_msg=system_msg)
