import os
import json
import re
from openai import OpenAI

_client = None

def get_client():
    global _client

    if _client is None:

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        _client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))

    return _client


def _strip_code_fence(content: str) -> str:
    """Strip ```json ... ``` / ``` ... ``` fences some models wrap JSON in."""
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _extract_json_object(content: str) -> str:
    """Fall back to the outermost {...} span if the model added stray prose."""
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start:end + 1]
    return content


def record_token_usage(response, usage_log) -> None:
    """Append this response's actual token usage (from the API's own
    usage metadata, never estimated) to a shared per-evaluation
    usage_log list. No-op if usage_log is None (tracking not requested
    by the caller). Never raises - missing/malformed usage data on a
    response must never break evaluation."""
    if usage_log is None:
        return
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        usage_log.append({
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        })
    except Exception:
        pass


def _call_gpt_with_model(prompt, model, usage_log=None, temperature: float = 0.0):
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    record_token_usage(response, usage_log)

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty GPT response")

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


def call_gpt(prompt, usage_log=None, temperature: float = 0.0):
    # temperature defaults to 0 (was: unset, i.e. the OpenAI API's own
    # default of ~1.0) - Speaking evaluation was non-deterministic between
    # identical runs of the same transcript, and nothing about scoring/
    # mistake-detection consistency is measurable until it's stable. See
    # the "Speaking evaluator: port proven deterministic Writing fixes"
    # plan - this mirrors Writing's own pinned-temperature call (though
    # Writing itself still runs at 0.2, not 0 - that's Writing's own,
    # separately-deferred Step 8, not changed here).
    return _call_gpt_with_model(
        prompt, os.getenv("OPENAI_MODEL_OVERRIDE", "gpt-4o-mini"),
        usage_log=usage_log, temperature=temperature,
    )


def call_gpt_strong(prompt, usage_log=None, temperature: float = 0.0):
    """Same contract as call_gpt(), but on a more capable model - reserved
    for judgment calls where gpt-4o-mini has shown a specific, repeatable
    failure mode (e.g. detect_answer_alignment_issues in speaking_audio.py,
    which mini kept confidently mis-judging on real test data even after
    the prompt was tightened with an explicit boolean commitment). Not a
    blanket upgrade - most calls in this codebase are fine on the cheaper
    model and switching all of them would multiply cost for no benefit."""
    return _call_gpt_with_model(
        prompt, os.getenv("OPENAI_MODEL_OVERRIDE", "gpt-4o"),
        usage_log=usage_log, temperature=temperature,
    )
