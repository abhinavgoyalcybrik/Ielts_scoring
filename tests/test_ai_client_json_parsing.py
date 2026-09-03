# utils/ai_client.py's _parse_json had no code-fence tolerance, unlike
# utils/gpt_client.py's equivalent - a fenced JSON response (observed live
# from gpt-4o) failed straight to ValueError, burning a retry or (worst
# case) exhausting both and silently landing on evaluate_writing()'s
# neutral fallback. Ported the same two-stage tolerant parsing
# (fence-strip, then outermost-{...} extraction) gpt_client.py already had.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils.ai_client as ai_client


def test_parse_json_handles_plain_json():
    assert ai_client._parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_json_fence():
    content = '```json\n{"a": 1}\n```'
    assert ai_client._parse_json(content) == {"a": 1}


def test_parse_json_strips_bare_fence_no_json_tag():
    content = '```\n{"a": 1}\n```'
    assert ai_client._parse_json(content) == {"a": 1}


def test_parse_json_strips_fence_case_insensitively():
    content = '```JSON\n{"a": 1}\n```'
    assert ai_client._parse_json(content) == {"a": 1}


def test_parse_json_falls_back_to_outermost_braces_with_stray_prose():
    content = 'Here is the result:\n{"a": 1}\nHope that helps!'
    assert ai_client._parse_json(content) == {"a": 1}


def test_parse_json_handles_fence_plus_stray_prose_together():
    content = '```json\nSure, here you go:\n{"a": 1}\n```'
    assert ai_client._parse_json(content) == {"a": 1}


def test_parse_json_still_raises_on_genuinely_invalid_content():
    import pytest
    with pytest.raises(ValueError):
        ai_client._parse_json("not json at all, no braces")


def test_strip_code_fence_leaves_unfenced_content_unchanged():
    assert ai_client._strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_extract_json_object_returns_unchanged_when_no_braces():
    assert ai_client._extract_json_object("no braces here") == "no braces here"


# ---------------------------------------------------------------------------
# End-to-end: call_gpt_writing() must tolerate a fenced response exactly
# the way call_gpt_extract/call_gpt_refine (which share _parse_json via
# _call_gpt_json) already implicitly will.

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def test_call_gpt_writing_tolerates_fenced_response(monkeypatch):
    fenced = '```json\n{"fluency": 7, "mistakes": []}\n```'
    monkeypatch.setattr(ai_client, "client", _FakeClient(fenced))
    result = ai_client.call_gpt_writing("some prompt")
    assert result == {"fluency": 7, "mistakes": []}


# ---------------------------------------------------------------------------
# _writing_model() - Writing-specific model config, independent of
# Speaking's own model selection. Default flipped to gpt-4o (confirmed:
# 5/5 detection on both test essays against a working parser, 0.5-band max
# spread, +5.6 cents/candidate approved) - WRITING_MODEL_OVERRIDE remains
# available as an instant rollback lever.

def test_writing_model_defaults_to_gpt4o_when_nothing_set(monkeypatch):
    monkeypatch.delenv("WRITING_MODEL_OVERRIDE", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_OVERRIDE", raising=False)
    assert ai_client._writing_model() == "gpt-4o"


def test_writing_model_override_takes_priority(monkeypatch):
    monkeypatch.setenv("WRITING_MODEL_OVERRIDE", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_MODEL_OVERRIDE", "gpt-4o-mini")
    assert ai_client._writing_model() == "gpt-4.1-mini"


def test_writing_model_override_enables_instant_rollback(monkeypatch):
    # The specific safety property this lever exists for: reverting to the
    # pre-flip model needs only an env var, no code deploy.
    monkeypatch.setenv("WRITING_MODEL_OVERRIDE", "gpt-4.1-mini")
    assert ai_client._writing_model() == "gpt-4.1-mini"


def test_writing_model_falls_back_to_global_override_for_backward_compat(monkeypatch):
    monkeypatch.delenv("WRITING_MODEL_OVERRIDE", raising=False)
    monkeypatch.setenv("OPENAI_MODEL_OVERRIDE", "gpt-4o-mini")
    assert ai_client._writing_model() == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# _looks_like_refusal() - makes an observed live failure mode (gpt-4o
# outright refusing on an ordinary essay) distinguishable from a generic
# parse error in the eval log, without changing what ai_evaluation_failed
# means or when it fires.

def test_looks_like_refusal_detects_raw_refusal_content():
    assert ai_client._looks_like_refusal("I'm sorry, I can't assist with that request.")
    assert ai_client._looks_like_refusal("I'm sorry, but I can't help with that.")
    assert ai_client._looks_like_refusal("I cannot assist with this request.")


def test_looks_like_refusal_detects_refusal_wrapped_in_exception_message():
    # evaluators/writing.py passes str(exception), not the bare model
    # content - _parse_json's own error format wraps it with a prefix.
    wrapped = "Invalid JSON from GPT:\nI'm sorry, I can't assist with that request."
    assert ai_client._looks_like_refusal(wrapped)


def test_looks_like_refusal_is_false_for_genuine_parse_errors():
    assert not ai_client._looks_like_refusal("Invalid JSON from GPT:\n{fluency: 7, broken json")
    assert not ai_client._looks_like_refusal("")
    assert not ai_client._looks_like_refusal(None)


def test_looks_like_refusal_is_false_for_ordinary_essay_text():
    # A candidate essay is exactly the kind of text this must never
    # false-positive on - it's the thing being scored, not a refusal.
    assert not ai_client._looks_like_refusal(
        "I believe that governments should invest more in public transport "
        "because it reduces congestion and pollution in major cities."
    )


# ---------------------------------------------------------------------------
# safe_gpt_call's new on_failure hook - additive, default None, must not
# change behaviour for any existing caller that doesn't pass it.

def test_safe_gpt_call_without_on_failure_is_unaffected(monkeypatch):
    from utils.safety import safe_gpt_call

    def always_fails(prompt):
        raise ValueError("boom")

    result = safe_gpt_call("prompt", fallback="FALLBACK", caller=always_fails, retries=1)
    assert result == "FALLBACK"


def test_safe_gpt_call_on_failure_receives_last_exception(monkeypatch):
    from utils.safety import safe_gpt_call

    captured = []

    def always_fails(prompt):
        raise ValueError("specific failure reason")

    result = safe_gpt_call(
        "prompt", fallback="FALLBACK", caller=always_fails, retries=2,
        on_failure=lambda e: captured.append(str(e)),
    )
    assert result == "FALLBACK"
    assert captured == ["specific failure reason"]


def test_safe_gpt_call_on_failure_not_called_when_call_succeeds():
    from utils.safety import safe_gpt_call

    captured = []
    result = safe_gpt_call(
        "prompt", fallback="FALLBACK", caller=lambda p: "a genuinely long real result here",
        on_failure=lambda e: captured.append(e),
    )
    assert result == "a genuinely long real result here"
    assert captured == []


def test_safe_gpt_call_on_failure_exception_does_not_break_fallback():
    from utils.safety import safe_gpt_call

    def always_fails(prompt):
        raise ValueError("boom")

    def broken_callback(e):
        raise RuntimeError("callback itself is broken")

    result = safe_gpt_call("prompt", fallback="FALLBACK", caller=always_fails, retries=1, on_failure=broken_callback)
    assert result == "FALLBACK"
