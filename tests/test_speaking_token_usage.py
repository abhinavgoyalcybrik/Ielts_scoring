import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils.gpt_client as gpt_client
import evaluators.speaking_audio as speaking_audio
from utils.gpt_client import record_token_usage, call_gpt


# ---------------------------------------------------------------------------
# Token usage tracking for the Speaking evaluation module. GPT-calling
# functions accept an optional `usage_log` list (default None, so every
# EXISTING call site that doesn't pass it behaves exactly as before);
# when provided, each LLM response's actual API usage metadata (never
# estimated) is appended to it. The final response sums these into a
# top-level "usage": {input_tokens, output_tokens, total_tokens} field.
# Uses a plain shared list rather than a global counter or ContextVar,
# since asyncio.to_thread() runs these calls across real OS threads -
# list.append() is atomic under the GIL, so concurrent appends are safe.
# ---------------------------------------------------------------------------

def _fake_usage(prompt_tokens, completion_tokens, total_tokens):
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)


def _fake_response(content, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


def test_record_token_usage_appends_actual_usage_fields():
    log = []
    record_token_usage(_fake_response("x", usage=_fake_usage(100, 50, 150)), log)
    assert log == [{"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}]


def test_record_token_usage_is_noop_when_usage_log_is_none():
    # Must not raise - this is the default for every existing call site.
    record_token_usage(_fake_response("x", usage=_fake_usage(1, 2, 3)), None)


def test_record_token_usage_is_noop_when_response_has_no_usage():
    log = []
    record_token_usage(_fake_response("x", usage=None), log)
    assert log == []

    log2 = []
    record_token_usage(SimpleNamespace(), log2)  # no .usage attribute at all
    assert log2 == []


def test_call_gpt_records_usage_without_changing_return_value(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return _fake_response('{"ok": true}', usage=_fake_usage(10, 20, 30))

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(gpt_client, "get_client", lambda: FakeClient())

    log = []
    result = call_gpt("some prompt", usage_log=log)
    assert result == {"ok": True}
    assert log == [{"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}]

    # Existing callers (e.g. evaluators/speaking.py) never pass usage_log -
    # must still work identically.
    result_no_log = call_gpt("some prompt")
    assert result_no_log == {"ok": True}


_SCORES_JSON_TEMPLATE = (
    '{{"fluency_bands": {bands}, "lexical_bands": {bands}, "grammar_bands": {bands}}}'
)
_ALL_TRUE_DOWN_TO_7 = '{"9": false, "8": false, "7": true, "6": true, "5": true, "4": true, "3": true, "2": true, "1": true}'


def test_generate_scores_records_usage_without_changing_scoring_output(monkeypatch):
    class FakeOpenAIClient:
        def __init__(self, *a, **k):
            pass
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    content = _SCORES_JSON_TEMPLATE.format(bands=_ALL_TRUE_DOWN_TO_7)
                    return _fake_response(content, usage=_fake_usage(200, 80, 280))

    monkeypatch.setattr(speaking_audio, "OpenAI", FakeOpenAIClient)

    log = []
    scores_with_log = speaking_audio.generate_scores(1, "some transcript text here", usage_log=log)
    scores_without_log = speaking_audio.generate_scores(1, "some transcript text here")

    assert scores_with_log == scores_without_log
    assert log == [{"input_tokens": 200, "output_tokens": 80, "total_tokens": 280}]


def test_generate_scores_missing_usage_metadata_does_not_break_evaluation(monkeypatch):
    class FakeOpenAIClientNoUsage:
        def __init__(self, *a, **k):
            pass
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    content = _SCORES_JSON_TEMPLATE.format(bands=_ALL_TRUE_DOWN_TO_7)
                    return _fake_response(content, usage=None)  # simulates missing usage metadata

    monkeypatch.setattr(speaking_audio, "OpenAI", FakeOpenAIClientNoUsage)

    log = []
    result = speaking_audio.generate_scores(1, "some transcript text here", usage_log=log)

    assert log == []
    assert result.get("fluency") == 7.0


def test_multiple_llm_calls_sum_into_one_shared_usage_log(monkeypatch):
    class FakeOpenAIClient:
        def __init__(self, *a, **k):
            pass
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    content = _SCORES_JSON_TEMPLATE.format(bands=_ALL_TRUE_DOWN_TO_7)
                    return _fake_response(content, usage=_fake_usage(50, 25, 75))

    monkeypatch.setattr(speaking_audio, "OpenAI", FakeOpenAIClient)

    shared_log = []
    speaking_audio.generate_scores(1, "transcript one", usage_log=shared_log)
    speaking_audio.generate_scores(2, "transcript two", usage_log=shared_log)

    totals = {
        "input_tokens": sum(u["input_tokens"] for u in shared_log),
        "output_tokens": sum(u["output_tokens"] for u in shared_log),
        "total_tokens": sum(u["total_tokens"] for u in shared_log),
    }
    assert totals == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
