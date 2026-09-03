# Writing had zero token-usage tracking before this - added specifically
# so real per-evaluation cost is computable from the eval log once the
# gpt-4o flip has real traffic (see utils/eval_log.py records). Mirrors
# the same {input_tokens, output_tokens, total_tokens} shape already used
# in evaluators/speaking_audio.py's final_response["usage"].

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing

_MAKE_BANDS = lambda band: {str(n): (n == band) for n in range(1, 10)}


def _complete_task2_bands(band=7):
    flags = _MAKE_BANDS(band)
    return {
        "task_response_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
    }


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


def test_evaluate_writing_result_includes_usage_totals(monkeypatch):
    essay = "The government should invest in education because it helps the economy grow."
    ai_response = {
        **_complete_task2_bands(),
        "mistakes": [], "strengths": "x", "improvement": "y", "topic_relevance": "on_topic",
    }

    # call_gpt_writing itself is mocked (as every other test in this suite
    # does), so it never reaches the real record_token_usage() call inside
    # utils/ai_client.py - simulate what that call does: append directly
    # to the usage_log list evaluate_writing() passes through.
    def fake_call_gpt_writing(prompt, image_url=None, usage_log=None):
        if usage_log is not None:
            usage_log.append({"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})
        return ai_response

    monkeypatch.setattr(writing, "call_gpt_writing", fake_call_gpt_writing)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined essay text.")

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": essay},
    })

    assert result["usage"] == {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}


def test_evaluate_writing_usage_defaults_to_zero_when_call_gpt_writing_mock_ignores_usage_log(monkeypatch):
    # Every OTHER existing test's mock lambda accepts **kwargs but doesn't
    # populate usage_log - confirms that's safe and just yields zeros,
    # not an error.
    essay = "The government should invest in education because it helps the economy grow."
    ai_response = {
        **_complete_task2_bands(),
        "mistakes": [], "strengths": "x", "improvement": "y", "topic_relevance": "on_topic",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined essay text.")

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": essay},
    })

    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_call_gpt_writing_forwards_usage_log_to_record_token_usage(monkeypatch):
    import utils.ai_client as ai_client

    class _FakeMessage:
        content = '{"a": 1}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]
        usage = _FakeUsage(500, 100, 600)

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(ai_client, "client", _FakeClient())
    usage_log = []
    ai_client.call_gpt_writing("some prompt", usage_log=usage_log)
    assert usage_log == [{"input_tokens": 500, "output_tokens": 100, "total_tokens": 600}]
