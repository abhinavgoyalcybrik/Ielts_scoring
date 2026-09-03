import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils.ai_client as ai_client


# ---------------------------------------------------------------------------
# call_gpt_writing() gained an optional image_url parameter so Academic
# Writing Task 1 can attach the actual chart/graph/diagram image to a
# vision-capable model call, instead of only ever sending text. When
# image_url is given, the user message's "content" becomes the multi-part
# list format the OpenAI Chat Completions API expects for image input;
# when omitted, the message is byte-for-byte the same plain string as
# before, so every existing text-only caller is unaffected.
# ---------------------------------------------------------------------------

class _FakeMessage:
    content = '{"ok": true}'


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self):
        self.captured_kwargs = {}

    def create(self, **kwargs):
        self.captured_kwargs = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def _install_fake_client(monkeypatch):
    completions = _FakeCompletions()
    monkeypatch.setattr(ai_client, "client", _FakeClient(completions))
    return completions


def test_call_gpt_writing_with_image_url_sends_multipart_content(monkeypatch):
    completions = _install_fake_client(monkeypatch)

    ai_client.call_gpt_writing("Evaluate this essay.", image_url="https://example.com/chart.png")

    user_message = completions.captured_kwargs["messages"][-1]
    assert user_message["role"] == "user"
    assert user_message["content"] == [
        {"type": "text", "text": "Evaluate this essay."},
        {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
    ]


def test_call_gpt_writing_without_image_url_sends_plain_text_unchanged(monkeypatch):
    completions = _install_fake_client(monkeypatch)

    ai_client.call_gpt_writing("Evaluate this essay.")

    user_message = completions.captured_kwargs["messages"][-1]
    assert user_message["content"] == "Evaluate this essay."


def test_call_gpt_text_and_refine_answer_unaffected_by_image_url_parameter(monkeypatch):
    # Neither of these callers ever passes image_url - confirm they still
    # produce a plain string message, not a multi-part list.
    completions = _install_fake_client(monkeypatch)
    ai_client.call_gpt_text("Some prompt")
    assert completions.captured_kwargs["messages"][-1]["content"] == "Some prompt"

    completions2 = _install_fake_client(monkeypatch)
    ai_client.call_gpt_refine_answer("question", "answer")
    content = completions2.captured_kwargs["messages"][-1]["content"]
    assert isinstance(content, str)
