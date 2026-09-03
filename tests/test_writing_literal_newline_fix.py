# Item 1 - literal "/n/n" escaping guard. Live payloads have arrived with
# paragraph breaks written as the literal two-character sequence "/n"
# (forward slash + n) instead of a real newline - the essay then has zero
# real newlines, the paragraph cap fires, and Coherence drops to 5.0 on an
# essay that genuinely has paragraphs. This tests _fix_literal_newline_
# escaping() directly (a pure function, no GPT call needed) plus one
# end-to-end check that the paragraph cap genuinely stops firing once the
# literal sequence is converted.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing


def test_literal_newline_escaping_converts_when_no_real_newlines_present():
    essay = "First paragraph here./n/nSecond paragraph here./n/nThird paragraph here."
    fixed, diagnostics = writing._fix_literal_newline_escaping(essay)

    assert fixed == "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    assert diagnostics["fired"] is True
    assert diagnostics["literals_converted"] == 4  # two "/n/n" pairs = four "/n" units


def test_literal_newline_escaping_leaves_text_with_real_newlines_untouched():
    essay = "First paragraph here.\n\nSecond paragraph mentions /n as an escape sequence."
    fixed, diagnostics = writing._fix_literal_newline_escaping(essay)

    assert fixed == essay
    assert diagnostics["fired"] is False
    assert diagnostics["literals_converted"] == 0


def test_literal_newline_escaping_leaves_genuine_prose_slash_n_untouched_alongside_real_newlines():
    # The exact scenario the "real newlines already present" guard exists
    # for: "/n" appearing as genuine content (e.g. explaining the escape
    # sequence itself), not a corrupted paragraph break, because real
    # paragraph breaks are already present elsewhere in the same text.
    essay = (
        "In programming, /n is commonly used to represent a newline character.\n\n"
        "This essay discusses that convention in more detail below."
    )
    fixed, diagnostics = writing._fix_literal_newline_escaping(essay)

    assert fixed == essay
    assert "/n" in fixed
    assert diagnostics["fired"] is False


def test_literal_newline_escaping_no_op_when_neither_present():
    essay = "A single block of text with no paragraph breaks and no literal escape sequences at all."
    fixed, diagnostics = writing._fix_literal_newline_escaping(essay)

    assert fixed == essay
    assert diagnostics["fired"] is False
    assert diagnostics["literals_converted"] == 0


def test_literal_newline_escaping_single_slash_n_without_pairing_also_converts():
    # The trigger is "contains at least one literal /n and zero real
    # newlines" - not specifically "/n/n" - a single stray "/n" with no
    # real newlines anywhere is the same corruption pattern.
    essay = "Intro sentence./nBody sentence with no other breaks at all."
    fixed, diagnostics = writing._fix_literal_newline_escaping(essay)

    assert fixed == "Intro sentence.\nBody sentence with no other breaks at all."
    assert diagnostics["fired"] is True
    assert diagnostics["literals_converted"] == 1


# ---------------------------------------------------------------------------
# End-to-end: confirm the paragraph cap genuinely stops firing once
# evaluate_writing() converts the literal sequence - not just that the
# string transform is correct in isolation.
# ---------------------------------------------------------------------------

def _make_bands(true_band):
    return {str(n): (n == true_band) for n in range(1, 10)}


def _complete_task2_bands(band=7):
    flags = _make_bands(band)
    return {
        "task_response_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
    }


def _paragraphed_essay_body(n_words_per_para=40):
    filler = ["This", "argument", "shows", "clearly", "that", "the",
              "situation", "requires", "careful", "consideration"]
    words = (filler * ((n_words_per_para // len(filler)) + 1))[:n_words_per_para]
    sentences = [" ".join(words[i:i + 10]) + "." for i in range(0, len(words), 10)]
    return " ".join(sentences)


def test_evaluate_writing_paragraph_cap_does_not_fire_after_literal_newline_conversion(monkeypatch):
    # Three genuinely substantial paragraphs, joined with the literal "/n/n"
    # bug instead of real newlines - before the fix this reads as one
    # giant unbroken block and trips the paragraph cap; after the fix it
    # should be treated exactly as if real "\n\n" had been submitted.
    paragraphs = [_paragraphed_essay_body(60) for _ in range(3)]
    essay = "/n/n".join(paragraphs)

    ai_response = {
        **_complete_task2_bands(7),
        "mistakes": [], "strengths": "x", "improvement": "y",
        "topic_relevance": "on_topic",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined text. " * 60)

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": essay},
    })

    assert result["newline_diagnostics"]["literal_newline_escaping"]["fired"] is True
    assert result["newline_diagnostics"]["newline_count"] == 4  # two "/n/n" pairs converted
    assert result["newline_diagnostics"]["paragraph_block_count"] == 3
    # The paragraph cap did NOT drag coherence down to a cap value despite
    # a genuinely high checklist band - it should reflect the checklist
    # band (7.0), not a capped-low value, now that the input has real
    # paragraph breaks.
    assert result["criteria_scores"]["coherence_cohesion"] == 7.0
