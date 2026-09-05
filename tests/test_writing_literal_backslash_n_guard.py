# Item 10 - the mirror bug to the confirmed-and-fixed /n/n literal
# escaping bug (Item 1), on the OUTPUT side: refined_answer sometimes
# contained the literal two-character sequence "\n" (backslash + n)
# instead of a real newline - live case: "...expatriates often
# encounter.\n\nOne of the most significant...", the escape sequence
# itself printed in the candidate's model answer.
#
# Root cause found and fixed: the three refine prompt files' RESPONSE
# FORMAT section showed a double-escaped example ("\\n\\n", two
# backslashes each) while their own STRUCTURE section, a few lines
# earlier, correctly used a single backslash for the identical
# instruction. A plain-text prompt only needs one backslash to show GPT
# what a real JSON-encoded newline looks like - showing two teaches GPT
# to double-escape its own output, which a correctly-functioning JSON
# decoder then un-escapes down to a literal backslash + 'n'. Fixed in
# all three prompt files (writing_refine_task1.txt, _task1_gt.txt,
# _task2.txt). These tests cover the backstop this session's own
# instruction requires regardless: normalise on the way out, but treat
# ever needing to as a hard validation failure that drives a retry, not
# a silently accepted fix.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing


# ---------------------------------------------------------------------------
# The predicate and normaliser themselves.
# ---------------------------------------------------------------------------

def test_detects_literal_backslash_n():
    text = "expatriates often encounter." + chr(92) + chr(110) + chr(92) + chr(110) + "One of the most"
    assert writing._contains_literal_backslash_n(text) is True


def test_does_not_flag_real_newlines():
    text = "expatriates often encounter.\n\nOne of the most"
    assert writing._contains_literal_backslash_n(text) is False


def test_fix_converts_literal_backslash_n_to_real_newline():
    text = "para one." + chr(92) + chr(110) + chr(92) + chr(110) + "para two."
    fixed, diag = writing._fix_literal_backslash_n_in_output(text)
    assert fixed == "para one.\n\npara two."
    assert diag == {"fired": True, "literals_converted": 2}


def test_fix_is_a_noop_on_clean_text():
    text = "para one.\n\npara two."
    fixed, diag = writing._fix_literal_backslash_n_in_output(text)
    assert fixed == text
    assert diag == {"fired": False, "literals_converted": 0}


def test_fix_handles_empty_text():
    fixed, diag = writing._fix_literal_backslash_n_in_output("")
    assert fixed == ""
    assert diag["fired"] is False


# ---------------------------------------------------------------------------
# Prompt files - the actual root-cause fix. Confirms the RESPONSE FORMAT
# example now matches the single-backslash convention already used in
# each file's own STRUCTURE section, rather than trusting the edit.
# ---------------------------------------------------------------------------

def test_refine_prompt_files_use_single_backslash_not_double():
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    double_backslash = chr(92) * 2 + "n" + chr(92) * 2 + "n"
    single_backslash = chr(92) + "n" + chr(92) + "n"
    for fname in ("writing_refine_task1.txt", "writing_refine_task1_gt.txt", "writing_refine_task2.txt"):
        text = (prompts_dir / fname).read_text(encoding="utf-8")
        assert double_backslash not in text, f"{fname} still shows GPT a double-escaped example"
        assert single_backslash in text, f"{fname} is missing the expected single-escaped example"


# ---------------------------------------------------------------------------
# _validate_refine_output - hard failure, not a silent pass.
# ---------------------------------------------------------------------------

def test_validate_flags_literal_backslash_n_as_a_hard_issue():
    refined = "Intro paragraph." + chr(92) + chr(110) + chr(92) + chr(110) + "Body paragraph here with enough words to count as real content for this test."
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_2", None, 250, 2000)
    assert any("literal" in i and "\\n" in i for i in issues)


def test_validate_does_not_flag_real_newlines():
    refined = "\n\n".join(["Paragraph with real content here for this test."] * 4)
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_2", None, 250, 2000)
    assert not any("literal" in i and "\\n" in i for i in issues)


# ---------------------------------------------------------------------------
# _generate_refined_answer_v2 - retries on the defect, normalises the
# final output regardless of outcome, and reports validation_passed=False
# (not silently True) when the defect persists after the retry.
# ---------------------------------------------------------------------------

def _good_task2_structured_answer():
    return "\n\n".join([
        "I believe that governments should invest more in public health campaigns nationwide.",
        "Firstly, prevention reduces long-term healthcare costs, since treating illness early is far cheaper than treating it once it becomes severe.",
        "Secondly, informed citizens make healthier choices, since well-designed campaigns give people practical steps they can act on immediately.",
        "In conclusion, I believe stronger investment in public health campaigns is worthwhile, since it reduces costs and helps citizens make healthier choices.",
    ])


def test_generate_refine_v2_retries_on_literal_backslash_n_then_succeeds(monkeypatch):
    bad_text = "I believe governments should invest." + chr(92) + chr(110) + chr(92) + chr(110) + "Firstly, prevention reduces costs substantially for everyone involved in the system."
    bad = {"refined_answer": bad_text, "vocabulary_suggestions": []}
    good = {"refined_answer": _good_task2_structured_answer(), "vocabulary_suggestions": []}

    call_n = {"n": 0}
    seen_prompts = []

    def fake_call_gpt_refine(prompt, image_url=None):
        seen_prompts.append(prompt)
        call_n["n"] += 1
        return bad if call_n["n"] == 1 else good

    monkeypatch.setattr(writing, "call_gpt_refine", fake_call_gpt_refine)

    result = writing._generate_refined_answer_v2(
        "task_2", None, "question text", "essay text", [], None, [], 250, 2000, None,
    )
    assert call_n["n"] == 2
    assert "literal" in seen_prompts[1].lower() or "\\n" in seen_prompts[1]
    assert result["diagnostics"]["validation_passed"] is True
    assert result["refined_answer"] == good["refined_answer"]


def test_generate_refine_v2_normalises_final_output_even_if_defect_persists(monkeypatch):
    # Both attempts have the defect - validation_passed must be False
    # (not silently True), but the text actually returned must still
    # never contain literal "\n" - the backstop applies regardless of
    # whether the retry succeeded at fixing the ROOT cause.
    bad_text = "I believe governments should invest." + chr(92) + chr(110) + chr(92) + chr(110) + "Firstly, prevention reduces costs substantially for everyone involved."
    bad = {"refined_answer": bad_text, "vocabulary_suggestions": []}

    def fake_call_gpt_refine(prompt, image_url=None):
        return bad

    monkeypatch.setattr(writing, "call_gpt_refine", fake_call_gpt_refine)

    result = writing._generate_refined_answer_v2(
        "task_2", None, "question text", "essay text", [], None, [], 250, 2000, None,
    )
    assert result["diagnostics"]["validation_passed"] is False
    assert not writing._contains_literal_backslash_n(result["refined_answer"])
    assert "\n\n" in result["refined_answer"]
    assert result["diagnostics"]["literal_backslash_n_normalized"]["fired"] is True


# ---------------------------------------------------------------------------
# Legacy (flag OFF) path - reuses safe_gpt_call's own retry, plus the
# same unconditional backstop normalisation.
# ---------------------------------------------------------------------------

def _make_bands(true_band):
    return {str(n): (n == true_band) for n in range(1, 10)}


def test_legacy_path_retries_on_literal_backslash_n_then_succeeds(monkeypatch):
    monkeypatch.delenv("WRITING_INDEPENDENT_MODEL_ANSWER", raising=False)
    flags = _make_bands(6)
    ai_response = {
        "task_response_bands": dict(flags), "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags), "grammar_bands": dict(flags),
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)

    bad_text = ("word " * 30) + chr(92) + chr(110) + chr(92) + chr(110) + ("word " * 80)
    good_text = "\n\n".join([("word " * 30)] * 4)
    call_n = {"n": 0}

    def fake_call_gpt_text(prompt, system_msg=None):
        call_n["n"] += 1
        return bad_text if call_n["n"] == 1 else good_text

    monkeypatch.setattr(writing, "call_gpt_text", fake_call_gpt_text)

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": "Some essay text about the topic at hand for the question given."},
    })
    assert call_n["n"] == 2
    assert not writing._contains_literal_backslash_n(result["refined_answer"])


def test_legacy_path_backstop_normalises_even_if_every_retry_has_the_defect(monkeypatch):
    monkeypatch.delenv("WRITING_INDEPENDENT_MODEL_ANSWER", raising=False)
    flags = _make_bands(6)
    ai_response = {
        "task_response_bands": dict(flags), "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags), "grammar_bands": dict(flags),
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)

    bad_text = ("word " * 30) + chr(92) + chr(110) + chr(92) + chr(110) + ("word " * 80)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: bad_text)

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": "Some essay text about the topic at hand for the question given."},
    })
    assert not writing._contains_literal_backslash_n(result["refined_answer"])
