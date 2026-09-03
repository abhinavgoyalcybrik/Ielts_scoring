# Tests for the v2 refine pipeline (WRITING_INDEPENDENT_MODEL_ANSWER flag,
# default OFF) - see the approved "refined_answer overhaul" plan. Covers the
# new standalone functions in isolation (no GPT calls) and the retry-loop
# orchestrator + full evaluate_writing() wiring with GPT calls mocked, same
# monkeypatch style as tests/test_writing_evaluation.py.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing


# ---------------------------------------------------------------------------
# _extract_chart_data - fails open (returns None) rather than blocking.
# ---------------------------------------------------------------------------

def test_extract_chart_data_returns_parsed_result_on_success(monkeypatch):
    response = {"chart_count": 1, "charts": [{"chart_type": "pie_chart", "categories": [{"name": "Sodium"}]}]}
    monkeypatch.setattr(writing, "call_gpt_extract", lambda prompt, image_url=None, **kwargs: response)

    result = writing._extract_chart_data("data:image/png;base64,xyz", "question text")
    assert result == response


def test_extract_chart_data_fails_open_to_none_when_gpt_call_errors(monkeypatch):
    def always_fail(prompt, image_url=None):
        raise ValueError("boom")
    monkeypatch.setattr(writing, "call_gpt_extract", always_fail)

    result = writing._extract_chart_data("data:image/png;base64,xyz", "question text")
    assert result is None


def test_extract_chart_data_returns_none_when_charts_list_is_empty(monkeypatch):
    # A genuinely empty extraction (image failed to load / no chart
    # present) must not be treated as a usable result.
    monkeypatch.setattr(writing, "call_gpt_extract", lambda prompt, image_url=None, **kwargs: {"chart_count": 0, "charts": []})
    result = writing._extract_chart_data("data:image/png;base64,xyz", "question text")
    assert result is None


# ---------------------------------------------------------------------------
# _derive_categories_from_question_text - best-effort, no GPT call.
# ---------------------------------------------------------------------------

def test_derive_categories_finds_comma_and_and_list():
    question = (
        "The charts below show the average percentages in typical meals "
        "of sodium, saturated fat, and added sugar. Summarise the "
        "information by selecting and reporting the main features."
    )
    categories = writing._derive_categories_from_question_text(question)
    assert categories == ["sodium", "saturated fat", "added sugar"]


def test_derive_categories_returns_empty_when_no_recognisable_pattern():
    question = "The line graph below shows internet access in three regions."
    # No "of X, Y and Z" list pattern here - must not guess.
    categories = writing._derive_categories_from_question_text(question)
    assert categories == []


def test_derive_categories_handles_empty_question():
    assert writing._derive_categories_from_question_text("") == []


# ---------------------------------------------------------------------------
# _derive_required_points_from_gt_question - best-effort, no GPT call.
# ---------------------------------------------------------------------------

def test_derive_gt_points_splits_comma_separated_bullets():
    question = (
        "You recently bought a piece of furniture online, but it arrived "
        "damaged. Write a letter to the company. In your letter: describe "
        "the item you bought, explain what is wrong with it, say what you "
        "would like the company to do."
    )
    points = writing._derive_required_points_from_gt_question(question)
    assert points == [
        "describe the item you bought",
        "explain what is wrong with it",
        "say what you would like the company to do",
    ]


def test_derive_gt_points_returns_empty_when_pattern_absent():
    question = "Write a letter to your friend about your holiday plans."
    assert writing._derive_required_points_from_gt_question(question) == []


# ---------------------------------------------------------------------------
# _validate_refine_output - every check in isolation.
# ---------------------------------------------------------------------------

_CHART_DATA = {
    "chart_count": 1,
    "charts": [{
        "chart_type": "pie_chart",
        "title": "Nutrients",
        "categories": [
            {"name": "Sodium", "values": {}},
            {"name": "Saturated Fat", "values": {}},
            {"name": "Added Sugar", "values": {}},
        ],
        "units": "%",
    }],
    "overview_pattern": "x",
}


def _good_task1_refined():
    return (
        "The chart illustrates nutrient data.\n\n"
        "Overall, the figures vary by meal.\n\n"
        "Sodium is highest at dinner. Saturated Fat follows a similar "
        "pattern.\n\n"
        "Added Sugar is highest at snacks."
    )


def test_validate_passes_clean_task1_output():
    parsed = {
        "refined_answer": _good_task1_refined(),
        "data_contradiction_flag": False,
        "vocabulary_suggestions": [],
    }
    issues = writing._validate_refine_output(
        parsed, _CHART_DATA, [], [], "candidate essay text",
        "task_1", "academic", 150, 170,
    )
    assert issues == []


def test_validate_flags_missing_chart_category():
    refined = "The chart illustrates nutrient data.\n\nSodium is highest at dinner.\n\nSaturated Fat follows.\n\nOverall pattern noted."
    parsed = {"refined_answer": refined, "data_contradiction_flag": False, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(
        parsed, _CHART_DATA, [], [], "candidate essay text",
        "task_1", "academic", 150, 170,
    )
    assert any("Added Sugar" in i for i in issues)


def test_validate_flags_self_reported_data_contradiction():
    parsed = {"refined_answer": _good_task1_refined(), "data_contradiction_flag": True, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(
        parsed, _CHART_DATA, [], [], "candidate essay text",
        "task_1", "academic", 150, 170,
    )
    assert any("data_contradiction_flag" in i for i in issues)


def test_validate_flags_task2_under_four_paragraphs():
    parsed = {"refined_answer": "One paragraph only, no breaks at all here really.", "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(
        parsed, None, [], [], "essay", "task_2", None, 250, 260,
    )
    assert any("paragraph" in i for i in issues)


def test_validate_passes_task2_with_four_paragraphs():
    refined = "Intro paragraph here with enough words to count. " * 3
    parsed = {"refined_answer": "\n\n".join([refined] * 4), "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(
        parsed, None, [], [], "essay", "task_2", None, 250, 2000,
    )
    assert not any("paragraph" in i and "Task 2" in i for i in issues)


def test_validate_flags_missing_paragraph_breaks_when_long_enough():
    long_flat = "word " * 90
    parsed = {"refined_answer": long_flat, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(
        parsed, None, [], [], "essay", "task_2", None, 250, 2000,
    )
    assert any("paragraph breaks" in i for i in issues)


def test_validate_flags_flagged_mistake_leaking_into_rewrite():
    essay = "The candidate wrote a very bad sentence structure here."
    mistakes = [{"original": "a very bad sentence structure"}]
    refined = "Intro.\n\nThis has a very bad sentence structure in it still.\n\nBody two.\n\nConclusion."
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(
        parsed, None, [], mistakes, essay, "task_2", None, 250, 2000,
    )
    assert any("still appears verbatim" in i for i in issues)


def test_validate_flags_vocabulary_suggestion_not_in_essay():
    essay = "The candidate wrote about remote work today."
    parsed = {
        "refined_answer": "Intro.\n\nBody.\n\nBody two.\n\nConclusion paragraph here.",
        "vocabulary_suggestions": [{"original_phrase": "a phrase never written", "stronger_alternative": "x"}],
    }
    issues = writing._validate_refine_output(
        parsed, None, [], [], essay, "task_2", None, 250, 2000,
    )
    assert any("not actually in the candidate" in i for i in issues)


def test_validate_flags_word_budget_exceeded():
    refined = "Intro para.\n\n" + ("word " * 300) + "\n\nBody two.\n\nConclusion."
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(
        parsed, None, [], [], "essay", "task_2", None, 250, 260,
    )
    assert any(i.startswith("word_budget_exceeded") for i in issues)


def test_validate_rejects_missing_refined_answer_field():
    issues = writing._validate_refine_output(
        {"rewrite_basis": "candidate_material"}, None, [], [], "essay", "task_2", None, 250, 260,
    )
    assert issues == ["refined_answer field is missing or empty"]


# ---------------------------------------------------------------------------
# _map_tied_vocabulary_suggestions - verbatim filter + shape mapping.
# ---------------------------------------------------------------------------

def test_map_vocab_keeps_verbatim_suggestion_and_maps_shape():
    essay = "The report shows consumed levels were very high."
    suggestions = [{"original_phrase": "consumed levels", "stronger_alternative": "recorded intake", "placement_context": "in the overview"}]
    mapped = writing._map_tied_vocabulary_suggestions(suggestions, essay, "task_1")
    assert len(mapped) == 1
    item = mapped[0]
    assert item["word"] == "recorded intake"
    assert "consumed levels" in item["usage_hint"]
    assert item["original_phrase"] == "consumed levels"
    assert item["placement_context"] == "in the overview"
    assert item["task_type"] == "task_1"
    assert item["task_specific"] is True


def test_map_vocab_drops_non_verbatim_suggestion():
    essay = "The report shows recorded levels were very high."
    suggestions = [{"original_phrase": "a phrase never in the essay", "stronger_alternative": "x"}]
    mapped = writing._map_tied_vocabulary_suggestions(suggestions, essay, "task_1")
    assert mapped == []


def test_map_vocab_drops_entries_missing_required_fields():
    essay = "Some real essay text here."
    suggestions = [{"original_phrase": "real essay text"}]  # no stronger_alternative
    mapped = writing._map_tied_vocabulary_suggestions(suggestions, essay, "task_2")
    assert mapped == []


# ---------------------------------------------------------------------------
# _generate_refined_answer_v2 - the bespoke retry loop.
# ---------------------------------------------------------------------------

def _task2_call_sequence(monkeypatch, responses):
    calls = {"n": 0}

    def fake_call_gpt_refine(prompt, image_url=None):
        idx = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[idx]

    monkeypatch.setattr(writing, "call_gpt_refine", fake_call_gpt_refine)
    return calls


# A Task 2 answer that satisfies the structure-enforcement checks added by
# the "Band 9 structure per task variant" plan (stated position in the
# intro, restated position in the conclusion, no addition-connective
# introducing a new idea) - used wherever a test needs "good" output that
# must actually pass validation, not just have >=4 paragraphs.
def _good_task2_structured_answer(pad_words: int = 0) -> str:
    pad = (" Extra detail sentence padding the word count further along." * pad_words) if pad_words else ""
    return "\n\n".join([
        "I believe that governments should invest more in public health campaigns nationwide." + pad,
        "Firstly, prevention reduces long-term healthcare costs, since treating illness early is far cheaper than treating it once it becomes severe." + pad,
        "Secondly, informed citizens make healthier choices, since well-designed campaigns give people practical steps they can act on immediately." + pad,
        "In conclusion, I believe stronger investment in public health campaigns is worthwhile, since it reduces costs and helps citizens make healthier choices." + pad,
    ])


def test_generate_refine_v2_passes_on_first_attempt(monkeypatch):
    good = {
        "refined_answer": _good_task2_structured_answer(),
        "rewrite_basis": "candidate_material",
        "vocabulary_suggestions": [],
    }
    calls = _task2_call_sequence(monkeypatch, [good])

    result = writing._generate_refined_answer_v2(
        "task_2", None, "question text", "essay text", [], None, [], 250, 2000, None,
    )
    assert calls["n"] == 1
    assert result["diagnostics"]["validation_passed"] is True
    assert result["diagnostics"]["attempts"] == 1
    assert result["refined_answer"] == good["refined_answer"]


def test_generate_refine_v2_retries_once_with_named_issues_then_passes(monkeypatch):
    bad = {"refined_answer": "Only one paragraph, not enough of them here at all really.", "vocabulary_suggestions": []}
    good = {
        "refined_answer": _good_task2_structured_answer(),
        "rewrite_basis": "candidate_material",
        "vocabulary_suggestions": [],
    }
    seen_prompts = []
    call_n = {"n": 0}

    def fake_call_gpt_refine(prompt, image_url=None):
        seen_prompts.append(prompt)
        call_n["n"] += 1
        return bad if call_n["n"] == 1 else good

    monkeypatch.setattr(writing, "call_gpt_refine", fake_call_gpt_refine)

    result = writing._generate_refined_answer_v2(
        "task_2", None, "question text", "essay text", [], None, [], 250, 2000, None,
    )
    assert call_n["n"] == 2
    # The retry prompt must name the specific failure from attempt 1.
    assert "PREVIOUS ATTEMPT" in seen_prompts[1]
    assert "paragraph" in seen_prompts[1].lower()
    assert result["diagnostics"]["validation_passed"] is True
    assert result["diagnostics"]["attempts"] == 2


def test_generate_refine_v2_two_failures_reports_issues_not_silent(monkeypatch):
    bad = {"refined_answer": "Still only one paragraph of text here, not enough at all.", "vocabulary_suggestions": []}
    calls = _task2_call_sequence(monkeypatch, [bad, bad])

    result = writing._generate_refined_answer_v2(
        "task_2", None, "question text", "essay text", [], None, [], 250, 2000, None,
    )
    assert calls["n"] == 2
    assert result["diagnostics"]["validation_passed"] is False
    assert result["diagnostics"]["issues"]
    # Must still return usable text, never blow up or return nothing.
    assert result["refined_answer"]


def test_generate_refine_v2_keeps_longer_text_when_still_over_budget_after_retry(monkeypatch):
    over_budget = {
        "refined_answer": _good_task2_structured_answer(pad_words=20),
        "vocabulary_suggestions": [],
    }
    calls = _task2_call_sequence(monkeypatch, [over_budget, over_budget])

    result = writing._generate_refined_answer_v2(
        "task_2", None, "question text", "essay text", [], None, [], 250, 260, None,
    )
    assert calls["n"] == 2
    # Never truncated - the full over-budget text is kept, not cut (allow
    # the orchestrator's own incidental .strip() of leading/trailing
    # whitespace - that's not content truncation).
    assert result["refined_answer"] == over_budget["refined_answer"].strip()
    assert any(i.startswith("word_budget_exceeded") for i in result["diagnostics"]["issues"])
    # A pure word-budget overshoot alone is non-fatal - not a hard failure.
    assert result["diagnostics"]["validation_passed"] is True


def test_generate_refine_v2_fails_open_when_gpt_call_exhausts_retries(monkeypatch):
    def always_fail(prompt, image_url=None):
        raise ValueError("boom")

    monkeypatch.setattr(writing, "call_gpt_refine", always_fail)

    result = writing._generate_refined_answer_v2(
        "task_2", None, "question text", "the original essay text", [], None, [], 250, 2000, None,
    )
    assert result["refined_answer"] == "the original essay text"
    assert result["diagnostics"]["validation_passed"] is False
    assert "gpt_call_failed_after_retries" in result["diagnostics"]["issues"]


# ---------------------------------------------------------------------------
# Full evaluate_writing() wiring, flag ON vs OFF - GPT calls mocked.
# ---------------------------------------------------------------------------

def _make_bands(true_band):
    return {str(n): (n == true_band) for n in range(1, 10)}


def _complete_task2_bands(band=6):
    flags = _make_bands(band)
    return {
        "task_response_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
    }


def test_evaluate_writing_flag_off_never_calls_call_gpt_refine(monkeypatch):
    monkeypatch.delenv("WRITING_INDEPENDENT_MODEL_ANSWER", raising=False)
    ai_response = {**_complete_task2_bands(), "mistakes": [], "strengths": "x", "improvement": "y"}
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined text. " * 60)

    def fail_if_called(prompt, image_url=None):
        raise AssertionError("call_gpt_refine must not be called when the flag is OFF")

    monkeypatch.setattr(writing, "call_gpt_refine", fail_if_called)

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Some question?"},
        "user_answers": {"text": "Some essay text about the topic at hand for the question given."},
    })
    assert "refine_diagnostics" not in result
    assert "refine_rewrite_basis" not in result


def test_evaluate_writing_flag_on_wires_v2_pipeline_end_to_end(monkeypatch):
    monkeypatch.setenv("WRITING_INDEPENDENT_MODEL_ANSWER", "true")
    ai_response = {
        **_complete_task2_bands(),
        "mistakes": [],
        "strengths": "x",
        "improvement": "y",
        "topic_relevance": "on_topic",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)

    def fail_if_called(prompt, system_msg=None):
        raise AssertionError("legacy call_gpt_text must not be called when the flag is ON")
    monkeypatch.setattr(writing, "call_gpt_text", fail_if_called)

    refine_response = {
        "refined_answer": _good_task2_structured_answer(),
        "vocabulary_suggestions": [
            {"original_phrase": "invest in public health", "stronger_alternative": "channel resources into public health", "placement_context": "opening sentence"}
        ],
    }
    monkeypatch.setattr(writing, "call_gpt_refine", lambda prompt, image_url=None, **kwargs: refine_response)

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Should governments fund public health campaigns?"},
        "user_answers": {"text": "Governments should invest in public health because it saves lives for everyone."},
    })

    assert result["refined_answer"] == refine_response["refined_answer"]
    assert result["refine_diagnostics"]["validation_passed"] is True
    assert result["relevance_status"] == "on_topic"
    assert result["relevance_notice"]["model_answer_source"] == "built from your answer"
    assert any(v["original_phrase"] == "invest in public health" for v in result["vocabulary"])


# ---------------------------------------------------------------------------
# Off-topic/irrelevant answers: relevance_status derivation, structured
# relevance_notice, and model_answer_source - see the "off-topic answers"
# plan. relevance_status/relevance_reasons are inline logic inside
# evaluate_writing(), not standalone functions, so these are tested the
# same way the rest of that function's internal logic already is
# throughout this repo: end-to-end through evaluate_writing() with GPT
# calls mocked.
# ---------------------------------------------------------------------------

_GOOD_REFINE_RESPONSE = {
    "refined_answer": "\n\n".join(["Paragraph with real content here for this test."] * 4),
    "vocabulary_suggestions": [],
}


def _task2_bands_with_relevance(band=6, topic_relevance="on_topic", what_you_did="", what_was_asked=""):
    return {
        **_complete_task2_bands(band),
        "mistakes": [], "strengths": "x", "improvement": "y",
        "topic_relevance": topic_relevance,
        "off_topic_what_you_did": what_you_did,
        "off_topic_what_was_asked": what_was_asked,
    }


def _evaluate_task2_v2(monkeypatch, ai_response, refine_response=None):
    monkeypatch.setenv("WRITING_INDEPENDENT_MODEL_ANSWER", "true")
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_refine", lambda prompt, image_url=None, **kwargs: refine_response or _GOOD_REFINE_RESPONSE)
    return writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": "Should governments fund public health campaigns?"},
        "user_answers": {"text": "Governments should invest in public health because it saves lives for everyone."},
    })


def test_relevance_status_on_topic(monkeypatch):
    result = _evaluate_task2_v2(monkeypatch, _task2_bands_with_relevance(topic_relevance="on_topic"))
    assert result["relevance_status"] == "on_topic"
    assert result["relevance_notice"]["headline"] == ""
    assert result["relevance_notice"]["model_answer_source"] == "built from your answer"


def test_relevance_status_completely_off_topic_maps_to_off_topic(monkeypatch):
    result = _evaluate_task2_v2(
        monkeypatch,
        _task2_bands_with_relevance(
            topic_relevance="completely_off_topic",
            what_you_did="Your answer discusses remote work.",
            what_was_asked="The question asked about foreign-language social problems.",
        ),
    )
    assert result["relevance_status"] == "off_topic"
    notice = result["relevance_notice"]
    assert notice["what_you_did"] == "Your answer discusses remote work."
    assert notice["what_was_asked"] == "The question asked about foreign-language social problems."
    assert notice["model_answer_source"] == "written from the question"
    assert "does not address" in notice["headline"].lower()


def test_relevance_status_partially_off_topic_maps_to_partial_with_topic_drift_reason(monkeypatch):
    result = _evaluate_task2_v2(monkeypatch, _task2_bands_with_relevance(topic_relevance="partially_off_topic"))
    assert result["relevance_status"] == "partially_off_topic"
    assert "drifts away from the question" in result["relevance_notice"]["headline"]
    assert result["relevance_notice"]["model_answer_source"] == "built from your answer"


def test_off_topic_band_scoring_cap_is_unaffected_by_relevance_notice_changes(monkeypatch):
    # Band scoring stays completely separate from the relevance_notice/
    # model-answer machinery - the existing topic_relevance cap must fire
    # exactly as it always has, regardless of the flag.
    result = _evaluate_task2_v2(monkeypatch, _task2_bands_with_relevance(band=9, topic_relevance="completely_off_topic"))
    assert result["criteria_scores"]["task_response"] <= 5.0


def _task1_bands_with_relevance(band=6, topic_relevance="on_topic", image_data_accuracy="not_applicable"):
    flags = {str(n): (n == band) for n in range(1, 10)}
    return {
        "task_achievement_bands": dict(flags), "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags), "grammar_bands": dict(flags),
        "mistakes": [], "strengths": "x", "improvement": "y",
        "topic_relevance": topic_relevance,
        "off_topic_what_you_did": "", "off_topic_what_was_asked": "",
        "image_data_accuracy": image_data_accuracy,
    }


def _evaluate_task1_v2(monkeypatch, ai_response, image_url=None, extract_response=None, refine_response=None):
    monkeypatch.setenv("WRITING_INDEPENDENT_MODEL_ANSWER", "true")
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_refine", lambda prompt, image_url=None, **kwargs: refine_response or _GOOD_REFINE_RESPONSE)
    if extract_response is not None:
        monkeypatch.setattr(writing, "call_gpt_extract", lambda prompt, image_url=None, **kwargs: extract_response)
    metadata = {"task_type": "task_1", "question": "The chart below shows sodium, saturated fat, and added sugar in meals."}
    if image_url:
        metadata["image_url"] = image_url
    return writing.evaluate_writing({
        "metadata": metadata,
        "user_answers": {"text": "The chart shows dinner has the highest sodium at 43 percent."},
    })


def test_relevance_status_wrong_data_maps_to_partially_off_topic(monkeypatch):
    # On-topic in subject, but the chart-data-accuracy self-report is
    # inaccurate - this is the Task-1-specific failure mode Task 2 doesn't
    # have (see the plan's "wrong_data" reason).
    result = _evaluate_task1_v2(
        monkeypatch,
        _task1_bands_with_relevance(topic_relevance="on_topic", image_data_accuracy="partially_inaccurate"),
        image_url="data:image/png;base64,xyz",
        extract_response={"chart_count": 1, "charts": [{"chart_type": "pie_chart", "categories": [{"name": "Sodium"}]}]},
    )
    assert result["relevance_status"] == "partially_off_topic"
    assert "data in your answer doesn't match" in result["relevance_notice"]["headline"]


def test_relevance_status_combines_topic_drift_and_wrong_data_reasons(monkeypatch):
    result = _evaluate_task1_v2(
        monkeypatch,
        _task1_bands_with_relevance(topic_relevance="partially_off_topic", image_data_accuracy="significantly_inaccurate"),
        image_url="data:image/png;base64,xyz",
        extract_response={"chart_count": 1, "charts": [{"chart_type": "pie_chart", "categories": [{"name": "Sodium"}]}]},
    )
    assert result["relevance_status"] == "partially_off_topic"
    headline = result["relevance_notice"]["headline"]
    assert "drifts from the question" in headline and "data doesn't match" in headline


def test_model_answer_source_off_topic_task1_with_successful_chart_extraction(monkeypatch):
    result = _evaluate_task1_v2(
        monkeypatch,
        _task1_bands_with_relevance(topic_relevance="completely_off_topic"),
        image_url="data:image/png;base64,xyz",
        extract_response={"chart_count": 1, "charts": [{"chart_type": "pie_chart", "categories": [{"name": "Sodium"}]}]},
    )
    assert result["relevance_notice"]["model_answer_source"] == "written from the question and chart"


def test_model_answer_source_off_topic_task1_without_image(monkeypatch):
    result = _evaluate_task1_v2(
        monkeypatch,
        _task1_bands_with_relevance(topic_relevance="completely_off_topic"),
        image_url=None,
    )
    assert result["relevance_notice"]["model_answer_source"] == "written from the question"


# ---------------------------------------------------------------------------
# _build_refine_prompt_v2: off_topic must never receive the real essay -
# a structural guarantee, verified directly on the built prompt string,
# not just trusted from an instruction.
# ---------------------------------------------------------------------------

_DISTINCTIVE_ESSAY_MARKER = "xyzzy_unique_candidate_phrase_12345"


def test_build_refine_prompt_v2_omits_essay_when_off_topic():
    prompt = writing._build_refine_prompt_v2(
        "task_2", None, "question text", f"essay containing {_DISTINCTIVE_ESSAY_MARKER}", [],
        None, [], 250, 260, relevance_status="off_topic",
    )
    assert _DISTINCTIVE_ESSAY_MARKER not in prompt


def test_build_refine_prompt_v2_includes_essay_when_partially_off_topic():
    prompt = writing._build_refine_prompt_v2(
        "task_2", None, "question text", f"essay containing {_DISTINCTIVE_ESSAY_MARKER}", [],
        None, [], 250, 260, relevance_status="partially_off_topic",
    )
    assert _DISTINCTIVE_ESSAY_MARKER in prompt
    assert "keep the on-topic" in prompt.lower()


def test_build_refine_prompt_v2_includes_essay_when_on_topic():
    prompt = writing._build_refine_prompt_v2(
        "task_2", None, "question text", f"essay containing {_DISTINCTIVE_ESSAY_MARKER}", [],
        None, [], 250, 260, relevance_status="on_topic",
    )
    assert _DISTINCTIVE_ESSAY_MARKER in prompt


# ---------------------------------------------------------------------------
# _validate_refine_output: off-topic content-leak check.
# ---------------------------------------------------------------------------

def test_validate_refine_output_flags_leaked_content_when_off_topic():
    essay = (
        "Remote work has transformed daily commuting patterns for millions "
        "of employees across major metropolitan areas worldwide recently."
    )
    # A "fresh" answer that actually reuses the essay's distinctive content
    # words - simulates the off-topic path failing to stay independent.
    leaked = {
        "refined_answer": "\n\n".join([
            "Remote work has transformed daily commuting patterns significantly.",
            "Employees across metropolitan areas now work differently.",
            "This transformation affects millions of workers worldwide.",
            "Commuting patterns continue changing across major cities.",
        ]),
        "vocabulary_suggestions": [],
    }
    issues = writing._validate_refine_output(
        leaked, None, [], [], essay, "task_2", None, 250, 2000, relevance_status="off_topic",
    )
    assert any("leaked candidate content" in i for i in issues)


def test_validate_refine_output_passes_genuinely_fresh_content_when_off_topic():
    essay = (
        "Remote work has transformed daily commuting patterns for millions "
        "of employees across major metropolitan areas worldwide recently."
    )
    fresh = {
        "refined_answer": "\n\n".join([
            "Governments should invest heavily in public health campaigns nationwide.",
            "Preventative medicine reduces long-term healthcare costs substantially.",
            "Community clinics provide accessible screening for vulnerable populations.",
            "Sustained funding ensures lasting improvements in population wellbeing.",
        ]),
        "vocabulary_suggestions": [],
    }
    issues = writing._validate_refine_output(
        fresh, None, [], [], essay, "task_2", None, 250, 2000, relevance_status="off_topic",
    )
    assert not any("leaked candidate content" in i for i in issues)


def test_validate_refine_output_skips_leak_check_when_on_topic():
    # The leak check is off_topic-specific - an on_topic rewrite is
    # SUPPOSED to share content with the essay (it's built from it), so
    # the same overlap that fails the off_topic check must never fire here.
    essay = "Remote work has transformed daily commuting patterns for millions of employees."
    similar = {
        "refined_answer": "\n\n".join(["Remote work has transformed daily commuting patterns considerably."] * 4),
        "vocabulary_suggestions": [],
    }
    issues = writing._validate_refine_output(
        similar, None, [], [], essay, "task_2", None, 250, 2000, relevance_status="on_topic",
    )
    assert not any("leaked candidate content" in i for i in issues)
