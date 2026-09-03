import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing


# ---------------------------------------------------------------------------
# evaluate_writing(): "feedback" and "improvement" in the returned result
# must reflect what the model actually said. Regression tests for a real
# bug: the code read ai.get("examiner_response", ...) and
# ai.get("feedback", {}).get("improvements", ...) - keys that don't exist
# anywhere in the prompt's actual RESPONSE FORMAT ("strengths" and
# "improvement" are the real top-level keys) - so both fields were ALWAYS
# empty and every essay silently fell back to the same two hardcoded
# generic strings regardless of task or content, defeating the prompt's
# own "no generic/boilerplate praise" instruction.
# ---------------------------------------------------------------------------

def _install_fake_gpt(monkeypatch, ai_response, refined_text="Refined essay text."):
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: refined_text)


def _evaluate(question="Should governments fund public health campaigns?",
              essay="Governments should invest in public health because it saves lives and reduces long-term costs for everyone in society as a whole."):
    return writing.evaluate_writing({
        "metadata": {"task_type": "task_2", "question": question},
        "user_answers": {"text": essay},
    })


def _complete_task2_bands(band=6):
    # A minimal but COMPLETE checklist shape - all 4 required band grids
    # present - so these tests exercise only what they intend to (feedback/
    # improvement extraction) without tripping the completeness validation
    # in evaluate_writing() that rejects a response missing required keys.
    flags = {str(n): (n == band) for n in range(1, 10)}
    return {
        "task_response_bands": dict(flags),
        "coherence_cohesion_bands": dict(flags),
        "lexical_resource_bands": dict(flags),
        "grammar_bands": dict(flags),
    }


def test_evaluate_writing_surfaces_real_strengths_and_improvement(monkeypatch):
    ai_response = {
        **_complete_task2_bands(),
        "mistakes": [],
        "strengths": "The essay effectively uses the collocation 'public health campaigns' and develops a balanced argument.",
        "improvement": "Work on linking paragraphs more explicitly with discourse markers like 'Furthermore'.",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate()

    assert result["feedback"] == ai_response["strengths"]
    assert result["improvement"] == ai_response["improvement"]
    # Must NOT be the generic fallback strings.
    assert result["feedback"] != "Clear and concise answer; focus on stronger linking."
    assert result["improvement"] != "Improve coherence with clearer transitions."


def test_evaluate_writing_falls_back_when_fields_genuinely_missing(monkeypatch):
    ai_response = {
        **_complete_task2_bands(),
        "mistakes": [],
        # strengths/improvement genuinely absent this time
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate()

    assert result["feedback"] == "Clear and concise answer; focus on stronger linking."
    assert result["improvement"] == "Improve coherence with clearer transitions."


def test_evaluate_writing_gives_different_feedback_for_different_essays(monkeypatch):
    # Two different essays with two different real "strengths" values must
    # produce two different "feedback" results, not the same boilerplate
    # both times - the exact symptom of the original bug.
    _install_fake_gpt(monkeypatch, {
        **_complete_task2_bands(),
        "mistakes": [],
        "strengths": "Strength specific to essay A.",
        "improvement": "Improvement specific to essay A.",
    })
    result_a = _evaluate(essay="Essay A content about health policy and government responsibility for citizens.")

    _install_fake_gpt(monkeypatch, {
        **_complete_task2_bands(),
        "mistakes": [],
        "strengths": "Strength specific to essay B.",
        "improvement": "Improvement specific to essay B.",
    })
    result_b = _evaluate(essay="Essay B content about technology and its impact on modern society.")

    assert result_a["feedback"] != result_b["feedback"]
    assert result_a["improvement"] != result_b["improvement"]


# ---------------------------------------------------------------------------
# Task 1 conjunctive band checklist (writing_task1_common.txt + the
# Academic/General Training criteria file, official Task 1 descriptors):
# GPT reports which features of each band hold, and
# _highest_fully_met_band() deterministically picks the band, the same
# pattern as evaluators/speaking_audio.py's checklist scoring.
# ---------------------------------------------------------------------------

def _make_bands(true_band):
    return {str(n): (n == true_band) for n in range(1, 10)}


def _realistic_essay(word_count):
    """A placeholder essay of exactly `word_count` words, but with real
    sentence punctuation and paragraph breaks - unlike a bare
    " ".join(["word"] * N) block, which evaluate_writing()'s
    deterministic structural caps (_coherence_paragraph_cap /
    _grammar_punctuation_cap in evaluators/writing.py) would otherwise
    flag as "no paragraph breaks" / "no sentence boundaries", an
    unrelated confound for tests that are actually about the checklist-
    based band picking, not about formatting."""
    filler = ["This", "argument", "shows", "clearly", "that", "the",
              "situation", "requires", "careful", "consideration"]
    words = (filler * ((word_count // len(filler)) + 1))[:word_count]
    sentences = [" ".join(words[i:i + 10]) + "." for i in range(0, len(words), 10)]
    paragraphs = [" ".join(sentences[i:i + 4]) for i in range(0, len(sentences), 4)]
    return "\n\n".join(paragraphs)


def _evaluate_task1(essay, question="Describe the chart below."):
    return writing.evaluate_writing({
        "metadata": {"task_type": "task_1", "question": question},
        "user_answers": {"text": essay},
    })


def test_evaluate_writing_task1_picks_band_from_checklist(monkeypatch):
    ai_response = {
        "task_achievement_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(6),
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate_task1(_realistic_essay(160))

    assert result["criteria_scores"] == {
        "task_response": 7.0,
        "coherence_cohesion": 6.0,
        "lexical_resource": 7.0,
        "grammar_accuracy": 6.0,
    }


def test_evaluate_writing_task1_defaults_to_neutral_when_checklist_missing(monkeypatch):
    # No band grids at all in the response - this is now caught by the
    # completeness validation in evaluate_writing() (a response missing
    # its required checklist keys is treated the same as a failed call,
    # not silently accepted) and falls back to the neutral default - the
    # same "don't silently produce a worst-case score" principle already
    # applied to a total API failure, not the old "quietly default to
    # Band 1" behavior, which was itself inconsistent with that
    # principle. Must not crash.
    _install_fake_gpt(monkeypatch, {"mistakes": [], "strengths": "x", "improvement": "y"})

    result = _evaluate_task1(_realistic_essay(160))
    assert result["ai_evaluation_failed"] is True
    assert all(v == 5.0 for v in result["criteria_scores"].values())


def test_evaluate_writing_task1_enforces_band_1_word_count_rule(monkeypatch):
    # The descriptors' own mechanical rule: "Responses of 20 words or
    # fewer are rated at Band 1." Must override even an (incorrectly)
    # generous GPT checklist.
    ai_response = {
        "task_achievement_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate_task1("This chart shows very little information overall.")  # 8 words

    assert all(v == 1.0 for v in result["criteria_scores"].values())


def test_evaluate_writing_task2_uses_checklist_with_task_response_key(monkeypatch):
    # Task 2 uses the official Task 2 descriptors' own conjunctive
    # checklist too, keyed "task_response_bands" (Task 2's first
    # criterion is "Task Response", not "Task Achievement" like Task 1).
    ai_response = {
        "task_response_bands": _make_bands(6),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(6),
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate(essay=_realistic_essay(260))

    assert result["criteria_scores"]["task_response"] == 6.0
    assert result["criteria_scores"]["lexical_resource"] == 7.0


def test_evaluate_writing_task2_enforces_band_1_word_count_rule(monkeypatch):
    ai_response = {
        "task_response_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate(essay="Short answer well under twenty words total here.")  # 8 words

    assert all(v == 1.0 for v in result["criteria_scores"].values())


def test_evaluate_writing_no_extra_band_cap_for_moderately_short_task1(monkeypatch):
    # There used to be an additional overall-band cap for "moderately"
    # underlength answers (Task 1 < 100 words -> capped at 5.5) with no
    # basis in the official IDP descriptors, which only state (1) <=20
    # words is Band 1 and (2) Lexical Resource Band 3 MAY apply "due to
    # the response being significantly underlength" - a per-criterion
    # checklist judgment, not a fixed overall-band threshold. A genuinely
    # strong 40-word answer (per GPT's own checklist) must now score its
    # real band, not get silently capped by word count alone.
    ai_response = {
        "task_achievement_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate_task1(_realistic_essay(40))  # under the old 50/100 thresholds

    assert result["overall_band"] == 7.0
    assert all(v == 7.0 for v in result["criteria_scores"].values())


def test_evaluate_writing_no_extra_band_cap_for_moderately_short_task2(monkeypatch):
    ai_response = {
        "task_response_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate(essay=_realistic_essay(60))  # under the old 80/150 thresholds

    assert result["overall_band"] == 7.0
    assert all(v == 7.0 for v in result["criteria_scores"].values())


def test_evaluate_writing_falls_back_to_neutral_band_on_total_failure(monkeypatch):
    # A genuine total failure (safe_gpt_call exhausts retries and returns
    # the fallback) must default to a NEUTRAL band (5), not the worst one
    # (1) - the same class of bug already fixed once in
    # evaluators/speaking_audio.py's generate_scores().
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: (_ for _ in ()).throw(ValueError("boom")))
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "refined text")

    result = _evaluate(essay=_realistic_essay(260))

    assert all(v == 5.0 for v in result["criteria_scores"].values())


# ---------------------------------------------------------------------------
# Topic relevance detection + off-topic Band 9 model answer handling.
# GPT self-reports "topic_relevance" in the same checklist call (same
# pattern as evaluators/speaking_audio.py's generate_scores()). Python
# validates it, applies a deterministic band cap, surfaces a
# "relevance_notice", and - the key new behaviour requested - builds the
# Band 9 model answer differently depending on relevance: refine the
# candidate's REAL essay when on-topic, but write a FRESH answer straight
# from the question when off-topic, since "refining" an off-topic essay
# would just produce a more fluent version of the wrong answer. Both paths
# must also carry explicit IELTS paragraph-structure instructions.
# ---------------------------------------------------------------------------

def _install_fake_gpt_capturing_refine_prompt(monkeypatch, ai_response, captured):
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)

    def fake_call_gpt_text(prompt, system_msg=None):
        captured["refine_prompt"] = prompt
        return "Refined answer text."

    monkeypatch.setattr(writing, "call_gpt_text", fake_call_gpt_text)


def test_evaluate_writing_on_topic_refines_the_real_essay(monkeypatch):
    captured = {}
    essay = "Governments should invest heavily in public transport infrastructure for cities."
    ai_response = {
        "task_response_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_refine_prompt(monkeypatch, ai_response, captured)

    result = _evaluate(essay=essay)

    assert essay in captured["refine_prompt"]
    assert result["topic_relevance"] == "on_topic"
    assert result["relevance_notice"] is None


def test_evaluate_writing_completely_off_topic_writes_fresh_answer_and_caps_band(monkeypatch):
    captured = {}
    question = "Should governments fund public health campaigns?"
    off_topic_essay = "My favorite hobby is playing football with my friends every weekend."
    ai_response = {
        "task_response_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "completely_off_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_refine_prompt(monkeypatch, ai_response, captured)

    result = _evaluate(question=question, essay=off_topic_essay)

    # The fresh Band 9 prompt must be built from the question, not the
    # candidate's off-topic essay.
    assert off_topic_essay not in captured["refine_prompt"]
    assert question in captured["refine_prompt"]
    assert result["topic_relevance"] == "completely_off_topic"
    assert result["relevance_notice"] == writing.RELEVANCE_NOTICE_MESSAGES["completely_off_topic"]
    # Deterministic band cap, even though the checklist itself claimed 7s.
    assert result["overall_band"] <= 5.0
    for score in result["criteria_scores"].values():
        assert score <= 5.0


def test_evaluate_writing_partially_off_topic_caps_band_at_six(monkeypatch):
    captured = {}
    ai_response = {
        "task_response_bands": _make_bands(8),
        "coherence_cohesion_bands": _make_bands(8),
        "lexical_resource_bands": _make_bands(8),
        "grammar_bands": _make_bands(8),
        "topic_relevance": "partially_off_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_refine_prompt(monkeypatch, ai_response, captured)

    result = _evaluate()

    assert result["topic_relevance"] == "partially_off_topic"
    assert result["relevance_notice"] == writing.RELEVANCE_NOTICE_MESSAGES["partially_off_topic"]
    assert result["overall_band"] <= 6.0
    for score in result["criteria_scores"].values():
        assert score <= 6.0


def test_evaluate_writing_invalid_topic_relevance_defaults_to_on_topic(monkeypatch):
    ai_response = {
        "task_response_bands": _make_bands(6),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(6),
        "grammar_bands": _make_bands(6),
        "topic_relevance": "not_a_real_value",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate()

    assert result["topic_relevance"] == "on_topic"
    assert result["relevance_notice"] is None


def test_evaluate_writing_refine_prompt_includes_paragraph_structure_task1(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_refine_prompt(monkeypatch, ai_response, captured)

    _evaluate_task1(_realistic_essay(160))

    assert writing.PARAGRAPH_STRUCTURE_INSTRUCTIONS["task_1"] in captured["refine_prompt"]


def test_evaluate_writing_refine_prompt_includes_paragraph_structure_task2_off_topic(monkeypatch):
    captured = {}
    ai_response = {
        "task_response_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "completely_off_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_refine_prompt(monkeypatch, ai_response, captured)

    _evaluate()

    assert writing.PARAGRAPH_STRUCTURE_INSTRUCTIONS["task_2"] in captured["refine_prompt"]


# ---------------------------------------------------------------------------
# Mistake severity classification + escalation, from the IELTS Writing
# error-taxonomy reference doc. GPT self-reports "severity"
# ("minor"/"significant") and "category" (a short slug) per mistake in the
# same checklist call; Python validates/defaults severity (mirrors
# Speaking's generate_mistakes() severity pattern) and applies two
# deterministic escalations the taxonomy specifies:
# 1. Frequency: the SAME minor (type, category) pair recurring 4+ times
#    across the answer is systematic, not invisible - escalate all of them.
# 2. Cluster: 3+ errors landing in the SAME sentence compound each other's
#    impact on the reader - escalate all of them, regardless of type.
# Sentences below are deliberately non-overlapping so each mechanism can be
# isolated (see the real bug this session found: two flagged phrases
# sharing one sentence unintentionally also triggers cluster escalation -
# which is correct behaviour, not a false positive, but means test data
# must be built carefully to isolate one mechanism at a time).
# ---------------------------------------------------------------------------

_SEVERITY_TEST_ESSAY = (
    "The government should invest in education for citizens. "
    "It is good idea to support students. "
    "Economical growth benefits society greatly. "
    "Country need better infrastructure overall. "
    "People depend of internet for daily tasks in nowadays society. "
    "Government support economic growth policies. "
    "It is a good idea to review this."
)

_SEVERITY_TEST_MISTAKES = [
    {"type": "grammar", "category": "article", "severity": "minor", "original": "invest in education", "corrected": "invest in the education", "explanation": "missing article"},
    # Item 2 (verbatim filter now requires 3+ words + unique occurrence):
    # these three used to be bare 2-word spans ("good idea", "Economical
    # growth", "Country need") - lengthened to genuine 3-4 word substrings
    # of the same sentences, same missing-article intent, still each
    # occurring exactly once in _SEVERITY_TEST_ESSAY.
    {"type": "grammar", "category": "article", "severity": "minor", "original": "is good idea to", "corrected": "is a good idea to", "explanation": "missing article"},
    {"type": "grammar", "category": "article", "severity": "minor", "original": "Economical growth benefits society", "corrected": "The economical growth benefits society", "explanation": "missing article"},
    {"type": "grammar", "category": "article", "severity": "minor", "original": "Country need better infrastructure", "corrected": "The country needs better infrastructure", "explanation": "missing article"},
    {"type": "grammar", "category": "preposition", "severity": "minor", "original": "depend of internet", "corrected": "depend on the internet", "explanation": "wrong preposition"},
    {"type": "lexical", "category": "formulaic", "severity": "minor", "original": "in nowadays society", "corrected": "in today's society", "explanation": "awkward phrasing"},
    {"type": "grammar", "category": "word_order", "severity": "minor", "original": "for daily tasks", "corrected": "for their daily tasks", "explanation": "missing article"},
    {"type": "lexical", "category": "spelling", "severity": "minor", "original": "economic growth policies", "corrected": "economic growth strategies", "explanation": "word choice"},
    {"type": "coherence", "category": "linking", "severity": "not_a_real_value", "original": "It is a good idea to review this", "corrected": "It is a good idea to review this.", "explanation": "weak conclusion"},
]


def _evaluate_severity_case(monkeypatch):
    ai_response = {
        "task_response_bands": _make_bands(6),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(6),
        "grammar_bands": _make_bands(6),
        "topic_relevance": "on_topic",
        "mistakes": [dict(m) for m in _SEVERITY_TEST_MISTAKES],
        "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)
    return _evaluate(essay=_SEVERITY_TEST_ESSAY)


def test_evaluate_writing_escalates_frequent_minor_category_to_systematic(monkeypatch):
    result = _evaluate_severity_case(monkeypatch)

    article_mistakes = [m for m in result["mistakes"] if m["category"] == "article"]
    assert len(article_mistakes) == 4
    assert all(m["severity"] == "significant" for m in article_mistakes)
    assert all(m["escalated_to"] == "systematic" for m in article_mistakes)
    assert all(m["occurrence_count"] == 4 for m in article_mistakes)


def test_evaluate_writing_leaves_isolated_minor_mistake_unescalated(monkeypatch):
    result = _evaluate_severity_case(monkeypatch)

    by_original = {m["original"]: m for m in result["mistakes"]}
    spelling = by_original["economic growth policies"]
    assert spelling["severity"] == "minor"
    assert "escalated_to" not in spelling


def test_evaluate_writing_defaults_invalid_severity_to_significant(monkeypatch):
    result = _evaluate_severity_case(monkeypatch)

    by_original = {m["original"]: m for m in result["mistakes"]}
    invalid = by_original["It is a good idea to review this"]
    assert invalid["severity"] == "significant"
    assert "escalated_to" not in invalid


def test_evaluate_writing_escalates_error_cluster_in_same_sentence(monkeypatch):
    result = _evaluate_severity_case(monkeypatch)

    cluster_mistakes = [m for m in result["mistakes"] if m["category"] in ("preposition", "formulaic", "word_order")]
    assert len(cluster_mistakes) == 3
    assert all(m["severity"] == "significant" for m in cluster_mistakes)
    assert all(m["escalated_to"] == "cluster" for m in cluster_mistakes)


def test_evaluate_writing_result_includes_severity_legend(monkeypatch):
    result = _evaluate_severity_case(monkeypatch)

    assert result["severity_legend"] == writing.SEVERITY_LEGEND
    assert "minor" in result["severity_legend"]
    assert "significant" in result["severity_legend"]


# ---------------------------------------------------------------------------
# Writing Task 1 Academic vs General Training: only the Task Achievement
# criterion genuinely differs between the two (Academic = chart/graph/
# diagram report, General Training = letter) - Coherence & Cohesion,
# Lexical Resource, and Grammatical Range & Accuracy stay one shared/
# common checklist for both. The old approach merged both variants into
# ONE prompt with inline "(Academic: X) OR (General Training: Y)"
# alternatives and asked GPT to infer which applied - now the variant is
# detected deterministically in Python (_detect_task1_variant) and only
# the genuinely-differing Task Achievement block is swapped in from a
# separate file; the shared criteria live in exactly one place
# (writing_task1_common.txt), never duplicated.
# ---------------------------------------------------------------------------

def _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured):
    def fake_call_gpt_writing(prompt, image_url=None, **kwargs):
        captured["prompt"] = prompt
        captured["image_url"] = image_url
        return ai_response

    monkeypatch.setattr(writing, "call_gpt_writing", fake_call_gpt_writing)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined text. " * 60)


def test_detect_task1_variant_academic_by_default():
    assert writing._detect_task1_variant(
        "The chart below shows subject enrollment in 2020.",
        "The chart illustrates the proportion of students who studied various subjects.",
    ) == "academic"


def test_detect_task1_variant_letter_salutation():
    assert writing._detect_task1_variant(
        "Write a letter to your landlord about a repair.",
        "Dear Mr. Smith,\n\nI am writing about the broken heater in my apartment.",
    ) == "general"


def test_detect_task1_variant_letter_signoff():
    assert writing._detect_task1_variant(
        "Some prompt with no explicit letter mention.",
        "Thank you for your consideration.\n\nYours faithfully,\nA. Candidate",
    ) == "general"


def test_detect_task1_variant_letter_from_question_wording():
    assert writing._detect_task1_variant(
        "Write a letter to a friend inviting them to visit.",
        "I hope you are doing well and I wanted to invite you to stay with us.",
    ) == "general"


def test_evaluate_writing_task1_academic_loads_academic_criteria_only(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    result = _evaluate_task1(
        "The chart illustrates the proportion of students who studied various subjects in 2020.",
        question="The chart below shows subject enrollment.",
    )

    prompt = captured["prompt"]
    assert "<<<TASK_ACHIEVEMENT_CHECKLIST>>>" not in prompt
    assert "TASK 1 - ACADEMIC - OFFICIAL BAND DESCRIPTORS" in prompt
    assert "TASK ACHIEVEMENT CATEGORY REFERENCE (Academic)" in prompt
    assert "TASK 1 - GENERAL TRAINING - OFFICIAL BAND DESCRIPTORS" not in prompt
    assert "TASK ACHIEVEMENT CATEGORY REFERENCE (General Training)" not in prompt
    assert result["task1_variant"] == "academic"


def test_evaluate_writing_task1_general_loads_general_criteria_only(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    letter_essay = "Dear Sir or Madam,\n\nI am writing to complain about the service I received.\n\nYours faithfully,\nJohn Smith"
    result = _evaluate_task1(letter_essay, question="Write a letter to the manager of a hotel.")

    prompt = captured["prompt"]
    assert "<<<TASK_ACHIEVEMENT_CHECKLIST>>>" not in prompt
    assert "TASK 1 - GENERAL TRAINING - OFFICIAL BAND DESCRIPTORS" in prompt
    assert "TASK ACHIEVEMENT CATEGORY REFERENCE (General Training)" in prompt
    assert "TASK 1 - ACADEMIC - OFFICIAL BAND DESCRIPTORS" not in prompt
    assert "TASK ACHIEVEMENT CATEGORY REFERENCE (Academic)" not in prompt
    assert result["task1_variant"] == "general"


def test_evaluate_writing_task1_common_criteria_identical_across_variants(monkeypatch):
    captured_academic = {}
    captured_general = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }

    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured_academic)
    _evaluate_task1("A chart-style report about data.", question="The chart below shows data.")

    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured_general)
    _evaluate_task1("Dear Sir,\n\nI am writing about an issue.\n\nYours faithfully,\nA.", question="Write a letter.")

    for shared_sentence in (
        "The message can be followed effortlessly.",  # CC Band 9
        "Full flexibility and precise use are evident within the scope of the",  # LR Band 9
        "A wide range of structures within the scope of the task is used with",  # GRA Band 9
    ):
        assert shared_sentence in captured_academic["prompt"]
        assert shared_sentence in captured_general["prompt"]


# ---------------------------------------------------------------------------
# Descriptor fidelity against the actual IDP IELTS Band Descriptors
# document. A careful band-by-band comparison against the real document
# found several gaps in the previously-reconstructed prompt text: Band 1
# across Task Achievement/Coherence & Cohesion/Lexical Resource/
# Grammatical Range used a generic placeholder ("there is little or no
# evidence...") instead of the descriptors' own actual Band 1 wording
# ("Responses of 20 words or fewer are rated at Band 1..."), several
# bands were missing a shared opening/closing sentence entirely (e.g.
# Band 8 CC's "The message can be followed with ease", Band 6 LR's
# "risk-taker" sentence), Academic
# Band 4 TA was missing its shared closing sentences, and GRA Band 2 had
# an extra unsanctioned sentence not in the source. These tests pin the
# corrected wording so it can't silently regress.
# ---------------------------------------------------------------------------

def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text)


def test_evaluate_writing_task1_academic_band4_includes_shared_closing_sentences(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    _evaluate_task1("The chart illustrates various trends.", question="The chart below shows data.")

    prompt = _normalize_whitespace(captured["prompt"]).lower()
    assert "the format may be inappropriate." in prompt
    assert "key features which are presented may be irrelevant, repetitive, inaccurate or inappropriate" in prompt


def test_evaluate_writing_task1_band1_uses_real_descriptor_wording_not_placeholder(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    _evaluate_task1("The chart illustrates various trends.", question="The chart below shows data.")

    prompt = _normalize_whitespace(captured["prompt"]).lower()
    assert "responses of 20 words or fewer are rated at band 1" in prompt
    assert "no rateable language is evident" in prompt
    assert "no resource is apparent, except for a few isolated words" in prompt
    assert "there is little or no evidence of organisation at all" not in prompt  # old placeholder text


def test_evaluate_writing_task1_common_criteria_includes_previously_dropped_sentences(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    _evaluate_task1("The chart illustrates various trends.", question="The chart below shows data.")

    prompt = _normalize_whitespace(captured["prompt"]).lower()
    assert "the message can be followed with ease" in prompt  # CC Band 8, was missing
    assert "if the writer is a risk-taker" in prompt  # LR Band 6, was missing
    # Task 1's CC Band 7 must NOT carry Task 2's paragraphing clause - that
    # cross-contamination was a separate, confirmed bug fixed by removing
    # it from Task 1's descriptors, not by adding it.
    assert "paragraphing is generally used effectively to support overall" not in prompt


def test_evaluate_writing_task1_academic_and_general_no_cross_contamination(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)
    _evaluate_task1("The chart illustrates various trends.", question="The chart below shows data.")
    academic_ta_block = captured["prompt"].split("TASK ACHIEVEMENT")[1].split("COHERENCE AND COHESION")[0]

    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)
    letter_essay = "Dear Sam,\n\nI hope you are well.\n\nYours sincerely,\nAlex"
    _evaluate_task1(letter_essay, question="Write a letter to a friend.")
    gt_ta_block = captured["prompt"].split("TASK ACHIEVEMENT")[1].split("COHERENCE AND COHESION")[0]

    for forbidden in ("overview", "trends", "figures/data", "key features"):
        assert forbidden not in gt_ta_block.lower()
    for forbidden in ("bullet points", "purpose of the letter", "tone is consistent"):
        assert forbidden not in academic_ta_block.lower()


# ---------------------------------------------------------------------------
# Academic Task 1 chart/graph/diagram image verification. Previously the
# model only ever saw the essay TEXT, never the actual image, so it could
# not verify whether the candidate's stated figures/trends genuinely
# matched the real chart - only whether the writing sounded coherent and
# complete. An optional image_url now gets attached as a real image to
# the vision-capable model, restricted to Academic Task 1 only (General
# Training's letter and Task 2 have no chart to verify against, so an
# image_url must never reach GPT for those, even if a caller mistakenly
# sends one).
# ---------------------------------------------------------------------------

_FAKE_IMAGE_URL = "https://example.com/charts/test18-bar-chart.png"


def test_evaluate_writing_task1_academic_with_image_forwards_it_to_gpt(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_1", "question": "The chart below shows data.", "image_url": _FAKE_IMAGE_URL},
        "user_answers": {"text": "The chart illustrates various trends over time."},
    })

    assert captured["image_url"] == _FAKE_IMAGE_URL
    assert "IMAGE VERIFICATION" in captured["prompt"]
    assert result["image_verification_used"] is True
    assert result["task1_variant"] == "academic"


def test_evaluate_writing_task1_academic_without_image_is_unaffected(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_1", "question": "The chart below shows data."},
        "user_answers": {"text": "The chart illustrates various trends over time."},
    })

    assert captured["image_url"] is None
    # A broad "IMAGE VERIFICATION" substring check isn't precise enough -
    # the schema instructions reference that section by name even when no
    # image is attached ("...not_applicable" unless attached). Check for
    # the actual instruction block content instead, which is only ever
    # substituted in when an image genuinely was provided.
    assert "Look at the actual image and verify" not in captured["prompt"]
    assert "<<<IMAGE_VERIFICATION_INSTRUCTIONS>>>" not in captured["prompt"]
    assert result["image_verification_used"] is False


def test_evaluate_writing_task1_general_training_ignores_image_url(monkeypatch):
    # A letter has no chart to verify against - even if a caller
    # mistakenly attaches an image_url, it must never reach GPT.
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    letter_essay = "Dear Sam,\n\nI hope you are well.\n\nYours sincerely,\nAlex"
    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_1", "question": "Write a letter to a friend.", "image_url": _FAKE_IMAGE_URL},
        "user_answers": {"text": letter_essay},
    })

    assert captured["image_url"] is None
    # A broad "IMAGE VERIFICATION" substring check isn't precise enough -
    # the schema instructions reference that section by name even when no
    # image is attached ("...not_applicable" unless attached). Check for
    # the actual instruction block content instead, which is only ever
    # substituted in when an image genuinely was provided.
    assert "Look at the actual image and verify" not in captured["prompt"]
    assert result["task1_variant"] == "general"
    assert result["image_verification_used"] is False


# ---------------------------------------------------------------------------
# Deterministic image_data_accuracy cap. Attaching the image alone was
# only a soft prompt instruction - GPT could be told to check the data
# but nothing enforced it. GPT now self-reports "image_data_accuracy" in
# the same checklist call, and Python deterministically caps Task
# Achievement (only TA - CC/LR/GR are about writing quality, not whether
# the specific figures are correct) when it indicates a genuine mismatch,
# the same pattern already used for topic_relevance capping the whole
# criteria set for an off-topic answer.
# ---------------------------------------------------------------------------

_IMAGE_TASK1_ESSAY = (
    "The chart illustrates various trends over time in great detail, showing how "
    "spending patterns changed across each of the given categories throughout the period."
)


def _evaluate_task1_with_image(image_data_accuracy, monkeypatch, image_url=_FAKE_IMAGE_URL):
    ai_response = {
        "task_achievement_bands": _make_bands(8), "coherence_cohesion_bands": _make_bands(8),
        "lexical_resource_bands": _make_bands(8), "grammar_bands": _make_bands(8),
        "topic_relevance": "on_topic", "image_data_accuracy": image_data_accuracy,
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined text. " * 60)
    data = {"metadata": {"task_type": "task_1", "question": "The chart below shows data."}, "user_answers": {"text": _IMAGE_TASK1_ESSAY}}
    if image_url is not None:
        data["metadata"]["image_url"] = image_url
    return writing.evaluate_writing(data)


def test_evaluate_writing_significantly_inaccurate_image_data_caps_task_achievement_only(monkeypatch):
    result = _evaluate_task1_with_image("significantly_inaccurate", monkeypatch)

    assert result["criteria_scores"]["task_response"] == 5.0
    assert result["criteria_scores"]["coherence_cohesion"] == 8.0
    assert result["criteria_scores"]["lexical_resource"] == 8.0
    assert result["criteria_scores"]["grammar_accuracy"] == 8.0
    assert result["image_data_accuracy"] == "significantly_inaccurate"
    assert result["image_accuracy_notice"] == writing.IMAGE_ACCURACY_NOTICE_MESSAGES["significantly_inaccurate"]


def test_evaluate_writing_partially_inaccurate_image_data_caps_task_achievement_at_six(monkeypatch):
    result = _evaluate_task1_with_image("partially_inaccurate", monkeypatch)

    assert result["criteria_scores"]["task_response"] == 6.0
    assert result["criteria_scores"]["coherence_cohesion"] == 8.0


def test_evaluate_writing_accurate_image_data_applies_no_cap(monkeypatch):
    result = _evaluate_task1_with_image("accurate", monkeypatch)

    assert result["criteria_scores"]["task_response"] == 8.0
    assert result["image_accuracy_notice"] is None


def test_evaluate_writing_ignores_image_data_accuracy_when_no_image_was_sent(monkeypatch):
    # Defensive: even if GPT hallucinates a verdict, it must never affect
    # scoring unless an image was genuinely attached to this call.
    result = _evaluate_task1_with_image("significantly_inaccurate", monkeypatch, image_url=None)

    assert result["criteria_scores"]["task_response"] == 8.0
    assert result["image_data_accuracy"] == "not_applicable"


def test_evaluate_writing_invalid_image_data_accuracy_defaults_to_not_applicable(monkeypatch):
    result = _evaluate_task1_with_image("some garbage value", monkeypatch)

    assert result["image_data_accuracy"] == "not_applicable"
    assert result["criteria_scores"]["task_response"] == 8.0


# ---------------------------------------------------------------------------
# image_verification_incomplete: real live test data surfaced a gap where
# an image WAS genuinely sent to GPT (image_verification_used: true) but
# GPT still reported "image_data_accuracy": "not_applicable" - despite the
# prompt now explicitly forbidding that combination - meaning it skipped
# the actual verification task. This flag detects and surfaces exactly
# that case, so it's visible/monitorable rather than silently accepted.
# ---------------------------------------------------------------------------

def test_evaluate_writing_flags_incomplete_verification_when_gpt_skips_it(monkeypatch):
    result = _evaluate_task1_with_image("not_applicable", monkeypatch)

    assert result["image_verification_incomplete"] is True
    assert result["image_data_accuracy"] == "not_applicable"


def test_evaluate_writing_no_incomplete_flag_when_gpt_gives_a_real_verdict(monkeypatch):
    result = _evaluate_task1_with_image("accurate", monkeypatch)

    assert result["image_verification_incomplete"] is False


def test_evaluate_writing_no_incomplete_flag_when_no_image_was_sent(monkeypatch):
    result = _evaluate_task1_with_image("not_applicable", monkeypatch, image_url=None)

    assert result["image_verification_incomplete"] is False


def test_evaluate_writing_incomplete_flag_true_when_field_entirely_missing(monkeypatch):
    ai_response = {
        "task_achievement_bands": _make_bands(8), "coherence_cohesion_bands": _make_bands(8),
        "lexical_resource_bands": _make_bands(8), "grammar_bands": _make_bands(8),
        "topic_relevance": "on_topic",
        # image_data_accuracy genuinely absent from GPT's response
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: ai_response)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined text. " * 60)

    result = writing.evaluate_writing({
        "metadata": {"task_type": "task_1", "question": "The chart below shows data.", "image_url": _FAKE_IMAGE_URL},
        "user_answers": {"text": _IMAGE_TASK1_ESSAY},
    })

    assert result["image_verification_incomplete"] is True


# ---------------------------------------------------------------------------
# IELTS Writing Error Taxonomy (110 named error types across Grammar,
# Lexical Resource, Coherence & Cohesion, Task Response, Academic Task 1,
# and General Training Task 1 Letters). Per the taxonomy's own
# implementation note: used to DETECT and NAME errors consistently, but
# final scoring still comes from the official band descriptor checklists
# (unchanged) - "category" is now drawn from this fixed taxonomy instead
# of a free-form slug, with a new "subtype" (specific instance) and
# "meaning_impact" (transparency signal) field alongside the existing
# "severity". The taxonomy's own frequency-escalation and error-
# interaction rules were already implemented (_escalate_frequent_minor_
# mistakes / _escalate_error_clusters) before this document was supplied.
# ---------------------------------------------------------------------------

def test_evaluate_writing_task1_academic_prompt_includes_taxonomy_and_academic_ta_categories(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    _evaluate_task1("The chart illustrates various trends over time in great detail across categories.")

    prompt = _normalize_whitespace(captured["prompt"])
    assert "Tense Errors" in prompt and "Article Errors" in prompt  # shared grammar categories
    assert "SEVERITY DECISION DIMENSIONS" in prompt
    assert "Missing Overview" in prompt and "Incorrect Trend" in prompt  # Academic Task 1 categories
    assert "Missing Bullet Point" not in prompt  # GT categories must not leak into Academic


def test_evaluate_writing_task1_general_prompt_includes_gt_ta_categories_only(monkeypatch):
    captured = {}
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt_capturing_writing_prompt(monkeypatch, ai_response, captured)

    letter_essay = "Dear Sam,\n\nI hope you are well and doing okay these days.\n\nYours sincerely,\nAlex"
    _evaluate_task1(letter_essay, question="Write a letter to a friend.")

    prompt = _normalize_whitespace(captured["prompt"])
    assert "Missing/Incomplete Bullet Point" in prompt and "Tone" in prompt
    # "Missing Overview" legitimately appears inside the file's own "Do NOT
    # use Academic categories (...)" clarifying note - that's the
    # anti-leakage mechanism itself, not a leak.
    assert "Do NOT use Academic categories" in prompt


def test_evaluate_writing_task2_prompt_includes_task_response_categories(monkeypatch):
    captured = {}
    ai_response = {
        "task_response_bands": _make_bands(6), "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(6), "grammar_bands": _make_bands(6),
        "topic_relevance": "on_topic", "mistakes": [], "strengths": "x", "improvement": "y",
    }
    monkeypatch.setattr(writing, "call_gpt_writing", lambda prompt, image_url=None, **kwargs: (captured.update(prompt=prompt), ai_response)[1])
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined text. " * 60)

    _evaluate()

    prompt = _normalize_whitespace(captured["prompt"])
    assert "Misunderstanding the Task/Question" in prompt and "Off-Topic Tangent" in prompt


def test_evaluate_writing_mistake_category_preserves_original_casing_but_escalates_case_insensitively(monkeypatch):
    # _normalize_mistake_severity_and_category() used to lowercase
    # "category" for grouping, which silently overwrote the taxonomy's
    # proper-case names ("Article Errors" -> "article errors") in the
    # field actually returned. Casing must now be preserved in the
    # output, while frequency escalation still groups case-insensitively
    # so a stray case difference doesn't split one real pattern in two.
    ai_response = {
        "task_achievement_bands": _make_bands(7), "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7), "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        # Item 2 (verbatim filter now requires 3+ words + unique
        # occurrence): all four "original" spans used to be bare 1-2 word
        # fragments ("government", "good idea", "economy grew", "engineer
        # works"). Lengthened to genuine 3-4 word substrings of the essay
        # below - the essay itself also drops the "a"/"the" the second and
        # third mistakes claim is missing (it already had them, which was
        # never actually consistent with "missing article" even before
        # this item; fixed here rather than carried forward), so the
        # missing-article intent is now genuinely true of the text, not
        # just schema-shaped.
        "mistakes": [
            {"type": "grammar", "category": "Article Errors", "subtype": "missing 'the'", "severity": "minor", "meaning_impact": "low", "original": "Government invests heavily in", "corrected": "The government invests heavily in", "explanation": "x"},
            {"type": "grammar", "category": "Article Errors", "subtype": "missing 'a'", "severity": "minor", "meaning_impact": "low", "original": "consider this good idea", "corrected": "consider this a good idea", "explanation": "x"},
            {"type": "grammar", "category": "article errors", "subtype": "missing 'the'", "severity": "minor", "meaning_impact": "low", "original": "Last year economy grew", "corrected": "Last year the economy grew", "explanation": "x"},
            {"type": "grammar", "category": "Article Errors", "subtype": "missing 'an'", "severity": "minor", "meaning_impact": "low", "original": "Every engineer works closely", "corrected": "An engineer works closely", "explanation": "x"},
        ],
        "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate_task1(
        "The chart illustrates trends. Government invests heavily in infrastructure. "
        "Many experts consider this good idea for growth. Last year economy "
        "grew steadily despite challenges. Every engineer works closely with local "
        "teams to support these long-term development projects across many regions."
    )

    categories = {m["category"] for m in result["mistakes"]}
    assert "Article Errors" in categories  # original proper-case preserved
    assert "article errors" in categories  # the one deliberately-lowercase instance is untouched too
    assert all(m["severity"] == "significant" for m in result["mistakes"])
    assert all(m["escalated_to"] == "systematic" for m in result["mistakes"])
    assert all(m["occurrence_count"] == 4 for m in result["mistakes"])
    assert all(m.get("subtype") for m in result["mistakes"])
    assert all(m.get("meaning_impact") == "low" for m in result["mistakes"])


def test_evaluate_writing_task2_has_no_task1_variant_field(monkeypatch):
    ai_response = {
        "task_response_bands": _make_bands(6),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(6),
        "grammar_bands": _make_bands(6),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate()

    assert "task1_variant" not in result


# ---------------------------------------------------------------------------
# evaluate_writing(): "mistakes" filtering must drop any object whose own
# "explanation" admits the flagged text isn't actually an error, even when
# it doesn't use the literal phrase "no error". Regression test for a real
# case: GPT's explanation read "...however, the original is acceptable.
# This is borderline but not an error, so no correction needed." while
# still emitting a "corrected" field that introduced a NEW error of its
# own ("a grade a B") - the old filter only matched the exact substring
# "no error" and let this contradictory object through untouched.
# ---------------------------------------------------------------------------

def test_evaluate_writing_drops_mistake_whose_explanation_admits_no_real_error(monkeypatch):
    ai_response = {
        "task_response_bands": _make_bands(6),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(6),
        "grammar_bands": _make_bands(6),
        "topic_relevance": "on_topic",
        "mistakes": [
            {
                "type": "lexical", "category": "Collocation Errors", "subtype": "grade phrasing",
                "severity": "minor", "meaning_impact": "low",
                "original": "a student received a grade B for his Physics examination",
                "corrected": "a student received a grade a B for his Physics examination",
                "explanation": (
                    "The phrase 'a grade B' is acceptable, but 'a grade a B' is "
                    "incorrect; however, the original is acceptable. This is "
                    "borderline but not an error, so no correction needed."
                ),
            },
            {
                "type": "grammar", "category": "Tense Errors", "subtype": "past vs present",
                "severity": "minor", "meaning_impact": "low",
                "original": "he go to school", "corrected": "he goes to school",
                "explanation": "Subject-verb agreement error: 'go' should be 'goes'.",
            },
        ],
        "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    essay = (
        _realistic_essay(240) + " For example, a student received a grade B for "
        "his Physics examination. Every morning he go to school on time."
    )
    result = _evaluate(essay=essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Collocation Errors" not in categories
    assert "Tense Errors" in categories


# ---------------------------------------------------------------------------
# evaluate_writing(): the candidate's own submitted text must be echoed
# back in the result as "answer_text" - previously absent entirely (only
# refined_answer, the Band 9 rewrite, was returned), making it impossible
# to see what a mistake's "original"/"corrected" snippets actually refer
# to without separately tracking the original submission.
# ---------------------------------------------------------------------------

def test_evaluate_writing_returns_original_answer_text(monkeypatch):
    ai_response = {
        "task_response_bands": _make_bands(6),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(6),
        "grammar_bands": _make_bands(6),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    essay = "Governments should invest in public health because it saves lives for everyone."
    result = _evaluate(essay=essay)

    assert result["answer_text"] == essay


# ---------------------------------------------------------------------------
# evaluate_writing(): the 4 criteria must combine into the task's
# overall_band via a SIMPLE EQUAL AVERAGE (25% each), matching real IELTS
# Writing scoring - not the weighted average (0.3/0.25/0.25/0.2 for Task
# 1, 0.4/0.3/0.2/0.1 for Task 2) that used to be applied here, which
# silently over-weighted Task Achievement/Response and under-weighted
# Grammar. Values are chosen so the two formulas land on DIFFERENT
# rounded bands, proving equal weighting is what's actually applied.
# ---------------------------------------------------------------------------

def test_evaluate_writing_task1_overall_band_is_simple_equal_average(monkeypatch):
    # TA=8, CC=7, LR=6, GR=3 -> equal average = 24/4 = 6.0 exactly.
    # The old weighted formula (TA*0.3+CC*0.25+LR*0.25+GR*0.2) gives 6.25,
    # which rounds UP to 6.5 under this codebase's own round-half-up
    # convention - a genuinely different result, so this pins the fix.
    ai_response = {
        "task_achievement_bands": _make_bands(8),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(6),
        "grammar_bands": _make_bands(3),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate_task1(_realistic_essay(160))

    assert result["overall_band"] == 6.0


def test_evaluate_writing_task2_overall_band_is_simple_equal_average(monkeypatch):
    # TR=3, CC=8, LR=7, GR=6 -> equal average = 24/4 = 6.0 exactly.
    # The old weighted formula (TR*0.4+CC*0.3+LR*0.2+GR*0.1) gives 5.6,
    # which rounds to 5.5 - a genuinely different result, so this pins
    # the fix.
    ai_response = {
        "task_response_bands": _make_bands(3),
        "coherence_cohesion_bands": _make_bands(8),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(6),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate(essay=_realistic_essay(260))

    assert result["overall_band"] == 6.0


# ---------------------------------------------------------------------------
# evaluate_writing(): regression test for a real bug found in a live API
# test - GPT flagged "people spend 35 percent" as a Subject-Verb
# Agreement error with explanation "This is correct; 'people' plural
# with 'spend' plural verb," while still emitting it as a mistake object
# with escalation metadata. "This is correct" wasn't in the original
# self-admission phrase list (which only caught "no error", "not an
# error", etc.), so the bogus mistake slipped through AND inflated the
# systematic-escalation occurrence_count for the real SVA errors in the
# same essay from 4 to 5.
# ---------------------------------------------------------------------------

def test_evaluate_writing_drops_mistake_whose_explanation_says_this_is_correct(monkeypatch):
    ai_response = {
        "task_achievement_bands": _make_bands(6),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(6),
        "grammar_bands": _make_bands(6),
        "topic_relevance": "on_topic",
        "mistakes": [
            {
                "type": "grammar", "category": "Subject-Verb Agreement Errors",
                "subtype": "plural subject with plural verb", "severity": "significant",
                "meaning_impact": "low",
                "original": "people spend 35 percent", "corrected": "people spend 35 percent",
                "explanation": "This is correct; 'people' plural with 'spend' plural verb.",
            },
            {
                "type": "grammar", "category": "Noun Number Errors", "subtype": "missing plural",
                "severity": "minor", "meaning_impact": "low",
                "original": "four different country", "corrected": "four different countries",
                "explanation": "'Four' requires the plural noun form 'countries'.",
            },
        ],
        "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    essay = (
        _realistic_essay(140) + " Overall, people spend 35 percent of their "
        "income on housing across four different country in this survey."
    )
    result = _evaluate_task1(essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Subject-Verb Agreement Errors" not in categories
    assert "Noun Number Errors" in categories


# ---------------------------------------------------------------------------
# evaluate_writing(): regression test for a real bug found from a live
# server result - when the underlying OpenAI call fails on every retry
# (e.g. an image_url the API can't fetch: openai.BadRequestError,
# "invalid_image_url"), safe_gpt_call() silently returns the neutral
# default_ai fallback (band 5 on every criterion, empty mistakes, empty
# strengths/improvement). Before this fix, that fallback was
# indistinguishable from a real, weak evaluation once feedback/
# improvement fell back to their own generic hardcoded strings - a
# candidate would see what looked like a plausible Band 5 result with no
# indication their answer was never actually evaluated.
# ---------------------------------------------------------------------------

def test_evaluate_writing_surfaces_ai_evaluation_failed_when_gpt_call_fails(monkeypatch):
    def always_fail(prompt, image_url=None, **kwargs):
        raise RuntimeError("simulated OpenAI failure")

    monkeypatch.setattr(writing, "call_gpt_writing", always_fail)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined essay text.")

    result = _evaluate(essay=_realistic_essay(260))

    assert result["ai_evaluation_failed"] is True
    assert result["ai_evaluation_failed_reason"] == "other"
    assert result["feedback"] == "Evaluation temporarily unavailable - please try submitting this answer again."
    assert "could not be evaluated" in result["improvement"]
    assert result["mistakes"] == []
    # Still a safe neutral band, not a crash and not the worst possible
    # score - matches the existing default_ai fallback behavior.
    assert result["overall_band"] == 5.0


def test_evaluate_writing_surfaces_refusal_as_the_failure_reason(monkeypatch):
    # Observed live: gpt-4o intermittently refuses outright on an ordinary
    # essay - a distinct failure mode from a generic parse/network error,
    # now made visible in the eval log instead of collapsing into "other".
    def always_refuses(prompt, image_url=None, **kwargs):
        raise ValueError("Invalid JSON from GPT:\nI'm sorry, I can't assist with that request.")

    monkeypatch.setattr(writing, "call_gpt_writing", always_refuses)
    monkeypatch.setattr(writing, "call_gpt_text", lambda prompt, system_msg=None: "Refined essay text.")

    result = _evaluate(essay=_realistic_essay(260))

    assert result["ai_evaluation_failed"] is True
    assert result["ai_evaluation_failed_reason"] == "refusal"


def test_evaluate_writing_ai_evaluation_failed_reason_is_none_on_success(monkeypatch):
    ai_response = {
        "task_response_bands": _make_bands(6),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(6),
        "grammar_bands": _make_bands(6),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "Real strength.", "improvement": "Real improvement.",
    }
    _install_fake_gpt(monkeypatch, ai_response)
    result = _evaluate(essay=_realistic_essay(260))
    assert result["ai_evaluation_failed"] is False
    assert result["ai_evaluation_failed_reason"] is None


def test_evaluate_writing_ai_evaluation_failed_false_on_success(monkeypatch):
    ai_response = {
        "task_response_bands": _make_bands(6),
        "coherence_cohesion_bands": _make_bands(6),
        "lexical_resource_bands": _make_bands(6),
        "grammar_bands": _make_bands(6),
        "topic_relevance": "on_topic",
        "mistakes": [], "strengths": "Real strength.", "improvement": "Real improvement.",
    }
    _install_fake_gpt(monkeypatch, ai_response)

    result = _evaluate(essay=_realistic_essay(260))

    assert result["ai_evaluation_failed"] is False
    assert result["feedback"] == "Real strength."
    assert result["improvement"] == "Real improvement."


# ---------------------------------------------------------------------------
# evaluate_writing(): Task 1 only - regression test for a real bug found in
# a live test. A mistake's "corrected" field was a long, unrelated sentence
# that happened to also appear verbatim in the independently-generated
# refined_answer rewrite - not an actual fix for the flagged "original"
# span at all. _mistake_correction_relates_to_original() drops such
# mistakes. Scoped to task_type == "task_1" only - the second test below
# proves Task 2 is completely unaffected by the identical mistake shape.
# ---------------------------------------------------------------------------

def test_evaluate_writing_task1_drops_mistake_with_unrelated_correction(monkeypatch):
    refined_text = (
        "The three pie charts depict the average proportions of three "
        "potentially harmful nutrients found in typical meals consumed "
        "in the United States, which can pose health risks if eaten to "
        "excess."
    )
    ai_response = {
        "task_achievement_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        "mistakes": [
            {
                "type": "coherence", "category": "Repetition of Ideas", "subtype": "x",
                "severity": "minor", "meaning_impact": "low",
                "original": "includes 23%. A typical dinner includes 23% added sugar",
                # Long, unrelated "correction" that happens to appear
                # verbatim in refined_text below - the confirmed bug shape.
                "corrected": refined_text,
                "explanation": "Repeated data point.",
            },
            {
                "type": "lexical", "category": "Incorrect Word Choice", "subtype": "y",
                "severity": "minor", "meaning_impact": "low",
                "original": "Through eating lunch, 29% sodium is consumed",
                "corrected": "Lunch accounts for 29% of sodium consumption",
                "explanation": "Awkward phrasing.",
            },
        ],
        "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response, refined_text=refined_text)

    essay = (
        _realistic_essay(130) + " Dinner includes 23%. A typical dinner "
        "includes 23% added sugar. Through eating lunch, 29% sodium is consumed."
    )
    result = _evaluate_task1(essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Repetition of Ideas" not in categories
    assert "Incorrect Word Choice" in categories


def test_evaluate_writing_task1_keeps_short_genuine_corrections(monkeypatch):
    # Short, genuine phrase-level corrections (the normal case) must
    # never be touched by this check, regardless of word overlap.
    ai_response = {
        "task_achievement_bands": _make_bands(7),
        "coherence_cohesion_bands": _make_bands(7),
        "lexical_resource_bands": _make_bands(7),
        "grammar_bands": _make_bands(7),
        "topic_relevance": "on_topic",
        "mistakes": [
            {
                "type": "lexical", "category": "Incorrect Word Choice", "subtype": "x",
                "severity": "minor", "meaning_impact": "low",
                "original": "Through eating lunch, 29% sodium is consumed",
                "corrected": "Lunch accounts for 29% of sodium consumption",
                "explanation": "Awkward phrasing.",
            },
        ],
        "strengths": "x", "improvement": "y",
    }
    _install_fake_gpt(monkeypatch, ai_response, refined_text="Some completely different refined text.")

    essay = _realistic_essay(140) + " Through eating lunch, 29% sodium is consumed."
    result = _evaluate_task1(essay)

    categories = [m["category"] for m in result["mistakes"]]
    assert "Incorrect Word Choice" in categories
