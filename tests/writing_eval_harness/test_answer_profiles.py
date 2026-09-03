# Regression tests for answer_profiles.py's transforms, focused on
# p6_memorised_template - the user asked specifically to verify this
# profile produces a genuine memorised-template answer (a learned essay
# frame with the question topic bolted on, matching what real candidates
# actually submit) rather than merely vague wording. An earlier version
# vocabulary-flattened the entire base essay and prepended one generic
# opener sentence, which kept all of the original essay's specific
# reasoning intact - these tests lock in the rewritten behaviour so that
# regression is impossible to reintroduce silently.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_profiles import make_template_answer, build_profile
from question_bank import QUESTION_BANK, QUESTION_BANK_BY_ID
from profile1_essays import PROFILE1_ESSAYS


def test_template_frame_is_identical_across_different_topics():
    """Two different Task 2 questions must produce the SAME underlying
    frame (once the topic phrase is stripped back out) - that's what
    makes this a genuine reusable template rather than per-essay
    generated content."""
    text_a = make_template_answer("task_2", topic="remote work")
    text_b = make_template_answer("task_2", topic="food waste")
    assert text_a.replace("remote work", "X") == text_b.replace("food waste", "X")


def test_template_does_not_reuse_any_base_essay_content():
    """The template output must share no distinctive sentence with the
    profile-1 base essay it's derived from - a genuine memorised template
    doesn't engage with the specific essay's own arguments/examples at
    all. Checked across every one of the 30 corpus questions."""
    for q in QUESTION_BANK:
        base = PROFILE1_ESSAYS[q["id"]]
        built = build_profile("p6_memorised_template", base, q["task_type"], t1_variant=q.get("t1_variant"), topic=q.get("topic"))
        template_text = built["text"]
        base_sentences = [s.strip() for s in base.replace("\n\n", " ").split(".") if len(s.strip()) > 25]
        for sentence in base_sentences:
            assert sentence not in template_text, (
                f"{q['id']}: template answer leaked a base-essay sentence: {sentence!r}"
            )


def test_template_includes_the_question_topic():
    for q in QUESTION_BANK:
        base = PROFILE1_ESSAYS[q["id"]]
        built = build_profile("p6_memorised_template", base, q["task_type"], t1_variant=q.get("t1_variant"), topic=q.get("topic"))
        assert q["topic"] in built["text"], f"{q['id']}: topic {q['topic']!r} not found in template output"


def test_template_meets_task_minimum_word_count():
    """The template must never dip below the task's real minimum word
    count (150 for Task 1, 250 for Task 2) purely as an artefact of a
    short topic phrase - that would confound this profile with
    p9_underlength, testing the wrong thing."""
    for q in QUESTION_BANK:
        base = PROFILE1_ESSAYS[q["id"]]
        built = build_profile("p6_memorised_template", base, q["task_type"], t1_variant=q.get("t1_variant"), topic=q.get("topic"))
        n_words = len(built["text"].split())
        minimum = 150 if q["task_type"] == "task_1" else 250
        assert n_words >= minimum, f"{q['id']}: template only {n_words} words, below the {minimum}-word task minimum"


def test_task1_academic_and_gt_frames_are_different():
    academic = make_template_answer("task_1", t1_variant="academic", topic="internet access rates")
    gt = make_template_answer("task_1", t1_variant="general", topic="internet access rates")
    assert academic != gt
    assert "Dear Sir/Madam" in gt
    assert "Dear Sir/Madam" not in academic


def test_task2_frame_is_distinct_from_task1_frames():
    t2 = make_template_answer("task_2", topic="remote work")
    t1_academic = make_template_answer("task_1", t1_variant="academic", topic="remote work")
    t1_gt = make_template_answer("task_1", t1_variant="general", topic="remote work")
    assert t2 != t1_academic != t1_gt


def test_template_falls_back_to_generic_topic_when_none_given():
    text = make_template_answer("task_2")
    assert "this issue" in text
