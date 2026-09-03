# Tests for the "Band 9 structure per task variant" plan (WRITING_INDEPENDENT_MODEL_ANSWER
# pipeline only, default OFF) - explicit structure templates per Task 1/Task 2 variant plus
# the Python-side validation that enforces them. Task 1 Academic is a data report: no thesis,
# no opinion, no conclusion. Task 2's structure genuinely varies by question type. Structure
# is additive to (never a replacement for) the existing coverage checks - see
# tests/test_writing_refine_v2.py for those.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.writing as writing


# ---------------------------------------------------------------------------
# _detect_task2_question_type - validated against the eval harness's own
# 10 hand-labelled Task 2 questions (question_bank.py) as ground truth.
# ---------------------------------------------------------------------------

def test_detect_opinion_from_agree_disagree_wording():
    q = "Some people believe schools should teach cooking. To what extent do you agree or disagree?"
    assert writing._detect_task2_question_type(q) == "opinion"


def test_detect_discussion_from_both_views_wording():
    q = "Some argue parks matter more than housing. Discuss both views and give your own opinion."
    assert writing._detect_task2_question_type(q) == "discussion"


def test_detect_advantages_disadvantages_from_outweigh_wording():
    q = "Remote work has become common. Do the advantages of this trend outweigh the disadvantages?"
    assert writing._detect_task2_question_type(q) == "advantages_disadvantages"


def test_detect_problem_solution_from_problems_and_solutions_wording():
    q = "Traffic congestion has become a serious problem. What problems does this cause, and what solutions can you suggest?"
    assert writing._detect_task2_question_type(q) == "problem_solution"


def test_detect_two_part_fallback_when_no_more_specific_pattern_matches():
    q = "Fewer young people choose skilled trades. Why is this happening, and what can be done about it?"
    assert writing._detect_task2_question_type(q) == "two_part"


def test_detect_defaults_to_opinion_when_nothing_matches():
    assert writing._detect_task2_question_type("A single unadorned statement with no question mark at all") == "opinion"


def test_detect_handles_empty_question():
    assert writing._detect_task2_question_type("") == "opinion"


def test_detect_against_full_harness_question_bank_ground_truth():
    # Real, honest calibration - not every case is separable from wording
    # alone (see the docstring on _detect_task2_question_type): the one
    # known miss is a "two_part" question phrased identically to a
    # "problem_solution" one ("what are the causes... what measures...").
    # 9/10 is the real, reported number - this test locks in that exact
    # number so a regression is visible if the heuristic gets worse.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "writing_eval_harness"))
    from question_bank import QUESTION_BANK

    correct = 0
    total = 0
    for item in QUESTION_BANK:
        if item.get("task_type") != "task_2":
            continue
        total += 1
        if writing._detect_task2_question_type(item["question"]) == item["t2_variant"]:
            correct += 1
    assert total == 10
    assert correct == 9


# ---------------------------------------------------------------------------
# _build_refine_prompt_v2 - correct structure template gets interpolated,
# no placeholder survives unfilled, for every variant.
# ---------------------------------------------------------------------------

def test_task2_prompt_selects_matching_structure_block_per_variant():
    markers = {
        "opinion": "OPINION (agree/disagree)",
        "discussion": "DISCUSSION (discuss both views",
        "advantages_disadvantages": "ADVANTAGES/DISADVANTAGES",
        "problem_solution": "PROBLEM/SOLUTION",
        "two_part": "TWO-PART QUESTION",
    }
    for variant, marker in markers.items():
        prompt = writing._build_refine_prompt_v2(
            "task_2", None, "question text", "essay text", [], None, [], 250, 260,
            task2_question_type=variant,
        )
        assert marker in prompt
        assert "<<<" not in prompt
        # Only this variant's block should be present, not the others'.
        other_markers = [m for v, m in markers.items() if v != variant]
        assert not any(m in prompt for m in other_markers)


def test_task2_prompt_defaults_to_opinion_structure_for_unknown_type():
    prompt = writing._build_refine_prompt_v2(
        "task_2", None, "question text", "essay text", [], None, [], 250, 260,
        task2_question_type="not_a_real_type",
    )
    assert "OPINION (agree/disagree)" in prompt


def test_task1_academic_prompt_states_no_thesis_no_conclusion_rule():
    prompt = writing._build_refine_prompt_v2(
        "task_1", "academic", "question text", "essay text", [], None, [], 150, 170,
    )
    assert "NOT AN ESSAY" in prompt
    assert "no conclusion" in prompt.lower()
    assert "<<<" not in prompt


def test_task1_gt_prompt_states_register_matching_rule():
    prompt = writing._build_refine_prompt_v2(
        "task_1", "general", "question text", "essay text", [], None, [], 150, 170,
    )
    assert "Yours faithfully" in prompt
    assert "Yours sincerely" in prompt
    assert "<<<" not in prompt


# ---------------------------------------------------------------------------
# _validate_refine_output - Task 1 Academic structure checks.
# ---------------------------------------------------------------------------

def _good_task1_structured():
    return (
        "The chart illustrates nutrient data across four meals.\n\n"
        "Overall, the figures vary considerably by meal and nutrient type.\n\n"
        "Sodium is highest at dinner, at 20%, while breakfast records the lowest figure.\n\n"
        "Saturated fat follows a broadly similar pattern, peaking at dinner as well."
    )


def test_task1_academic_clean_structured_output_passes():
    parsed = {"refined_answer": _good_task1_structured(), "data_contradiction_flag": False, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "academic", 150, 170)
    assert issues == []


def test_task1_academic_flags_figures_in_overview_paragraph():
    refined = (
        "The chart illustrates nutrient data across four meals.\n\n"
        "Overall, sodium reaches 20% at dinner while breakfast is lowest.\n\n"
        "Saturated fat is highest at dinner too.\n\n"
        "Added sugar peaks at snacks."
    )
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "academic", 150, 170)
    assert any("overview paragraph" in i and "figures" in i for i in issues)


def test_task1_academic_flags_conclusion_paragraph():
    refined = (
        "The chart illustrates nutrient data.\n\n"
        "Overall, figures vary by meal.\n\n"
        "Sodium is highest at dinner.\n\n"
        "In conclusion, the data shows clear differences between meals."
    )
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "academic", 150, 170)
    assert any("must not have a conclusion" in i for i in issues)


def test_task1_academic_flags_opinion_language():
    refined = (
        "The chart illustrates nutrient data.\n\n"
        "Overall, figures vary by meal.\n\n"
        "Sodium is highest at dinner, and people should reduce dinner sodium intake.\n\n"
        "Saturated fat is highest at dinner too."
    )
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "academic", 150, 170)
    assert any("opinion/recommendation language" in i for i in issues)


def test_task1_academic_flags_too_few_paragraphs():
    refined = "The chart shows data.\n\nOverall, it varies.\n\nSodium is highest at dinner."
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "academic", 150, 170)
    assert any("Task 1 needs a minimum" in i for i in issues)


def test_task1_academic_allows_five_paragraphs_for_full_coverage():
    # STRUCTURE MUST NOT OVERRIDE CONTENT: a 5th detail paragraph covering
    # a chart that would otherwise be dropped must not itself be flagged.
    refined = (
        "The charts illustrate nutrient data.\n\n"
        "Overall, figures vary across both charts.\n\n"
        "Sodium is highest at dinner.\n\n"
        "Saturated fat is highest at dinner too.\n\n"
        "Added sugar peaks at snacks, unlike the other two nutrients."
    )
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "academic", 150, 170)
    assert not any("Task 1 should be 4" in i for i in issues)


def test_task1_academic_flags_excessive_fragmentation():
    refined = "\n\n".join([
        "Intro paragraph here.", "Overview paragraph here.", "Detail one.",
        "Detail two.", "Detail three.", "Detail four.", "Detail five.",
    ])
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "academic", 150, 170)
    assert any("consolidate rather than fragment" in i for i in issues)


def test_task1_academic_overview_year_mention_is_not_a_figure():
    # Live-measured false positive (n=5, 5/5 runs on a genuine 2015-2020
    # bar chart): a date-range chart's overview unavoidably mentions a
    # calendar year ("...over the five-year period... by 2020"), which
    # isn't a reported data value and can't be removed without breaking
    # the sentence - this is the exact real text that was wrongly flagged
    # before the fix.
    refined = (
        "The bar chart illustrates the average weekly hours adults in four countries spent on "
        "leisure activities in 2015 and 2020.\n\n"
        "Overall, leisure time increased in all four countries over the five-year period. "
        "Country A consistently had the highest leisure hours, while Country D had the lowest. "
        "The gap between the countries narrowed slightly by 2020.\n\n"
        "In 2015, adults in Country A spent the most time on leisure activities, averaging "
        "around eighteen hours per week. Country B followed with fifteen hours, while "
        "Countries C and D recorded lower averages at eleven and nine hours, respectively.\n\n"
        "By 2020, all countries experienced an increase in leisure time. Country A rose to "
        "twenty-one hours, maintaining its lead. Country D, despite having the lowest figure "
        "in 2015, showed a significant increase to thirteen hours. Countries B and C both "
        "increased by approximately three hours, preserving their relative positions."
    )
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "academic", 150, 170)
    assert not any("overview paragraph" in i for i in issues)


def test_task1_gt_closing_split_across_a_blank_line_is_still_recognised():
    # Live-measured false positive (n=5, 3/5 runs on a genuine letter): a
    # real letter sometimes puts the signature on its own blank-line-
    # separated block ("Yours faithfully,\n\nA Customer") rather than
    # joined with a single newline - this is the exact real text that was
    # wrongly flagged as missing a closing before the fix.
    refined = (
        "Dear Sir or Madam,\n\n"
        "I am writing to express my dissatisfaction with a piece of furniture I ordered from "
        "your website last week. I purchased a wooden bookshelf, which was delivered to me "
        "yesterday.\n\n"
        "Upon arrival, I noticed that the bookshelf was damaged. There is a significant crack "
        "running down one side, and one of the shelves is missing entirely.\n\n"
        "I would appreciate it if you could send a replacement bookshelf at your earliest "
        "convenience. Alternatively, if a replacement is not available, I would like to "
        "request a full refund.\n\n"
        "Yours faithfully,\n\n"
        "A Customer"
    )
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "general", 150, 170)
    assert not any("no closing" in i for i in issues)
    assert not any("register mismatch" in i for i in issues)


def test_task1_academic_coverage_check_still_fires_alongside_structure_checks():
    # Coverage wins where it conflicts with structure - both kinds of
    # issue must be able to appear together, not one suppressing the
    # other.
    chart_data = {"charts": [{"categories": [{"name": "Added Sugar"}]}]}
    refined = _good_task1_structured()  # never mentions "Added Sugar"
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, chart_data, [], [], "essay", "task_1", "academic", 150, 170)
    assert any("Added Sugar" in i for i in issues)
    assert issues == [i for i in issues if "Added Sugar" in i]  # nothing else wrong here


# ---------------------------------------------------------------------------
# _validate_refine_output - Task 1 GT structure checks.
# ---------------------------------------------------------------------------

def _good_gt_letter(salutation="Dear Sir or Madam,", closing="Yours faithfully,\nA Candidate"):
    return "\n\n".join([
        salutation,
        "I am writing to report a fault with a product I purchased from your store recently.",
        "The item arrived damaged, with a large crack across the main panel that was visible immediately.",
        "I would like a full refund or a replacement to be sent as soon as possible.",
        closing,
    ])


def test_task1_gt_clean_formal_letter_passes():
    parsed = {"refined_answer": _good_gt_letter(), "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "general", 150, 170)
    assert issues == []


def test_task1_gt_flags_missing_salutation():
    refined = _good_gt_letter(salutation="I am writing to report a problem.")
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "general", 150, 170)
    assert any("no salutation" in i for i in issues)


def test_task1_gt_flags_missing_closing():
    refined = _good_gt_letter(closing="Thank you for your attention to this matter.")
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "general", 150, 170)
    assert any("no closing" in i for i in issues)


def test_task1_gt_flags_unnamed_salutation_with_sincerely_closing():
    refined = _good_gt_letter(salutation="Dear Sir or Madam,", closing="Yours sincerely,\nA Candidate")
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "general", 150, 170)
    assert any("register mismatch" in i and "unnamed recipient" in i for i in issues)


def test_task1_gt_flags_named_salutation_with_faithfully_closing():
    refined = _good_gt_letter(salutation="Dear Mr Smith,", closing="Yours faithfully,\nA Candidate")
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "general", 150, 170)
    assert any("register mismatch" in i and "names the recipient" in i for i in issues)


def test_task1_gt_named_salutation_with_sincerely_is_correct():
    refined = _good_gt_letter(salutation="Dear Mr Smith,", closing="Yours sincerely,\nA Candidate")
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_1", "general", 150, 170)
    assert not any("register mismatch" in i for i in issues)


def test_task1_gt_bullet_coverage_check_unaffected_by_structure_checks():
    result = writing._validate_refine_output(
        {"refined_answer": _good_gt_letter(), "vocabulary_suggestions": []},
        None, ["explain what went wrong"], [], "essay", "task_1", "general", 150, 170,
    )
    assert any("missing coverage of required point" in i for i in result)


# ---------------------------------------------------------------------------
# _validate_refine_output - Task 2 structure checks (position/conclusion),
# applied uniformly across variants per the approved plan.
# ---------------------------------------------------------------------------

def _good_task2_answer():
    return "\n\n".join([
        "I believe that governments should invest more in public health campaigns nationwide.",
        "Firstly, prevention reduces long-term healthcare costs, since treating illness early is cheaper than treating it once severe.",
        "Secondly, informed citizens make healthier choices, since campaigns give people practical steps to act on.",
        "In conclusion, I believe stronger investment in public health campaigns is worthwhile, since it reduces costs and improves choices.",
    ])


def test_task2_clean_structured_answer_passes():
    parsed = {"refined_answer": _good_task2_answer(), "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_2", None, 250, 260)
    assert issues == []


def test_task2_flags_intro_with_no_stated_position():
    refined = "\n\n".join([
        "This essay is about public health campaigns and government spending on them.",
        "Firstly, prevention reduces long-term healthcare costs substantially for everyone involved.",
        "Secondly, informed citizens make healthier choices when properly educated on the matter.",
        "In conclusion, I believe stronger investment in public health campaigns is worthwhile overall.",
    ])
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_2", None, 250, 260)
    assert any("does not state" in i and "position" in i for i in issues)


def test_task2_flags_conclusion_that_does_not_restate_position():
    refined = "\n\n".join([
        "I believe that governments should invest more in public health campaigns nationwide.",
        "Firstly, prevention reduces long-term healthcare costs substantially for everyone involved.",
        "Secondly, informed citizens make healthier choices when properly educated on the matter.",
        "Public health campaigns exist in many countries around the world today.",
    ])
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_2", None, 250, 260)
    assert any("does not restate the position" in i for i in issues)


def test_task2_flags_new_idea_introduced_via_addition_connective_in_conclusion():
    refined = "\n\n".join([
        "I believe that governments should invest more in public health campaigns nationwide.",
        "Firstly, prevention reduces long-term healthcare costs substantially for everyone involved.",
        "Secondly, informed citizens make healthier choices when properly educated on the matter.",
        "In conclusion, I believe investment is worthwhile. Furthermore, schools should also teach nutrition classes to children.",
    ])
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(parsed, None, [], [], "essay", "task_2", None, 250, 260)
    assert any("new idea" in i for i in issues)


def test_task2_advantages_disadvantages_position_via_outweigh_language():
    refined = "\n\n".join([
        "Remote work has become far more common recently, and the advantages clearly outweigh the disadvantages overall.",
        "Firstly, employees save significant commuting time, which improves their daily wellbeing considerably.",
        "On the other hand, some workers feel isolated, though this is a manageable drawback in comparison.",
        "Overall, the advantages outweigh the disadvantages, since time saved and flexibility matter more than isolation.",
    ])
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(
        parsed, None, [], [], "essay", "task_2", None, 250, 260, task2_question_type="advantages_disadvantages",
    )
    assert not any("does not state" in i for i in issues)
    assert not any("does not restate" in i for i in issues)


def test_task2_position_issue_message_mentions_outweigh_for_adv_dis_type():
    refined = "\n\n".join([
        "Remote work has become far more common in recent years across many industries.",
        "Firstly, employees save significant commuting time, which improves daily wellbeing considerably.",
        "On the other hand, some workers feel isolated at home without office company.",
        "Public health campaigns exist in many countries around the world today.",
    ])
    parsed = {"refined_answer": refined, "vocabulary_suggestions": []}
    issues = writing._validate_refine_output(
        parsed, None, [], [], "essay", "task_2", None, 250, 260, task2_question_type="advantages_disadvantages",
    )
    assert any("which side outweighs" in i for i in issues)
