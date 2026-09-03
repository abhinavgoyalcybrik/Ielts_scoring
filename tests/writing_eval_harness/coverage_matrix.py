# Coverage matrix scaffolding (question variant x answer profile).
# Structure only in this file - no live numbers. Every cell is populated
# by run_coverage_matrix() in run_eval_harness.py, which needs the live
# API and does not run until credits are confirmed restored.
#
# ROWS: the 15 question variants in question_bank.py (7 Task 1 Academic
# chart types, 3 Task 1 GT letter registers, 5 Task 2 question types).
# Each variant has 2 questions; a row's numbers are aggregated across both.
#
# COLUMNS: the 13 answer profiles in answer_profiles.py.
#
# Not every (variant, profile) cell is meaningful - profile 12
# (misreads_data) is only meaningful for Task 1 Academic chart variants
# (General Training letters and Task 2 essays have no chart data to
# misread). Those cells are marked not_applicable rather than left blank,
# so a genuinely untested gap is never confused with a deliberately
# skipped one.

from question_bank import QUESTION_BANK, ALL_VARIANT_ROW_LABELS, QUESTION_BANK_BY_ID
from answer_profiles import PROFILE_NAMES

MATRIX_ROWS = ALL_VARIANT_ROW_LABELS  # 15 variant labels
MATRIX_COLUMNS = PROFILE_NAMES  # 13 profile names


def _questions_for_row(row_label: str) -> list:
    """Maps a matrix row label back to its 2 questions in QUESTION_BANK."""
    matches = []
    for q in QUESTION_BANK:
        if row_label.startswith("t1_academic_") and q["task_type"] == "task_1" and q.get("t1_variant") == "academic":
            chart_type = row_label[len("t1_academic_"):]
            if q.get("chart_type") == chart_type:
                matches.append(q["id"])
        elif row_label.startswith("t1_gt_") and q["task_type"] == "task_1" and q.get("t1_variant") == "general":
            register = row_label[len("t1_gt_"):]
            if q.get("letter_register") == register:
                matches.append(q["id"])
        elif row_label.startswith("t2_") and q["task_type"] == "task_2":
            variant = row_label[len("t2_"):]
            if q.get("t2_variant") == variant:
                matches.append(q["id"])
    return matches


def _cell_is_applicable(row_label: str, profile_name: str) -> bool:
    if profile_name == "p12_misreads_data":
        # Only meaningful where there is chart data to misread.
        return row_label.startswith("t1_academic_")
    return True


# ---------------------------------------------------------------------------
# Smoke subset - a much smaller slice of the matrix for fast, cheap
# iteration during development. A full run (187 applicable cells x up to 2
# questions) gets materially more expensive once the adversarial verifier
# (plan Step 3) lands - up to ~13 GPT calls per cell instead of today's ~2 -
# so re-running the full matrix after every small change stops being
# practical. The full matrix remains the only thing that counts as a
# before/after baseline for a plan step; the smoke subset exists purely to
# catch an obviously broken change quickly.
#
# Coverage: one representative variant per task type (so every task type's
# prompt path - Task 1 Academic, Task 1 GT, Task 2 - is exercised at least
# once) across all 13 profiles, PLUS every one of the 15 variants on
# P1/P3/P7 (strong / weak / fully off-topic), so a variant-specific
# regression on those three high-signal profiles is still visible without
# needing all 13 profiles on every row.
# ---------------------------------------------------------------------------
SMOKE_REPRESENTATIVE_ROWS = {
    "t1_academic_line_graph",  # stands in for all 7 Task 1 Academic chart types
    "t1_gt_formal",            # stands in for all 3 Task 1 GT letter registers
    "t2_opinion",              # stands in for all 5 Task 2 question types
}
SMOKE_ALWAYS_PROFILES = {"p1_strong", "p3_weak", "p7_off_topic"}


def _cell_is_in_smoke_subset(row_label: str, profile_name: str) -> bool:
    """True if (row_label, profile_name) belongs to the smoke subset - see
    module comment above. Independent of _cell_is_applicable(); callers
    should check both (a not-applicable cell isn't in the smoke subset
    either, even if it would otherwise match)."""
    return row_label in SMOKE_REPRESENTATIVE_ROWS or profile_name in SMOKE_ALWAYS_PROFILES


def build_empty_matrix() -> dict:
    """Returns the full (row, column) -> cell scaffolding, every cell
    either PENDING (needs a live run) or NOT_APPLICABLE (structurally
    doesn't apply, e.g. profile 12 outside Task 1 Academic). No API calls
    happen here - this is pure structure, safe to build and inspect with
    zero cost."""
    matrix = {}
    for row in MATRIX_ROWS:
        question_ids = _questions_for_row(row)
        for col in MATRIX_COLUMNS:
            key = (row, col)
            if not _cell_is_applicable(row, col):
                matrix[key] = {"status": "NOT_APPLICABLE", "question_ids": question_ids}
            else:
                matrix[key] = {"status": "PENDING", "question_ids": question_ids}
    return matrix


def print_matrix_structure(smoke_only: bool = False):
    """Prints the row x column structure with PENDING/NOT_APPLICABLE
    markers and, per row, which 2 questions feed it - the "report the
    coverage matrix structure before filling in numbers" deliverable.
    Pass smoke_only=True to see the smoke subset's shape instead (see
    SMOKE_REPRESENTATIVE_ROWS/SMOKE_ALWAYS_PROFILES above) - cells outside
    the subset print as 'x' rather than '.'."""
    matrix = build_empty_matrix()
    col_width = 6
    row_label_width = max(len(r) for r in MATRIX_ROWS) + 2
    header = "VARIANT".ljust(row_label_width) + "".join(f"P{i+1:<{col_width-1}}" for i in range(len(MATRIX_COLUMNS)))
    print(header)
    print("-" * len(header))
    n_smoke_cells = 0
    for row in MATRIX_ROWS:
        line = row.ljust(row_label_width)
        for col in MATRIX_COLUMNS:
            status = matrix[(row, col)]["status"]
            if status == "NOT_APPLICABLE":
                marker = "-"
            elif smoke_only and not _cell_is_in_smoke_subset(row, col):
                marker = "x"
            else:
                marker = "."
                if smoke_only:
                    n_smoke_cells += 1
            line += marker.ljust(col_width)
        print(line)
    print()
    if smoke_only:
        n_pending_full = sum(1 for v in matrix.values() if v["status"] == "PENDING")
        print("Legend: '.' = IN SMOKE SUBSET   'x' = skipped in smoke mode   '-' = NOT_APPLICABLE")
        print(f"Smoke subset: {n_smoke_cells} cells (vs {n_pending_full} in the full matrix)")
    else:
        print("Legend: '.' = PENDING (awaiting live run)   '-' = NOT_APPLICABLE")
    print("Columns P1-P13 map to:", ", ".join(f"P{i+1}={name}" for i, name in enumerate(MATRIX_COLUMNS)))
    print()
    print("Row -> question coverage:")
    for row in MATRIX_ROWS:
        qids = _questions_for_row(row)
        flag = "" if len(qids) >= 2 else "  <-- FEWER THAN 2 QUESTIONS, gap"
        print(f"  {row:24s} {qids}{flag}")

    n_pending = sum(1 for v in matrix.values() if v["status"] == "PENDING")
    n_na = sum(1 for v in matrix.values() if v["status"] == "NOT_APPLICABLE")
    print()
    print(f"Total cells: {len(matrix)} ({n_pending} pending live data, {n_na} not applicable)")


# ---------------------------------------------------------------------------
# Relational assertions - ready to run against populated cell data once a
# live run exists. Each takes the matrix (row -> profile -> result dict,
# where result carries criteria_scores/mistake_count/etc. from an actual
# evaluate_writing() call) and returns a list of violation strings (empty
# = passed). Verifiable by construction, per the user's "assert relational
# properties, not exact bands" instruction - no assertion here depends on
# a specific expected band number.
# ---------------------------------------------------------------------------
def assert_profile_ordering(results_by_profile: dict) -> list:
    """profile 1 > profile 2 > profile 3 on every criterion."""
    violations = []
    criteria = ["task_response", "coherence_cohesion", "lexical_resource", "grammar_accuracy"]
    p1, p2, p3 = results_by_profile.get("p1_strong"), results_by_profile.get("p2_competent_occasional_errors"), results_by_profile.get("p3_weak")
    if not (p1 and p2 and p3):
        return ["missing p1/p2/p3 results - cannot check ordering"]
    for key in criteria:
        # Task 1 uses "task_response" as the dict key name too (see
        # evaluators/writing.py's criteria_scores construction - both
        # task types use the same 4 output key names regardless of
        # whether the underlying criterion is Task Achievement or Task
        # Response).
        v1, v2, v3 = p1["criteria_scores"].get(key), p2["criteria_scores"].get(key), p3["criteria_scores"].get(key)
        if None in (v1, v2, v3):
            continue
        if not (v1 >= v2 >= v3):
            violations.append(f"{key}: expected p1({v1}) >= p2({v2}) >= p3({v3})")
    return violations


def assert_grammar_vocab_independence(results_by_profile: dict) -> list:
    """Profile 4 (strong grammar, weak vocab): grammar score materially
    above lexical score. Profile 5: the reverse. "Materially" = at least
    1 full band apart - anything less could plausibly be noise rather
    than genuine independence."""
    violations = []
    p4 = results_by_profile.get("p4_strong_grammar_weak_vocab")
    p5 = results_by_profile.get("p5_weak_grammar_strong_vocab")
    if p4:
        gr, lr = p4["criteria_scores"].get("grammar_accuracy"), p4["criteria_scores"].get("lexical_resource")
        if gr is not None and lr is not None and (gr - lr) < 1.0:
            violations.append(f"p4: expected grammar({gr}) materially above lexical({lr}), gap={gr - lr}")
    if p5:
        gr, lr = p5["criteria_scores"].get("grammar_accuracy"), p5["criteria_scores"].get("lexical_resource")
        if gr is not None and lr is not None and (lr - gr) < 1.0:
            violations.append(f"p5: expected lexical({lr}) materially above grammar({gr}), gap={lr - gr}")
    return violations


def assert_off_topic_cap(results_by_profile: dict) -> list:
    """Profile 7 (fully off-topic): Task Achievement/Response should be
    low (this codebase's own topic_relevance cap logic caps it at 5 for
    completely_off_topic - see evaluators/writing.py). Profile 8
    (partially off-topic): Task Achievement/Response below the same
    answer without the drift (profile 1)."""
    violations = []
    p1 = results_by_profile.get("p1_strong")
    p7 = results_by_profile.get("p7_off_topic")
    p8 = results_by_profile.get("p8_partially_off_topic")
    if p7:
        tr = p7["criteria_scores"].get("task_response")
        if tr is not None and tr > 5.0:
            violations.append(f"p7 (fully off-topic): expected task_response <= 5.0, got {tr}")
    if p1 and p8:
        tr1, tr8 = p1["criteria_scores"].get("task_response"), p8["criteria_scores"].get("task_response")
        if tr1 is not None and tr8 is not None and tr8 >= tr1:
            violations.append(f"p8 (partial drift): expected task_response({tr8}) < p1's({tr1})")
    return violations


def assert_paragraphing_caps(results_by_profile: dict) -> list:
    """Profile 10 (no paragraphing): coherence capped, lexical NOT
    capped - directly re-testing evaluators/writing.py's own
    _coherence_paragraph_cap()/_grammar_punctuation_cap() split (built
    earlier this session specifically because a prior bug let a
    paragraphing problem wrongly drag Lexical Resource down too)."""
    violations = []
    p1 = results_by_profile.get("p1_strong")
    p10 = results_by_profile.get("p10_no_paragraphing")
    if p1 and p10:
        cc1, cc10 = p1["criteria_scores"].get("coherence_cohesion"), p10["criteria_scores"].get("coherence_cohesion")
        lr1, lr10 = p1["criteria_scores"].get("lexical_resource"), p10["criteria_scores"].get("lexical_resource")
        if cc1 is not None and cc10 is not None and cc10 >= cc1:
            violations.append(f"p10: expected coherence to be capped below p1 ({cc1}), got {cc10}")
        if lr1 is not None and lr10 is not None and lr10 < lr1:
            violations.append(f"p10: lexical resource dropped ({lr1} -> {lr10}) - should NOT be affected by paragraphing")
    return violations


def assert_underlength_penalty(results_by_profile: dict) -> list:
    """Profile 9: underlength answer should score at or below the
    equivalent full-length answer on task achievement/response - the
    official descriptors only mandate a hard Band-1 cap at <=20 words
    (already enforced elsewhere in evaluators/writing.py), so this
    assertion is deliberately soft (<=, not strictly <) rather than
    assuming a fixed penalty beyond that rule."""
    violations = []
    p1 = results_by_profile.get("p1_strong")
    p9 = results_by_profile.get("p9_underlength")
    if p1 and p9:
        tr1, tr9 = p1["criteria_scores"].get("task_response"), p9["criteria_scores"].get("task_response")
        if tr1 is not None and tr9 is not None and tr9 > tr1:
            violations.append(f"p9: expected underlength task_response({tr9}) <= full-length({tr1})")
    return violations


def assert_template_answer_not_rewarded(results_by_profile: dict) -> list:
    """Profile 6 (memorised template): should not score as well on task
    achievement/response as a genuine, specific answer to the same
    question."""
    violations = []
    p1 = results_by_profile.get("p1_strong")
    p6 = results_by_profile.get("p6_memorised_template")
    if p1 and p6:
        tr1, tr6 = p1["criteria_scores"].get("task_response"), p6["criteria_scores"].get("task_response")
        if tr1 is not None and tr6 is not None and tr6 >= tr1:
            violations.append(f"p6: expected template task_response({tr6}) < genuine answer's({tr1})")
    return violations


ALL_RELATIONAL_ASSERTIONS = [
    ("profile_ordering (p1>p2>p3)", assert_profile_ordering),
    ("grammar_vocab_independence (p4/p5)", assert_grammar_vocab_independence),
    ("off_topic_cap (p7/p8)", assert_off_topic_cap),
    ("paragraphing_caps (p10)", assert_paragraphing_caps),
    ("underlength_penalty (p9)", assert_underlength_penalty),
    ("template_answer_not_rewarded (p6)", assert_template_answer_not_rewarded),
]


# ---------------------------------------------------------------------------
# P6 format/register mismatch - a CROSS-ROW assertion, unlike the six
# above (which each compare profiles within one row's essay identity).
# p6_memorised_template deliberately uses the SAME frame regardless of
# chart_type/letter_register - see answer_profiles.py's module comment:
# "the mismatch is the point" - a candidate who memorised one frame really
# does bolt it onto the wrong chart type or send a formal letter to a
# friend, rather than correctly adapting to the task. This assertion makes
# that mismatch measurable: Task Achievement/Response on P6 rows where the
# frame's language structurally fits the variant must be HIGHER than on P6
# rows where it doesn't. If both score the same, the evaluator isn't
# detecting inappropriate format/register at all - a real, examinable Task
# Achievement requirement that was previously untested.
#
# MATCHED rows: the frame's "chart... categories... trend/data over a
# period" language genuinely fits (line graph, bar chart, pie chart,
# table, mixed charts - all categorical/numeric "trends and comparisons"
# chart types), and the formal-letter frame fits (formal register).
# MISMATCHED rows: process_diagram/map (the frame has no "categories" or
# numeric trend language that fits either - a process diagram describes
# stages, a map describes spatial change, neither is "data... over the
# given period") and the informal letter register (a strict formal "Dear
# Sir/Madam... Yours faithfully" frame sent to a friend).
#
# t1_gt_semi_formal is deliberately in NEITHER set - whether the strict
# formal frame counts as a mismatch there is genuinely ambiguous
# (semi-formal letters sit between "Dear Sir/Madam" and a first-name
# greeting), so this assertion doesn't take a position on it. Task 2 rows
# have only one frame with no variant-specific mismatch to test, so
# they're excluded too - see build_profile()/make_template_answer()'s use
# of the same Task 2 frame for every t2_variant.
# ---------------------------------------------------------------------------
P6_FORMAT_MATCHED_ROWS = {
    "t1_academic_line_graph", "t1_academic_bar_chart", "t1_academic_pie_chart",
    "t1_academic_table", "t1_academic_mixed_charts", "t1_gt_formal",
}
P6_FORMAT_MISMATCHED_ROWS = {
    "t1_academic_process_diagram", "t1_academic_map", "t1_gt_informal",
}


def assert_p6_format_mismatch_scores_lower(p6_task_achievement_by_row: dict) -> list:
    """p6_task_achievement_by_row: {row_label: task_response_score} for
    every row that has a P6 result (task_response is this codebase's
    shared field name for both Task Achievement and Task Response - see
    assert_profile_ordering's inline comment on this). Compares the MEAN across
    MATCHED rows against the MEAN across MISMATCHED rows - a single
    aggregate comparison, not pairwise per-row, since individual rows are
    different questions/essays and only the aggregate is meaningful
    evidence of whether the evaluator detects the mismatch at all rather
    than reacting to one row's particular content.

    Returns a "missing" message (not a silent pass) if either side has no
    data - e.g. this cannot be checked from a smoke-subset run, since P6
    only runs on the 3 representative rows there and none of them are in
    P6_FORMAT_MISMATCHED_ROWS. That is a real coverage gap of the smoke
    subset, not a bug - report it as "not checked in smoke mode", not as
    a pass."""
    matched = [v for row, v in p6_task_achievement_by_row.items() if row in P6_FORMAT_MATCHED_ROWS and v is not None]
    mismatched = [v for row, v in p6_task_achievement_by_row.items() if row in P6_FORMAT_MISMATCHED_ROWS and v is not None]
    if not matched or not mismatched:
        return [
            f"missing matched ({len(matched)}) or mismatched ({len(mismatched)}) P6 "
            f"results - cannot check format-mismatch detection (expected on a smoke run)"
        ]
    mean_matched = sum(matched) / len(matched)
    mean_mismatched = sum(mismatched) / len(mismatched)
    if mean_mismatched >= mean_matched:
        return [
            f"P6 format mismatch NOT detected: mean task_response on MISMATCHED rows "
            f"({mean_mismatched:.2f}, n={len(mismatched)}) >= mean on MATCHED rows "
            f"({mean_matched:.2f}, n={len(matched)}) - inappropriate format/register isn't "
            f"lowering the score"
        ]
    return []


if __name__ == "__main__":
    print_matrix_structure()
