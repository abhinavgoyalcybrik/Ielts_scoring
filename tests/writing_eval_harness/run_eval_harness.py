"""
writing_eval_harness - Step 0 of the writing-evaluator overhaul plan.

Run: venv/Scripts/python.exe tests/writing_eval_harness/run_eval_harness.py

Runs live against the real evaluate_writing() (real GPT calls, real cost -
this is not a mocked unit test). Prints one results table covering all six
label-free checks, plus instrumentation output and (if the local-only
official-sample PDFs are present) the optional band-accuracy check.

Nothing here changes evaluate_writing()'s behaviour - instrumentation
captures the raw GPT response by wrapping call_gpt_writing() and restoring
it afterward; every other code path is untouched.
"""
import os
import sys
import time
import statistics
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _REPO_ROOT)

from clean_corpus import CLEAN_CORPUS
from error_injection import build_damaged_corpus
from degradation_ladder import build_ladder
from perturbation import build_perturbations
from question_bank import QUESTION_BANK_BY_ID
from profile1_essays import PROFILE1_ESSAYS
from answer_profiles import build_profile
from coverage_matrix import (
    MATRIX_ROWS, MATRIX_COLUMNS, _questions_for_row, _cell_is_applicable,
    _cell_is_in_smoke_subset, ALL_RELATIONAL_ASSERTIONS,
    assert_p6_format_mismatch_scores_lower,
)

import evaluators.writing as writing_module
from evaluators.writing import evaluate_writing


class HarnessAbortError(Exception):
    """Raised when a single evaluate_writing() call inside a harness run
    came back degraded - ai_evaluation_failed=True, meaning the real GPT
    call failed on every retry and evaluate_writing() silently fell back
    to its neutral default (band 5 on every criterion, empty mistakes).
    That fallback is INDISTINGUISHABLE at a glance from a genuine clean
    result - a fallback's "0 mistakes" looks exactly like Check 1's
    entire success criterion, and would silently corrupt every check that
    trusts the data. A harness run must never render a results table
    built on any fraction of fallback data, so this is raised to abort
    the whole run rather than let one degraded call slip through into a
    clean-looking number. See print_report()'s except block - it prints a
    loud failure banner and explicitly does not print a results table."""


# ---------------------------------------------------------------------------
# Thin call wrapper matching evaluate_writing()'s actual input/output shape
# (confirmed directly from evaluators/writing.py: it returns a FLAT dict -
# {overall_band, criteria_scores, mistakes, ...} - not wrapped in a "tasks"
# key; that wrapping only happens one layer up, in the /writing/evaluate API
# route, which this harness does not go through).
#
# Every call is validated before it's handed back to a check - see
# HarnessAbortError above. This is deliberately NOT optional/toggleable:
# there is no legitimate reason for a harness run to accept a degraded
# result, so every caller gets this for free rather than needing to
# remember to check for it themselves.
# ---------------------------------------------------------------------------
def _run_eval_once(task_type: str, question: str, text: str, image_url: str | None = None, _context: str = "") -> dict:
    metadata = {"task_type": task_type, "question": question}
    if image_url:
        metadata["image_url"] = image_url
    result = evaluate_writing({"metadata": metadata, "user_answers": {"text": text}})
    if result.get("ai_evaluation_failed") is True:
        where = f" (during: {_context})" if _context else ""
        raise HarnessAbortError(
            f"evaluate_writing() returned ai_evaluation_failed=True{where}. "
            f"The underlying GPT call failed on every retry and fell back "
            f"to the neutral default (band 5 on every criterion, 0 "
            f"mistakes) - this is NOT a genuine result. Aborting rather "
            f"than reporting numbers built on fallback data. "
            f"result={result!r}"
        )
    return result


_CRITERIA_KEYS = ("task_response", "task_achievement", "coherence_cohesion", "lexical_resource", "grammar_accuracy")


def _mistake_key(m: dict) -> str:
    """Same normalization used in the live n=5 determinism measurement -
    (type, category, normalized original span) as the identity of one
    mistake across runs, so repeated runs can be compared by content, not
    by object identity."""
    return f"{m.get('type')}|{m.get('category')}|{str(m.get('original') or '').strip().lower()}"


def _aggregate_runs(runs: list) -> dict:
    """Turns N raw evaluate_writing() results (same input, temperature 0)
    into range/hit-rate data instead of a single value - single-run
    before/after comparisons were confirmed this session to sometimes be
    reading run-to-run noise as signal (see the temperature-0 determinism
    measurement: bands moved by a full band on 3 of 4 tested run-sets, and
    mistake detection itself flickered independently of the band)."""
    overall_bands = [r.get("overall_band") for r in runs if r.get("overall_band") is not None]
    criteria_values = defaultdict(list)
    for r in runs:
        scores = r.get("criteria_scores") or {}
        for key in _CRITERIA_KEYS:
            if key in scores and scores[key] is not None:
                criteria_values[key].append(scores[key])

    mistake_counts = [len(r.get("mistakes") or []) for r in runs]

    # Hit rate per distinct mistake (by content key) - what fraction of the
    # N runs actually surfaced this specific mistake, not just "did the
    # mistake count change". A mistake present in every run has hit_rate
    # 1.0; one that flickered in/out has something less.
    key_hits = defaultdict(int)
    for r in runs:
        seen_this_run = {_mistake_key(m) for m in (r.get("mistakes") or [])}
        for key in seen_this_run:
            key_hits[key] += 1
    mistake_hit_rates = {key: count / len(runs) for key, count in key_hits.items()} if runs else {}

    return {
        "repeats": len(runs),
        "runs": runs,
        "overall_band_values": overall_bands,
        "overall_band_min": min(overall_bands) if overall_bands else None,
        "overall_band_max": max(overall_bands) if overall_bands else None,
        "overall_band_spread": (max(overall_bands) - min(overall_bands)) if overall_bands else None,
        "criteria_ranges": {
            key: {"values": vals, "min": min(vals), "max": max(vals)}
            for key, vals in criteria_values.items()
        },
        "mistake_counts": mistake_counts,
        "mistake_hit_rates": mistake_hit_rates,
    }


def run_eval(
    task_type: str, question: str, text: str, image_url: str | None = None,
    _context: str = "", repeats: int = 1,
) -> dict:
    """repeats=1 (default): identical behaviour and return shape to before
    this function had a repeat count - every existing call site in this
    harness is completely unaffected. repeats>1: runs the same input that
    many times and returns the aggregated range/hit-rate dict from
    _aggregate_runs() instead of a single flat result - callers that want
    per-run detail can still read it back out of the "runs" list."""
    if repeats <= 1:
        return _run_eval_once(task_type, question, text, image_url, _context)
    runs = [_run_eval_once(task_type, question, text, image_url, _context) for _ in range(repeats)]
    return _aggregate_runs(runs)


# ---------------------------------------------------------------------------
# Instrumentation: capture every raw band-checklist boolean grid GPT
# returns, without altering what evaluate_writing() actually does with it.
# Wraps call_gpt_writing() (imported by name inside evaluators.writing) so
# the real call still happens and its real return value still flows through
# unchanged - only a copy is siphoned off into _captured_band_flags.
# ---------------------------------------------------------------------------
_captured_band_flags = []


def _install_instrumentation():
    original = writing_module.call_gpt_writing

    def capturing(prompt, image_url=None):
        result = original(prompt, image_url=image_url)
        if isinstance(result, dict):
            _captured_band_flags.append(dict(result))
        return result

    writing_module.call_gpt_writing = capturing
    return original


def _uninstall_instrumentation(original):
    writing_module.call_gpt_writing = original


def instrumentation_report() -> dict:
    """Frequency table: for each (criterion_key, band) pair, how often was
    that band's checklist marked false across every captured clean-corpus
    call - i.e. which specific descriptor features are being judged unmet
    even on essays that should score high. This is the evidence the
    deferred Step 6 decision depends on: if failures concentrate on a
    small number of specific bands/criteria, Steps 1-5 (better mistake
    detection) plausibly fix the band problem as a side effect; if
    failures are broad and diffuse across many different features, that
    points toward the conjunctive rule itself being the bottleneck."""
    band_keys = [
        "task_achievement_bands", "task_response_bands",
        "coherence_cohesion_bands", "lexical_resource_bands", "grammar_bands",
    ]
    false_counts = defaultdict(int)
    total_counts = defaultdict(int)
    for response in _captured_band_flags:
        for key in band_keys:
            grid = response.get(key)
            if not isinstance(grid, dict):
                continue
            for band_str, value in grid.items():
                total_counts[(key, band_str)] += 1
                if value is not True:
                    false_counts[(key, band_str)] += 1
    return {
        "false_counts": dict(false_counts),
        "total_counts": dict(total_counts),
        "n_captured_calls": len(_captured_band_flags),
    }


# ---------------------------------------------------------------------------
# CHECK 1 - False positives on the clean corpus.
# ---------------------------------------------------------------------------
def check_1_false_positives() -> dict:
    rows = []
    category_counts = defaultdict(int)
    for entry in CLEAN_CORPUS:
        result = run_eval(entry["task_type"], entry["question"], entry["text"], _context=f"check 1 / {entry['id']}")
        mistakes = result.get("mistakes", [])
        for m in mistakes:
            category_counts[m.get("category", "(none)")] += 1
        rows.append({
            "id": entry["id"],
            "mistake_count": len(mistakes),
            "band": result.get("overall_band"),
            "ai_evaluation_failed": result.get("ai_evaluation_failed"),
        })
    total_mistakes = sum(r["mistake_count"] for r in rows)
    return {"rows": rows, "total_mistakes": total_mistakes, "category_counts": dict(category_counts)}


# ---------------------------------------------------------------------------
# CHECKS 2 & 3 - Recall and category accuracy on the injected-error corpus.
# ---------------------------------------------------------------------------
def _diff_core(find: str, replace: str) -> str:
    """The minimal substring that actually changed between find/replace,
    by trimming the common prefix and suffix - used as one signal (not the
    only one) for whether a returned mistake plausibly caught a specific
    injection."""
    i = 0
    while i < len(find) and i < len(replace) and find[i] == replace[i]:
        i += 1
    find_rest, replace_rest = find[i:], replace[i:]
    j = 0
    while j < len(find_rest) and j < len(replace_rest) and find_rest[len(find_rest) - 1 - j] == replace_rest[len(replace_rest) - 1 - j]:
        j += 1
    core = replace_rest[: len(replace_rest) - j] if j else replace_rest
    return core.strip()


def _mistake_matches_injection(mistake: dict, spec: dict) -> bool:
    original = (mistake.get("original") or mistake.get("sentence") or "").lower()
    if not original:
        return False
    replace_lower = spec["replace"].lower()
    # Signal A: substring overlap either direction (same technique already
    # used in evaluators/writing.py's own sentence-matching, e.g.
    # _escalate_error_clusters).
    substring_overlap = original in replace_lower or replace_lower in original
    # Signal B: the specific word(s) that actually changed appear somewhere
    # in the mistake's own text (original, subtype, or explanation) - looser
    # than pure substring containment, to tolerate GPT quoting a slightly
    # different span boundary than the injection's full find/replace pair.
    core = _diff_core(spec["find"], spec["replace"]).lower()
    haystack = " ".join([
        original,
        (mistake.get("subtype") or "").lower(),
        (mistake.get("explanation") or "").lower(),
    ])
    core_present = bool(core) and core in haystack
    return substring_overlap or core_present


def check_2_and_3_recall_and_category() -> dict:
    damaged_corpus = build_damaged_corpus()
    per_injection_results = []
    confusion = defaultdict(lambda: defaultdict(int))  # injected_type -> reported_category -> count

    for damaged in damaged_corpus:
        result = run_eval(damaged["task_type"], damaged["question"], damaged["text"], _context=f"checks 2/3 / {damaged['id']}")
        mistakes = result.get("mistakes", [])

        for spec in damaged["injections"]:
            matched_mistakes = [m for m in mistakes if _mistake_matches_injection(m, spec)]
            caught = len(matched_mistakes) > 0
            category_correct = False
            reported_category = None
            if caught:
                reported_category = matched_mistakes[0].get("category", "(none)")
                expected = {spec["expected_category"], *spec.get("expected_category_alts", [])}
                category_correct = reported_category in expected
                confusion[spec["error_type"]][reported_category] += 1
            else:
                confusion[spec["error_type"]]["(missed)"] += 1
            per_injection_results.append({
                "injection_id": spec["id"],
                "error_type": spec["error_type"],
                "expected_category": spec["expected_category"],
                "caught": caught,
                "category_correct": category_correct,
                "reported_category": reported_category,
            })

    n = len(per_injection_results)
    n_caught = sum(1 for r in per_injection_results if r["caught"])
    n_category_correct = sum(1 for r in per_injection_results if r["category_correct"])
    recall_pct = 100.0 * n_caught / n if n else 0.0
    category_accuracy_pct = 100.0 * n_category_correct / n_caught if n_caught else 0.0

    missed_by_type = defaultdict(int)
    for r in per_injection_results:
        if not r["caught"]:
            missed_by_type[r["error_type"]] += 1

    return {
        "per_injection_results": per_injection_results,
        "recall_pct": recall_pct,
        "category_accuracy_pct": category_accuracy_pct,
        "n_total": n,
        "n_caught": n_caught,
        "n_category_correct": n_category_correct,
        "missed_by_type": dict(missed_by_type),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


# ---------------------------------------------------------------------------
# CHECK 4 - Ordering (degradation ladder monotonicity).
# ---------------------------------------------------------------------------
def check_4_ordering() -> dict:
    ladder = build_ladder()
    from clean_corpus import CLEAN_CORPUS_BY_ID
    source = CLEAN_CORPUS_BY_ID["clean_t2_01"]

    scored_levels = []
    for level in ladder:
        result = run_eval(source["task_type"], source["question"], level["text"], _context=f"check 4 / level {level['level']}")
        scored_levels.append({
            "level": level["level"],
            "description": level["description"],
            "overall_band": result.get("overall_band"),
            "criteria_scores": result.get("criteria_scores", {}),
        })

    inversions = []
    criteria_keys = ["task_response", "coherence_cohesion", "lexical_resource", "grammar_accuracy"]
    for i in range(len(scored_levels) - 1):
        a, b = scored_levels[i], scored_levels[i + 1]
        if b["overall_band"] is not None and a["overall_band"] is not None and b["overall_band"] > a["overall_band"]:
            inversions.append(f"overall_band: level {a['level']} ({a['overall_band']}) -> level {b['level']} ({b['overall_band']}) increased")
        for key in criteria_keys:
            av, bv = a["criteria_scores"].get(key), b["criteria_scores"].get(key)
            if av is not None and bv is not None and bv > av:
                inversions.append(f"{key}: level {a['level']} ({av}) -> level {b['level']} ({bv}) increased")

    return {"scored_levels": scored_levels, "inversions": inversions}


# ---------------------------------------------------------------------------
# CHECK 5 - Stability (same essay, 5 runs).
# ---------------------------------------------------------------------------
def check_5_stability(n_runs: int = 5) -> dict:
    from clean_corpus import CLEAN_CORPUS_BY_ID
    entry = CLEAN_CORPUS_BY_ID["clean_t2_01"]
    runs = []
    for run_index in range(n_runs):
        result = run_eval(entry["task_type"], entry["question"], entry["text"], _context=f"check 5 / run {run_index + 1}")
        mistake_categories = frozenset(m.get("category", "?") for m in result.get("mistakes", []))
        runs.append({
            "overall_band": result.get("overall_band"),
            "criteria_scores": result.get("criteria_scores", {}),
            "mistake_count": len(result.get("mistakes", [])),
            "mistake_categories": mistake_categories,
        })

    bands = [r["overall_band"] for r in runs if r["overall_band"] is not None]
    band_spread = (max(bands) - min(bands)) if bands else None

    jaccards = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i]["mistake_categories"], runs[j]["mistake_categories"]
            union = a | b
            jaccards.append(1.0 if not union else len(a & b) / len(union))
    mean_jaccard = statistics.mean(jaccards) if jaccards else 1.0

    return {"runs": runs, "band_spread": band_spread, "mean_mistake_category_jaccard": mean_jaccard}


# ---------------------------------------------------------------------------
# CHECK 6 - Perturbation invariance.
# ---------------------------------------------------------------------------
def check_6_perturbation_invariance(corpus_subset: list | None = None) -> dict:
    # Full CLEAN_CORPUS x ~4 variants each is expensive (this check alone
    # is the single largest share of the harness's live-call budget) for a
    # check whose transformations are purely textual/mechanical (spelling,
    # quote glyphs, line endings, a name), not dependent on essay content
    # or topic. A representative subset is scientifically equivalent for
    # what this check is actually testing - if a bug like the verbatim-
    # substring/curly-quote issue exists, it will surface on any essay
    # containing an apostrophe, not only on essays this subset excludes.
    # Pass corpus_subset explicitly to run the full corpus instead.
    entries = corpus_subset if corpus_subset is not None else CLEAN_CORPUS[:6]
    results = []
    for entry in entries:
        variants = build_perturbations(entry)
        baseline_result = None
        for v in variants:
            result = run_eval(entry["task_type"], entry["question"], v["text"], _context=f"check 6 / {entry['id']} / {v['variant']}")
            mistake_categories = frozenset(m.get("category", "?") for m in result.get("mistakes", []))
            record = {
                "corpus_id": entry["id"],
                "variant": v["variant"],
                "overall_band": result.get("overall_band"),
                "mistake_count": len(result.get("mistakes", [])),
                "mistake_categories": mistake_categories,
            }
            if v["variant"] == "baseline":
                baseline_result = record
            else:
                band_changed = baseline_result is not None and record["overall_band"] != baseline_result["overall_band"]
                mistakes_changed = baseline_result is not None and record["mistake_categories"] != baseline_result["mistake_categories"]
                record["band_changed_vs_baseline"] = band_changed
                record["mistakes_changed_vs_baseline"] = mistakes_changed
                results.append(record)
    n = len(results)
    n_band_stable = sum(1 for r in results if not r["band_changed_vs_baseline"])
    n_mistakes_stable = sum(1 for r in results if not r["mistakes_changed_vs_baseline"])
    return {
        "results": results,
        "n_total": n,
        "band_stable_pct": 100.0 * n_band_stable / n if n else 0.0,
        "mistakes_stable_pct": 100.0 * n_mistakes_stable / n if n else 0.0,
    }


# ---------------------------------------------------------------------------
# COVERAGE MATRIX - question variant x answer profile. NOT run as part of
# print_report()'s default six checks (it is a separate, much larger live
# run - see the cost estimate this function prints before starting). Call
# run_coverage_matrix() directly when ready, or run this file with
# `--coverage-matrix` (add `--smoke` for the cheap subset, see below).
#
# Full fidelity runs BOTH questions per row for every applicable cell -
# 187 applicable cells x 2 questions = 374 evaluate_writing() calls, each
# ~2 real GPT calls, so roughly 750 API calls for one full pass. Pass
# questions_per_row=1 for a cheaper first pass at roughly half the cost,
# at the cost of losing the "aggregated across 2 independently-authored
# questions" signal the coverage requirement asked for. The full matrix
# (whichever questions_per_row) is the only thing that counts as a
# before/after baseline for a plan step - see smoke below for iteration.
#
# smoke=True runs SMOKE_REPRESENTATIVE_ROWS x all 13 profiles, plus all 15
# rows x SMOKE_ALWAYS_PROFILES (p1/p3/p7) - see coverage_matrix.py for the
# exact membership. This is for fast, cheap iteration once the adversarial
# verifier (plan Step 3) lands and per-cell cost goes up (up to ~13 GPT
# calls/cell instead of today's ~2) - a full run stops being practical to
# re-run after every small change. Never report a smoke run's numbers as a
# step's before/after baseline.
# ---------------------------------------------------------------------------
def run_coverage_matrix(questions_per_row: int = 2, smoke: bool = False) -> dict:
    all_applicable = [(row, col) for row in MATRIX_ROWS for col in MATRIX_COLUMNS if _cell_is_applicable(row, col)]
    cells = [rc for rc in all_applicable if not smoke or _cell_is_in_smoke_subset(*rc)]
    cells_set = set(cells)
    n_calls = len(cells) * min(questions_per_row, 2)
    label = "SMOKE SUBSET" if smoke else "FULL"
    print(f"Coverage matrix ({label}): {len(cells)}/{len(all_applicable)} applicable cells x "
          f"{min(questions_per_row, 2)} question(s)/row = {n_calls} evaluate_writing() calls "
          f"(~{n_calls * 2} raw API calls).")

    cell_results = {}  # (row, profile) -> {question_id: result_dict}
    for row in MATRIX_ROWS:
        question_ids = _questions_for_row(row)[:questions_per_row]
        for profile_name in MATRIX_COLUMNS:
            if not _cell_is_applicable(row, profile_name):
                cell_results[(row, profile_name)] = {"status": "NOT_APPLICABLE"}
                continue
            if (row, profile_name) not in cells_set:
                cell_results[(row, profile_name)] = {"status": "SKIPPED_SMOKE"}
                continue
            per_question = {}
            for qid in question_ids:
                q = QUESTION_BANK_BY_ID[qid]
                base = PROFILE1_ESSAYS[qid]
                built = build_profile(
                    profile_name, base, q["task_type"],
                    t1_variant=q.get("t1_variant"), topic=q.get("topic"),
                )
                if built["text"] is None:
                    continue  # shouldn't happen given _cell_is_applicable, defensive only
                result = run_eval(
                    q["task_type"], q["question"], built["text"],
                    _context=f"coverage matrix / {row} / {profile_name} / {qid}",
                )
                per_question[qid] = {
                    "overall_band": result.get("overall_band"),
                    "criteria_scores": result.get("criteria_scores", {}),
                    "mistake_count": len(result.get("mistakes", [])),
                    "profile_note": built["note"],
                }
            cell_results[(row, profile_name)] = {"status": "OK", "per_question": per_question}

    # Relational assertions, per row, using that row's first question's
    # per-profile results (the assertions are about profile-to-profile
    # relationships within one consistent essay identity, not aggregated
    # across the row's 2 questions - aggregating first would blur exactly
    # the signal these checks exist to catch).
    assertion_results = {}
    for row in MATRIX_ROWS:
        question_ids = _questions_for_row(row)[:questions_per_row]
        if not question_ids:
            continue
        first_qid = question_ids[0]
        results_by_profile = {}
        for profile_name in MATRIX_COLUMNS:
            cell = cell_results.get((row, profile_name), {})
            if cell.get("status") == "OK" and first_qid in cell.get("per_question", {}):
                results_by_profile[profile_name] = cell["per_question"][first_qid]
        row_violations = {}
        for name, fn in ALL_RELATIONAL_ASSERTIONS:
            violations = fn(results_by_profile)
            if violations:
                row_violations[name] = violations
        assertion_results[row] = row_violations

    # P6 format/register mismatch check - cross-row, computed once over
    # the whole matrix rather than per-row (see
    # assert_p6_format_mismatch_scores_lower's docstring for why this
    # can't live inside the per-row loop above). Uses each row's first
    # question's P6 task_response, same first-question convention as the
    # per-row assertions.
    p6_task_response_by_row = {}
    for row in MATRIX_ROWS:
        question_ids = _questions_for_row(row)[:questions_per_row]
        if not question_ids:
            continue
        cell = cell_results.get((row, "p6_memorised_template"), {})
        per_question = cell.get("per_question", {}) if cell.get("status") == "OK" else {}
        first_result = per_question.get(question_ids[0])
        if first_result:
            p6_task_response_by_row[row] = first_result["criteria_scores"].get("task_response")
    p6_format_mismatch_violations = assert_p6_format_mismatch_scores_lower(p6_task_response_by_row)

    n_ok = sum(1 for v in cell_results.values() if v["status"] == "OK")
    n_skipped = sum(1 for v in cell_results.values() if v["status"] == "SKIPPED_SMOKE")
    n_rows_with_violations = sum(1 for v in assertion_results.values() if v)
    print(f"\nDone: {n_ok} cells OK, {n_skipped} skipped (smoke), "
          f"{n_rows_with_violations}/{len(assertion_results)} rows with assertion violations.")
    for row, row_violations in assertion_results.items():
        if not row_violations:
            continue
        print(f"  {row}:")
        for name, vlist in row_violations.items():
            for v in vlist:
                print(f"    [{name}] {v}")
    print("  p6_format_mismatch (cross-row):")
    for v in p6_format_mismatch_violations:
        print(f"    [p6_format_mismatch] {v}")

    return {
        "cell_results": cell_results,
        "assertion_results": assertion_results,
        "p6_format_mismatch_violations": p6_format_mismatch_violations,
        "smoke": smoke,
    }


# ---------------------------------------------------------------------------
# ROUND-TRIP CHECK - the v2 refine pipeline's strongest free validation
# (see the "refined_answer overhaul" plan). Feeds each qualifying profile's
# generated refined_answer back through evaluate_writing() as if it were a
# fresh submission. NOT part of print_report()'s default six checks - real
# cost, up to ~5 GPT calls for the first pass (flag ON) plus ~2 for the
# round-trip pass (flag OFF, since the round-trip's own refined_answer is
# immediately discarded - generating one under the v2 pipeline there would
# waste calls for nothing). Call run_refine_round_trip_check() directly, or
# run this file with `--round-trip`.
# ---------------------------------------------------------------------------
WRITING_INDEPENDENT_MODEL_ANSWER_FLAG = "WRITING_INDEPENDENT_MODEL_ANSWER"


def run_refine_round_trip_check(
    profile_names: tuple = ("p3_weak", "p6_memorised_template", "p7_off_topic"),
    questions_per_row: int = 1,
) -> dict:
    """Assertions are RELATIONAL, not an absolute band gate - matching
    every assertion in coverage_matrix.py. There is no labelled ground
    truth in this corpus, and this harness's own instrumentation_report()
    already suggests the scorer may under-award strong text on some
    descriptor features, so a hardcoded "must land 8.5-9.0" target
    couldn't distinguish a refine-quality bug from a pre-existing
    scorer-strictness one - exactly the ambiguity this check exists to
    resolve, not reproduce. Checks: round-trip band strictly above the
    original submission's band; round-trip mistake_count <= 2. Raw
    round-trip band/mistake numbers are reported in full regardless, for
    human eyeballing.

    KNOWN LIMITATION: this harness's 30 questions carry no real chart
    images, so Task 1 Academic rows run this check without the
    image-extraction path (falls back to the question-text-derived
    category guess, or no coverage requirement at all) - only the general
    refine-quality machinery (retry/validation/vocabulary/no-truncation)
    is exercised there. The chart-coverage fix itself is verified
    separately, live, against a real chart image - see the plan's
    Verification section. This is a documented gap, not a silent one."""
    results = []
    original_flag = os.environ.get(WRITING_INDEPENDENT_MODEL_ANSWER_FLAG)
    try:
        for row in MATRIX_ROWS:
            question_ids = _questions_for_row(row)[:questions_per_row]
            for qid in question_ids:
                q = QUESTION_BANK_BY_ID[qid]
                base = PROFILE1_ESSAYS[qid]
                for profile_name in profile_names:
                    if not _cell_is_applicable(row, profile_name):
                        continue
                    built = build_profile(
                        profile_name, base, q["task_type"],
                        t1_variant=q.get("t1_variant"), topic=q.get("topic"),
                    )
                    if built["text"] is None:
                        continue

                    os.environ[WRITING_INDEPENDENT_MODEL_ANSWER_FLAG] = "true"
                    first_pass = run_eval(
                        q["task_type"], q["question"], built["text"],
                        _context=f"round-trip first pass / {row} / {profile_name} / {qid}",
                    )
                    refined_text = first_pass.get("refined_answer")
                    original_band = first_pass.get("overall_band")
                    if not refined_text:
                        continue

                    os.environ[WRITING_INDEPENDENT_MODEL_ANSWER_FLAG] = "false"
                    round_trip = run_eval(
                        q["task_type"], q["question"], refined_text,
                        _context=f"round-trip second pass / {row} / {profile_name} / {qid}",
                    )

                    round_trip_band = round_trip.get("overall_band")
                    round_trip_mistake_count = len(round_trip.get("mistakes", []))
                    violations = []
                    if original_band is not None and round_trip_band is not None and round_trip_band <= original_band:
                        violations.append(
                            f"round-trip band ({round_trip_band}) did not improve on original ({original_band})"
                        )
                    if round_trip_mistake_count > 2:
                        violations.append(f"round-trip mistake_count ({round_trip_mistake_count}) > 2")

                    results.append({
                        "row": row, "profile": profile_name, "question_id": qid,
                        "original_band": original_band,
                        "round_trip_band": round_trip_band,
                        "round_trip_mistake_count": round_trip_mistake_count,
                        "violations": violations,
                    })
    finally:
        if original_flag is None:
            os.environ.pop(WRITING_INDEPENDENT_MODEL_ANSWER_FLAG, None)
        else:
            os.environ[WRITING_INDEPENDENT_MODEL_ANSWER_FLAG] = original_flag

    n_passed = sum(1 for r in results if not r["violations"])
    print(f"Round-trip check: {len(results)} cells run, {n_passed}/{len(results)} passed relational assertions.")
    for r in results:
        flag = "" if not r["violations"] else f"  <-- {'; '.join(r['violations'])}"
        print(
            f"  {r['row']:28s} {r['profile']:32s} orig_band={r['original_band']} "
            f"round_trip_band={r['round_trip_band']} mistakes={r['round_trip_mistake_count']}{flag}"
        )

    return {"results": results, "n_total": len(results), "n_passed": n_passed}


def check_off_topic_round_trip(
    profile_names: tuple = ("p7_off_topic", "p8_partially_off_topic"),
    questions_per_row: int = 1,
) -> dict:
    """Off-topic/partially-off-topic specific round-trip check - see the
    "off-topic answers" plan. Deliberately HARD-gated (band >= 8.5 AND
    round-trip relevance_status == "on_topic"), unlike the general
    run_refine_round_trip_check() above, which is relational-only. The two
    checks validate different claims: the general one asks "does refine
    improve on a weak/constrained input" (where scorer strictness is a
    real confound); this one asks "does the off-topic path - which writes
    entirely fresh with no weak-input constraint at all - reliably produce
    something strong and on-topic" (where an absolute floor is more
    defensible, since there's no weak-input excuse left). An off-topic
    model answer that itself fails its own relevance check is a hard
    failure, not a "your call" - per the plan.

    Runs the round-trip (second) pass with the flag ON too, unlike
    run_refine_round_trip_check()'s flag-OFF second pass - relevance_status
    only exists in the result under the flag, and this check specifically
    needs the round-trip's OWN relevance_status. Real cost: both passes
    now run the full v2 pipeline, not just the first."""
    results = []
    original_flag = os.environ.get(WRITING_INDEPENDENT_MODEL_ANSWER_FLAG)
    try:
        os.environ[WRITING_INDEPENDENT_MODEL_ANSWER_FLAG] = "true"
        for row in MATRIX_ROWS:
            question_ids = _questions_for_row(row)[:questions_per_row]
            for qid in question_ids:
                q = QUESTION_BANK_BY_ID[qid]
                base = PROFILE1_ESSAYS[qid]
                for profile_name in profile_names:
                    if not _cell_is_applicable(row, profile_name):
                        continue
                    built = build_profile(
                        profile_name, base, q["task_type"],
                        t1_variant=q.get("t1_variant"), topic=q.get("topic"),
                    )
                    if built["text"] is None:
                        continue

                    first_pass = run_eval(
                        q["task_type"], q["question"], built["text"],
                        _context=f"off-topic round-trip first pass / {row} / {profile_name} / {qid}",
                    )
                    refined_text = first_pass.get("refined_answer")
                    if not refined_text:
                        continue

                    round_trip = run_eval(
                        q["task_type"], q["question"], refined_text,
                        _context=f"off-topic round-trip second pass / {row} / {profile_name} / {qid}",
                    )

                    round_trip_band = round_trip.get("overall_band")
                    round_trip_status = round_trip.get("relevance_status")
                    violations = []
                    if round_trip_band is None or round_trip_band < 8.5:
                        violations.append(f"round-trip band ({round_trip_band}) < 8.5")
                    if round_trip_status != "on_topic":
                        violations.append(f"round-trip relevance_status ({round_trip_status!r}) != 'on_topic'")

                    results.append({
                        "row": row, "profile": profile_name, "question_id": qid,
                        "first_pass_relevance_status": first_pass.get("relevance_status"),
                        "model_answer_source": (first_pass.get("relevance_notice") or {}).get("model_answer_source"),
                        "round_trip_band": round_trip_band,
                        "round_trip_relevance_status": round_trip_status,
                        "violations": violations,
                    })
    finally:
        if original_flag is None:
            os.environ.pop(WRITING_INDEPENDENT_MODEL_ANSWER_FLAG, None)
        else:
            os.environ[WRITING_INDEPENDENT_MODEL_ANSWER_FLAG] = original_flag

    n_passed = sum(1 for r in results if not r["violations"])
    print(f"Off-topic round-trip check: {len(results)} cells run, {n_passed}/{len(results)} passed (hard gate: band >= 8.5 AND on_topic).")
    for r in results:
        flag = "" if not r["violations"] else f"  <-- {'; '.join(r['violations'])}"
        print(
            f"  {r['row']:28s} {r['profile']:32s} source={r['model_answer_source']} "
            f"round_trip_band={r['round_trip_band']} round_trip_status={r['round_trip_relevance_status']}{flag}"
        )

    return {"results": results, "n_total": len(results), "n_passed": n_passed}


# ---------------------------------------------------------------------------
# OPTIONAL - band accuracy against local-only official samples.
# ---------------------------------------------------------------------------
def check_band_accuracy_optional() -> dict | None:
    local_dir = os.path.join(_THIS_DIR, "official_samples_local")
    if not os.path.isdir(local_dir) or not os.listdir(local_dir):
        return None
    try:
        from official_samples_extractor import iter_official_samples
    except ImportError:
        return {"skipped": True, "reason": "official_samples_extractor.py not present yet"}

    rows = []
    for sample in iter_official_samples(local_dir):
        result = run_eval(sample["task_type"], sample["question"], sample["text"], _context=f"band-accuracy / {sample['script_id']}")
        rows.append({
            "script_id": sample["script_id"],
            "expected_band": sample["expected_band"],
            "actual_band": result.get("overall_band"),
            "delta": (result.get("overall_band") - sample["expected_band"]) if result.get("overall_band") is not None else None,
            "mistake_count": len(result.get("mistakes", [])),
        })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Manual QA audit mining - PENDING. The user was asked for the Portal Tests
# 1-50 audit material (tagged Unfair Correction / Overcorrection / Incorrect
# Spelling / Major difference) and it has not been provided yet. This is a
# deliberate placeholder, not a silent omission - do not fabricate audit
# findings. Once provided, this becomes: negative examples appended to the
# detection prompt, category-mislabel entries added to the closed enum
# mapping (Step 2), and one regression-test case per distinct false-positive
# pattern added to this harness.
# ---------------------------------------------------------------------------
def audit_mining_status() -> dict:
    return {
        "status": "PENDING - manual QA audit (Portal Tests 1-50) not yet provided",
        "action_needed": "user to provide the audit material",
    }


# ---------------------------------------------------------------------------
# Report printing.
# ---------------------------------------------------------------------------
def print_report():
    print("=" * 78)
    print("WRITING EVAL HARNESS - baseline run")
    print("=" * 78)

    original_call = _install_instrumentation()
    t0 = time.time()
    try:
        try:
            c1 = check_1_false_positives()
            c23 = check_2_and_3_recall_and_category()
            c4 = check_4_ordering()
            c5 = check_5_stability()
            c6 = check_6_perturbation_invariance()
            band_acc = check_band_accuracy_optional()
            instr = instrumentation_report()
        except HarnessAbortError as e:
            # Deliberately no results table below this point - see
            # HarnessAbortError's docstring. Re-raised after printing so
            # the process also exits non-zero and prints a traceback,
            # rather than silently continuing.
            print("\n" + "!" * 78)
            print("HARNESS RUN ABORTED - a degraded result (ai_evaluation_failed=True)")
            print("was returned during this run. NO RESULTS TABLE WILL BE PRINTED.")
            print("Any numbers computed so far are built on incomplete/fallback data")
            print("and must not be trusted or reported as a baseline.")
            print("!" * 78)
            print(f"\n{e}\n")
            raise
    finally:
        _uninstall_instrumentation(original_call)
    elapsed = time.time() - t0

    print(f"\nTotal wall-clock time: {elapsed:.1f}s\n")

    print("-" * 78)
    print("CHECK 1 - False positives on clean corpus (expect 0 mistakes everywhere)")
    print("-" * 78)
    for r in c1["rows"]:
        flag = "  <-- FALSE POSITIVE(S)" if r["mistake_count"] > 0 else ""
        print(f"  {r['id']:15s} band={r['band']:<5} mistakes={r['mistake_count']}{flag}")
    print(f"  TOTAL false-positive mistakes across {len(c1['rows'])} clean texts: {c1['total_mistakes']}")
    if c1["category_counts"]:
        print(f"  By category: {c1['category_counts']}")

    print("-" * 78)
    print("CHECKS 2/3 - Recall and category accuracy on injected errors")
    print("-" * 78)
    print(f"  Recall: {c23['n_caught']}/{c23['n_total']} ({c23['recall_pct']:.1f}%)")
    print(f"  Category accuracy (of caught): {c23['n_category_correct']}/{c23['n_caught']} ({c23['category_accuracy_pct']:.1f}%)")
    if c23["missed_by_type"]:
        print(f"  Missed by error type: {c23['missed_by_type']}")
    print("  Confusion (injected_type -> {reported_category: count}):")
    for etype, cats in c23["confusion"].items():
        print(f"    {etype}: {cats}")

    print("-" * 78)
    print("CHECK 4 - Ordering (degradation ladder must be non-increasing)")
    print("-" * 78)
    for lvl in c4["scored_levels"]:
        print(f"  Level {lvl['level']} ({lvl['description']}): overall_band={lvl['overall_band']} criteria={lvl['criteria_scores']}")
    if c4["inversions"]:
        print(f"  INVERSIONS FOUND ({len(c4['inversions'])}):")
        for inv in c4["inversions"]:
            print(f"    - {inv}")
    else:
        print("  No inversions - monotonically non-increasing on every criterion.")

    print("-" * 78)
    print("CHECK 5 - Stability (5 runs, same essay)")
    print("-" * 78)
    print(f"  Bands across runs: {[r['overall_band'] for r in c5['runs']]}")
    print(f"  Max band spread: {c5['band_spread']}")
    print(f"  Mean mistake-category Jaccard similarity: {c5['mean_mistake_category_jaccard']:.2f}")

    print("-" * 78)
    print("CHECK 6 - Perturbation invariance")
    print("-" * 78)
    print(f"  Band stable across perturbations: {c6['band_stable_pct']:.1f}% ({c6['n_total']} variant checks)")
    print(f"  Mistake set stable across perturbations: {c6['mistakes_stable_pct']:.1f}%")
    for r in c6["results"]:
        if r["band_changed_vs_baseline"] or r["mistakes_changed_vs_baseline"]:
            print(f"  CHANGED: {r['corpus_id']} / {r['variant']} - band_changed={r['band_changed_vs_baseline']} mistakes_changed={r['mistakes_changed_vs_baseline']}")

    print("-" * 78)
    print("INSTRUMENTATION - band-checklist feature failure frequency (clean corpus)")
    print("-" * 78)
    print(f"  Captured {instr['n_captured_calls']} raw GPT scoring responses.")
    interesting = sorted(instr["false_counts"].items(), key=lambda kv: -kv[1])[:15]
    for (criterion, band), false_n in interesting:
        total_n = instr["total_counts"].get((criterion, band), 0)
        print(f"    {criterion} band={band}: false {false_n}/{total_n} times")

    print("-" * 78)
    print("BAND ACCURACY (optional, local-only official samples)")
    print("-" * 78)
    if band_acc is None:
        print("  SKIPPED - tests/writing_eval_harness/official_samples_local/ not present.")
        print("  This is expected for anyone without the local-only PDFs; the six checks")
        print("  above ran regardless and are the required baseline.")
    elif band_acc.get("skipped"):
        print(f"  SKIPPED - {band_acc['reason']}")
    else:
        for r in band_acc["rows"]:
            print(f"  {r['script_id']:20s} expected={r['expected_band']} actual={r['actual_band']} delta={r['delta']}")

    print("-" * 78)
    print("MANUAL QA AUDIT MINING")
    print("-" * 78)
    audit = audit_mining_status()
    print(f"  {audit['status']}")

    print("=" * 78)
    print("END OF REPORT")
    print("=" * 78)


if __name__ == "__main__":
    # `python run_eval_harness.py` (no flags) runs the default six-check
    # baseline report - the standing default, unaffected by anything below.
    # `--coverage-matrix` runs the (much larger, much more expensive)
    # question-variant x answer-profile matrix instead; add `--smoke` for
    # the cheap iteration subset and/or `--questions-per-row=1` for the
    # cheaper half-fidelity pass. Both are real GPT calls, real cost -
    # nothing here runs until the user has confirmed API credits.
    if "--coverage-matrix" in sys.argv:
        run_coverage_matrix(
            questions_per_row=1 if "--questions-per-row=1" in sys.argv else 2,
            smoke="--smoke" in sys.argv,
        )
    elif "--round-trip" in sys.argv:
        run_refine_round_trip_check()
    elif "--off-topic-round-trip" in sys.argv:
        check_off_topic_round_trip()
    else:
        print_report()
