"""
speaking_eval_harness - label-free eval harness for the Speaking evaluator,
mirroring tests/writing_eval_harness/. See the "Speaking evaluator: port
proven deterministic Writing fixes + build measurement" plan.

Run: venv/Scripts/python.exe tests/speaking_eval_harness/run_eval_harness.py

Runs live against the real generate_scores()/generate_question_mistakes()
(evaluators/speaking_audio.py, the only Speaking scoring engine now that
the legacy pipeline has been retired - see SPEAKING_ENGINE_CONSOLIDATION.md)
- real GPT calls, real cost, not a mocked unit test. Targets each
function's plain-text entry point directly (no audio/Whisper involved),
the same pattern tests/test_speaking_token_usage.py already uses.

Determinism (Check 1) is the first check because nothing else is
trustworthy to measure until it's stable - matches the "TEMPERATURE 0...
this is the first thing" ordering from the approved plan.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _REPO_ROOT)

from clean_corpus import ALL_CLEAN_ITEMS, combined_transcript_for_part
from asr_artifact_corpus import ASR_ARTIFACT_ITEMS
from error_injection import build_damaged_corpus

from evaluators.speaking_audio import generate_scores, generate_question_mistakes


class HarnessAbortError(Exception):
    """Raised when a GPT call returns something unusable (exception,
    unparseable response) partway through a harness run - same principle
    as the Writing harness's HarnessAbortError: a run must never render a
    results table built on partial/degraded data as if it were a
    trustworthy result."""


def _safe_generate_question_mistakes(question: str, answer: str, _context: str = "") -> dict:
    try:
        result = generate_question_mistakes(question, answer)
    except Exception as e:
        raise HarnessAbortError(f"generate_question_mistakes failed{f' (during: {_context})' if _context else ''}: {e}")
    if not isinstance(result, dict) or "mistakes" not in result:
        raise HarnessAbortError(f"generate_question_mistakes returned an unusable result{f' (during: {_context})' if _context else ''}: {result!r}")
    return result


def _safe_generate_scores(part: int, transcript: str, _context: str = "") -> dict:
    try:
        result = generate_scores(part, transcript)
    except Exception as e:
        raise HarnessAbortError(f"generate_scores failed{f' (during: {_context})' if _context else ''}: {e}")
    if not isinstance(result, dict) or "fluency" not in result:
        raise HarnessAbortError(f"generate_scores returned an unusable result{f' (during: {_context})' if _context else ''}: {result!r}")
    return result


# ---------------------------------------------------------------------------
# CHECK 1 - Determinism (temperature=0). Run FIRST - nothing else is
# trustworthy to measure until this is stable.
# ---------------------------------------------------------------------------
def check_1_determinism(n_runs: int = 5) -> dict:
    fixed_item = ALL_CLEAN_ITEMS[0]  # p1_hometown
    part1_transcript = combined_transcript_for_part(1)

    score_runs = []
    mistake_runs = []
    for _ in range(n_runs):
        scores = _safe_generate_scores(1, part1_transcript, _context="determinism/generate_scores")
        score_runs.append({k: scores.get(k) for k in ("fluency", "lexical", "grammar", "pronunciation")})
        mistakes_result = _safe_generate_question_mistakes(fixed_item["question"], fixed_item["answer"], _context="determinism/generate_question_mistakes")
        mistake_runs.append(mistakes_result.get("mistakes", []))

    def _spread(dicts, key):
        values = [d.get(key) for d in dicts if d.get(key) is not None]
        return (min(values), max(values)) if values else (None, None)

    score_spread = {k: _spread(score_runs, k) for k in ("fluency", "lexical", "grammar", "pronunciation")}
    mistake_counts = [len(m) for m in mistake_runs]
    mistake_category_sets = [frozenset((m.get("type"), m.get("original")) for m in run) for run in mistake_runs]
    mistake_sets_identical = len(set(mistake_category_sets)) == 1

    return {
        "n_runs": n_runs,
        "score_runs": score_runs,
        "score_spread": score_spread,
        "mistake_counts": mistake_counts,
        "mistake_sets_identical": mistake_sets_identical,
    }


# ---------------------------------------------------------------------------
# CHECK 2 - False positives on the clean corpus (expect 0 mistakes).
# ---------------------------------------------------------------------------
def check_2_false_positives_clean() -> dict:
    rows = []
    for item in ALL_CLEAN_ITEMS:
        result = _safe_generate_question_mistakes(item["question"], item["answer"], _context=f"clean/{item['id']}")
        mistakes = result.get("mistakes", [])
        rows.append({"id": item["id"], "part": item["part"], "mistake_count": len(mistakes), "mistakes": mistakes})
    return {"rows": rows, "total_mistakes": sum(r["mistake_count"] for r in rows)}


# ---------------------------------------------------------------------------
# CHECK 3 - False positives on ASR-artifact transcripts (expect 0
# mistakes) - the direct test of the punctuation/filler/self-correction/
# contraction guards.
# ---------------------------------------------------------------------------
def check_3_false_positives_asr_artifacts() -> dict:
    rows = []
    for item in ASR_ARTIFACT_ITEMS:
        result = _safe_generate_question_mistakes(item["question"], item["answer"], _context=f"asr_artifact/{item['id']}")
        mistakes = result.get("mistakes", [])
        rows.append({"id": item["id"], "part": item["part"], "mistake_count": len(mistakes), "mistakes": mistakes})
    return {"rows": rows, "total_mistakes": sum(r["mistake_count"] for r in rows)}


# ---------------------------------------------------------------------------
# CHECK 4 - Recall + category accuracy on injected errors.
# ---------------------------------------------------------------------------
def check_4_recall_and_category() -> dict:
    damaged_corpus = build_damaged_corpus()
    rows = []
    n_caught = 0
    n_category_correct = 0
    for item in damaged_corpus:
        result = _safe_generate_question_mistakes(item["question"], item["answer"], _context=f"injected/{item['id']}")
        mistakes = result.get("mistakes", [])
        signal = item["detection_signal"].lower()
        matches = [m for m in mistakes if signal in (m.get("original") or "").lower()]
        caught = len(matches) > 0
        category_correct = caught and any(m.get("type") == item["expected_type"] for m in matches)
        if caught:
            n_caught += 1
        if category_correct:
            n_category_correct += 1
        rows.append({
            "id": item["id"], "expected_type": item["expected_type"],
            "caught": caught, "category_correct": category_correct,
            "returned_mistakes": mistakes,
        })
    n_total = len(damaged_corpus)
    return {
        "rows": rows, "n_total": n_total, "n_caught": n_caught,
        "n_category_correct": n_category_correct,
        "recall_pct": 100.0 * n_caught / n_total if n_total else 0.0,
        "category_accuracy_pct": 100.0 * n_category_correct / n_caught if n_caught else 0.0,
    }


# ---------------------------------------------------------------------------
# Report printing.
# ---------------------------------------------------------------------------
def print_report():
    print("=" * 78)
    print("SPEAKING EVAL HARNESS - baseline run")
    print("=" * 78)

    try:
        c1 = check_1_determinism()
        c2 = check_2_false_positives_clean()
        c3 = check_3_false_positives_asr_artifacts()
        c4 = check_4_recall_and_category()
    except HarnessAbortError as e:
        print("\n" + "!" * 78)
        print("HARNESS RUN ABORTED - a GPT call returned something unusable.")
        print("NO RESULTS TABLE WILL BE PRINTED. Any numbers computed so far are")
        print("built on incomplete data and must not be trusted or reported.")
        print("!" * 78)
        print(f"\n{e}\n")
        raise

    print("-" * 78)
    print("CHECK 1 - Determinism (temperature=0, 5 runs)")
    print("-" * 78)
    print(f"  speaking_audio.py generate_scores spread: {c1['score_spread']}")
    print(f"  speaking_audio.py mistake counts across runs: {c1['mistake_counts']}")
    print(f"  speaking_audio.py mistake sets identical across all runs: {c1['mistake_sets_identical']}")

    print("-" * 78)
    print("CHECK 2 - False positives on clean corpus (expect 0 mistakes)")
    print("-" * 78)
    for r in c2["rows"]:
        flag = "  <-- FALSE POSITIVE(S)" if r["mistake_count"] > 0 else ""
        print(f"  {r['id']:20s} part={r['part']} mistakes={r['mistake_count']}{flag}")
    print(f"  TOTAL false-positive mistakes across {len(c2['rows'])} clean transcripts: {c2['total_mistakes']}")

    print("-" * 78)
    print("CHECK 3 - False positives on ASR-artifact transcripts (expect 0 mistakes)")
    print("-" * 78)
    for r in c3["rows"]:
        flag = "  <-- FALSE POSITIVE(S)" if r["mistake_count"] > 0 else ""
        print(f"  {r['id']:20s} part={r['part']} mistakes={r['mistake_count']}{flag}")
    print(f"  TOTAL false-positive mistakes across {len(c3['rows'])} ASR-artifact transcripts: {c3['total_mistakes']}")

    print("-" * 78)
    print("CHECK 4 - Recall + category accuracy on injected errors")
    print("-" * 78)
    print(f"  Recall: {c4['n_caught']}/{c4['n_total']} ({c4['recall_pct']:.1f}%)")
    print(f"  Category accuracy (of caught): {c4['n_category_correct']}/{c4['n_caught'] or 1} ({c4['category_accuracy_pct']:.1f}%)")
    for r in c4["rows"]:
        status = "caught+correct" if r["category_correct"] else ("caught, wrong category" if r["caught"] else "MISSED")
        print(f"  {r['id']:30s} expected={r['expected_type']:12s} {status}")

    print("=" * 78)
    print("END OF REPORT")
    print("=" * 78)


if __name__ == "__main__":
    print_report()
