"""Speaking determinism runner - measurement only, no behaviour change.

Runs Engine A's generate_scores()/generate_question_mistakes() N times on
identical text input and reports band spread, mistake-detection hit rates,
and checklist-boolean stability - the same shape of measurement built for
Writing's harness (tests/writing_eval_harness/run_eval_harness.py's
repeats=N aggregation), reimplemented here rather than shared, since the
two domains' result shapes don't match cleanly (Speaking has no
criteria_scores dict, no single combined mistakes list per part - see that
module's own comment about not sharing this under time pressure).

Everything here is read-only observation:
- generate_scores()/generate_question_mistakes() are called exactly as any
  real caller would call them - same arguments, same return value used.
- Instrumentation captures a COPY of the raw GPT response (band-flags grid
  for scores; a copy of the raw candidate mistake list before this file's
  own validation/filtering runs) without altering what either function
  actually returns.
- No flag is read or changed. No prompt is edited. Nothing is fixed.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from openai import OpenAI as _RealOpenAI

import evaluators.speaking_audio as speaking_audio
from evaluators.speaking_audio import generate_scores, generate_question_mistakes


# ---------------------------------------------------------------------------
# Instrumentation - proxies the real OpenAI client used inline inside
# generate_scores() (it builds its own client per call, not through
# utils.gpt_client, so this can't reuse a simple function-wrap the way
# generate_question_mistakes' call_gpt() capture below does). Every method
# call is forwarded to the real client unchanged; only the response content
# is additionally copied into _captured_score_raw.
# ---------------------------------------------------------------------------
_captured_score_raw = []


class _InstrumentedCompletions:
    def __init__(self, real_completions):
        self._real = real_completions

    def create(self, *args, **kwargs):
        response = self._real.create(*args, **kwargs)
        try:
            content = response.choices[0].message.content
            _captured_score_raw.append(content)
        except Exception:
            pass
        return response


class _InstrumentedChat:
    def __init__(self, real_chat):
        self.completions = _InstrumentedCompletions(real_chat.completions)


class _InstrumentedOpenAI:
    def __init__(self, *args, **kwargs):
        self._real = _RealOpenAI(*args, **kwargs)
        self.chat = _InstrumentedChat(self._real.chat)


def install_score_instrumentation():
    original = speaking_audio.OpenAI
    speaking_audio.OpenAI = _InstrumentedOpenAI
    return original


def uninstall_score_instrumentation(original):
    speaking_audio.OpenAI = original


# generate_question_mistakes() goes through utils.gpt_client.call_gpt,
# imported by name into speaking_audio - same wrap-and-restore pattern as
# the Writing harness's _install_instrumentation().
_captured_mistakes_raw = []


def install_mistakes_instrumentation():
    original = speaking_audio.call_gpt

    def capturing(prompt, usage_log=None, temperature=0.0):
        result = original(prompt, usage_log=usage_log, temperature=temperature)
        if isinstance(result, dict):
            _captured_mistakes_raw.append(dict(result))
        return result

    speaking_audio.call_gpt = capturing
    return original


def uninstall_mistakes_instrumentation(original):
    speaking_audio.call_gpt = original


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _mistake_key(m: dict) -> str:
    return f"{m.get('type')}|{(m.get('original') or '').strip().lower()}"


def run_scores_n(part_number: int, question: str, transcript: str, repeats: int = 5) -> dict:
    """Runs generate_scores() `repeats` times on identical input. Returns
    band values/spread per criterion plus every raw band-flags grid
    captured for this batch (for checklist-flicker analysis)."""
    qas_clean = [{"question": question, "user_answer": transcript}]
    combined = f"Q: {question}\nA: {transcript}"

    _captured_score_raw.clear()
    original = install_score_instrumentation()
    runs = []
    try:
        for _ in range(repeats):
            runs.append(generate_scores(part_number, combined, qas_clean, acoustic_pronunciation=None, speech_timing=None, usage_log=[]))
    finally:
        uninstall_score_instrumentation(original)

    raw_grids = []
    for content in _captured_score_raw:
        try:
            raw_grids.append(json.loads(content.strip().replace("```json", "").replace("```", "").strip()))
        except Exception:
            raw_grids.append(None)

    result = {"runs": runs, "raw_grids": raw_grids}
    for key in ("fluency", "lexical", "grammar"):
        values = [r.get(key) for r in runs if r.get(key) is not None]
        result[key] = {"values": values, "min": min(values) if values else None, "max": max(values) if values else None}
    return result


def run_mistakes_n(question: str, transcript: str, repeats: int = 5) -> dict:
    """Runs generate_question_mistakes() `repeats` times on identical
    input. Returns per-run final (validated) mistake lists, per-run raw
    (pre-validation) mistake lists, and hit rates by content across runs."""
    _captured_mistakes_raw.clear()
    original = install_mistakes_instrumentation()
    runs = []
    try:
        for _ in range(repeats):
            runs.append(generate_question_mistakes(question, transcript, usage_log=[]))
    finally:
        uninstall_mistakes_instrumentation(original)

    raw_per_run = [call.get("mistakes", []) for call in _captured_mistakes_raw]

    key_hits = defaultdict(int)
    for r in runs:
        seen = {_mistake_key(m) for m in (r.get("mistakes") or [])}
        for key in seen:
            key_hits[key] += 1
    hit_rates = {key: count / repeats for key, count in key_hits.items()}

    return {"runs": runs, "raw_per_run": raw_per_run, "hit_rates": hit_rates}


def checklist_frequency_table(raw_grids: list) -> dict:
    """Same purpose as the Writing harness's instrumentation_report():
    for each (criterion, band) pair, how often was it marked false across
    the captured calls - i.e. which specific descriptor features are
    denied even on answers that plausibly deserve them."""
    band_keys = ["fluency_bands", "lexical_bands", "grammar_bands"]
    false_counts = defaultdict(int)
    total_counts = defaultdict(int)
    for grid in raw_grids:
        if not isinstance(grid, dict):
            continue
        for key in band_keys:
            sub = grid.get(key)
            if not isinstance(sub, dict):
                continue
            for band_str, value in sub.items():
                total_counts[(key, band_str)] += 1
                if value is not True:
                    false_counts[(key, band_str)] += 1
    return {"false_counts": dict(false_counts), "total_counts": dict(total_counts)}
