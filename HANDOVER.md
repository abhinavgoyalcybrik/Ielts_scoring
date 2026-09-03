# Handover — Writing/Speaking scoring work (2026-09-03)

This repo was 1 unpushed commit + 78 uncommitted changes behind what was actually
tested. Everything below is now committed and pushed to `origin/main`. Read this
before deploying.

## Config flags

| Flag | Default | What it does |
|---|---|---|
| `WRITING_MODEL_OVERRIDE` | **`gpt-4o`** (ships ON) | Writing's scoring+detection model. Was `gpt-4.1-mini`; flipped after n=5 testing showed detection reliability going from 1/15→5/5 and 0/15→5/5 on two real error cases. Adds ~**$0.056/submission** (~5.6¢) vs. the old model — see `scripts/eval_log_report.py` for real per-submission cost once traffic exists. |
| `WRITING_INDEPENDENT_MODEL_ANSWER` | OFF | v2 refine pipeline — builds the "model answer" independently instead of rewriting a possibly-weak submission. Safe to leave off. |
| `SPEAKING_MISTAKE_SEVERITY_SPLIT` | OFF | Splits Speaking mistakes into significant/minor + adds deterministic word-repetition detection. Repetition thresholds are provisional (not calibrated on real data). Safe to leave off. |
| `SPEAKING_VOICED_WPM` | OFF | Switches Speaking's "active" WPM figure from raw-duration to voiced-duration basis. Safe to leave off. |
| `EVAL_LOG_ENABLED` | ON (infra, not a scoring behavior change) | Fire-and-forget append-only JSONL logging of every evaluation to `eval_logs/`. Cannot block or alter a response — write happens off the request path. |
| `EVAL_LOG_PATH` | `eval_logs` | Where those files go. **Must be a writable directory on the server** — this is the only way to get real usage/traffic data; see next section. |

Everything except `WRITING_MODEL_OVERRIDE` is off by default — no behavior changes
unless someone deliberately flips a flag.

## `eval_logs/` must be writable on the server

Nothing reads production traffic today. `scripts/eval_log_report.py` answers cost,
reliability, and (for Speaking) which endpoint real candidates use — but only once
`eval_logs/*.jsonl` files exist. If the deploy target doesn't have write access to
`EVAL_LOG_PATH`, logging fails silently (fire-and-forget, by design) and this stays
unanswerable. Confirm write access, let it run a few days, then:

```
python scripts/eval_log_report.py --path eval_logs --days 7
```

## Update: Speaking is down to one engine

The "two live Speaking scoring engines" issue below is resolved, not open. Confirmed
via the frontend that only `/speaking/audio/question-wise` had a consumer — the other
two routes (`/speaking/part/{part}/audio`, `/speaking/evaluate`) are gone, along with
the legacy engine behind them (`evaluators/speaking.py` and everything that only
existed to serve it — full list in `SPEAKING_ENGINE_CONSOLIDATION.md`'s "Resolution"
section). That legacy engine had a hard 4.0 floor on every criterion (a true beginner
scored 4.2–4.7 instead of Band 1) and its own separate WPM bug — both gone with it.
`SPEAKING_LEGACY_VOICED_WPM` is removed from the flag table above for the same reason.

One Speaking endpoint remains: `/speaking/audio/question-wise`, unchanged and verified
byte-identical throughout this cleanup. `evaluators/speaking_final.py` (an already-
unreachable, never-registered route) was deliberately left alone rather than also
removed — noted as a known orphan for a future cleanup, not a risk.

## Also flagged, not fixed (Writing Task 1 prompts)

Three pieces of guidance from the old single-file `writing_task1_prompt.txt` (now
split into `writing_task1_common.txt` + academic/general criteria files) don't
appear verbatim in any replacement file: a Band-5 floor rule, short-answer scoring
guidance, and mistake-volume guidance. Flagged for a separate decision, not silently
dropped or silently restored.

## Update: whole-essay mistake guard (Item 7)

`evaluators/writing.py`'s `_mistake_spans_whole_essay()` drops any mistake whose
`"original"` span covers 70%+ of the essay's word count - unconditional, no new flag,
same pattern as the other five deterministic mistake filters (verbatim check,
correction-relatedness, severity normalization/escalation, dedupe) already shipping
unconditionally. Catches a confirmed bug: a mistake object that's really essay-level
feedback (in every observed case, "needs paragraphing") rather than a per-span error.
Calibrated against 206 real mistakes from every saved run this session produced;
70% sits in the gap between the largest confirmed-genuine mistake (69%, on a short
essay) and the smallest confirmed whole-essay case (73%). In every case this guard
drops, the same guidance was already present independently in the `improvement`
field - nothing is lost, it's just no longer double-counted as a per-span error.
