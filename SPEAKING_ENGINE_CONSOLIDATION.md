# Speaking engine consolidation - tracking notes

Living reference for the "two scoring engines, three live routes" finding.
Nothing here has been actioned - traffic data decides what happens next.
See `scripts/eval_log_report.py` for the traffic report and
`speaking_engine_comparison.json` for the raw divergence run.

## Current state (as of this note)

| Route | Engine | Status |
|---|---|---|
| `/speaking/audio/question-wise` | A (`speaking_audio.py::generate_scores`) | Actively maintained - all fixes this session went here |
| `/speaking/part/{part}/audio` | B (`speaking.py::evaluate_speaking_part`) | Logging wired, not touched |
| `/speaking/evaluate` (multipart) | B, same function | Logging wired, not touched |
| `/speaking/evaluate` (JSON) | B, via `evaluator.py::evaluate_attempt` → `evaluate_speaking` (loops `evaluate_speaking_part`) | Logging wired, not touched |

Flags, all default OFF, nothing shipped:
- `SPEAKING_MISTAKE_SEVERITY_SPLIT` (Engine A minor/significant split + repetition)
- `SPEAKING_VOICED_WPM` (Engine A WPM denominator fix)
- `SPEAKING_LEGACY_VOICED_WPM` (Engine B WPM denominator fix, prepared)
- `EVAL_LOG_ENABLED` (default ON - passive logging only, no scoring effect)

## Divergence, quantified (see `speaking_engine_comparison.json` for full raw output)

| Transcript | Engine A (F/L/G) | Engine B (F/L/G) | Gap |
|---|---|---|---|
| Moderate - Tokyo/desk job | 5.0/5.0/5.0 | 7.0/6.5/6.0 | +2.0/+1.5/+1.0 |
| Moderate - exercise habit | 6.0/6.0/6.0 | 4.7/5.5/5.2 | -1.3/-0.5/-0.8 |
| Moderate - healthy lifestyle | 6.0/6.0/6.0 | 5.2/5.2/5.2 | -0.8/-0.8/-0.8 |
| Weak (synthetic, authored - no real weak transcript exists in this repo) | 1.0/1.0/1.0 | 4.7/4.2/4.2 | +3.7/+3.2/+3.2 |

Direction is inconsistent across transcripts - not a fixed bias, can't be corrected with an offset.
The weak-transcript gap is the serious one: Engine B structurally cannot report a true beginner below ~4.0-4.7.

## The 4.0 floor - investigation

`evaluators/speaking.py:1627-1630`, `calibrate_score()`:
```python
def calibrate_score(val):
    return max(4.0, min(9.0, round(calibrate(val), 1)))
for key in ["fluency", "lexical", "grammar", "pronunciation"]:
    result[key] = calibrate_score(result.get(key, 0))
```
Under the header comment `FINAL SCORE CALIBRATION & PENALTY SAFEGUARD`, immediately followed by:
```python
# Ensure no criterion drops more than 1.0 below GPT baselines due to stacking penalties
if fluency_from_gpt is not None and result["fluency"] < fluency_from_gpt - 1.0:
    result["fluency"] = round(fluency_from_gpt - 1.0, 1)
```

**Git history**: present, byte-identical, in the earliest commit this repo has (`99272e2`,
"final clean upload", Apr 2026) - not present at all in the very first commit (`562a48f`).
No commit message or PR description explains the original rationale; `99272e2` is a
bulk initial import (adds `__pycache__` files, whole app scaffolding at once), so this
repo's git history doesn't reach back to whatever development process actually
introduced it.

**What the surrounding code tells us instead**: `evaluate_speaking_part()` has an
EARLIER, separate, more targeted safeguard for a genuine GPT failure - all four scores
returned as exactly 0 (`speaking.py:1148-1157`) - which substitutes conservative defaults
(5/5/5/6), not a floor. The 4.0 floor in `calibrate_score` is downstream of that and
unrelated to it. So: **not protecting against a null/crash** - that's already handled,
separately and more precisely, earlier in the function.

What it IS protecting against, by its own comment: many independent, uncoordinated
`-=` penalty rules scattered through this function (WPM <70 -1, WPM >200 -0.5, pauses >8
-1, no discourse markers -0.5, relevance <0.2 -0.2, consistency-check averaging, etc.)
stacking additively and crashing a score inappropriately far below GPT's own judgment.
That's a real, legitimate concern given how many uncoordinated penalties this file has.

But the file already has a MORE PRECISE tool for exactly that, sitting right next to the
floor: "no criterion drops more than 1.0 below GPT baseline" - which is RELATIVE to
GPT's own value, so it respects a genuinely low GPT judgment instead of overriding it.
The 4.0 floor is an ABSOLUTE bound on top of that, and fires regardless of whether any
penalty stacking happened at all - including when GPT's own raw judgment was already
low with zero Python-side penalties involved. That's the actual, additional effect
beyond the two legitimate protections already in the code: silently overriding a
genuine low-band GPT judgment. That's exactly the true-beginner harm found in the
divergence run.

**Conclusion**: removing the 4.0 floor does not reintroduce the null/crash case (separate
safeguard, untouched) and does not remove the stacking-penalty protection (separate
"1.0 below baseline" rule, untouched). Its only remaining effect is the one causing the
harm. Low-risk removal by this evidence - not zero-risk, since "why was it 4.0
specifically and not something else" was never documented anywhere findable.

**Residual floor elsewhere, smaller than first thought**: `compute_pause_score()`
(`speaking.py:689-706`) still clamps its own output to `[4.0, 8.0]` - but this only
feeds the audio-based FLUENCY fusion signal (blended with GPT's fluency via
`smooth_score()`, weighted ~20-50% by ASR confidence) when real audio is present, not a
hard floor on the final value. `compute_pronunciation_score()` (`speaking.py:745-799`)
has NO floor (clamps `[1.0, 9.0]`) - already fixed at some point, shared with Engine A.
In the text-only case (no `audio_metrics`, e.g. most `/speaking/evaluate` JSON traffic),
neither of these fires at all - `calibrate_score`'s floor is the only one in play, so
removing it alone is a complete fix for that path. For audio-bearing requests, removing
`calibrate_score`'s floor alone leaves a smaller, indirect upward pull on fluency
specifically (not lexical/grammar/pronunciation) via the pause-based fusion weight -
worth knowing, not necessarily worth fixing in the same pass.

## Decision, on record (pending the traffic numbers)

- **Zero traffic on Engine B routes** -> option (b), consolidate properly. No rush, no
  compat layer needed since there are no real callers to break.
- **Any traffic at all** -> (a) ships the same day (one line stops the beginner harm
  immediately), then (b) follows as the real fix. Not (c) - an error helps nobody when a
  one-line change fixes the actual harm.

Not yet triggered - waiting on `scripts/eval_log_report.py` numbers.

## If any live traffic reaches Engine B routes - three options, not chosen

### (a) Remove the 4.0 floor only
- **Change**: one line in `calibrate_score` (`evaluators/speaking.py:1628`).
- **Fixes**: the true-beginner harm specifically - GPT's own low judgment can surface.
- **Doesn't fix**: the rest of the divergence (different methodology entirely - holistic
  GPT guess + flat regex-triggered bonuses vs. Engine A's conjunctive checklist), the
  flat-prose mistakes shape, the WPM path differences, the ~26-vs-1-3 GPT call richness
  gap. Also the smaller residual fluency pull noted above (audio-bearing requests only).
- **Cost**: smallest of the three - a handful of unit tests, same day.

### (b) Route Engine B's endpoints to Engine A behind a shim
- **Change**: reshape both endpoints' inputs (multipart audio; `/speaking/evaluate`'s
  three JSON format variants) into what Engine A's scoring functions expect, call those
  instead of `evaluate_speaking_part`, reshape the richer response back out.
- **Fixes**: everything - one methodology, one accuracy bar, regardless of which URL is
  hit.
- **Cost**: real engineering, not a same-day patch. Engine A's per-part orchestration
  (vocabulary generation, band9 answers, whole-test systematic-error detection) assumes
  all 3 parts are known together - calling it for a single part (what
  `/speaking/part/{part}/audio` needs) means either a genuine refactor or accepting
  degraded/skipped cross-part features for that path. Response shape changes
  completely (flat mistakes dict -> rich array, different field names) - breaks any
  existing consumer of the current shape unless a compatibility translation layer is
  ALSO built. Needs real traffic/request-shape data first to build correctly rather than
  guess.
- **Breaks**: nothing immediately if built and validated properly first: the risk is
  building it wrong without knowing the real request/response shapes in use.

### (c) Return "unavailable, use the other endpoint" explicitly
- **Change**: a few lines per endpoint, early return before `evaluate_speaking_part` is
  ever called (e.g. HTTP 410 or a clear deprecation JSON body).
- **Fixes**: immediately and completely stops any further wrong-methodology bands from
  these routes.
- **Cost**: cheapest of the three to implement - literally same day.
- **Breaks**: any real client currently depending on these endpoints gets an error
  instead of a result, with no transition period unless one is deliberately built in.
  "A wrong band is worse than no band" is the tradeoff this makes explicit, per the
  instruction that named it.

No option chosen. Traffic data (item 1) decides which, if any, ships - and whether it's
urgent at all.

## Retirement list (report only, nothing removed)

- `evaluators/speaking.py::evaluate_speaking_part` / `evaluate_speaking` - Engine B's
  scoring logic. Candidate for full removal once traffic confirms zero live usage.
- `evaluator.py`'s `test_type == "speaking"` branch - thin wrapper calling
  `evaluate_speaking`, retires with it.
- `utils/audio_features.py` - retires with Engine B (its `compute_speech_rate_wpm` /
  `SPEAKING_LEGACY_VOICED_WPM` prep becomes moot if Engine B goes away entirely -
  removal costs nothing to reverse since it was never turned on).
- `evaluate_speaking_part`'s `time_seconds`-based internal WPM recomputation
  (`speaking.py:1204-1207`) - bypasses `audio_metrics` entirely, computed from a bare
  caller-supplied scalar with no waveform available to derive voiced time from. Correctly
  NOT given the same fix as the other two WPM paths (structurally can't be). **If Engine
  B is retired, this path retires with it and needs no fix at all** - noting it here
  specifically so it isn't mistaken for an outstanding gap in the WPM fix.
- `utils/wpm.py::calculate_speaking_wpm` - a third, independent WPM formula (same
  raw-duration bug), imported into `evaluator.py` but never actually called anywhere.
  Dead code, not a live risk. The fragmentation pattern (three independent WPM
  definitions for one concept) in miniature.
