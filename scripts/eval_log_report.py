"""Report request counts per evaluator/route from the append-only eval log
(utils/eval_log.py). This is the tool that answers "which speaking endpoint
does real traffic actually use" - run it after a few days of live logging.
Also reports Writing's real per-evaluation cost, ai_evaluation_failed rate,
and latency once real gpt-4o traffic exists (see the model-flip decision) -
so those numbers don't require a bespoke script when the time comes.

Usage:
    python scripts/eval_log_report.py [--path eval_logs] [--days N]
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Official OpenAI pricing, USD per 1M tokens - source:
# https://developers.openai.com/api/docs/pricing (checked 2026-09-02).
# Update here if pricing changes; nothing else in this script hardcodes it.
_PRICING_PER_MILLION = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}


def _cost_for_record(record: dict) -> float | None:
    usage = (record.get("response") or {}).get("usage") or {}
    model = record.get("model")
    pricing = _PRICING_PER_MILLION.get(model)
    if not pricing or not usage:
        return None
    return (
        usage.get("input_tokens", 0) * pricing["input"]
        + usage.get("output_tokens", 0) * pricing["output"]
    ) / 1_000_000


def iter_records(log_dir: Path):
    for path in sorted(log_dir.glob("eval_log_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="eval_logs", help="Directory containing eval_log_*.jsonl files")
    parser.add_argument("--days", type=int, default=None, help="Only count records from the last N days")
    args = parser.parse_args()

    log_dir = Path(args.path)
    if not log_dir.exists():
        print(f"No log directory at {log_dir} - nothing has been logged yet.")
        return

    cutoff = None
    if args.days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    by_evaluator = Counter()
    total = 0
    for record in iter_records(log_dir):
        if cutoff is not None:
            ts = record.get("timestamp")
            try:
                record_time = datetime.fromisoformat(ts)
            except (TypeError, ValueError):
                record_time = None
            if record_time is not None and record_time < cutoff:
                continue
        by_evaluator[record.get("evaluator", "unknown")] += 1
        total += 1

    if total == 0:
        print("No records found in range.")
        return

    print(f"Total evaluations logged: {total}\n")
    print(f"{'evaluator':40s} {'count':>8s} {'% of total':>12s}")
    for evaluator, count in by_evaluator.most_common():
        pct = 100 * count / total
        print(f"{evaluator:40s} {count:8d} {pct:11.1f}%")

    print("\nSpeaking-specific breakdown (the three live routes):")
    speaking_keys = {
        "speaking": "/speaking/audio/question-wise (Engine A, current)",
        "speaking_legacy_part_audio": "/speaking/part/{part}/audio (Engine B)",
        "speaking_legacy_evaluate_multipart": "/speaking/evaluate, multipart audio (Engine B)",
        "speaking_legacy_evaluate_json": "/speaking/evaluate, JSON transcript (Engine B)",
    }
    for key, description in speaking_keys.items():
        print(f"  {description}: {by_evaluator.get(key, 0)}")

    # Writing cost/reliability/latency, per model actually used - the
    # numbers the gpt-4o flip decision needs watched. Re-derives records
    # (not reusing the loop above) so --days filtering still applies.
    writing_records = [
        r for r in iter_records(log_dir)
        if r.get("evaluator") == "writing"
        and (cutoff is None or _record_time_ok(r, cutoff))
    ]
    if writing_records:
        print("\nWriting - cost / reliability / latency (per model in use):")
        by_model = defaultdict(list)
        for r in writing_records:
            by_model[r.get("model", "unknown")].append(r)
        for model, records in by_model.items():
            costs = [c for c in (_cost_for_record(r) for r in records) if c is not None]
            fails = sum(1 for r in records if (r.get("response") or {}).get("ai_evaluation_failed") is True)
            refusals = sum(1 for r in records if (r.get("response") or {}).get("ai_evaluation_failed_reason") == "refusal")
            other_fails = fails - refusals
            latencies = [r["latency_seconds"] for r in records if isinstance(r.get("latency_seconds"), (int, float))]
            print(f"  {model}: {len(records)} evaluations")
            if costs:
                print(f"    avg cost/evaluation: ${sum(costs)/len(costs):.4f}  (estimate quoted: $0.0279/task)")
            else:
                print(f"    avg cost/evaluation: no usage data on these records (pre-usage-tracking, or unrecognized model in pricing table)")
            print(f"    ai_evaluation_failed rate: {fails}/{len(records)} ({100*fails/len(records):.1f}%)")
            # Refusal is its own line, not folded into the rate above - a
            # candidate's submission failing because the model outright
            # declined (no discernible trigger, observed non-reproducible
            # on retry) is a materially different production risk than a
            # transient parse/network error, even though both currently
            # get the same safe neutral fallback.
            print(f"      of which refusal-shaped: {refusals}/{fails or 1} ({100*refusals/fails:.1f}% of failures)" if fails else "      of which refusal-shaped: 0 (no failures)")
            if fails:
                print(f"      of which other (parse/network/etc): {other_fails}/{fails}")
            if latencies:
                print(f"    latency: min {min(latencies):.2f}s / avg {sum(latencies)/len(latencies):.2f}s / max {max(latencies):.2f}s")
            else:
                print(f"    latency: no latency data on these records")


def _record_time_ok(record: dict, cutoff: datetime) -> bool:
    ts = record.get("timestamp")
    try:
        record_time = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return True
    return record_time >= cutoff


if __name__ == "__main__":
    main()
