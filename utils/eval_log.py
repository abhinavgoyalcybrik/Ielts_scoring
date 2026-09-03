"""Minimal, privacy-safe evaluation log.

Append-only JSONL, one record per evaluation, written off the request path
so it can never add latency or ever break a real response. Exists because
this backend has no other persistence anywhere (see the earlier no-
persistence finding) - every evaluation the product produces is otherwise
gone the moment the response is sent, which makes it impossible to
calibrate anything (repetition thresholds, WPM prompt language, whether
this month's model is better than last month's) against real usage, or to
audit a disputed score after the fact.

No names, emails, audio, or any other user identifier is ever written here
- not because this module scrubs them, but because there is nothing of the
kind available anywhere in this backend to begin with (no auth layer, no
user_id, audio is never retained past feature extraction). This module
only ever writes exactly what its caller explicitly builds into `record` -
callers are responsible for not putting anything identifying into it.
"""

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Both env-configurable per the request: EVAL_LOG_ENABLED so this can be
# switched off without a deploy (e.g. if disk usage or write volume ever
# becomes a concern), EVAL_LOG_PATH so where it writes isn't hardcoded.
# Default ON (not the usual default-OFF pattern used elsewhere in this
# codebase for behaviour-changing flags) - this is a passive, additive
# side-channel write that changes nothing about any response, and the
# entire point is to start collecting real data as soon as this ships.
EVAL_LOG_ENABLED = os.getenv("EVAL_LOG_ENABLED", "true").strip().lower() == "true"
EVAL_LOG_PATH = os.getenv("EVAL_LOG_PATH", "eval_logs")

# Single-worker executor for the sync (non-async) request-handler case -
# FastAPI runs a plain `def` endpoint in a worker thread with no running
# event loop, so asyncio.create_task isn't available there at all. The
# async endpoint path below uses real create_task, matching the request
# for that context; this is the correctly-different mechanism for the
# context where create_task cannot work.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval_log")
_write_lock = threading.Lock()


def _log_file_path() -> Path:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    directory = Path(EVAL_LOG_PATH)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"eval_log_{month}.jsonl"


def _write_record(record: dict) -> None:
    # Never let a logging failure be visible to anything - this is
    # diagnostic infrastructure, not part of the evaluation contract, and
    # must never be the reason a real request fails.
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except Exception:
        return
    try:
        path = _log_file_path()
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass


def log_evaluation(record: dict) -> None:
    """Fire-and-forget: queue `record` to be appended to the current
    month's JSONL log file without blocking the caller or the response
    already being returned. Safe to call from both an async endpoint
    (uses asyncio.create_task on the running loop, wrapping the actual
    disk write in asyncio.to_thread so the blocking I/O never runs on the
    event loop thread itself) and a sync `def` endpoint (FastAPI runs
    those in a worker thread with no running loop - create_task would
    raise RuntimeError there, so this falls back to a small dedicated
    background thread pool instead). A single lock inside _write_record
    serializes the actual file append regardless of which path queued it,
    so concurrent writers from both contexts can never interleave a line.
    Entirely a no-op if EVAL_LOG_ENABLED is false."""
    if not EVAL_LOG_ENABLED:
        return
    record = dict(record)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        loop.create_task(asyncio.to_thread(_write_record, record))
    else:
        _executor.submit(_write_record, record)
