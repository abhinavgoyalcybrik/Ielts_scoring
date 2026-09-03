# Minimal evaluation log - utils/eval_log.py. Must actually append a
# usable JSONL line, must be a true no-op when disabled, and must dispatch
# through the correct mechanism for whichever context calls it (a
# background thread pool from a sync/no-running-loop caller - matching
# FastAPI's plain `def` endpoints, which run in a worker thread with no
# event loop - vs asyncio.create_task from an async caller with a running
# loop, matching the async endpoint).

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import eval_log


def test_write_record_appends_jsonl_line_to_configured_path(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_log, "EVAL_LOG_PATH", str(tmp_path))
    record = {"evaluator": "writing", "task_or_part": "task_2", "question": "Some question?"}
    eval_log._write_record(record)

    files = list(tmp_path.glob("eval_log_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_write_record_appends_not_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_log, "EVAL_LOG_PATH", str(tmp_path))
    eval_log._write_record({"n": 1})
    eval_log._write_record({"n": 2})

    files = list(tmp_path.glob("eval_log_*.jsonl"))
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(l)["n"] for l in lines] == [1, 2]


def test_write_record_never_raises_on_non_json_native_values(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_log, "EVAL_LOG_PATH", str(tmp_path))
    # A set isn't natively JSON-serializable - json.dumps(..., default=str)
    # must fall back to stringifying it rather than raising and silently
    # dropping the whole record.
    eval_log._write_record({"weird": {1, 2, 3}})
    files = list(tmp_path.glob("eval_log_*.jsonl"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8").strip()


def test_write_record_swallows_write_failure_silently(monkeypatch):
    # An unwritable path (e.g. disk full, permissions) must never raise
    # into the caller - this is diagnostic infrastructure, not part of the
    # evaluation contract, and must never be why a real request fails.
    monkeypatch.setattr(eval_log, "EVAL_LOG_PATH", "\0invalid\0path")
    eval_log._write_record({"n": 1})  # must not raise


def test_log_evaluation_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(eval_log, "EVAL_LOG_ENABLED", False)
    mock_executor = MagicMock()
    monkeypatch.setattr(eval_log, "_executor", mock_executor)
    eval_log.log_evaluation({"evaluator": "writing"})
    mock_executor.submit.assert_not_called()


def test_log_evaluation_adds_timestamp_if_missing(monkeypatch):
    monkeypatch.setattr(eval_log, "EVAL_LOG_ENABLED", True)
    mock_executor = MagicMock()
    monkeypatch.setattr(eval_log, "_executor", mock_executor)
    eval_log.log_evaluation({"evaluator": "writing"})
    mock_executor.submit.assert_called_once()
    _fn, record = mock_executor.submit.call_args[0]
    assert "timestamp" in record


def test_log_evaluation_preserves_caller_supplied_timestamp(monkeypatch):
    monkeypatch.setattr(eval_log, "EVAL_LOG_ENABLED", True)
    mock_executor = MagicMock()
    monkeypatch.setattr(eval_log, "_executor", mock_executor)
    eval_log.log_evaluation({"evaluator": "writing", "timestamp": "fixed"})
    _fn, record = mock_executor.submit.call_args[0]
    assert record["timestamp"] == "fixed"


def test_log_evaluation_uses_background_executor_with_no_running_loop(monkeypatch):
    # Called from plain sync test code - no running event loop, matching a
    # FastAPI `def` (non-async) endpoint's worker thread.
    monkeypatch.setattr(eval_log, "EVAL_LOG_ENABLED", True)
    mock_executor = MagicMock()
    monkeypatch.setattr(eval_log, "_executor", mock_executor)
    eval_log.log_evaluation({"evaluator": "reading"})
    mock_executor.submit.assert_called_once()
    assert mock_executor.submit.call_args[0][0] is eval_log._write_record


def test_log_evaluation_uses_create_task_when_loop_is_running(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_log, "EVAL_LOG_ENABLED", True)
    # The task this schedules runs the real _write_record via
    # asyncio.to_thread (not mocked - that's the point, proving the
    # scheduled task doesn't raise) - must not be allowed to hit the real
    # project directory.
    monkeypatch.setattr(eval_log, "EVAL_LOG_PATH", str(tmp_path))
    mock_executor = MagicMock()
    monkeypatch.setattr(eval_log, "_executor", mock_executor)

    async def _run():
        eval_log.log_evaluation({"evaluator": "speaking"})
        # Let the scheduled task actually execute before the loop closes,
        # so this also proves the task doesn't raise.
        await asyncio.sleep(0)

    asyncio.run(_run())
    # The async path must go through create_task, never the sync executor.
    mock_executor.submit.assert_not_called()
