import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from utils import eval_log


@pytest.fixture(autouse=True)
def _disable_eval_log_by_default(monkeypatch):
    """utils/eval_log.py defaults to ON in real usage (see its own module
    docstring - the whole point is collecting real data as soon as this
    ships), and several existing tests call the actual API router
    functions (not just the evaluator functions directly), which triggers
    a real log_evaluation() call. Without this, running the test suite
    silently writes real files into the repo's eval_logs/ directory on
    every run - test pollution, not anything to do with what's being
    tested. Off by default for every test; a test that specifically wants
    to exercise logging should monkeypatch EVAL_LOG_ENABLED back to True
    itself (see tests/test_eval_log.py)."""
    monkeypatch.setattr(eval_log, "EVAL_LOG_ENABLED", False)
