import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluators.api.writing as api_writing


# ---------------------------------------------------------------------------
# The /writing/evaluate endpoint's overall_writing_band. Two real bugs
# found from the same live test case (Task 1 = 7, Task 2 = 5.5):
# 1. It used a straight 1:1 average of the two task bands, when official
#    IELTS convention weights Task 2 at roughly twice Task 1 (Task 2 =
#    2/3, Task 1 = 1/3).
# 2. It rounded with Python's round(), which is round-half-to-even and
#    silently rounds a .25/.75 average DOWN about half the time, instead
#    of round_band() (round-half-up), the same rounding bug already fixed
#    elsewhere in this codebase.
# The two bugs happened to cancel out for that exact test case's numbers
# (6.0 either way) - these tests use numbers where they do NOT cancel out,
# so a regression back to either bug would be caught.
# ---------------------------------------------------------------------------

class _FakeTask:
    def __init__(self, question="q", answer="a", image_url=None):
        self.question = question
        self.answer = answer
        self.image_url = image_url


class _FakeRequest:
    def __init__(self, task_1=None, task_2=None):
        self.task_1 = task_1
        self.task_2 = task_2 or _FakeTask()


def _install_fake_evaluate_writing(monkeypatch, task1_band=None, task2_band=None):
    def fake_evaluate_writing(data):
        if data["metadata"]["task_type"] == "task_1":
            return {"overall_band": task1_band, "mistakes": []}
        return {"overall_band": task2_band, "mistakes": []}
    monkeypatch.setattr(api_writing, "evaluate_writing", fake_evaluate_writing)


def test_overall_writing_band_uses_official_1_to_2_weighting(monkeypatch):
    # Equal-average would give 7.0; correct 1:2 weighting gives 6.5.
    _install_fake_evaluate_writing(monkeypatch, task1_band=9, task2_band=5)

    result = api_writing.evaluate(_FakeRequest(task_1=_FakeTask()))

    assert result["overall_writing_band"] == 6.5


def test_overall_writing_band_rounds_half_up_not_half_to_even(monkeypatch):
    # Isolates the rounding fix specifically (task-2-only, no weighting
    # involved): round(6.25*2)/2 (the old buggy code) gives 6.0, since
    # Python's round() rounds 12.5 to the nearest EVEN integer (12).
    # round_band(6.25) (correct, round-half-up) gives 6.5.
    _install_fake_evaluate_writing(monkeypatch, task2_band=6.25)

    result = api_writing.evaluate(_FakeRequest(task_1=None))

    # round(6.25*2)/2 (the old buggy code) = round(12.5)/2 = 12/2 = 6.0
    # round_band(6.25) (correct) = 6.5
    assert result["overall_writing_band"] == 6.5


def test_overall_writing_band_falls_back_to_task2_when_task1_missing(monkeypatch):
    _install_fake_evaluate_writing(monkeypatch, task2_band=6.5)

    result = api_writing.evaluate(_FakeRequest(task_1=None))

    assert result["overall_writing_band"] == 6.5


def test_overall_writing_band_matches_real_observed_case(monkeypatch):
    # The exact live test case that surfaced both bugs - confirms the fix
    # produces the mathematically correct combined result.
    _install_fake_evaluate_writing(monkeypatch, task1_band=7, task2_band=5.5)

    result = api_writing.evaluate(_FakeRequest(task_1=_FakeTask()))

    assert result["overall_writing_band"] == 6.0
