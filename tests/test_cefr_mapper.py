import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.cefr_mapper import map_ielts_to_cefr


# ---------------------------------------------------------------------------
# map_ielts_to_cefr(): the previous version used invented intermediate
# labels ("High B1", "High B2") that don't match the official, widely-
# published IELTS-to-CEFR correlation table used by Cambridge/IELTS/
# British Council materials - Band 7 mapped to "High B2" (should be C1,
# squarely in the official table) and the mapping never returned "C2" at
# all (Band 8.5-9.0 is C2 officially, but everything >= 8.0 mapped to
# "C1"). Regression tests pin every official breakpoint.
# ---------------------------------------------------------------------------

def test_map_ielts_to_cefr_matches_official_correlation_table():
    cases = {
        9.0: "C2", 8.5: "C2",
        8.0: "C1", 7.5: "C1", 7.0: "C1",
        6.5: "B2", 6.0: "B2", 5.5: "B2",
        5.0: "B1", 4.5: "B1", 4.0: "B1",
        3.5: "A2", 3.0: "A2",
        2.5: "A1",
    }
    for band, expected in cases.items():
        assert map_ielts_to_cefr(band) == expected, f"band {band} expected {expected}"


def test_map_ielts_to_cefr_band_7_is_c1_not_high_b2():
    # The specific real case that surfaced this: a Task 1 result with
    # overall_band 7 came back with cefr_level "High B2" - not a real
    # CEFR level, and one full tier below where the official table puts
    # Band 7 (C1).
    assert map_ielts_to_cefr(7.0) == "C1"


def test_map_ielts_to_cefr_reaches_c2_for_top_bands():
    # The old mapping had no path to "C2" at all - everything >= 8.0
    # collapsed into "C1", even Band 9.
    assert map_ielts_to_cefr(8.5) == "C2"
    assert map_ielts_to_cefr(9.0) == "C2"


def test_map_ielts_to_cefr_falls_back_to_a1_on_invalid_input():
    assert map_ielts_to_cefr(None) == "A1"
    assert map_ielts_to_cefr("not a number") == "A1"
