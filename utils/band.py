import math

# =========================
# IELTS ACADEMIC READING
# =========================
# Kept as its OWN named constant, completely separate from
# GENERAL_TRAINING_READING_BAND_MAP below - band_from_correct() only ever
# reads from this map and never falls back to the General Training one.
ACADEMIC_READING_BAND_MAP = {
    40: 9.0, 39: 9.0,
    38: 8.5, 37: 8.5,
    36: 8.0, 35: 8.0,
    34: 7.5, 33: 7.5,
    32: 7.0, 31: 7.0, 30: 7.0,
    29: 6.5, 28: 6.5, 27: 6.5,
    26: 6.0, 25: 6.0, 24: 6.0, 23: 6.0,
    22: 5.5, 21: 5.5, 20: 5.5, 19: 5.5,
    18: 5.0, 17: 5.0, 16: 5.0, 15: 5.0,
    14: 4.5, 13: 4.5,
    12: 4.0, 11: 4.0, 10: 4.0,
    9: 3.5, 8: 3.5,
    7: 3.0, 6: 3.0,
    5: 2.5, 4: 2.5,
}
_ACADEMIC_READING_FLOOR_BAND = 2.0  # raw scores 0-3 (below the lowest listed band)


def band_from_correct(correct: int) -> float:
    """Academic Reading raw-score -> band, via ACADEMIC_READING_BAND_MAP
    ONLY. Deterministic dict lookup - never LLM-decided, never shares or
    falls back to the General Training map."""
    return ACADEMIC_READING_BAND_MAP.get(int(correct), _ACADEMIC_READING_FLOOR_BAND)


# =========================
# IELTS BAND SKILL LEVEL DESCRIPTORS (official qualitative descriptions)
# =========================
# These official descriptions are only defined for WHOLE bands (0-9) -
# there is no separate IELTS-published descriptor for a half band like
# 7.5. get_band_skill_description() floors to the whole-band tier below,
# since a candidate at 7.5 has not yet fully demonstrated the NEXT whole
# level's descriptor.
BAND_SKILL_DESCRIPTORS = {
    0: ("Did not attempt test", "No assessable information provided"),
    1: ("Non-user", "Essentially has no ability to use the language beyond possibly a few isolated words"),
    2: ("Intermittent user", "No real communication is possible except for the most basic information using isolated words or short formulae in familiar situations and to meet immediate needs. Has great difficulty understanding spoken and written English"),
    3: ("Extremely limited user", "Conveys and understands only general meaning in very familiar situations. Frequent breakdowns in communication occur"),
    4: ("Limited user", "Basic competence is limited to familiar situations. Has frequent problems with understanding and expression. Is not able to use complex language."),
    5: ("Modest user", "Has partial command of the language, coping with overall meaning in most situations, though is likely to make many mistakes. Should be able to handle basic communication in own field"),
    6: ("Competent user", "Has generally effective command of the language despite some inaccuracies, inappropriacies and misunderstandings. Can use and understand fairly complex language, particularly in familiar situations"),
    7: ("Good user", "Has operational command of the language, though with occasional inaccuracies, inappropriacies and misunderstandings in some situations. Generally handles complex language well and understands detailed reasoning"),
    8: ("Very good user", "Has fully operational command of the language with only occasional unsystematic inaccuracies and inappropriacies. Misunderstandings may occur in unfamiliar situations. Handles complex detailed argumentation well"),
    9: ("Expert user", "Has fully operational command of the language: appropriate, accurate and fluent with complete understanding"),
}


def get_band_skill_description(band_score) -> tuple[str, str]:
    """Deterministic band -> (skill_level, description) lookup - never
    LLM-decided. Floors a half-band score to its whole-band tier and
    clamps out-of-range/invalid input to the nearest valid band (0-9)
    rather than raising, so a malformed score can never break the
    response."""
    try:
        b = math.floor(float(band_score))
    except (TypeError, ValueError):
        b = 0
    b = max(0, min(9, b))
    return BAND_SKILL_DESCRIPTORS[b]


# =========================
# IELTS GENERAL TRAINING READING
# =========================
# Kept as its OWN named constant, completely separate from
# ACADEMIC_READING_BAND_MAP above - general_reading_band() only ever
# reads from this map and never falls back to the Academic one.
#
# Raw scores 9-40 come from the official General Training conversion
# table (corrected here - several top-end boundaries previously matched
# the Academic table's boundaries instead of General Training's own,
# e.g. raw score 39 used to be grouped with 40 at Band 9.0, but General
# Training splits it out to 8.5; raw score 33 used to give 7.0, but
# General Training gives 6.5). Raw scores below 9 are NOT specified in
# the supplied reference and are preserved UNCHANGED from this project's
# existing mapping (6-8 -> 2.5, 0-5 -> 2.0), per explicit instruction not
# to invent or assume values for that range.
GENERAL_TRAINING_READING_BAND_MAP = {
    40: 9.0,
    39: 8.5,
    38: 8.0, 37: 8.0,
    36: 7.5,
    35: 7.0, 34: 7.0,
    33: 6.5, 32: 6.5,
    31: 6.0, 30: 6.0,
    29: 5.5, 28: 5.5, 27: 5.5,
    26: 5.0, 25: 5.0, 24: 5.0, 23: 5.0,
    22: 4.5, 21: 4.5, 20: 4.5, 19: 4.5,
    18: 4.0, 17: 4.0, 16: 4.0, 15: 4.0,
    14: 3.5, 13: 3.5, 12: 3.5,
    11: 3.0, 10: 3.0, 9: 3.0,
    # Preserved unchanged (not covered by the supplied reference table):
    8: 2.5, 7: 2.5, 6: 2.5,
}
_GENERAL_TRAINING_READING_FLOOR_BAND = 2.0  # raw scores 0-5, preserved from the existing mapping


def general_reading_band(correct: int) -> float:
    """General Training Reading raw-score -> band, via
    GENERAL_TRAINING_READING_BAND_MAP ONLY. Deterministic dict lookup -
    never LLM-decided, never shares or falls back to the Academic map."""
    return GENERAL_TRAINING_READING_BAND_MAP.get(int(correct), _GENERAL_TRAINING_READING_FLOOR_BAND)


# =========================
# IELTS ACADEMIC LISTENING
# =========================
# Kept as its OWN named constant, completely separate from
# GENERAL_TRAINING_LISTENING_BAND_MAP below - the current raw-score ->
# band values happen to be identical between Academic and General
# Training Listening, but the two are stored as two independent literal
# dicts (never one aliasing/copying the other) so each can be edited or
# diverge in the future without any risk of accidentally affecting the
# other. academic_listening_band() only ever reads from this map.
#
# Raw scores below 11 are not covered by the supplied reference table
# (which only goes down to 11-12 -> 4.0) and are preserved unchanged
# from this project's existing Listening mapping.
ACADEMIC_LISTENING_BAND_MAP = {
    40: 9.0, 39: 9.0,
    38: 8.5, 37: 8.5,
    36: 8.0, 35: 8.0,
    34: 7.5, 33: 7.5, 32: 7.5,
    31: 7.0, 30: 7.0,
    29: 6.5, 28: 6.5, 27: 6.5, 26: 6.5,
    25: 6.0, 24: 6.0, 23: 6.0,
    22: 5.5, 21: 5.5, 20: 5.5, 19: 5.5, 18: 5.5,
    17: 5.0, 16: 5.0,
    15: 4.5, 14: 4.5, 13: 4.5,
    12: 4.0, 11: 4.0,
    # Preserved unchanged (not covered by the supplied reference table):
    10: 4.0, 9: 3.5, 8: 3.5, 7: 3.0, 6: 3.0, 5: 2.5, 4: 2.5,
}
_ACADEMIC_LISTENING_FLOOR_BAND = 2.0  # raw scores 0-3, preserved from the existing mapping


def academic_listening_band(correct: int) -> float:
    """Academic Listening raw-score -> band, via
    ACADEMIC_LISTENING_BAND_MAP ONLY. Deterministic dict lookup - never
    LLM-decided, never shares or falls back to the General Training map."""
    return ACADEMIC_LISTENING_BAND_MAP.get(int(correct), _ACADEMIC_LISTENING_FLOOR_BAND)


# =========================
# IELTS GENERAL TRAINING LISTENING
# =========================
# A separate literal dict from ACADEMIC_LISTENING_BAND_MAP above (not a
# reference/copy of it) - see the note above for why, even though the
# values currently match.
GENERAL_TRAINING_LISTENING_BAND_MAP = {
    40: 9.0, 39: 9.0,
    38: 8.5, 37: 8.5,
    36: 8.0, 35: 8.0,
    34: 7.5, 33: 7.5, 32: 7.5,
    31: 7.0, 30: 7.0,
    29: 6.5, 28: 6.5, 27: 6.5, 26: 6.5,
    25: 6.0, 24: 6.0, 23: 6.0,
    22: 5.5, 21: 5.5, 20: 5.5, 19: 5.5, 18: 5.5,
    17: 5.0, 16: 5.0,
    15: 4.5, 14: 4.5, 13: 4.5,
    12: 4.0, 11: 4.0,
    # Preserved unchanged (not covered by the supplied reference table):
    10: 4.0, 9: 3.5, 8: 3.5, 7: 3.0, 6: 3.0, 5: 2.5, 4: 2.5,
}
_GENERAL_TRAINING_LISTENING_FLOOR_BAND = 2.0  # raw scores 0-3, preserved from the existing mapping


def general_training_listening_band(correct: int) -> float:
    """General Training Listening raw-score -> band, via
    GENERAL_TRAINING_LISTENING_BAND_MAP ONLY. Deterministic dict lookup -
    never LLM-decided, never shares or falls back to the Academic map."""
    return GENERAL_TRAINING_LISTENING_BAND_MAP.get(int(correct), _GENERAL_TRAINING_LISTENING_FLOOR_BAND)


# =========================
# COMMON (WRITING / SPEAKING)
# =========================
def round_band(band: float) -> float:
    """Official IELTS band rounding: an average ending in .25 rounds UP to
    the next half band, and .75 rounds UP to the next whole band - i.e.
    round-half-up on the doubled value. Python's built-in round() uses
    round-half-to-even, which silently rounds .25/.75 averages DOWN roughly
    half the time (e.g. round(6.25*2)/2 = 6.0, not the correct 6.5)."""
    return math.floor(band * 2 + 0.5) / 2