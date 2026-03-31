# =========================
# IELTS ACADEMIC READING
# =========================
def band_from_correct(correct: int) -> float:
    if 39 <= correct <= 40:
        return 9.0
    elif 37 <= correct <= 38:
        return 8.5
    elif 35 <= correct <= 36:
        return 8.0
    elif 33 <= correct <= 34:
        return 7.5
    elif 30 <= correct <= 32:
        return 7.0
    elif 27 <= correct <= 29:
        return 6.5
    elif 23 <= correct <= 26:
        return 6.0
    elif 19 <= correct <= 22:
        return 5.5
    elif 15 <= correct <= 18:
        return 5.0
    elif 13 <= correct <= 14:
        return 4.5
    elif 10 <= correct <= 12:
        return 4.0
    elif 8 <= correct <= 9:
        return 3.5
    elif 6 <= correct <= 7:
        return 3.0
    elif 4 <= correct <= 5:
        return 2.5
    else:
        return 2.0

# =========================
# IELTS LISTENING
# =========================
def listening_band_from_correct(correct: int) -> float:
    if 39 <= correct <= 40:
        return 9.0
    elif 37 <= correct <= 38:
        return 8.5
    elif 35 <= correct <= 36:
        return 8.0
    elif 32 <= correct <= 34:
        return 7.5
    elif 30 <= correct <= 31:
        return 7.0
    elif 26 <= correct <= 29:
        return 6.5
    elif 23 <= correct <= 25:
        return 6.0
    elif 18 <= correct <= 22:
        return 5.5
    elif 16 <= correct <= 17:
        return 5.0
    elif 13 <= correct <= 15:
        return 4.5
    elif 10 <= correct <= 12:
        return 4.0
    elif 8 <= correct <= 9:
        return 3.5
    elif 6 <= correct <= 7:
        return 3.0
    elif 4 <= correct <= 5:
        return 2.5
    elif 1 <= correct <= 3:
        return 2.0
    else:  # 0 correct
        return 2.0


# =========================
# COMMON (WRITING / SPEAKING)
# =========================
def round_band(band: float) -> float:
    """
    Rounds band score to nearest 0.5 (IELTS standard)
    """
    return round(band * 2) / 2