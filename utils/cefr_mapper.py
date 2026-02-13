def map_ielts_to_cefr(band) -> str:
    """
    Strict CEFR mapping based on the provided band (no score changes).

    Mapping rules (explicit):
    - Band 4–4.5 -> A2
    - Band 5 -> B1
    - Band 5.5 -> High B1
    - Band 6–6.5 -> B2
    - Band 7–<8 -> High B2
    - Band >= 8 -> C1

    Returns one of: A2, B1, High B1, B2, High B2, C1
    """
    try:
        b = float(band)
    except Exception:
        # If input isn't a number, fall back conservatively
        return "A2"

    # Normalize to nearest 0.5 (bands are reported in 0.5 steps elsewhere)
    # but respect exact values provided
    if 4.0 <= b <= 4.5:
        return "A2"
    if abs(b - 5.0) < 1e-9:
        return "B1"
    if abs(b - 5.5) < 1e-9:
        return "High B1"
    if 6.0 <= b <= 6.5:
        return "B2"
    if 7.0 <= b < 8.0:
        return "High B2"
    if b >= 8.0:
        return "C1"

    # Conservative defaults for other bands
    if b >= 5.0:
        return "B1"
    return "A2"
