def map_ielts_to_cefr(band) -> str:
    """
    CEFR mapping based on the official, widely-published IELTS-to-CEFR
    "common scale" correlation used by Cambridge/IELTS/British Council
    materials:
    - Band 8.5-9.0 -> C2
    - Band 7.0-8.0 -> C1
    - Band 5.5-6.5 -> B2
    - Band 4.0-5.0 -> B1
    - Band 3.0-3.5 -> A2
    - Band < 3.0   -> A1

    The previous version used its own invented intermediate labels
    ("High B1", "High B2") and mapped Band 7-<8 to "High B2" and only
    Band >= 8 to "C1" - neither matches the official correlation, where
    Band 7 is squarely C1 and Band 8.5+ is C2 (a level this mapping
    previously never returned at all).

    Returns one of: A1, A2, B1, B2, C1, C2
    """
    try:
        b = float(band)
    except Exception:
        # If input isn't a number, fall back conservatively to the floor.
        return "A1"

    if b >= 8.5:
        return "C2"
    if b >= 7.0:
        return "C1"
    if b >= 5.5:
        return "B2"
    if b >= 4.0:
        return "B1"
    if b >= 3.0:
        return "A2"
    return "A1"
