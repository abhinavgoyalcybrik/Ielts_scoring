def calculate_writing_wpm(word_count: int, time_minutes: float) -> float:
    if not time_minutes or time_minutes <= 0:
        return 0.0
    return round(word_count / time_minutes, 2)
