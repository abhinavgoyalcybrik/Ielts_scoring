def calculate_writing_wpm(word_count: int, time_minutes: float) -> float:
    if not time_minutes or time_minutes <= 0:
        return 0.0
    return round(word_count / time_minutes, 2)


def calculate_speaking_wpm(word_count: int, audio_seconds: float) -> float:
    if not audio_seconds or audio_seconds <= 0:
        return 0.0
    return round((word_count / audio_seconds) * 60, 2)
