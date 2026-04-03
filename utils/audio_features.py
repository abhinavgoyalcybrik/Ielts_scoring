import librosa

def extract_audio_features(wav_path: str):
    y, sr = librosa.load(wav_path, sr=16000)

    duration = librosa.get_duration(y=y, sr=sr)
    intervals = librosa.effects.split(y, top_db=25)
    pause_count = max(0, len(intervals) - 1)

    return {
        "duration_sec": round(duration, 2),
        "pause_count": pause_count
    }
