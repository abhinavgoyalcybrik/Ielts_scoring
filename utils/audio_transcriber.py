import whisper

model = whisper.load_model("base")

def transcribe_audio(wav_path: str):
    result = model.transcribe(wav_path)
    return result["text"]
