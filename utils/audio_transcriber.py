import os
import tempfile
import whisper

model = whisper.load_model("base")


def _coerce_audio_path(source):
    if isinstance(source, str):
        return source, None

    data = None
    filename = None

    if hasattr(source, "filename"):
        filename = source.filename
    if hasattr(source, "file"):
        data = source.file.read()
        try:
            source.file.seek(0)
        except Exception:
            pass
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)

    if data is None:
        raise TypeError("Unsupported audio source type")

    suffix = os.path.splitext(filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name, tmp.name


def transcribe_audio(source):
    path, cleanup_path = _coerce_audio_path(source)
    try:
        result = model.transcribe(path)
        return result["text"]
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except Exception:
                pass
