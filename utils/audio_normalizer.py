import subprocess
import tempfile
from pathlib import Path

def normalize_to_wav(upload_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as src:
        src.write(upload_file.file.read())
        src_path = src.name

    dst_path = Path(src_path).with_suffix(".wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", src_path,
        "-ar", "16000",
        "-ac", "1",
        str(dst_path)
    ], check=True)

    return str(dst_path)
