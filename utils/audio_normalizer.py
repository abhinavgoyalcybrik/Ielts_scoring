import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _add_to_process_path(directory: str) -> None:
    """Make `directory` visible to this process's PATH so any subprocess call
    that shells out to bare "ffmpeg" (e.g. inside the whisper library) can
    find it too, not just our own explicit ffmpeg_path calls."""
    current = os.environ.get("PATH", "")
    if directory not in current.split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + current


def _find_ffmpeg() -> str | None:
    for exe in ("ffmpeg", "ffmpeg.exe"):
        path = shutil.which(exe)
        if path:
            return path

    # Fall back to well-known install locations. This matters on Windows,
    # where a freshly winget-installed tool isn't visible to processes
    # (e.g. an already-running IDE/server) that started before the PATH
    # update, even after the user "restarts" a terminal tab.
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidate_dirs = [
        Path(local_app_data) / "Microsoft" / "WinGet" / "Links",
        Path("C:/ffmpeg/bin"),
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "ffmpeg" / "bin",
    ]
    winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    if winget_packages.is_dir():
        candidate_dirs.extend(winget_packages.glob("Gyan.FFmpeg_*/ffmpeg-*/bin"))

    for directory in candidate_dirs:
        candidate = directory / "ffmpeg.exe"
        if candidate.is_file():
            _add_to_process_path(str(directory))
            return str(candidate)

    return None


def normalize_to_wav(upload_file):
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required for audio conversion but was not found. "
            "Install ffmpeg and ensure it is available on your PATH."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as src:
        audio_data = upload_file.file.read()
        if not audio_data:
            raise RuntimeError("No audio data available for conversion.")
        src.write(audio_data)
        src_path = src.name

    dst_path = Path(src_path).with_suffix(".wav")

    try:
        subprocess.run(
            [
                ffmpeg_path, "-y",
                "-i", src_path,
                "-ar", "16000",
                "-ac", "1",
                str(dst_path)
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Audio conversion failed: {exc.stderr.strip() or exc.stdout.strip()}"
        ) from exc

    return str(dst_path)
