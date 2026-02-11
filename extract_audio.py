# utils/extract_audio.py
# Robust WAV extraction: prefer ffmpeg, fallback to librosa.load (kaiser_fast), then soundfile fallback.

from __future__ import annotations
import os
import tempfile
import subprocess
import shutil

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import librosa
except Exception:
    librosa = None

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v"}


def _tmp_wav_path() -> str:
    fd, p = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return p


def _write_wav(y: np.ndarray, sr: int, out_path: str) -> None:
    y = y.astype(np.float32)
    if sf is not None:
        sf.write(out_path, y, sr)
    else:
        try:
            from scipy.io import wavfile as wavwrite
            y16 = np.clip(y, -1.0, 1.0)
            y16 = (y16 * 32767.0).astype(np.int16)
            wavwrite.write(out_path, sr, y16)  # type: ignore
        except Exception:
            raise RuntimeError("No soundfile and scipy not available to write WAV.")


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_wav_mono(input_path: str, target_sr: int = 22050) -> str | None:
    """
    Returns path to a temp mono WAV at target_sr made from input (audio OR video).
    On failure returns None and prints brief debug lines.
    """
    if not os.path.isfile(input_path):
        print("extract_wav_mono: input file not found:", input_path)
        return None

    ext = os.path.splitext(input_path)[1].lower()
    out_wav = _tmp_wav_path()

    # ---- 1) Try ffmpeg conversion if available (preferred)
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is not None:
        cmd = [
            ffmpeg_exe, "-y", "-i", input_path,
            "-vn",             # no video
            "-ac", "1",        # mono
            "-ar", str(target_sr),
            "-f", "wav", out_wav
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print("extract_wav_mono: ffmpeg conversion succeeded.")
            return out_wav
        except Exception as e:
            print("extract_wav_mono: ffmpeg conversion failed:", repr(e))
            # fall through to librosa path

    # ---- 2) Librosa path (use res_type for more compatibility)
    if librosa is not None:
        try:
            y, sr = librosa.load(input_path, sr=target_sr, mono=True, res_type="kaiser_fast")
            _write_wav(y, target_sr, out_wav)
            print("extract_wav_mono: librosa.load succeeded.")
            return out_wav
        except Exception as e:
            print("extract_wav_mono: librosa.load failed:", repr(e))

    # ---- 3) soundfile direct read fallback (with simple resample)
    if sf is not None:
        try:
            y, sr = sf.read(input_path, always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != target_sr:
                duration = y.shape[0] / float(sr)
                new_len = int(round(duration * target_sr))
                if new_len <= 0:
                    print("extract_wav_mono: invalid resample length.")
                    return None
                old_idx = np.linspace(0, 1, num=y.shape[0])
                new_idx = np.linspace(0, 1, num=new_len)
                y = np.interp(new_idx, old_idx, y).astype(np.float32)
            _write_wav(y, target_sr, out_wav)
            print("extract_wav_mono: soundfile.read fallback succeeded.")
            return out_wav
        except Exception as e:
            print("extract_wav_mono: soundfile.read fallback failed:", repr(e))

    # Nothing worked
    print("extract_wav_mono: all methods failed. Ensure ffmpeg installed and/or librosa + soundfile/libsndfile available.")
    return None
