"""Decode the soundtrack of a local video for playback and YAMNet analysis."""
from __future__ import annotations
import importlib
import shutil
import subprocess
import numpy as np


def _ffmpeg_path():
    try:
        return importlib.import_module("imageio_ffmpeg").get_ffmpeg_exe()
    except ImportError:
        path = shutil.which("ffmpeg")
        if not path:
            raise RuntimeError("FFmpeg tidak ditemukan. Instal imageio-ffmpeg.")
        return path


def decode_video_audio(video_path, sample_rate=44100):
    """Decode video audio to stereo float32 PCM; does not write temporary files."""
    command = [
        _ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vn", "-ac", "2", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=900, check=False
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or "Gagal mengekstrak audio dari video.")
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if samples.size == 0 or samples.size % 2:
        raise RuntimeError("Video tidak memiliki audio stereo yang dapat dibaca.")
    audio = samples.reshape(-1, 2).copy()
    if not np.all(np.isfinite(audio)):
        raise RuntimeError("Audio video mengandung nilai non-finite.")
    return audio, int(sample_rate)
