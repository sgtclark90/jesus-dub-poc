"""ffmpeg/ffprobe helpers — all audio & video I/O goes through here.

Locates a bundled binary in ./tools first (the static build this repo downloads on
Windows), then falls back to whatever 'ffmpeg' is on PATH. Keeping every shell-out in
one place means the pipeline never assumes a codec or container.
"""
from __future__ import annotations

import json
import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bin(name: str) -> str:
    exe = name + (".exe" if os.name == "nt" else "")
    local = os.path.join(_ROOT, "tools", exe)
    return local if os.path.exists(local) else name


def ffmpeg() -> str:
    return _bin("ffmpeg")


def ffprobe() -> str:
    return _bin("ffprobe")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def extract_audio(media: str, out_wav: str, sr: int = 16000) -> str:
    """Pull a mono wav out of any video/audio file (16k is ideal for ASR)."""
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    _run([ffmpeg(), "-y", "-i", media, "-vn", "-ac", "1", "-ar", str(sr), out_wav])
    return out_wav


def duration(path: str) -> float:
    out = subprocess.run(
        [ffprobe(), "-v", "quiet", "-print_format", "json", "-show_format", path],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def to_wav(src: str, out_wav: str, sr: int = 24000) -> str:
    """Normalize any audio (e.g. edge-tts mp3) to a mono wav at `sr`."""
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    _run([ffmpeg(), "-y", "-i", src, "-ac", "1", "-ar", str(sr), out_wav])
    return out_wav


def atempo_fit(in_wav: str, out_wav: str, target_seconds: float) -> str:
    """Stretch/compress speech to `target_seconds`, pitch-preserved, via ffmpeg atempo.

    atempo only accepts 0.5–2.0 per filter; we clamp to a natural-sounding 0.7–1.4.
    """
    cur = duration(in_wav)
    factor = 1.0 if target_seconds <= 0 else cur / target_seconds
    factor = max(0.7, min(1.4, factor))
    _run([ffmpeg(), "-y", "-i", in_wav, "-filter:a", f"atempo={factor:.4f}", out_wav])
    return out_wav


def mux_audio(video: str, audio_wav: str, out_video: str) -> str:
    """Replace a video's audio track with `audio_wav` (the no-GPU lip-sync fallback)."""
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    _run([ffmpeg(), "-y", "-i", video, "-i", audio_wav,
          "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-shortest", out_video])
    return out_video
