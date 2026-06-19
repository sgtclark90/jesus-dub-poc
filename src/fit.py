"""Stage 5 — duration fitting.

Translated speech rarely matches the original line length. We time-stretch each
synthesized segment to fit its original slot (without changing pitch) and lay the
segments back on the original timeline, so the dub stays in sync with the action
before lip-sync even runs.
"""
from __future__ import annotations

from typing import List

from .schema import Segment


def fit_segment(in_wav: str, out_wav: str, target_seconds: float) -> None:
    """Time-stretch `in_wav` to `target_seconds`, preserving pitch."""
    import librosa            # lazy
    import soundfile as sf
    import pyrubberband as pyrb

    y, sr = librosa.load(in_wav, sr=None)
    cur = librosa.get_duration(y=y, sr=sr)
    if cur > 0 and target_seconds > 0:
        rate = cur / target_seconds          # >1 speeds up, <1 slows down
        rate = max(0.7, min(1.4, rate))      # clamp so speech stays natural
        y = pyrb.time_stretch(y, sr, rate)
    sf.write(out_wav, y, sr)


def fit_all(segments: List[Segment], out_dir: str) -> List[Segment]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    for seg in segments:
        if not seg.audio_path:
            continue
        fitted = os.path.join(out_dir, f"seg_{seg.id:03d}_fit.wav")
        fit_segment(seg.audio_path, fitted, seg.duration)
        seg.audio_path = fitted
    return segments


def assemble_timeline(segments: List[Segment], total_seconds: float, out_wav: str) -> None:
    """Place each fitted segment at its original start time onto a silent bed."""
    import numpy as np            # lazy
    import soundfile as sf
    import librosa

    sr = 24000
    bed = np.zeros(int(total_seconds * sr) + sr, dtype="float32")
    for seg in segments:
        if not seg.audio_path:
            continue
        y, _ = librosa.load(seg.audio_path, sr=sr)
        start = int(seg.start * sr)
        end = min(start + len(y), len(bed))
        bed[start:end] += y[: end - start]
    sf.write(out_wav, bed, sr)
