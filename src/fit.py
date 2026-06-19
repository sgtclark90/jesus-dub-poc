"""Stage 5 — duration fitting + timeline assembly.

Translated speech rarely matches the original line length. We time-stretch each
synthesized segment to fit its slot (pitch preserved, via ffmpeg atempo) and lay the
segments back on the original timeline, so the dub stays in sync with the action.
"""
from __future__ import annotations

import os
from typing import List

from . import audioio
from .schema import Segment

SR = 24000


def fit_all(segments: List[Segment], out_dir: str) -> List[Segment]:
    os.makedirs(out_dir, exist_ok=True)
    for seg in segments:
        if not seg.audio_path:
            continue
        fitted = os.path.join(out_dir, f"seg_{seg.id:03d}_fit.wav")
        audioio.atempo_fit(seg.audio_path, fitted, seg.duration)
        seg.audio_path = fitted
    return segments


def assemble_timeline(segments: List[Segment], total_seconds: float, out_wav: str) -> str:
    """Place each fitted segment at its original start time onto a silent bed."""
    import numpy as np            # lazy (installed with the CPU deps)
    import soundfile as sf

    bed = np.zeros(int(total_seconds * SR) + SR, dtype="float32")
    for seg in segments:
        if not seg.audio_path:
            continue
        y, sr = sf.read(seg.audio_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        start = int(seg.start * SR)
        end = min(start + len(y), len(bed))
        bed[start:end] += y[: end - start]
    peak = float(np.max(np.abs(bed))) or 1.0
    if peak > 1.0:
        bed = bed / peak
    sf.write(out_wav, bed, SR)
    return out_wav
