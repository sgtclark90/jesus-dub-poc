"""Background music / SFX preservation.

Instead of replacing the whole audio track, we split the original voice from the music
& sound effects, dub only the voice, and lay it back over the original score — so the
film's music survives the translation.

Two backends:
  * "demucs" — Meta's Demucs source separation (best quality). Needs torch → the GPU/
               Colab path. Auto-used when torch + demucs are installed.
  * "center" — no-ML stereo center-channel cancellation (CPU, runs anywhere, stereo only).
"""
from __future__ import annotations

import importlib.util
import os

from . import audioio


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def background_bed(source_media: str, out_wav: str, backend: str = "auto") -> str | None:
    """Return a path to a music/SFX bed (no original vocals), or None if separation
    isn't possible (e.g. mono audio with the center fallback)."""
    if backend == "auto":
        backend = "demucs" if (_has("torch") and _has("demucs")) else "center"

    if backend == "demucs":
        return _demucs_bed(source_media, out_wav)
    return _center_bed(source_media, out_wav)


def _center_bed(source_media: str, out_wav: str) -> str | None:
    stereo = audioio.extract_audio_stereo(source_media, out_wav.replace(".wav", "_stereo.wav"))
    # mono input -> the two channels are identical -> cancellation yields silence: bail out.
    import soundfile as sf
    import numpy as np
    y, _ = sf.read(stereo, dtype="float32")
    if y.ndim < 2 or np.allclose(y[:, 0], y[:, 1], atol=1e-4):
        return None
    return audioio.remove_center_vocals(stereo, out_wav)


def _demucs_bed(source_media: str, out_wav: str) -> str | None:
    import subprocess
    import shutil

    work = os.path.join(os.path.dirname(out_wav) or ".", "_demucs")
    os.makedirs(work, exist_ok=True)
    audio = audioio.extract_audio_stereo(source_media, os.path.join(work, "in.wav"))
    # --two-stems vocals -> produces vocals.wav and no_vocals.wav
    subprocess.run(["demucs", "--two-stems", "vocals", "-o", work, audio], check=True)
    for root, _, files in os.walk(work):
        if "no_vocals.wav" in files:
            shutil.copy(os.path.join(root, "no_vocals.wav"), out_wav)
            return out_wav
    return None
