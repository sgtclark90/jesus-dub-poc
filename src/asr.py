"""Stage 1 — ASR + segmentation.

Real path: faster-whisper (use Meta MMS for very low-resource source langs).
Models are imported lazily so this module loads even when deps/GPU are absent
(important: keeps `--demo` runnable on any laptop).
"""
from __future__ import annotations

from typing import List

from .schema import Segment


def transcribe(audio_or_video: str, model_size: str = "large-v3",
               language: str | None = None) -> List[Segment]:
    """Transcribe a media file into timestamped source-language segments.

    Args:
        audio_or_video: path to a .wav/.mp3/.mp4 (ffmpeg required for video/non-wav).
        model_size: faster-whisper model ("tiny" for CPU smoke tests, "large-v3" for quality).
        language: ISO code to force, or None to auto-detect.
    """
    from faster_whisper import WhisperModel  # lazy

    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception:
        # No CUDA on this box — int8 on CPU is the fast, low-memory choice.
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_or_video, language=language, vad_filter=True)

    out: List[Segment] = []
    for i, s in enumerate(segments):
        out.append(Segment(id=i, start=round(s.start, 3), end=round(s.end, 3),
                           text_source=s.text.strip()))
    return out
