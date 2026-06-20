"""Stage 4 — text-to-speech.

Two backends, same output (one mono 24k wav per segment):

  * "edge"  — Microsoft edge-tts. CPU-only, fast, no model download, many languages.
              Default so the pipeline runs on a laptop. (Needs network at synth time.)
  * "xtts"  — Coqui XTTS-v2 voice cloning. GPU path. Clones ONE reference voice into the
              target language — this is how you turn a single actor into a full cast
              (man / woman / child), the need Jesus Film Project named.
"""
from __future__ import annotations

import asyncio
import os
from typing import List

from . import audioio
from .schema import Segment

# Minimal language -> edge voice map. Pass --voice to override with any edge voice.
EDGE_VOICES = {
    "sw": "sw-TZ-RehemaNeural",   # Swahili (Tanzania), female
    "en": "en-US-GuyNeural",
    "es": "es-ES-AlvaroNeural",
    "hi": "hi-IN-SwaraNeural",
    "fr": "fr-FR-DeniseNeural",
    "ar": "ar-EG-SalmaNeural",
    "pt": "pt-BR-AntonioNeural",
    "id": "id-ID-ArdiNeural",
}


def _edge_synth(segments: List[Segment], voice: str, out_dir: str) -> List[Segment]:
    import edge_tts  # lazy

    async def one(seg: Segment) -> None:
        mp3 = os.path.join(out_dir, f"seg_{seg.id:03d}.mp3")
        await edge_tts.Communicate(seg.text_target, voice).save(mp3)
        wav = os.path.join(out_dir, f"seg_{seg.id:03d}.wav")
        audioio.to_wav(mp3, wav, sr=24000)
        os.remove(mp3)
        seg.audio_path = wav

    async def run_all() -> None:
        for seg in segments:                      # sequential keeps edge happy
            if seg.text_target:
                await one(seg)

    asyncio.run(run_all())
    return segments


def _xtts_synth(segments: List[Segment], speaker_wav: str, language: str,
                out_dir: str, model_name: str) -> List[Segment]:
    # coqui-tts imports `LogitsWarper`, which newer transformers removed (merged into
    # LogitsProcessor). Restore the name before importing TTS so the import succeeds.
    import transformers
    if not hasattr(transformers, "LogitsWarper"):
        from transformers import LogitsProcessor as _LW
        transformers.LogitsWarper = _LW

    from TTS.api import TTS  # lazy
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name).to(device)
    for seg in segments:
        if not seg.text_target:
            continue
        wav = os.path.join(out_dir, f"seg_{seg.id:03d}.wav")
        tts.tts_to_file(text=seg.text_target, speaker_wav=speaker_wav,
                        language=language, file_path=wav)
        seg.audio_path = wav
    return segments


def synthesize(segments: List[Segment], language: str, out_dir: str,
               backend: str = "edge", voice: str | None = None, speaker_wav: str | None = None,
               model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> List[Segment]:
    os.makedirs(out_dir, exist_ok=True)
    if backend == "edge":
        v = voice or EDGE_VOICES.get(language)
        if not v:
            raise SystemExit(f"No default edge voice for '{language}'. Pass --voice "
                             f"(see `edge-tts --list-voices`).")
        return _edge_synth(segments, v, out_dir)
    if backend == "xtts":
        if not speaker_wav:
            raise SystemExit("xtts backend needs --speaker (a ~6s reference wav).")
        return _xtts_synth(segments, speaker_wav, language, out_dir, model_name)
    raise SystemExit(f"Unknown TTS backend: {backend}")
