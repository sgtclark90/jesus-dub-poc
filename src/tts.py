"""Stage 4 — text-to-speech with voice cloning.

Real path: Coqui XTTS-v2. One ~6s reference clip clones a voice into the target
language. This is also how you demo Jesus Film's stated need: turn ONE voice actor
into many characters (old woman, child) where a full cast isn't available — just swap
the `speaker_wav` reference per character.
"""
from __future__ import annotations

import os
from typing import List

from .schema import Segment


def synthesize(segments: List[Segment], speaker_wav: str, language: str,
               out_dir: str, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> List[Segment]:
    """Render each segment's target text to a wav cloned from `speaker_wav`.

    `language` is the XTTS lang code (e.g. "es", "hi", "sw", "ar").
    """
    from TTS.api import TTS  # lazy
    import torch

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name).to(device)

    for seg in segments:
        if not seg.text_target:
            continue
        wav_path = os.path.join(out_dir, f"seg_{seg.id:03d}.wav")
        tts.tts_to_file(text=seg.text_target, speaker_wav=speaker_wav,
                        language=language, file_path=wav_path)
        seg.audio_path = wav_path
    return segments
