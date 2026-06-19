"""Stage 6 — lip-sync.

Real path: Wav2Lip (easy, well-known) — or LatentSync / VideoReTalking for higher
quality if GPU time allows. We shell out to a Wav2Lip checkout so this PoC stays
model-agnostic; swap the command for a better model without touching the pipeline.
"""
from __future__ import annotations

import subprocess


def wav2lip(face_video: str, audio_wav: str, out_video: str,
            wav2lip_dir: str, checkpoint: str = "checkpoints/wav2lip_gan.pth") -> None:
    """Remux `face_video` so the lips match `audio_wav`. Requires a Wav2Lip checkout + GPU."""
    cmd = [
        "python", "inference.py",
        "--checkpoint_path", checkpoint,
        "--face", face_video,
        "--audio", audio_wav,
        "--outfile", out_video,
    ]
    subprocess.run(cmd, cwd=wav2lip_dir, check=True)
