"""Stage 6 — lip-sync.

Real path: Wav2Lip (easy, well-known) — or LatentSync / VideoReTalking for higher
quality if GPU time allows. We shell out to a Wav2Lip checkout so this PoC stays
model-agnostic; swap the command for a better model without touching the pipeline.
"""
from __future__ import annotations

import os
import subprocess


def wav2lip(face_video: str, audio_wav: str, out_video: str,
            wav2lip_dir: str, checkpoint: str = "checkpoints/wav2lip_gan.pth") -> str:
    """Remux `face_video` so the lips match `audio_wav`. Requires a Wav2Lip checkout + GPU.

    inference.py runs with cwd=wav2lip_dir, so the input/output paths must be ABSOLUTE
    (relative ones would resolve inside the Wav2Lip folder and not be found).
    """
    face = os.path.abspath(face_video)
    audio = os.path.abspath(audio_wav)
    out = os.path.abspath(out_video)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    ckpt_path = os.path.join(wav2lip_dir, checkpoint)
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Wav2Lip checkpoint missing at {ckpt_path}. "
                           f"Re-run the Wav2Lip setup cell (7a).")

    cmd = [
        "python", "inference.py",
        "--checkpoint_path", checkpoint,   # relative to wav2lip_dir (the cwd) — correct
        "--face", face,
        "--audio", audio,
        "--outfile", out,
    ]
    r = subprocess.run(cmd, cwd=wav2lip_dir, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stdout[-1200:] + "\n" + r.stderr[-3000:]).strip()
        raise RuntimeError(f"Wav2Lip failed (exit {r.returncode}):\n{tail}")
    return out
