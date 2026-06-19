"""Callable pipeline API for the UI (and anything non-CLI).

`dub()` runs the full flow and reports progress through a callback so a UI can show a
real progress bar. Returns the dubbed video path plus the transcript/translation table,
the QA report, and a subtitle (.srt) file.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from . import asr, translate, glossary_qa, tts, fit, audioio
from .languages import BY_NAME

OUT = "outputs"
Progress = Callable[[float, str], None]


def _srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def write_srt(segments, path: str) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_srt_time(seg.start)} --> {_srt_time(seg.end)}")
        lines.append(seg.text_target or seg.text_source)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def dub(input_path: str, target_name: str, whisper_model: str = "base",
        src_lang: str = "eng_Latn", gate: bool = False,
        progress: Optional[Progress] = None) -> dict:
    """Translate + voice a clip into `target_name`. Returns paths + structured results."""
    lang = BY_NAME[target_name]
    os.makedirs(OUT, exist_ok=True)

    def report(p: float, msg: str) -> None:
        if progress:
            progress(p, msg)

    report(0.05, "Extracting audio…")
    src_wav = audioio.extract_audio(input_path, os.path.join(OUT, "source_16k.wav"))

    report(0.15, f"Transcribing (Whisper {whisper_model})…")
    segments = asr.transcribe(src_wav, model_size=whisper_model)
    if not segments:
        raise RuntimeError("No speech found in the clip.")

    report(0.45, f"Translating → {lang.name}…")
    segments = translate.translate_segments(segments, src_lang, lang.flores, backend="ct2")

    report(0.6, "Checking key terms…")
    report_qa = glossary_qa.check(segments, "data/glossary/terms.csv")
    if gate and report_qa.needs_review:
        return {"gated": True, "qa": report_qa, "segments": segments, "video": None, "srt": None}

    report(0.7, f"Generating {lang.name} speech…")
    segments = tts.synthesize(segments, lang.voice.split("-")[0], os.path.join(OUT, "audio"),
                              backend="edge", voice=lang.voice)

    report(0.85, "Fitting timing…")
    segments = fit.fit_all(segments, os.path.join(OUT, "audio_fit"))
    timeline = os.path.join(OUT, "dub_track.wav")
    total = max(s.end for s in segments)
    fit.assemble_timeline(segments, total, timeline)

    report(0.93, "Muxing dub onto video…")
    safe = lang.name.split()[0].lower()
    out_video = os.path.join(OUT, f"dubbed_{safe}.mp4")
    audioio.mux_audio(input_path, timeline, out_video)
    srt = write_srt(segments, os.path.join(OUT, f"subtitles_{safe}.srt"))

    report(1.0, "Done.")
    return {"gated": False, "qa": report_qa, "segments": segments, "video": out_video, "srt": srt}
