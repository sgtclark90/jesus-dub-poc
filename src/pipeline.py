"""End-to-end orchestrator: clip -> dubbed, lip-synced clip in a new language.

Two ways to run:

  # Runs on ANY laptop — no GPU, no models, no ffmpeg. Great for a live demo.
  python -m src.pipeline --demo

  # Full pipeline (needs GPU + ffmpeg + the optional deps in requirements.txt)
  python -m src.pipeline \
      --input data/input/clip.mp4 \
      --speaker data/input/voice_ref.wav \
      --src-lang eng_Latn --tgt-lang swh_Latn \
      --xtts-lang sw --whisper-lang en \
      --wav2lip-dir ../Wav2Lip

Every stage writes an inspectable artifact to outputs/. The QA report is the point:
the AI flags uncertain sacred terms for a human instead of shipping them silently.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

# Windows consoles default to cp1252; make sure Swahili text + glyphs render.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import glossary_qa
from .schema import Segment

OUT = "outputs"


def _save_json(name: str, payload) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _print_qa(report) -> None:
    s = report.summary()
    print("\n" + "=" * 64)
    print("  GLOSSARY QA  —  human-in-the-loop key-term check")
    print("=" * 64)
    print(f"  key terms checked : {s['terms_checked']}")
    print(f"  auto-passed       : {s['auto_passed']}")
    print(f"  NEEDS REVIEW      : {s['needs_human_review']}")
    if report.needs_review:
        print("-" * 64)
        for f in report.needs_review:
            print(f"  ⚠  seg {f.segment_id}: '{f.term_source}' -> expected "
                  f"'{f.expected_target}'  [{f.status}]")
            if f.note:
                print(f"       {f.note}")
    print("=" * 64 + "\n")


def run_demo() -> None:
    from . import demo_fixtures

    print("\n▶  DEMO MODE — no GPU/models/ffmpeg required.\n")

    # [1] ASR (fixture)
    segments: List[Segment] = demo_fixtures.source_segments()
    print(f"[1] ASR            -> {len(segments)} segments transcribed (source: English)")

    # [2] Translate (fixture)
    segments = demo_fixtures.apply_demo_translation(segments)
    print("[2] Translate      -> English -> Swahili (swh_Latn)")

    # [3] Glossary QA — runs FOR REAL (it's dependency-free)
    report = glossary_qa.check(segments, "data/glossary/terms.csv")
    print("[3] Glossary QA    -> checking approved key terms ...")

    # [4-6] stubbed in demo (need GPU/ffmpeg)
    for seg in segments:
        seg.audio_path = f"outputs/audio/seg_{seg.id:03d}.wav  (stub — run full pipeline for real audio)"
    print("[4] TTS / clone    -> [skipped in demo]  (XTTS-v2 voice clone)")
    print("[5] Duration fit   -> [skipped in demo]")
    print("[6] Lip-sync       -> [skipped in demo]  (Wav2Lip)")

    seg_path = _save_json("segments.json", [s.to_dict() for s in segments])
    qa_path = _save_json("qa_report.json", report.to_dict())
    _print_qa(report)
    print(f"  wrote {seg_path}")
    print(f"  wrote {qa_path}")
    print("\n  ☞ Notice seg 3: the draft dropped 'Messiah' (Masihi). A human reviewer is")
    print("    told exactly where to look — the AI never silently ships an uncertain term.\n")


def run_full(args) -> None:
    from . import asr, translate, tts, fit, lipsync

    segments = asr.transcribe(args.input, model_size=args.whisper_model, language=args.whisper_lang)
    _save_json("segments.json", [s.to_dict() for s in segments])

    segments = translate.translate_segments(segments, args.src_lang, args.tgt_lang)
    report = glossary_qa.check(segments, args.glossary)
    _save_json("qa_report.json", report.to_dict())
    _print_qa(report)
    if report.needs_review and not args.no_gate:
        print("  ⛔ Key terms need human review. Resolve them (or pass --no-gate) before dubbing.")
        return

    segments = tts.synthesize(segments, args.speaker, args.xtts_lang, os.path.join(OUT, "audio"))
    segments = fit.fit_all(segments, os.path.join(OUT, "audio_fit"))
    timeline = os.path.join(OUT, "dub_track.wav")
    total = max(s.end for s in segments)
    fit.assemble_timeline(segments, total, timeline)

    out_video = os.path.join(OUT, "clip_dubbed.mp4")
    lipsync.wav2lip(args.input, timeline, out_video, args.wav2lip_dir)
    print(f"\n✅ Done -> {out_video}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI dubbing PoC for the JESUS film.")
    p.add_argument("--demo", action="store_true", help="run offline with fixtures (any laptop)")
    p.add_argument("--input", help="source clip (.mp4/.wav)")
    p.add_argument("--speaker", help="~6s reference wav for voice cloning")
    p.add_argument("--glossary", default="data/glossary/terms.csv")
    p.add_argument("--src-lang", default="eng_Latn", help="NLLB source code")
    p.add_argument("--tgt-lang", default="swh_Latn", help="NLLB target code")
    p.add_argument("--whisper-lang", default=None, help="force ASR language (ISO), or auto")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--xtts-lang", default="sw", help="XTTS target language code")
    p.add_argument("--wav2lip-dir", default="../Wav2Lip")
    p.add_argument("--no-gate", action="store_true", help="dub even if QA flags terms")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.demo:
        run_demo()
    else:
        if not args.input or not args.speaker:
            raise SystemExit("Full mode needs --input and --speaker (or use --demo).")
        run_full(args)


if __name__ == "__main__":
    main()
