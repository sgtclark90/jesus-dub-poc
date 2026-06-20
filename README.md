# JESUS Film — AI Dubbing PoC

[![Open the GPU notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sgtclark90/jesus-dub-poc/blob/master/notebooks/colab_demo.ipynb)

**Turn one filmed scene into a fully dubbed, lip-synced clip in a new language — in minutes, with a human kept in charge of every sacred word.**

The JESUS film has reached **2,100+ languages** through decades of manual dubbing. There are ~7,000 languages. Jesus Film Project has publicly said it's turning to AI to close that gap — for translation, for lip-sync, and for **synthesizing a full voice cast (old woman, child) from a single voice actor** where one isn't available. This is a working proof-of-concept of exactly that pipeline.

> **The framing that matters:** AI does **not** get the final word on Scripture here. It produces a *draft* in minutes and **flags every uncertain key term for a human reviewer**. Acceleration, not automation of doctrine.

---

## See it run in 10 seconds (any laptop — no GPU, no models, no internet)

```bash
python -m src.pipeline --demo
```

This runs the real orchestration on a built-in scene (Luke 2:8–14, English → Swahili) and prints the **glossary QA report**. One line in the draft deliberately drops the word **"Messiah"** — watch the tool catch it and tell a reviewer exactly where to look:

```
  GLOSSARY QA — human-in-the-loop key-term check
  key terms checked : 11
  auto-passed       : 10
  NEEDS REVIEW      : 1
  ⚠  seg 3: 'Messiah' -> expected 'Masihi'  [review]
```

Artifacts land in `outputs/segments.json` and `outputs/qa_report.json`.

---

## The pipeline

```
clip.mp4
   │
   ▼
[1] ASR + segmentation     faster-whisper / Meta MMS   → timestamped source text
[2] Translate              NLLB-200 (offline-capable)  → target text, segment-aligned
[3] Glossary QA   ★        approved key-term check      → flags for HUMAN review
[4] TTS + voice clone      Coqui XTTS-v2                → target audio (1 actor → many voices)
[5] Duration fit           pitch-preserving stretch     → audio fits original timing
[6] Lip-sync               Wav2Lip / LatentSync         → lips match the new language
   │
   ▼
clip_dubbed.mp4
```

★ Stage 3 is the trust layer and runs **for real even in `--demo`** — it's pure logic, no model needed.

## The web UI (best for a live demo)

```bash
python app.py        # opens http://127.0.0.1:7860
```

Drag in a clip, pick a **target language** (15 curated, NLLB + natural TTS), choose
transcription quality, hit **Dub it**. You get a **live progress bar**, the dubbed video,
a downloadable **.srt subtitle** file, the **transcript → translation** table, and the
**key-term QA report** — all in the browser.

Options:
- **Keep original music / SFX** — splits voice from the score and dubs only the voice, so
  the music survives. Uses Demucs on GPU/Colab; on a CPU laptop it falls back to a
  stereo center-channel split (stereo audio only).
- **Clone original voice** — re-voice in the speaker's own voice across languages (XTTS,
  GPU/Colab path).
- **Language Suite** tab — dub one clip into **several languages at once**.

## Produce a real dubbed clip — **no GPU required** (CLI)

The default pipeline runs on a plain CPU laptop. It uses `edge-tts` for speech and, with
no GPU/Wav2Lip present, **muxes the dubbed audio onto the original video** — you still get
a watchable dubbed clip. (Lip-sync and voice-cloning are the GPU upgrades, below.)

```bash
pip install -r requirements.txt          # ffmpeg auto-bundled in ./tools on Windows
pwsh -File scripts/get_model.ps1         # one-time: fetch the ~600MB NLLB model via curl
# drop your clip at data/input/clip.mp4, then:
python -m src.pipeline --input data/input/clip.mp4 \
    --src-lang eng_Latn --tgt-lang swh_Latn --tts-lang sw
```

Output: `outputs/clip_dubbed.mp4` (the Swahili dub muxed onto your clip). If QA flags any
key term, the pipeline **stops before dubbing** (override with `--no-gate`).

> **Verified end-to-end on a CPU-only Windows laptop** (no GPU, no PyTorch): ASR
> (faster-whisper) → translation (NLLB-200 on CTranslate2) → glossary QA → Swahili TTS
> (edge-tts) → timeline fit → mux. Real NLLB output, e.g.
> *"He is the Messiah, the Lord." → "Yeye ndiye Masihi, Bwana."*

No clip yet? Generate a synthetic English test clip to try the whole thing:

```bash
python scripts/make_test_clip.py         # writes data/input/clip.mp4
```

### The GPU upgrades (Colab)

➡ **`notebooks/colab_demo.ipynb`** — one-click runner that adds:
- **Voice cloning** — `--tts-backend xtts --speaker ref.wav` turns one actor into a full cast.
- **Lip-sync** — `--lipsync --wav2lip-dir ../Wav2Lip` so the lips match the new language.

## Why these building blocks

Everything here is chosen to fit the existing Bible-tech ecosystem (SIL/XRI, Wycliffe, unfoldingWord) rather than reinvent it:

- **NLLB-200, Meta MMS, XTTS-v2** all run **offline** — the same constraint as XRI Global's "Truffle" offline device used in the field.
- **Glossary QA** mirrors how SIL/Wycliffe already work: AI drafts, humans + community approve.
- **Voice cloning** directly answers Jesus Film's stated need to build a cast from one speaker.

## Project layout

```
src/pipeline.py      orchestrator (--demo and full modes)
src/asr.py           [1] transcribe + segment
src/translate.py     [2] NLLB translation
src/glossary_qa.py   [3] ★ key-term QA — the trust layer
src/tts.py           [4] XTTS voice cloning
src/fit.py           [5] duration fitting + timeline assembly
src/lipsync.py       [6] Wav2Lip
data/glossary/terms.csv   approved source→target key terms
```

## Honest scope

This is a single-clip, single-speaker proof of concept meant to *demonstrate the workflow*, not a production system. It is built to start a conversation with Jesus Film Project / Cru / SIL about where developer help is most useful. Voice cloning and recordings of native speakers raise real consent/ethics questions that a production system must handle deliberately.
