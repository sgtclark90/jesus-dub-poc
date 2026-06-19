# JESUS Film — AI Dubbing PoC

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

## Run the full pipeline (GPU)

The real dub needs a GPU + ffmpeg. Easiest path is the one-click Colab notebook:

➡ **`notebooks/colab_demo.ipynb`** — upload a clip + a 6-second voice reference, run all cells, download the dubbed video.

Locally:

```bash
pip install -r requirements.txt          # + install ffmpeg and a Wav2Lip checkout
python -m src.pipeline \
    --input data/input/clip.mp4 \
    --speaker data/input/voice_ref.wav \
    --src-lang eng_Latn --tgt-lang swh_Latn --xtts-lang sw \
    --wav2lip-dir ../Wav2Lip
```

If QA flags any key term, the pipeline **stops before dubbing** (override with `--no-gate`).

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
