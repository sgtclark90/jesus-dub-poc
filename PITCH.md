# Pitch sheet — AI Dubbing for the JESUS Film
*For Jesus Film Project partners · CMA Rally*

## The one sentence
> I built a working proof-of-concept that takes a scene from the JESUS film and produces a **dubbed, lip-synced clip in a new language in minutes** — while keeping a human in charge of every sacred word.

## The problem (their own numbers)
- The film: **2,100+ languages** after **decades** of manual dubbing.
- The world: **~7,000 languages**. The gap is the unreached.
- Jesus Film Project has *already said publicly* it's turning to AI — for translation, lip-sync, and **building a voice cast from one actor**. This PoC does all three.

## The demo (90 seconds, live, on this laptop)
1. Run `python -m src.pipeline --demo` — no internet, no special hardware.
2. It transcribes → translates (English → Swahili) → **checks every key term**.
3. It catches a draft that quietly dropped the word **"Messiah"** and hands it to a human reviewer with the exact location.
4. Point to `outputs/qa_report.json`.

**The line to land:** *"The AI never gets the final word on Scripture. It does the months of grunt work in minutes — and it tells a human exactly where to double-check."*

## Why this is safe to put in front of a translation ministry
- **Human-in-the-loop by design** — the pipeline *stops before dubbing* if any key term is unverified.
- **Runs offline** — same constraint as the field devices SIL/XRI already deploy.
- **Fits the existing world** — built on the same open tools (NLLB, Meta MMS, XTTS) Wycliffe/SIL/unfoldingWord use, not a competing silo.

## What it would unlock
- Manual dub: **months per language.** This draft: **minutes.**
- One voice actor → a **full cast** (man, woman, child) for languages that have no studio.
- A QA layer that lets a reviewer who *doesn't even speak the target language* still catch errors (via back-translation — next on the roadmap).

## The ask
- Connect me with the **Jesus Film Project / Cru product + engineering team** (they have one).
- A **real clip + a glossary** to run a full GPU dub as the next milestone.
- Intros to **SIL / XRI Global** — the Bible-tech AI groups already deep in this; I want to build *with* them, not around them.

## Roadmap if there's interest
1. Full GPU dub of a real 60-sec clip with a native-speaker reaction video.
2. Back-translation QA so non-speakers can review.
3. Offline packaging for field/"Truffle"-class devices.
4. One-actor multi-voice casting demo.

---
*Repo: `jesus-dub-poc/` — README has the full pipeline + Colab notebook for the live GPU dub.*
