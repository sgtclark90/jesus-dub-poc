"""Stage 2 — translation.

Default backend "ct2": NLLB-200 running on **CTranslate2** (the same fast, CPU/offline
engine faster-whisper uses) — no torch, no GPU required. A pre-converted NLLB model is
pulled from the Hub once and cached. This is the path that runs on a plain laptop and on
field/'Truffle'-class offline hardware.

Optional backend "nllb": the original transformers + torch path (GPU-friendly).

Language codes are NLLB FLORES codes: eng_Latn, swh_Latn, hin_Deva, arb_Arab, ...
"""
from __future__ import annotations

import os
from typing import List

from .schema import Segment

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_CT2 = os.path.join(_ROOT, "models", "nllb-ct2-int8")

# Prefer a local model dir (set up by scripts/get_model.ps1) so we never depend on the
# flaky HF python downloader; otherwise fall back to pulling the repo from the Hub.
CT2_REPO = _LOCAL_CT2 if os.path.isdir(_LOCAL_CT2) else "JustFrederik/nllb-200-distilled-600M-ct2-int8"
HF_TOKENIZER = "facebook/nllb-200-distilled-600M"


def _translate_ct2(segments: List[Segment], src_lang: str, tgt_lang: str, repo: str) -> List[Segment]:
    import ctranslate2
    import transformers

    if os.path.isdir(repo):
        model_dir = repo
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo)
    try:
        tok = transformers.AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang)
    except Exception:
        tok = transformers.AutoTokenizer.from_pretrained(HF_TOKENIZER, src_lang=src_lang)

    translator = ctranslate2.Translator(model_dir, device="cpu", compute_type="int8")

    for seg in segments:
        src_tokens = tok.convert_ids_to_tokens(tok.encode(seg.text_source))
        res = translator.translate_batch([src_tokens], target_prefix=[[tgt_lang]], beam_size=4)
        hyp = res[0].hypotheses[0]
        if hyp and hyp[0] == tgt_lang:
            hyp = hyp[1:]
        seg.text_target = tok.decode(tok.convert_tokens_to_ids(hyp)).strip()
    return segments


def _translate_nllb(segments: List[Segment], src_lang: str, tgt_lang: str, model_name: str) -> List[Segment]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    bos = tok.convert_tokens_to_ids(tgt_lang)
    for seg in segments:
        enc = tok(seg.text_source, return_tensors="pt").to(device)
        gen = model.generate(**enc, forced_bos_token_id=bos, max_length=512)
        seg.text_target = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return segments


def translate_segments(segments: List[Segment], src_lang: str, tgt_lang: str,
                       backend: str = "ct2", repo: str = CT2_REPO,
                       model_name: str = "facebook/nllb-200-distilled-600M") -> List[Segment]:
    if backend == "ct2":
        return _translate_ct2(segments, src_lang, tgt_lang, repo)
    if backend == "nllb":
        return _translate_nllb(segments, src_lang, tgt_lang, model_name)
    raise SystemExit(f"Unknown MT backend: {backend}")
