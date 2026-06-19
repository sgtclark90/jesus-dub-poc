"""Stage 2 — translation.

Real path: NLLB-200 (offline-capable, 200 languages) for the deterministic baseline.
Optionally followed by an LLM naturalness pass. We keep NLLB as the default because it
runs offline on the kind of hardware XRI's 'Truffle' targets.
"""
from __future__ import annotations

from typing import List

from .schema import Segment


def translate_segments(segments: List[Segment], src_lang: str, tgt_lang: str,
                       model_name: str = "facebook/nllb-200-distilled-600M") -> List[Segment]:
    """Fill in `text_target` for each segment.

    src_lang / tgt_lang are NLLB FLORES codes, e.g. "eng_Latn", "swh_Latn", "hin_Deva".
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy
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
