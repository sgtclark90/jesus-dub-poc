"""Curated target languages: display name -> NLLB FLORES code + edge-tts voice.

Only languages where BOTH NLLB-200 (translation) and edge-tts (speech) are solid.
The pipeline can target far more via NLLB's 200 languages; this is the demo shortlist,
weighted toward languages with large unreached populations relevant to the JESUS film.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    name: str          # display label
    flores: str        # NLLB FLORES-200 code
    voice: str         # edge-tts voice ShortName


LANGUAGES = [
    Language("Swahili",            "swh_Latn", "sw-TZ-RehemaNeural"),
    Language("Spanish",            "spa_Latn", "es-ES-ElviraNeural"),
    Language("French",             "fra_Latn", "fr-FR-DeniseNeural"),
    Language("Hindi",              "hin_Deva", "hi-IN-SwaraNeural"),
    Language("Arabic",             "arb_Arab", "ar-EG-SalmaNeural"),
    Language("Portuguese (Brazil)","por_Latn", "pt-BR-FranciscaNeural"),
    Language("Indonesian",         "ind_Latn", "id-ID-GadisNeural"),
    Language("Chinese (Mandarin)", "zho_Hans", "zh-CN-XiaoxiaoNeural"),
    Language("Russian",            "rus_Cyrl", "ru-RU-SvetlanaNeural"),
    Language("Vietnamese",         "vie_Latn", "vi-VN-HoaiMyNeural"),
    Language("Bengali",            "ben_Beng", "bn-IN-TanishaaNeural"),
    Language("Urdu",               "urd_Arab", "ur-PK-UzmaNeural"),
    Language("Amharic",            "amh_Ethi", "am-ET-MekdesNeural"),
    Language("Tamil",              "tam_Taml", "ta-IN-PallaviNeural"),
    Language("Filipino",           "fil_Latn", "fil-PH-BlessicaNeural"),
]

BY_NAME = {lang.name: lang for lang in LANGUAGES}
NAMES = [lang.name for lang in LANGUAGES]
