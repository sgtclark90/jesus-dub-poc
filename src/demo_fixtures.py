"""Offline fixtures so the whole pipeline runs on ANY laptop (no GPU, no models, no ffmpeg).

Scene: Luke 2:8-14 — the angel announces Jesus' birth to the shepherds. (The JESUS film
is based on Luke, so this is on-theme.) Source = English, target = Swahili (swh_Latn),
a major language for the film's African distribution.

The QA stage runs FOR REAL on this data — including one deliberately imperfect line,
so a live audience can watch the tool catch a dropped key term ("Messiah").
"""
from __future__ import annotations

from typing import Dict, List

from .schema import Segment

# (id, start, end, English source)
_SOURCE = [
    (0, 0.0, 5.2, "And there were shepherds living out in the fields nearby, keeping watch over their flocks at night."),
    (1, 5.2, 10.6, "An angel of the Lord appeared to them, and the glory of the Lord shone around them, and they were terrified."),
    (2, 10.6, 15.1, "But the angel said to them, 'Do not be afraid. I bring you good news of great joy for all the people.'"),
    (3, 15.1, 20.4, "Today in the town of David a Savior has been born to you; he is the Messiah, the Lord."),
    (4, 20.4, 25.0, "Glory to God in the highest heaven, and on earth peace to those on whom his favor rests."),
    (5, 25.0, 29.3, "This is the good news about Jesus the Christ, the Son of God."),
]

# Pre-baked Swahili "machine translation". Segment 3 intentionally DROPS 'Masihi'
# (Messiah) — the QA stage will flag it for human review. Everything else is clean.
_TARGET: Dict[int, str] = {
    0: "Kulikuwa na wachungaji waliokaa mashambani karibu, wakilinda makundi yao usiku.",
    1: "Malaika wa Bwana akawatokea, na utukufu wa Bwana ukawaangazia, nao wakaogopa sana.",
    2: "Lakini malaika akawaambia, 'Msiogope. Nawaletea habari njema ya furaha kuu kwa watu wote.'",
    3: "Leo katika mji wa Daudi mmezaliwa Mwokozi; yeye ni Bwana.",  # <- 'Masihi' missing on purpose
    4: "Utukufu kwa Mungu juu mbinguni, na duniani amani kwa wale anaowapenda.",
    5: "Hii ni habari njema kuhusu Yesu Kristo, Mwana wa Mungu.",
}


def source_segments() -> List[Segment]:
    return [Segment(id=i, start=s, end=e, text_source=t) for (i, s, e, t) in _SOURCE]


def apply_demo_translation(segments: List[Segment]) -> List[Segment]:
    for seg in segments:
        seg.text_target = _TARGET.get(seg.id)
    return segments


TOTAL_SECONDS = _SOURCE[-1][2]
