"""Stage 3 — glossary / key-term QA.   ★ The trust layer ★

This is what makes the tool *welcome* in a translation ministry rather than feared:
the AI does not get the final word on sacred terms. For every approved key term
(Jesus, Holy Spirit, Son of God, Lord, ...) that appears in the source line, we check
whether the expected rendering shows up in the translation. Anything uncertain is
*surfaced to a human reviewer* instead of being silently shipped.

Deliberately dependency-free (plain string logic) so it runs everywhere, including the
rally laptop. A production version would add embeddings/morphology-aware matching.
"""
from __future__ import annotations

import csv
import unicodedata
from typing import Dict, List

from .schema import QAFlag, QAReport, Segment


def _norm(text: str) -> str:
    """Lowercase + strip accents so 'Yesú' and 'yesu' compare equal."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def load_glossary(path: str) -> List[Dict[str, str]]:
    """CSV columns: term_source, term_target, aliases (optional, ';'-separated)."""
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "term_source": row["term_source"].strip(),
                "term_target": row["term_target"].strip(),
                "aliases": [a.strip() for a in row.get("aliases", "").split(";") if a.strip()],
            })
    return rows


def check(segments: List[Segment], glossary_path: str) -> QAReport:
    glossary = load_glossary(glossary_path)
    report = QAReport()

    for seg in segments:
        src_n = _norm(seg.text_source)
        tgt_n = _norm(seg.text_target or "")
        for entry in glossary:
            if _norm(entry["term_source"]) not in src_n:
                continue  # this key term isn't in this line
            accepted = [entry["term_target"], *entry["aliases"]]
            found = any(_norm(a) in tgt_n for a in accepted if a)
            if found:
                status, note = "ok", ""
            elif not tgt_n:
                status, note = "review", "no translation produced"
            else:
                status = "review"
                note = (f"source uses '{entry['term_source']}' but expected target "
                        f"'{entry['term_target']}' not found — human check needed")
            report.flags.append(QAFlag(
                segment_id=seg.id,
                term_source=entry["term_source"],
                expected_target=entry["term_target"],
                found_in_translation=found,
                status=status,
                note=note,
            ))
    return report
