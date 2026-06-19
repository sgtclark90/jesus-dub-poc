"""Shared data contracts passed between pipeline stages.

Kept as plain dataclasses (no heavy deps) so the demo path runs on any laptop.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Segment:
    """One spoken line, with timing and its source + target text."""
    id: int
    start: float                       # seconds
    end: float                         # seconds
    text_source: str
    text_target: Optional[str] = None
    audio_path: Optional[str] = None   # synthesized target-language wav

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


@dataclass
class QAFlag:
    """A glossary QA finding a human reviewer should look at."""
    segment_id: int
    term_source: str
    expected_target: str
    found_in_translation: bool
    status: str                        # "ok" | "missing" | "review"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QAReport:
    flags: List[QAFlag] = field(default_factory=list)

    @property
    def needs_review(self) -> List[QAFlag]:
        return [f for f in self.flags if f.status != "ok"]

    def summary(self) -> dict:
        total = len(self.flags)
        ok = sum(1 for f in self.flags if f.status == "ok")
        return {
            "terms_checked": total,
            "auto_passed": ok,
            "needs_human_review": total - ok,
        }

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "flags": [f.to_dict() for f in self.flags]}
