"""ForecastBench question and resolution sets: a minimal loader (extended in M6).

ForecastBench (Forecasting Research Institute, CC BY-SA 4.0) publishes a
question set every two weeks and, as questions resolve, a resolution set for
it. This module reads one pair, keeps the questions a backtest may score, and
turns each into a binary :class:`~augur.questions.Question` so the bot's own
forecasting code runs on it unchanged.

Eligibility, applied in this order, with the count each step removes:

1. *market questions only*: ``manifold``, ``metaculus``, ``polymarket`` and
   ``infer``. Dataset questions (ACLED, FRED, ...) ask about several horizons
   at once and are left to M6;
2. *resolved to 0 or 1*: a market question still open is "resolved" by
   ForecastBench to its current price, which is not an outcome;
3. *opened after the model's release*: the question's freeze date (the
   instant the question was posed) must fall after the model's release date,
   so the model cannot remember the outcome;
4. *a market close date*: the prompt states when the question closes, and
   the only other date on hand is the resolution date, which is after the
   fact.

The market price at the freeze date is kept for a reference row, never shown
to the model: a price-only arm is M5's experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from augur.clock import ensure_utc, parse_iso
from augur.questions import Question

MARKET_SOURCES: frozenset[str] = frozenset({"manifold", "metaculus", "polymarket", "infer"})
SITE = "https://www.forecastbench.org"
LICENCE = "ForecastBench datasets, Forecasting Research Institute, CC BY-SA 4.0"


class FBQuestion(BaseModel):
    """One eligible ForecastBench market question with its outcome."""

    model_config = ConfigDict(frozen=True)

    fb_id: str
    source: str
    question: str
    resolution_criteria: str
    background: str
    source_intro: str
    url: str
    freeze_datetime: datetime
    freeze_value: float | None
    market_open: datetime | None
    market_close: datetime | None
    resolution_date: date
    outcome: int

    @property
    def question_id(self) -> int:
        """A stable integer id (52 bits of the SHA-256 of source and id)."""
        digest = hashlib.sha256(f"{self.source}:{self.fb_id}".encode()).hexdigest()
        return int(digest[:13], 16)

    def to_question(self) -> Question:
        """The question as the forecaster sees it: no market price, no outcome."""
        if self.market_close is None:
            raise ValueError(f"{self.fb_id} has no market close date; the loader drops these")
        close = max(self.market_close, self.freeze_datetime)
        description = "\n\n".join(p for p in (self.source_intro, self.background) if p)
        return Question(
            post_id=self.question_id,
            question_id=self.question_id,
            title=self.question,
            type="binary",
            status="open",
            url=self.url,
            open_time=self.freeze_datetime,
            scheduled_close_time=close,
            scheduled_resolve_time=None,
            description=description,
            resolution_criteria=self.resolution_criteria,
        )


@dataclass
class LoadReport:
    question_set: str
    forecast_due_date: str
    total: int = 0
    removed: dict[str, int] = field(default_factory=dict)
    kept: list[FBQuestion] = field(default_factory=list)

    def remove(self, reason: str) -> None:
        self.removed[reason] = self.removed.get(reason, 0) + 1

    def lines(self) -> list[str]:
        out = [f"{self.question_set}: {self.total} questions"]
        out.extend(f"  removed {n:>4}: {reason}" for reason, n in self.removed.items())
        out.append(f"  kept    {len(self.kept):>4}")
        return out


def _text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text in ("N/A", "nan") else text


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None


def _instant(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return ensure_utc(parse_iso(text))
    except ValueError:
        return None


def load_set(
    question_set: Path, resolution_set: Path, *, released: date, limit: int | None = None
) -> LoadReport:
    """Read one question set and its resolution set; keep the eligible questions.

    ``released`` is the model's release date. Questions keep the order of the
    question set, so a ``limit`` always takes the same ones.
    """
    qs = json.loads(question_set.read_text(encoding="utf-8"))
    rs = json.loads(resolution_set.read_text(encoding="utf-8"))
    if qs.get("question_set") != rs.get("question_set"):
        raise ValueError(
            f"{resolution_set.name} resolves {rs.get('question_set')!r}, "
            f"not {qs.get('question_set')!r}; pass the matching pair"
        )
    outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rs.get("resolutions", []):
        if isinstance(r.get("id"), str):
            outcomes[(str(r["source"]), r["id"])] = r
    report = LoadReport(str(qs.get("question_set")), str(qs.get("forecast_due_date")))
    for raw in qs.get("questions", []):
        report.total += 1
        source = str(raw.get("source"))
        if source not in MARKET_SOURCES or not isinstance(raw.get("id"), str):
            report.remove("not a market question (dataset questions are M6's)")
            continue
        r = outcomes.get((source, raw["id"]))
        if r is None or not r.get("resolved") or r.get("resolved_to") not in (0, 1, 0.0, 1.0):
            report.remove("not resolved to 0 or 1 in the resolution set")
            continue
        freeze = _instant(raw.get("freeze_datetime"))
        if freeze is None:
            report.remove("no freeze date")
            continue
        if freeze.date() <= released:
            report.remove("posed on or before the model's release")
            continue
        market_close = _instant(raw.get("market_info_close_datetime"))
        if market_close is None:
            # The fallback would be the resolution date, which is after the fact
            # and hints at the outcome ("by" markets that resolve early are Yes).
            report.remove("no market close date")
            continue
        report.kept.append(
            FBQuestion(
                fb_id=raw["id"],
                source=source,
                question=_text(raw.get("question")),
                resolution_criteria=_text(raw.get("resolution_criteria")),
                background=_text(raw.get("background")),
                source_intro=_text(raw.get("source_intro")),
                url=_text(raw.get("url")) or SITE,
                freeze_datetime=freeze,
                freeze_value=_float(raw.get("freeze_datetime_value")),
                market_open=_instant(raw.get("market_info_open_datetime")),
                market_close=market_close,
                resolution_date=date.fromisoformat(str(r["resolution_date"])),
                outcome=int(float(r["resolved_to"])),
            )
        )
    if limit is not None:
        dropped = len(report.kept) - limit
        if dropped > 0:
            report.kept = report.kept[:limit]
            report.removed["over the limit"] = dropped
    return report


def subset_files(
    report: LoadReport, question_set: Path, resolution_set: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The question and resolution set cut down to the kept questions.

    Committed so the run replays without the network; the cut keeps every
    field of each kept question as published.
    """
    keep = {(q.source, q.fb_id) for q in report.kept}
    qs = json.loads(question_set.read_text(encoding="utf-8"))
    rs = json.loads(resolution_set.read_text(encoding="utf-8"))
    qs["questions"] = [q for q in qs["questions"] if (str(q.get("source")), q.get("id")) in keep]
    rs["resolutions"] = [
        r for r in rs["resolutions"] if (str(r.get("source")), r.get("id")) in keep
    ]
    return qs, rs


def base_rate(questions: Sequence[FBQuestion]) -> float:
    if not questions:
        raise ValueError("no questions")
    return sum(q.outcome for q in questions) / len(questions)
