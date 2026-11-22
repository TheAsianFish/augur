"""Draws on an arm's recorded briefs: rebuilt from the ledger and checked by request hash."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from augur.backtest import sample
from augur.errors import AugurError
from augur.forecastbench import FBQuestion, load_set
from augur.redraw import Recorded, check_rebuilt, recorded_research, research_from

SET = Path("data/forecastbench/2026-06-21")
OPEN = Path("data/ledgers/m6-c2-open.sqlite")
ARM = "forecastbench:2026-06-21-llm.json+onepass-asof-open"


def questions() -> list[FBQuestion]:
    report = load_set(
        Path(f"{SET}-llm.subset.json"),
        Path(f"{SET}_resolution_set.subset.json"),
        released=date(2026, 6, 4),
    )
    return sample(report.kept, 64)


def test_every_recorded_brief_rebuilds_the_exact_request() -> None:
    recorded = recorded_research(OPEN, ARM)
    qs = questions()
    assert {q.question_id for q in qs} <= recorded.keys()
    for fb in qs:
        check_rebuilt(fb, recorded[fb.question_id])  # raises on any difference


def test_the_sources_are_the_ones_the_forecasters_cited() -> None:
    recorded = recorded_research(OPEN, ARM)
    for rec in recorded.values():
        for i, source in enumerate(rec.research.sources, start=1):
            if f"[S{i}]" in rec.research.brief:
                assert source.published_at is not None
                assert source.published_at <= source.as_of


def test_a_changed_brief_is_refused() -> None:
    recorded = recorded_research(OPEN, ARM)
    fb = questions()[0]
    rec = recorded[fb.question_id]
    rec.research.brief += " (edited)"
    tampered = Recorded(rec.research, rec.request, rec.request_hash)
    with pytest.raises(AugurError, match="does not match"):
        check_rebuilt(fb, tampered)


def test_research_from_skips_questions_without_a_brief() -> None:
    fb = questions()[0]
    assert research_from({})(fb) is None


def test_an_unknown_arm_has_no_briefs() -> None:
    assert recorded_research(OPEN, ARM + "+nothing") == {}
