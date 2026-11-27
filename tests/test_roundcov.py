"""M17: a round's coverage by opening day, counted from the ledgers; never a verdict."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from augur import floor
from augur.cli import main
from augur.roundcov import by_day
from augur.standings import RoundQuestion, snapshot_text

from .test_pooled import DAY_A, Q, _ledger, _snapshot

DAY_B = datetime(2026, 12, 1, 2, tzinfo=UTC)


def test_questions_are_counted_by_opening_day_against_each_row(tmp_path: Path) -> None:
    # Posts 1 and 2 opened on day A, 3 on day B (all with Augur and the template; only 1
    # with a blend); post 4 opened on day B and was never reached; post 5 has no date.
    ledger = _ledger(
        tmp_path / "l.sqlite",
        [Q(1, 0.8, 0.6, DAY_A, blend=0.7), Q(2, 0.8, 0.6, DAY_A), Q(3, 0.8, 0.6, DAY_B)],
    )
    snap = _snapshot([1, 2, 3, 4, 5], "minibench-2026-11-30")
    questions = [
        q.model_copy(update={"opens_at": "2026-12-01T05:00:00Z"}) if q.post_id == 4 else q
        for q in snap.questions
    ]
    snap = snap.model_copy(update={"questions": questions})
    rows = {r.day: r for r in by_day(snap, [ledger])}
    assert list(rows) == ["2026-11-30", "2026-12-01", "unknown", "total"]
    a, b, unknown, total = rows.values()
    assert (a.opened, a.held["augur"], a.held["template"], a.held["blend"]) == (2, 2, 2, 1)
    assert (b.opened, b.held["augur"], b.held["template"]) == (2, 1, 1)
    assert (unknown.opened, unknown.held["augur"]) == (1, 0)
    assert (total.opened, total.held["augur"], total.held["stat"], total.all_four) == (5, 3, 0, 0)


def test_all_four_needs_every_row(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "l.sqlite", [Q(1, 0.8, 0.6, DAY_A, blend=0.7)])
    snap = _snapshot([1], "minibench-2026-11-30")
    (row, _total) = by_day(snap, [ledger])
    assert row.held == {"augur": 1, "template": 1, "stat": 0, "blend": 1}
    assert row.all_four == 0


def test_cli_by_day_prints_counts_and_no_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = _ledger(tmp_path / "l.sqlite", [Q(1, 0.8, 0.6, DAY_A)])
    path = tmp_path / "s.json"
    path.write_text(snapshot_text(_snapshot([1, 2], "minibench-2026-11-30")), encoding="utf-8")
    args = ["floor", "--round", "minibench-2026-11-30", "--standings", str(path)]
    assert main([*args, "--ledger", str(ledger), "--by-day"]) == 0
    out = capsys.readouterr().out
    assert "2026-11-30: opened 1; augur 1, template 1, stat 0, blend 0; all four 0" in out
    assert "total: opened 2" in out
    for word in ("admitted", "decidable", "verdict:"):
        assert word not in out.replace("not a verdict", "")
    assert floor.AUGUR == "augur"


def test_a_snapshot_records_when_each_question_opened() -> None:
    from augur.standings import questions_from_posts

    (q,) = questions_from_posts(
        [{"id": 9, "open_time": "2026-11-30T14:00:00Z", "question": {"id": 19}}]
    )
    assert q.opens_at == "2026-11-30T14:00:00Z"
    old = RoundQuestion.model_validate(
        {"post_id": 1, "question_id": None, "title": "", "status": ""}
    )
    assert old.opens_at is None


def test_an_opening_time_with_an_offset_counts_on_its_utc_day() -> None:
    from augur.roundcov import utc_day

    assert utc_day("2026-11-30T20:00:00-08:00") == "2026-12-01"
    assert utc_day("2026-11-30T20:00:00Z") == "2026-11-30"
    assert utc_day("2026-11-30T20:00:00") == "2026-11-30"
    assert utc_day("not a time") == ""
    assert utc_day(None) == ""
