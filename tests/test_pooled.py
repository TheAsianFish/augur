"""``augur floor --pooled``: M14's pooled analysis, which can never print a floor verdict."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from augur import floor, pooled
from augur.cli import main
from augur.clock import FixedClock
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.questions import question_from_post
from augur.score import clustered_bootstrap
from augur.standings import RoundStandings, snapshot_text

from .conftest import T0
from .helpers import (
    binary_post,
    leaderboard_entry,
    record_aggregate,
    record_template,
    round_snapshot,
)

DAY_A = datetime(2026, 11, 30, 9, tzinfo=UTC)
BEFORE = FixedClock(datetime(2026, 12, 20, tzinfo=UTC))
ON_THE_DAY = FixedClock(datetime(2027, 1, 11, 6, tzinfo=UTC))


@dataclass(frozen=True)
class Q:
    post_id: int
    augur: float
    template: float
    opened: datetime = DAY_A
    blend: float | None = None
    resolution: str = "yes"


def _h(p_a: float, p_t: float) -> float:
    return 100 * math.log(p_a / p_t)


def _ledger(path: Path, questions: list[Q]) -> Path:
    with Ledger.open(path, FixedClock(T0)) as ledger:
        run_id = ledger.start_run(
            mode=RunMode.DRY_RUN,
            tournament="minibench",
            provider="openrouter",
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            settings={},
            code_version="t",
        )
        for spec in questions:
            post = binary_post(spec.post_id, spec.post_id + 1000, opened=spec.opened)
            q = question_from_post(post)
            ledger.record_question(run_id, q, T0, post)
            record_aggregate(ledger, run_id, q, {"probability": spec.augur})
            record_template(ledger, q, {"probability": spec.template})
            if spec.blend is not None:
                ledger.record_reference(
                    question_id=q.question_id,
                    post_id=q.post_id,
                    question_type=q.type,
                    forecaster=floor.BLEND,
                    detail="one-third Augur, two-thirds price",
                    values={"probability": spec.blend},
                    as_of=T0,
                    source="augur statarm (M12)",
                )
            annulled = spec.resolution in ("annulled", "ambiguous")
            ledger.record_resolution(
                question_id=q.question_id,
                post_id=q.post_id,
                question_type=q.type,
                status=spec.resolution if annulled else "resolved",
                value=None if annulled else spec.resolution,
                resolved_at=None,
                source="manual",
                evidence="https://example.test/",
            )
    return path


def _snapshot(posts: list[int], slug: str, s: float = 0.0) -> RoundStandings:
    entries = [leaderboard_entry(floor.PEER, s, len(posts))]
    return round_snapshot(sorted(posts), entries=entries, slug=slug)


def test_needed_matches_the_registered_arithmetic() -> None:
    # Item 7 of the registration: at sigma = 32, +5 needs 254 and +2 needs 1,583.
    assert pooled.needed(32.0, 5.0) == 254
    assert pooled.needed(32.0, 2.0) == 1583
    with pytest.raises(ValueError, match="must be positive"):
        pooled.needed(0.0, 5.0)


def test_clustering_by_opening_day_matches_a_hand_computed_case(tmp_path: Path) -> None:
    # Day A (2026-11-30 UTC): two questions, Augur 0.9 against 0.6, h = +40.55 each.
    # Day B (2026-12-01 UTC): one question, Augur 0.6 against 0.9, h = -40.55. It opened
    # at 16:30 on 2026-11-30 at UTC-8, which is already 2026-12-01 in UTC.
    late_evening = datetime(2026, 11, 30, 16, 30, tzinfo=timezone(timedelta(hours=-8)))
    qs = [
        Q(1, 0.9, 0.6, DAY_A),
        Q(2, 0.9, 0.6, DAY_A.replace(hour=23, minute=30)),
        Q(3, 0.6, 0.9, late_evening),
    ]
    ledger = _ledger(tmp_path / "l.sqlite", qs)
    report = pooled.pool([_snapshot([1, 2, 3], "minibench-2026-11-30")], [ledger], clock=BEFORE)
    arm = report.primary
    assert arm.by_day is not None
    h = _h(0.9, 0.6)
    # Two clusters, so a resample is AA, AB, BA or BB with mean h, h/3, h/3 or -h. BB and AA
    # each have probability 1/4, far above 5%, so the 90% interval's ends are -h and +h.
    assert arm.by_day.clusters == 2
    assert arm.by_day.n == 3
    assert arm.by_day.mean == pytest.approx(h / 3)
    assert arm.by_day.low == pytest.approx(-h)
    assert arm.by_day.high == pytest.approx(h)
    # Below eight clusters the question-level interval is printed beside it.
    assert arm.by_question is not None
    assert arm.by_question.clusters == 3
    want = clustered_bootstrap([h, h, -h], [0, 1, 2], level=0.90, resamples=10_000, seed=20261119)
    assert (arm.by_question.low, arm.by_question.high) == pytest.approx((want.low, want.high))
    assert arm.spread == pytest.approx(math.sqrt(((2 * h / 3) ** 2 * 2 + (4 * h / 3) ** 2) / 2))
    text = "\n".join(report.lines())
    assert "clusters (opening days) = 2" in text
    assert "the optimistic one" in text
    assert report.reading == pooled.UNRESOLVED


def test_eight_or_more_days_print_only_the_day_clustered_interval(tmp_path: Path) -> None:
    qs = [Q(i, 0.8, 0.6, DAY_A + timedelta(days=i)) for i in range(1, 9)]
    ledger = _ledger(tmp_path / "l.sqlite", qs)
    report = pooled.pool(
        [_snapshot([q.post_id for q in qs], "minibench-2026-11-30")], [ledger], clock=BEFORE
    )
    assert report.primary.by_day is not None
    assert report.primary.by_day.clusters == 8
    assert report.primary.by_question is None
    assert "optimistic" not in "\n".join(report.lines())


def test_the_pooled_report_cannot_emit_an_admission(tmp_path: Path) -> None:
    # Two rounds, each with a lead the floor rule itself admits.
    first = [Q(i, 0.9, 0.6, DAY_A + timedelta(hours=i), blend=0.95) for i in range(1, 25)]
    second = [
        Q(i, 0.9, 0.6, DAY_A + timedelta(days=14, hours=i), blend=0.95) for i in range(101, 125)
    ]
    ledger = _ledger(tmp_path / "l.sqlite", first + second)
    snaps = [
        _snapshot([q.post_id for q in first], "minibench-2026-11-30"),
        _snapshot([q.post_id for q in second], "minibench-2026-12-14"),
    ]
    for snap in snaps:
        assert floor.decide(snap, [ledger]).verdict == floor.ADMITTED
    for clock in (BEFORE, ON_THE_DAY):
        report = pooled.pool(snaps, [ledger], clock=clock)
        assert report.reading == pooled.ABOVE
        text = "\n".join(report.lines()) + report.json_text()
        assert "admit" not in text.lower()
        assert floor.ADMITTED not in text
        assert "decides nothing about submission" in text
    names = {f.name for f in fields(pooled.PooledReport)}
    assert not names & {"verdict", "admitted", "reason"}


def test_cli_pooled_never_prints_a_verdict_and_writes_stable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    qs = [Q(i, 0.9, 0.6, DAY_A + timedelta(hours=i)) for i in range(1, 25)]
    fe = [Q(i, 0.7, 0.6, DAY_A + timedelta(days=3, hours=i)) for i in range(201, 206)]
    ledger = _ledger(tmp_path / "l.sqlite", qs + fe)
    standings = tmp_path / "data" / "standings"
    standings.mkdir(parents=True)
    for posts, slug in (
        ([q.post_id for q in qs], "minibench-2026-11-30"),
        ([q.post_id for q in fe], "fall-futureeval-2026"),
    ):
        (standings / f"{slug}.json").write_text(
            snapshot_text(_snapshot(posts, slug)), encoding="utf-8"
        )
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "pooled.json"
    args = ["floor", "--pooled", "--round", "minibench-2026-11-30"]
    args += ["--round", "fall-futureeval-2026", "--ledger", str(ledger), "--json", str(out)]
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "admit" not in printed.lower()
    assert "verdict" not in printed
    assert "unsubmitted shadow comparison on a below-floor model" in printed
    assert "n = 29" in printed
    first = out.read_text(encoding="utf-8")
    body = json.loads(first)
    assert [r["round"] for r in body["rounds"]] == ["minibench-2026-11-30", "fall-futureeval-2026"]
    assert [r["scored"] for r in body["rounds"]] == [24, 5]
    assert body["registration"]["analysis_date"] == "2027-01-11"
    assert "admit" not in first.lower()
    # The one-round rule refuses several rounds, and --pooled refuses unmatched snapshots.
    two = ["floor", "--round", "minibench-2026-11-30", "--round", "fall-futureeval-2026"]
    assert main([*two, "--ledger", str(ledger)]) == 2
    assert "reads one round" in capsys.readouterr().err
    bad = [*two, "--pooled", "--ledger", str(ledger), "--standings", "x.json"]
    assert main(bad) == 2
    assert "one --standings per --round" in capsys.readouterr().err


def test_round_coverage_is_the_floor_rules_own_count(tmp_path: Path) -> None:
    qs = [
        Q(1, 0.8, 0.6, blend=0.85),
        Q(2, 0.8, 0.6),
        Q(3, 0.8, 0.6, resolution="annulled"),
    ]
    ledger = _ledger(tmp_path / "l.sqlite", qs)
    snap = _snapshot([1, 2, 3, 4], "minibench-2026-11-30")
    report = pooled.pool([snap], [ledger], clock=BEFORE)
    (cov,) = report.rounds
    own = floor.decide(snap, [ledger])
    assert (cov.posts, cov.augur, cov.template, cov.both) == (4, 3, 3, 3)
    assert (cov.scored, cov.annulled, cov.uncovered, cov.unresolved) == (
        own.n,
        own.annulled,
        own.uncovered,
        own.unresolved,
    )
    assert cov.scored == 2
    assert cov.blend_scored == 1
    assert report.secondary.by_day is not None
    assert report.secondary.by_day.n == 1


def test_a_post_in_two_snapshots_is_pooled_once(tmp_path: Path) -> None:
    qs = [Q(1, 0.8, 0.6), Q(2, 0.7, 0.6)]
    ledger = _ledger(tmp_path / "l.sqlite", qs)
    snaps = [_snapshot([1, 2], "minibench-2026-11-30"), _snapshot([2], "fall-futureeval-2026")]
    report = pooled.pool(snaps, [ledger], clock=BEFORE)
    assert report.duplicates == 1
    assert report.primary.by_day is not None
    assert report.primary.by_day.n == 2


def test_a_reading_before_the_analysis_date_is_labelled_progress(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "l.sqlite", [Q(1, 0.8, 0.6), Q(2, 0.6, 0.8)])
    snaps = [_snapshot([1, 2], "minibench-2026-11-30")]
    early = pooled.pool(snaps, [ledger], clock=BEFORE)
    assert early.reading_time == pooled.PROGRESS
    assert "progress reading" in early.lines()[0]
    assert "not the registered result" in early.lines()[0]
    on_the_day = pooled.pool(snaps, [ledger], clock=ON_THE_DAY)
    assert on_the_day.reading_time == pooled.REGISTERED
    assert on_the_day.lines()[0].endswith(f"(analysis date {date(2027, 1, 11).isoformat()})")
    later = pooled.pool(snaps, [ledger], clock=FixedClock(datetime(2027, 1, 12, tzinfo=UTC)))
    assert later.reading_time == pooled.LATE
    assert "late reading" in later.lines()[0]
    assert "not the registered result" in later.lines()[0]
    # Nothing scored yet still prints a report, with the assumed-spread table.
    empty = pooled.pool([_snapshot([9], "minibench-2026-12-14")], [ledger], clock=BEFORE)
    assert empty.primary.by_day is None
    assert empty.needed_measured is None
    assert any("+5 needs 254" in line for line in empty.lines())


def test_the_interval_does_not_depend_on_the_order_of_the_rounds(tmp_path: Path) -> None:
    first = [Q(i, 0.5 + 0.05 * i, 0.6, DAY_A + timedelta(days=i % 3)) for i in range(1, 7)]
    second = [
        Q(i, 0.9 - 0.04 * (i - 10), 0.6, DAY_A + timedelta(days=i % 3)) for i in range(11, 15)
    ]
    ledger = _ledger(tmp_path / "l.sqlite", first + second)
    a = _snapshot([q.post_id for q in first], "minibench-2026-11-30")
    b = _snapshot([q.post_id for q in second], "minibench-2026-12-14")
    one = pooled.pool([a, b], [ledger], clock=BEFORE)
    other = pooled.pool([b, a], [ledger], clock=BEFORE)
    assert one.primary == other.primary
    assert one.primary.by_question is not None


def test_a_round_outside_the_registration_is_flagged() -> None:
    assert pooled.registered_round("minibench")
    assert pooled.registered_round("minibench-2026-11-30")
    assert pooled.registered_round("fall-futureeval-2026")
    assert pooled.registered_round("minibench-2026-12-14")
    assert not pooled.registered_round("bot-testing-area")
    # Item 1: rounds up to and including 2026-11-16 are not registered, nor any after the two.
    assert not pooled.registered_round("minibench-2026-11-16")
    assert not pooled.registered_round("minibench-2026-11-02")
    assert not pooled.registered_round("minibench-2026-12-28")
    assert not pooled.registered_round("minibench-latest")


def test_an_unregistered_round_gets_a_note(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "l.sqlite", [Q(1, 0.8, 0.6)])
    report = pooled.pool([_snapshot([1], "bot-testing-area")], [ledger], clock=BEFORE)
    assert any("bot-testing-area is not one of the registered rounds" in x for x in report.lines())


def test_a_futureeval_round_with_no_augur_forecast_says_why(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "l.sqlite", [Q(1, 0.8, 0.6)])
    snaps = [_snapshot([1], "minibench-2026-11-30"), _snapshot([7, 8], "fall-futureeval-2026")]
    lines = pooled.pool(snaps, [ledger], clock=BEFORE).lines()
    notes = [x for x in lines if "holds no Augur forecast" in x]
    assert len(notes) == 1
    assert notes[0].startswith("  note: fall-futureeval-2026 ")
    assert "polls MiniBench only" in notes[0]
