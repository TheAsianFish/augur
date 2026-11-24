"""``augur floor``: PLAN.md's floor rule, every verdict branch, on small ledgers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from augur import floor
from augur.cli import main
from augur.clock import FixedClock
from augur.errors import AugurError
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.questions import Question, question_from_post
from augur.score import clustered_bootstrap
from augur.standings import RoundStandings, snapshot_text

from .conftest import T0
from .fakes.metaculus import NUMERIC_DETAIL, load_fixture
from .helpers import (
    binary_post,
    leaderboard_entry,
    record_aggregate,
    record_template,
    round_snapshot,
)

OPENED = datetime(2026, 11, 30, 9, tzinfo=UTC)
NUMERIC_POST = 43323


@dataclass(frozen=True)
class Q:
    """One question: Augur's and the template's P(Yes), and what it resolved to."""

    post_id: int
    augur: float | None
    template: float | None
    resolution: str | None = "yes"


def _ledger(path: Path, questions: list[Q], *, numeric: bool = False) -> Path:
    """A shadow ledger: one dry run, Augur's aggregates, template references, resolutions."""
    clock = FixedClock(T0)
    with Ledger.open(path, clock) as ledger:
        run_id = ledger.start_run(
            mode=RunMode.DRY_RUN,
            tournament="minibench",
            provider="openrouter",
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            settings={},
            code_version="t",
        )
        for spec in questions:
            post = binary_post(spec.post_id, spec.post_id + 1000, opened=OPENED)
            q = question_from_post(post)
            ledger.record_question(run_id, q, T0, post)
            if spec.augur is not None:
                record_aggregate(ledger, run_id, q, {"probability": spec.augur})
            if spec.template is not None:
                record_template(ledger, q, {"probability": spec.template})
            if spec.resolution is not None:
                _resolve(ledger, q, spec.resolution)
        if numeric:
            post = load_fixture(NUMERIC_DETAIL)
            q = question_from_post(post)
            assert q.post_id == NUMERIC_POST
            assert q.scaling is not None
            ledger.record_question(run_id, q, T0, post)
            n = q.scaling.inbound_outcome_count
            # 0.10 below the range for Augur, 0.05 for the template; flat inside.
            record_aggregate(ledger, run_id, q, {"cdf": [0.10 + 0.8 * i / n for i in range(n + 1)]})
            record_template(ledger, q, {"cdf": [0.05 + 0.9 * i / n for i in range(n + 1)]})
            _resolve(ledger, q, "below_lower_bound")
    return path


def _resolve(ledger: Ledger, q: Question, value: str) -> None:
    annulled = value in ("annulled", "ambiguous")
    ledger.record_resolution(
        question_id=q.question_id,
        post_id=q.post_id,
        question_type=q.type,
        status=value if annulled else "resolved",
        value=None if annulled else value,
        resolved_at=None,
        source="manual",
        evidence="https://example.test/",
    )


def _strong(first: int, count: int) -> list[Q]:
    """Augur clearly ahead: 0.9 or 0.8 on what happened, the template 0.6."""
    return [Q(first + i, 0.9 if i % 2 else 0.8, 0.6) for i in range(count)]


def _snapshot(
    questions: list[Q],
    *,
    s: float | None = -5.0,
    extra_posts: tuple[int, ...] = (),
    max_coverage: float | None = None,
    fetched_at: datetime = datetime(2026, 12, 13, 12, tzinfo=UTC),
) -> RoundStandings:
    posts = sorted({q.post_id for q in questions} | set(extra_posts))
    entries = [leaderboard_entry("metac-gpt-5+asknews", 3.0, len(posts))]
    if s is not None:
        entries.append(leaderboard_entry(floor.PEER, s, len(posts)))
    return round_snapshot(posts, entries=entries, max_coverage=max_coverage, fetched_at=fetched_at)


def _h(p_a: float, p_t: float) -> float:
    return 100 * math.log(p_a / p_t)


def test_a_strong_edge_on_twenty_questions_is_admitted(tmp_path: Path) -> None:
    qs = _strong(1, 20)
    result = floor.decide(_snapshot(qs), [_ledger(tmp_path / "l.sqlite", qs)])
    assert result.verdict == floor.ADMITTED
    assert (result.n, result.augur, result.template, result.both) == (20, 20, 20, 20)
    want = (10 * _h(0.9, 0.6) + 10 * _h(0.8, 0.6)) / 20
    assert result.mean_h == pytest.approx(want)
    assert result.h_low is not None
    assert result.h_high is not None
    assert 0 < result.h_low < want < result.h_high
    assert result.sd_h == pytest.approx(
        math.sqrt(20 / 19) * (_h(0.9, 0.6) - _h(0.8, 0.6)) / 2, rel=1e-9
    )
    assert result.s == pytest.approx(-5.0)
    assert result.implied == pytest.approx(want - 5.0)
    assert result.implied_low == pytest.approx(result.h_low - 5.0)
    assert result.implied_high == pytest.approx(result.h_high - 5.0)
    assert not result.after_decision_day


def test_the_interval_is_the_registered_bootstrap(tmp_path: Path) -> None:
    qs = [*_strong(1, 18), Q(19, 0.3, 0.6), Q(20, 0.5, 0.6)]
    result = floor.decide(_snapshot(qs), [_ledger(tmp_path / "l.sqlite", qs)])
    est = clustered_bootstrap(
        [x.h for x in result.scored], level=0.90, resamples=10_000, seed=20261119
    )
    assert (result.h_low, result.h_high) == (est.low, est.high)


def test_fewer_than_twenty_scored_is_not_admitted(tmp_path: Path) -> None:
    qs = _strong(1, 19)
    result = floor.decide(_snapshot(qs), [_ledger(tmp_path / "l.sqlite", qs)])
    assert result.verdict == floor.NOT_ADMITTED
    assert result.reason == "n = 19 scored questions, below 20"


def test_a_head_to_head_edge_with_a_very_negative_peer_is_not_admitted(tmp_path: Path) -> None:
    qs = _strong(1, 20)
    result = floor.decide(_snapshot(qs, s=-100.0), [_ledger(tmp_path / "l.sqlite", qs)])
    assert result.h_low is not None
    assert result.h_low > 0
    assert result.implied_low is not None
    assert result.implied_low < 0
    assert result.verdict == floor.NOT_ADMITTED
    assert "lower ends" in result.reason


def test_a_question_without_a_resolution_is_not_yet_decidable(tmp_path: Path) -> None:
    qs = [*_strong(1, 20), Q(21, 0.7, None, resolution=None)]
    result = floor.decide(_snapshot(qs), [_ledger(tmp_path / "l.sqlite", qs)])
    assert result.verdict == floor.UNDECIDABLE
    assert result.unresolved == 1
    assert "1 of the round's questions" in result.reason


def test_a_leaderboard_that_scored_under_half_is_not_yet_decidable(tmp_path: Path) -> None:
    qs = _strong(1, 20)
    ledger = _ledger(tmp_path / "l.sqlite", qs)
    result = floor.decide(_snapshot(qs, max_coverage=9.5), [ledger])
    assert result.verdict == floor.UNDECIDABLE
    assert "scored 9.5 of the round's 20 questions" in result.reason
    assert floor.decide(_snapshot(qs, max_coverage=10), [ledger]).verdict == floor.ADMITTED


def test_no_same_model_template_entry_is_not_yet_decidable(tmp_path: Path) -> None:
    qs = _strong(1, 20)
    result = floor.decide(_snapshot(qs, s=None), [_ledger(tmp_path / "l.sqlite", qs)])
    assert result.verdict == floor.UNDECIDABLE
    assert result.s is None
    assert result.implied is None
    assert floor.PEER in result.reason


def test_numeric_head_to_head_is_halved(tmp_path: Path) -> None:
    qs = _strong(1, 20)
    snap = _snapshot(qs, extra_posts=(NUMERIC_POST,))
    result = floor.decide(snap, [_ledger(tmp_path / "l.sqlite", qs, numeric=True)])
    [numeric] = [x for x in result.scored if x.post_id == NUMERIC_POST]
    assert numeric.question_type == "numeric"
    assert numeric.h == pytest.approx(_h(0.10, 0.05) / 2)
    assert result.n == 21


def test_annulled_uncovered_and_outside_posts_are_counted_not_scored(tmp_path: Path) -> None:
    qs = [
        *_strong(1, 20),
        Q(21, 0.9, 0.1, resolution="annulled"),
        Q(22, 0.9, 0.1, resolution="ambiguous"),
        Q(23, 0.9, None),  # Augur only
        Q(24, None, 0.1),  # template only
    ]
    outside = Q(99, 0.01, 0.99)  # in the ledger, not in the round: a huge loss if counted
    ledger = _ledger(tmp_path / "l.sqlite", [*qs, outside])
    result = floor.decide(_snapshot(qs), [ledger])
    assert (result.round_posts, result.augur, result.template, result.both) == (24, 23, 23, 22)
    assert (result.n, result.annulled, result.uncovered, result.unresolved) == (20, 2, 2, 0)
    assert {x.post_id for x in result.scored} == set(range(1, 21))
    assert result.verdict == floor.ADMITTED


def test_probabilities_are_held_to_the_metaculus_range(tmp_path: Path) -> None:
    qs = [Q(1, 0.9995, 0.0)]
    result = floor.decide(_snapshot(qs), [_ledger(tmp_path / "l.sqlite", qs)])
    assert result.scored[0].h == pytest.approx(_h(0.999, 0.001))


def test_the_template_may_sit_in_another_ledger(tmp_path: Path) -> None:
    qs = _strong(1, 20)
    augur = _ledger(tmp_path / "augur.sqlite", [Q(q.post_id, q.augur, None, None) for q in qs])
    template = _ledger(tmp_path / "template.sqlite", qs)
    result = floor.decide(_snapshot(qs), [augur, template])
    assert (result.n, result.verdict) == (20, floor.ADMITTED)


def test_a_snapshot_after_the_decision_day_decides_nothing_for_the_next_round(
    tmp_path: Path,
) -> None:
    qs = _strong(1, 20)
    late = datetime(2026, 12, 14, 0, 5, tzinfo=UTC)
    result = floor.decide(_snapshot(qs, fetched_at=late), [_ledger(tmp_path / "l.sqlite", qs)])
    assert result.after_decision_day
    assert "decides nothing for the 2026-12-14 round" in result.lines()[-1]
    on_time = _snapshot(qs, fetched_at=late - timedelta(minutes=10))
    assert not floor.decide(on_time, [tmp_path / "l.sqlite"]).after_decision_day


def test_a_missing_ledger_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(AugurError, match="no ledger at"):
        floor.decide(_snapshot(_strong(1, 2)), [tmp_path / "nope.sqlite"])


def test_cli_prints_the_verdict_and_writes_stable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    qs = _strong(1, 20)
    ledger = _ledger(tmp_path / "l.sqlite", qs)
    snap = _snapshot(qs)
    (tmp_path / "data" / "standings").mkdir(parents=True)
    (tmp_path / "data" / "standings" / f"{snap.round.slug}.json").write_text(
        snapshot_text(snap), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "floor.json"
    args = ["floor", "--round", snap.round.slug, "--ledger", str(ledger), "--json", str(out)]
    assert main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"round: {snap.round.slug}"
    assert lines[-1].startswith("verdict: admitted (lower ends: head-to-head ")
    assert any(line.startswith("proxy biases: ") for line in lines)
    first = out.read_text(encoding="utf-8")
    body = json.loads(first)
    assert body["verdict"] == "admitted"
    assert body["n"] == 20
    assert body["rule"]["seed"] == 20261119
    assert main(args) == 0
    assert out.read_text(encoding="utf-8") == first
    assert main(["floor", "--round", "minibench-2026-12-14", "--ledger", str(ledger)]) == 2


def test_cli_refuses_a_snapshot_of_another_round(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    qs = _strong(1, 2)
    path = tmp_path / "snap.json"
    path.write_text(snapshot_text(_snapshot(qs)), encoding="utf-8")
    code = main(
        [
            "floor",
            "--round",
            "minibench-2026-12-14",
            "--ledger",
            str(_ledger(tmp_path / "l.sqlite", qs)),
            "--standings",
            str(path),
        ]
    )
    assert code == 2
    assert "not minibench-2026-12-14" in capsys.readouterr().err
