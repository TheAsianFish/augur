"""The scorecard: which forecast is scored, against what, and the numbers by hand."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from augur.clock import FixedClock
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.questions import question_from_post
from augur.score import ScoringError
from augur.scorecard import Loaded, ScoredQuestion, format_summary, load, summarise

from .conftest import T0
from .fakes.metaculus import MC_DETAIL, NUMERIC_DETAIL, load_fixture
from .helpers import binary_post

RESAMPLES = 300
OPENED = {
    "q1": datetime(2026, 11, 2, 9, tzinfo=UTC),  # Monday, ISO week 45
    "q2": datetime(2026, 11, 4, 9, tzinfo=UTC),  # Wednesday, week 45
    "q3": datetime(2026, 11, 9, 9, tzinfo=UTC),  # Monday, week 46
}
# MC fixture opened 2026-10-17 (week 42); numeric fixture 2026-04-27 (week 18).


def b(x: float) -> float:
    """Baseline score of a binary forecast that put ``x`` on what happened."""
    return 100 * (math.log(x) + math.log(2)) / math.log(2)


@dataclass
class World:
    path: Path
    ids: dict[str, int]
    clock: FixedClock


def _run(ledger: Ledger, mode: RunMode, tournament: str) -> str:
    return ledger.start_run(
        mode=mode,
        tournament=tournament,
        provider="replay",
        model=f"model-{mode.value}",
        settings={},
        code_version="t",
    )


def _post(ledger: Ledger, run_id: str, post: dict[str, Any]) -> int:
    q = question_from_post(post)
    ledger.record_question(run_id, q, T0, post)
    return q.question_id


def _aggregate(
    ledger: Ledger, run_id: str, qid: int, values: dict[str, Any], qtype: str = "binary"
) -> None:
    ledger.record_forecast(
        run_id=run_id,
        question_id=qid,
        kind="aggregate",
        member=None,
        method="median",
        probability=values.get("probability"),
        reasoning="r",
        call_id=None,
        question_type=qtype,
        values=values,
    )


def _resolve(ledger: Ledger, qid: int, qtype: str, status: str, value: str | None) -> None:
    ledger.record_resolution(
        question_id=qid,
        post_id=qid - 1000 if qtype == "binary" else qid,
        question_type=qtype,
        status=status,
        value=value,
        resolved_at=None,
        source="manual",
        evidence="test",
    )


def _reference(ledger: Ledger, qid: int, qtype: str, name: str, values: dict[str, Any]) -> None:
    ledger.record_reference(
        question_id=qid,
        post_id=qid,
        question_type=qtype,
        forecaster=name,
        detail="d",
        values=values,
        as_of=T0,
        source="s",
    )


@pytest.fixture
def world(tmp_path: Path) -> World:
    """Five scored questions, one annulled, one unresolved, one unscorable.

    Binary q1 (Augur 0.8, Yes), q2 (0.3, No), q3 (0.6, No); a multiple-choice question
    (0.5 on the winning option of three); a numeric one resolved below its range (the CDF
    puts 0.05 there). A template bot forecast q1 (0.6) and q3 (0.5). q1 was posted from a
    live run at 0.8 and forecast again later in a dry run at 0.2; q2 was forecast twice in
    dry runs (0.9, then 0.3).
    """
    clock = FixedClock(T0)
    path = tmp_path / "score.sqlite"
    ids: dict[str, int] = {}
    with Ledger.open(path, clock) as ledger:
        live = _run(ledger, RunMode.LIVE, "bot-testing-area")
        dry1 = _run(ledger, RunMode.DRY_RUN, "minibench")
        for name, qid, post_id in (("q1", 2001, 1001), ("q2", 2002, 1002), ("q3", 2003, 1003)):
            post = binary_post(post_id, qid, opened=OPENED[name], title=f"Question {name}?")
            ids[name] = qid
            _post(ledger, dry1, post)
            if name == "q1":
                _post(ledger, live, post)
        # q1: posted 0.8 in the live run ...
        _aggregate(ledger, live, ids["q1"], {"probability": 0.8})
        ledger.record_submission(
            run_id=live,
            question_id=ids["q1"],
            post_id=1001,
            kind="forecast",
            payload={"probability_yes": 0.8},
            status="submitted",
            http_status=201,
            response_excerpt=None,
        )
        # ... then a later dry-run forecast of 0.2, which must not replace it.
        clock.advance(3600)
        _aggregate(ledger, dry1, ids["q1"], {"probability": 0.2})
        # q2: an earlier dry-run forecast of 0.9, replaced by a later one of 0.3.
        _aggregate(ledger, dry1, ids["q2"], {"probability": 0.9})
        clock.advance(60)
        _aggregate(ledger, dry1, ids["q2"], {"probability": 0.3})
        _aggregate(ledger, dry1, ids["q3"], {"probability": 0.6})

        mc_post = load_fixture(MC_DETAIL)
        ids["mc"] = _post(ledger, dry1, mc_post)
        options = question_from_post(mc_post).options
        assert options
        mc_probs = {options[0]: 0.3, options[1]: 0.5, options[2]: 0.2}
        _aggregate(ledger, dry1, ids["mc"], {"probabilities": mc_probs}, "multiple_choice")

        num_post = load_fixture(NUMERIC_DETAIL)
        ids["num"] = _post(ledger, dry1, num_post)
        scaling = question_from_post(num_post).scaling
        assert scaling is not None
        n = scaling.inbound_outcome_count
        cdf = [0.05 + 0.9 * i / n for i in range(n + 1)]  # 0.05 below, 0.05 above, flat inside
        _aggregate(ledger, dry1, ids["num"], {"cdf": cdf}, "numeric")

        for name, qid in (("annulled", 2004), ("open", 2005), ("odd", 2006)):
            post = binary_post(qid - 1000, qid, opened=OPENED["q1"], title=name)
            ids[name] = qid
            _post(ledger, dry1, post)
            _aggregate(ledger, dry1, qid, {"probability": 0.5})

        _resolve(ledger, ids["q1"], "binary", "resolved", "yes")
        _resolve(ledger, ids["q2"], "binary", "resolved", "no")
        _resolve(ledger, ids["q3"], "binary", "resolved", "no")
        _resolve(ledger, ids["mc"], "multiple_choice", "resolved", options[1])
        _resolve(ledger, ids["num"], "numeric", "resolved", "below_lower_bound")
        _resolve(ledger, ids["annulled"], "binary", "annulled", None)
        _resolve(ledger, ids["odd"], "binary", "resolved", "maybe")

        _reference(ledger, ids["q1"], "binary", "template", {"probability": 0.6})
        _reference(ledger, ids["q3"], "binary", "template", {"probability": 0.5})
        for score_type, value in (
            ("baseline_score", 70.0),
            ("spot_baseline_score", 65.0),
            ("peer_score", 5.0),
        ):
            ledger.record_metaculus_score(
                question_id=ids["q1"],
                post_id=1001,
                forecaster="augur",
                score_type=score_type,
                score=value,
            )
        ledger.record_metaculus_score(
            question_id=ids["q2"],
            post_id=1002,
            forecaster="community",
            score_type="baseline_score",
            score=99.0,
        )
    return World(path, ids, clock)


def _by_id(loaded: Loaded, world: World) -> dict[str, ScoredQuestion]:
    names = {v: k for k, v in world.ids.items()}
    return {names[s.question_id]: s for s in loaded.scored}


def test_load_counts(world: World) -> None:
    loaded = load([world.path])
    assert sorted(_by_id(loaded, world)) == ["mc", "num", "q1", "q2", "q3"]
    assert loaded.unresolved == 1
    assert loaded.annulled == 1
    assert len(loaded.unscorable) == 1
    assert loaded.unscorable[0].startswith(f"{world.ids['odd']}:")
    assert "not yes or no" in loaded.unscorable[0]


def test_load_orders_by_open_time_then_id(world: World) -> None:
    loaded = load([world.path])
    opens = [s.opened for s in loaded.scored]
    assert opens == sorted(opens)
    assert loaded.scored[0].question_type == "numeric"  # opened 2026-04-27


def test_a_posted_forecast_beats_a_later_dry_run(world: World) -> None:
    q1 = _by_id(load([world.path]), world)["q1"]
    assert q1.augur == {"probability": 0.8}
    assert q1.submitted is True
    assert q1.model == "model-live"
    assert q1.source == "bot-testing-area"


def test_without_a_post_the_latest_aggregate_wins(world: World) -> None:
    loaded = _by_id(load([world.path]), world)
    assert loaded["q2"].augur == {"probability": 0.3}
    assert loaded["q2"].submitted is False
    assert loaded["q2"].source == "minibench"


def test_mode_filter_keeps_only_those_runs(world: World) -> None:
    live_only = load([world.path], modes=["live"])
    assert [s.question_id for s in live_only.scored] == [world.ids["q1"]]
    assert load([world.path], modes=["backtest"]).scored == []
    # With the live run excluded, q1's only forecast is the dry-run 0.2.
    dry = _by_id(load([world.path], modes=["dry_run"]), world)
    assert dry["q1"].augur == {"probability": 0.2}
    assert dry["q1"].submitted is False


def test_the_same_ledger_twice_counts_each_question_once(world: World) -> None:
    assert len(load([world.path, world.path]).scored) == 5


def test_an_empty_file_contributes_nothing(tmp_path: Path) -> None:
    import sqlite3

    old = tmp_path / "old.sqlite"
    sqlite3.connect(old).close()
    loaded = load([old])
    assert loaded.scored == []
    assert (loaded.unresolved, loaded.annulled) == (0, 0)


def _two_ledgers(tmp_path: Path) -> tuple[Path, Path, int]:
    """A dry-run 0.9 in one ledger; the posted 0.2 and the resolution (No) in another."""
    clock = FixedClock(T0)
    post = binary_post(7001, 8001, opened=OPENED["q1"])
    dry, live = tmp_path / "a-dry.sqlite", tmp_path / "b-live.sqlite"
    with Ledger.open(dry, clock) as ledger:
        run = _run(ledger, RunMode.DRY_RUN, "minibench")
        qid = _post(ledger, run, post)
        _aggregate(ledger, run, qid, {"probability": 0.9})
    with Ledger.open(live, clock) as ledger:
        run = _run(ledger, RunMode.LIVE, "bot-testing-area")
        _post(ledger, run, post)
        _aggregate(ledger, run, qid, {"probability": 0.2})
        ledger.record_submission(
            run_id=run,
            question_id=qid,
            post_id=7001,
            kind="forecast",
            payload=[{}],
            status="submitted",
            http_status=201,
            response_excerpt="",
        )
        _resolve(ledger, qid, "binary", "resolved", "no")
    return dry, live, qid


def test_the_posted_forecast_wins_whatever_the_ledger_order(tmp_path: Path) -> None:
    dry, live, qid = _two_ledgers(tmp_path)
    for paths in ([dry, live], [live, dry]):
        [scored] = load(paths).scored
        assert scored.question_id == qid
        assert scored.augur == {"probability": 0.2}
        assert scored.submitted


def test_a_question_in_several_ledgers_is_counted_once(tmp_path: Path) -> None:
    dry, _live, _qid = _two_ledgers(tmp_path)
    loaded = load([dry, dry])
    assert (len(loaded.scored), loaded.unresolved) == (0, 1)


def test_summarise_binary_numbers_by_hand(world: World) -> None:
    summary = summarise(load([world.path]).scored, resamples=RESAMPLES)
    assert summary.n == 5
    assert summary.by_type == {"binary": 3, "multiple_choice": 1, "numeric": 1}
    assert summary.binary_n == 3
    # Probability put on what happened: q1 0.8, q2 1 - 0.3 = 0.7, q3 1 - 0.6 = 0.4,
    # MC 0.5 (of 3 options), numeric 0.05 (the mass below the range).
    # Baseline: 100 * (ln p + ln N) / ln N with N = 2 for binary:
    #   q1 67.807, q2 48.543, q3 -32.193; MC N = 3: 36.907;
    #   numeric: 100 * (ln 0.05 - ln 0.05) / 2 = 0 (the baseline puts 0.05 beyond an open bound).
    assert b(0.8) == pytest.approx(67.807, abs=1e-3)
    assert b(0.7) == pytest.approx(48.543, abs=1e-3)
    assert b(0.4) == pytest.approx(-32.193, abs=1e-3)
    mc = 100 * (math.log(0.5) + math.log(3)) / math.log(3)
    assert mc == pytest.approx(36.907, abs=1e-3)
    numeric = 0.0
    expected = (b(0.8) + b(0.7) + b(0.4) + mc + numeric) / 5
    assert expected == pytest.approx(24.213, abs=1e-3)
    assert summary.baseline.mean == pytest.approx(expected)
    assert summary.baseline.n == 5
    assert summary.baseline.clusters == 5
    assert summary.baseline.resamples == RESAMPLES
    assert summary.baseline.low <= summary.baseline.mean <= summary.baseline.high
    # Log score: mean of ln p.
    logs = [math.log(p) for p in (0.8, 0.7, 0.4, 0.5, 0.05)]
    assert summary.log.mean == pytest.approx(sum(logs) / 5)
    # Brier on the binary questions only: (0.2^2 + 0.3^2 + 0.6^2) / 3 = 0.49 / 3.
    assert summary.brier is not None
    assert summary.brier.mean == pytest.approx(0.49 / 3)
    assert summary.base_rate == pytest.approx(1 / 3)


def test_head_to_head_against_the_template_on_the_shared_questions(world: World) -> None:
    summary = summarise(load([world.path]).scored, resamples=RESAMPLES)
    [template] = [c for c in summary.comparisons if c.against == "template"]
    assert template.n == 2  # q1 and q3 only
    # q1: 100 ln(0.8 / 0.6) = 28.768; q3 (No): Augur put 0.4, the template 0.5:
    # 100 ln(0.4 / 0.5) = -22.314; the mean is 3.227.
    assert template.head_to_head.mean == pytest.approx(
        (100 * math.log(0.8 / 0.6) + 100 * math.log(0.4 / 0.5)) / 2
    )
    assert template.head_to_head.mean == pytest.approx(3.227, abs=1e-3)
    # Brier: Augur (0.2^2 + 0.6^2) / 2 = 0.2; template (0.4^2 + 0.5^2) / 2 = 0.205.
    assert template.augur_brier == pytest.approx(0.2)
    assert template.other_brier == pytest.approx(0.205)
    assert template.brier_diff is not None
    assert template.brier_diff.mean == pytest.approx(-0.005)
    # Baseline scores on the shared two: Augur (67.807 - 32.193) / 2 = 17.807;
    # template (b(0.6) + b(0.5)) / 2 = (26.303 + 0) / 2.
    assert template.augur_baseline == pytest.approx((b(0.8) + b(0.4)) / 2)
    assert template.other_baseline == pytest.approx((b(0.6) + 0.0) / 2)
    assert template.other_baseline == pytest.approx(13.151, abs=1e-3)


def test_comparisons_against_the_constants(world: World) -> None:
    summary = summarise(load([world.path]).scored, resamples=RESAMPLES)
    assert [c.against for c in summary.comparisons] == [
        "template",
        "always 50%",
        "base rate (33%)",
    ]
    fifty = summary.comparisons[1]
    assert fifty.n == 3  # binary questions only
    # 100 ln(p / 0.5): 47.000, 33.647, -22.314; mean 19.444.
    assert fifty.head_to_head.mean == pytest.approx(
        100 * (math.log(0.8 / 0.5) + math.log(0.7 / 0.5) + math.log(0.4 / 0.5)) / 3
    )
    assert fifty.head_to_head.mean == pytest.approx(19.444, abs=1e-3)
    assert fifty.other_baseline == pytest.approx(0.0, abs=1e-9)
    assert fifty.other_brier == pytest.approx(0.25)
    # Base rate 1/3: on a Yes it puts 1/3, on a No 2/3.
    rate = summary.comparisons[2]
    expected = 100 * (math.log(0.8 / (1 / 3)) + math.log(0.7 / (2 / 3)) + math.log(0.4 / (2 / 3)))
    assert rate.head_to_head.mean == pytest.approx(expected / 3)
    assert rate.other_brier == pytest.approx(((2 / 3) ** 2 + (1 / 3) ** 2 * 2) / 3)


def test_metaculus_checks_pick_the_baseline_scores_of_this_account(world: World) -> None:
    summary = summarise(load([world.path]).scored, resamples=RESAMPLES)
    q1 = world.ids["q1"]
    # No peer_score, and not the community's 99.0 on q2.
    assert [(c[0], c[1], c[3]) for c in summary.metaculus_checks] == [
        (q1, "baseline_score", 70.0),
        (q1, "spot_baseline_score", 65.0),
    ]
    assert all(c[2] == pytest.approx(b(0.8)) for c in summary.metaculus_checks)


def test_cluster_by_week_groups_questions_opened_in_the_same_iso_week(world: World) -> None:
    scored = load([world.path]).scored
    by_question = summarise(scored, resamples=RESAMPLES)
    by_week = summarise(scored, cluster="week", resamples=RESAMPLES)
    # q1 and q2 share week 45; q3 is week 46, the MC question week 42, the numeric week 18.
    assert by_question.baseline.clusters == 5
    assert by_week.baseline.clusters == 4
    assert by_week.cluster == "week"
    assert by_week.baseline.mean == by_question.baseline.mean  # same point estimate
    by_source = summarise(scored, cluster="source", resamples=RESAMPLES)
    assert by_source.baseline.clusters == 2  # bot-testing-area (q1) and minibench


def test_cluster_labels() -> None:
    s = ScoredQuestion(
        question_id=1,
        post_id=1,
        question_type="binary",
        title="t",
        opened=datetime(2026, 1, 1, 12, tzinfo=UTC),  # Thursday of ISO week 1, 2026
        resolution="yes",
        source="minibench",
        submitted=False,
        model="m",
        augur={"probability": 0.5},
        references={},
    )
    assert s.cluster("week") == "2026-W01"
    assert s.cluster("source") == "minibench"
    assert s.cluster("question") == 1
    # ISO years differ from calendar years at the turn of the year.
    dec = ScoredQuestion(**{**s.__dict__, "opened": datetime(2026, 12, 31, tzinfo=UTC)})
    jan = ScoredQuestion(**{**s.__dict__, "opened": datetime(2027, 1, 1, tzinfo=UTC)})
    assert dec.cluster("week") == jan.cluster("week") == "2026-W53"


def test_summarise_with_nothing_scored_raises() -> None:
    with pytest.raises(ScoringError, match="no resolved question"):
        summarise([])


def test_a_summary_without_binary_questions_has_no_brier_or_constants(world: World) -> None:
    scored = [s for s in load([world.path]).scored if s.question_type != "binary"]
    summary = summarise(scored, resamples=RESAMPLES)
    assert summary.brier is None
    assert summary.base_rate is None
    assert summary.comparisons == []


def test_format_summary_reports_the_counts_and_the_comparisons(world: World) -> None:
    loaded = load([world.path])
    text = format_summary(summarise(loaded.scored, resamples=RESAMPLES), loaded)
    first = text.splitlines()[0]
    assert first.startswith("5 scored questions (binary 3, multiple_choice 1, numeric 1); ")
    assert "1 without a resolution, 1 annulled" in first
    assert "300 resamples" in first
    assert "  vs template (n = 2)" in text
    assert "  vs always 50% (n = 3)" in text
    assert "  vs base rate (33%) (n = 3)" in text
    assert "check " in text


def test_references_and_resolutions_pool_across_ledgers(world: World, tmp_path: Path) -> None:
    """A template forecast kept in its own ledger pairs with Augur's on the same question."""
    other = tmp_path / "shadow.sqlite"
    with Ledger.open(other, FixedClock(T0)) as ledger:
        _reference(ledger, world.ids["q2"], "binary", "template", {"probability": 0.5})
    alone = {s.question_id: s for s in load([world.path]).scored}
    pooled = {s.question_id: s for s in load([world.path, other]).scored}
    assert "template" not in alone[world.ids["q2"]].references
    assert pooled[world.ids["q2"]].references["template"] == {"probability": 0.5}
    comparison = next(
        c
        for c in summarise(list(pooled.values()), resamples=200).comparisons
        if c.against == "template"
    )
    assert comparison.n == 3
    # A ledger holding only references adds no scored question of its own.
    assert len(load([other]).scored) == 0


def test_a_ledger_from_before_values_json_still_contributes_its_forecast() -> None:
    # Schema 1 kept a binary forecast as a bare probability column.
    old = Path(__file__).resolve().parent.parent / "data/ledgers/m1-live-bot-testing-area.sqlite"
    loaded = load([old])
    assert (len(loaded.scored), loaded.unresolved, loaded.unscorable) == (0, 1, [])


def test_a_bad_reference_is_listed_not_fatal_and_a_later_one_is_flagged(
    world: World, tmp_path: Path
) -> None:
    """One unscorable reference must not stop the scorecard, and a reference
    dated after Augur's forecast is counted, since it knew more."""
    other = tmp_path / "shadow.sqlite"
    with Ledger.open(other, FixedClock(T0 + timedelta(days=1))) as ledger:
        ledger.record_reference(
            question_id=world.ids["q2"],
            post_id=world.ids["q2"],
            question_type="binary",
            forecaster="template",
            detail="d",
            values={"probability": 0.5},
            as_of=T0 + timedelta(hours=3),  # Augur forecast q2 at T0 + 61 minutes
            source="s",
        )
        # Replaces q3's good template row (later recorded_at) with one that has no probability.
        _reference(ledger, world.ids["q3"], "binary", "template", {"p": 0.5})
    loaded = load([world.path, other])
    assert any(f"{world.ids['q3']} (template)" in u for u in loaded.unscorable)
    by_id = {s.question_id: s for s in loaded.scored}
    assert "template" not in by_id[world.ids["q3"]].references
    summary = summarise(loaded.scored, resamples=RESAMPLES)
    template = next(c for c in summary.comparisons if c.against == "template")
    assert (template.n, template.later) == (2, 1)  # q1 (as of T0, not later) and q2
    assert "1 of 2 dated after Augur's forecast" in format_summary(summary, loaded)
