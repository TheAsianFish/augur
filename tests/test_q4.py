"""Q4's harness: the time order, the measures, the live choice and the committed snapshot."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import pytest

from augur import q4
from augur.calibrate import CALIBRATORS
from augur.errors import LeakageError
from augur.report import REPO

D0 = date(2026, 7, 1)


def _inst(key: str, opened: date, resolved: date, outcome: int, p: float) -> q4.Instance:
    return q4.Instance(
        key=key,
        market=f"m:{key}",
        source="polymarket",
        question_set=opened.isoformat(),
        opened=opened,
        resolved=resolved,
        known=resolved,
        outcome=outcome,
        p=p,
        price=0.5,
    )


def _corpus(seed: int = 3, days: int = 6, per_day: int = 40) -> list[q4.Instance]:
    rng = random.Random(seed)
    out: list[q4.Instance] = []
    for d in range(days):
        opened = D0 + timedelta(days=14 * d)
        for k in range(per_day):
            p = rng.uniform(0.02, 0.98)
            truth = 0.3 + 0.4 * p
            out.append(
                _inst(
                    f"{d}:{k}",
                    opened,
                    opened + timedelta(days=rng.randint(1, 40)),
                    int(rng.random() < truth),
                    p,
                )
            )
    return out


def test_history_is_strictly_before_the_open_day() -> None:
    a = _inst("a", D0, D0 + timedelta(days=3), 1, 0.4)
    same_day = _inst("b", D0, D0 + timedelta(days=10), 0, 0.4)
    later = _inst("c", D0 + timedelta(days=10), D0 + timedelta(days=20), 0, 0.6)
    assert [i.key for i in q4.history([a, same_day, later], D0 + timedelta(days=10))] == ["a"]


def test_history_uses_the_market_s_own_resolution_time() -> None:
    # ForecastBench says it resolved on day 3 (the close); the market says day 20.
    late = replace(_inst("late", D0, D0 + timedelta(days=3), 1, 0.4), known=D0 + timedelta(days=20))
    unknown = replace(_inst("infer", D0, D0 + timedelta(days=3), 0, 0.4), known=None)
    opened = D0 + timedelta(days=10)
    assert q4.history([late, unknown], opened) == []
    with pytest.raises(LeakageError):
        q4.check_time_order([late], opened)
    with pytest.raises(LeakageError):
        q4.check_time_order([unknown], opened)
    assert q4.history([late], D0 + timedelta(days=21)) == [late]


def test_check_time_order_refuses_a_later_resolution() -> None:
    train = [_inst("x", D0, D0 + timedelta(days=5), 1, 0.3)]
    q4.check_time_order(train, D0 + timedelta(days=6))
    with pytest.raises(LeakageError, match="not shown to be public before"):
        q4.check_time_order(train, D0 + timedelta(days=5))


def test_walk_forward_refuses_a_history_rule_that_leaks() -> None:
    corpus = _corpus()

    def leaky(instances: Sequence[q4.Instance], opened: date) -> list[q4.Instance]:
        return list(instances)  # every outcome, including later ones

    with pytest.raises(LeakageError):
        q4.walk_forward(corpus, min_history=1, select=leaky)


def test_every_fit_sees_only_earlier_resolutions() -> None:
    corpus = _corpus()
    c = q4.walk_forward(corpus, min_history=20)
    assert c.fits
    for f in c.fits:
        assert f.latest_resolution < f.opened
        assert f.n == len(q4.history(corpus, f.opened))
    assert len(c.tests) == sum(f.tests for f in c.fits)
    assert all(len(v) == len(c.tests) for v in c.forecasts.values())


def test_window_keeps_the_most_recent_resolutions() -> None:
    corpus = _corpus()
    c = q4.walk_forward(corpus, min_history=100, window=25)
    for f in c.fits:
        recent = q4.history(corpus, f.opened)[-25:]
        assert f.n == 25
        assert f.latest_resolution == max(i.resolved for i in recent)


def test_days_below_the_minimum_are_not_tested() -> None:
    corpus = _corpus()
    c = q4.walk_forward(corpus, min_history=10_000)
    assert c.tests == []
    assert c.fits == []


def test_raw_forecasts_pass_through() -> None:
    corpus = _corpus()
    c = q4.walk_forward(corpus, CALIBRATORS, min_history=20)
    assert c.forecasts["raw"] == [min(max(i.p, 0.01), 0.99) for i in c.tests]


def test_equal_mass_bins_and_ece_by_hand() -> None:
    ps = [0.1, 0.2, 0.3, 0.4]
    ys = [0, 0, 1, 1]
    bins = q4.equal_mass_bins(ps, ys, bins=2)
    assert [(b.n, round(b.mean_p, 6), b.freq) for b in bins] == [(2, 0.15, 0.0), (2, 0.35, 1.0)]
    assert q4.ece(ps, ys, bins=2) == pytest.approx(0.5 * 0.15 + 0.5 * 0.65)
    assert q4.ece([0.5, 0.5], [0, 1], bins=1) == pytest.approx(0.0)
    # Sizes differ by at most one.
    assert [b.n for b in q4.equal_mass_bins([0.1] * 7, [0] * 7, bins=3)] == [2, 2, 3]


def test_sharpness_is_zero_for_a_constant_forecaster() -> None:
    assert q4.sharpness([0.3, 0.3, 0.3]) == 0.0
    assert q4.sharpness([0.0, 1.0]) == pytest.approx(0.5)


def _snap(rows: list[tuple[str, float, float]]) -> dict[str, object]:
    return {
        "walk_forward": {
            "rows": [
                {"calibrator": n, "vs_raw_brier": {"mean": m, "low": m - 0.01, "high": h}}
                for n, m, h in rows
            ]
        }
    }


def test_live_choice_follows_the_replacement_rule() -> None:
    assert q4.live_choice(_snap([("raw", 0.0, 0.0), ("platt", -0.01, 0.002)])) == "raw"
    assert q4.live_choice(_snap([("platt", -0.02, -0.01), ("isotonic", -0.03, -0.001)])) == (
        "isotonic"
    )
    assert q4.verdict({"high": -0.001}).startswith("replaces")
    assert q4.verdict({"high": 0.0}) == "raw stays"


# -- the committed snapshot ---------------------------------------------------


SNAP = REPO / q4.SNAPSHOT


@pytest.mark.skipif(not SNAP.exists(), reason="augur calibrate has not been run")
def test_committed_snapshot_respects_the_time_order() -> None:
    snap = json.loads(SNAP.read_text(encoding="utf-8"))
    for design in ("walk_forward", "learning_curve"):
        fits = snap[design]["fits"]
        assert fits
        for f in fits:
            assert f["latest_resolution"] < f["opened"]
    evidence = snap["corpus"]["resolution_evidence"]
    assert evidence["markets"] + evidence["none"] == snap["corpus"]["markets"]
    assert all(f["n"] >= q4.W_MIN for f in snap["walk_forward"]["fits"])
    assert {f["n"] for f in snap["learning_curve"]["fits"]} == {q4.KS[-1]}


def test_edge_reports_the_calibrator_s_effect_within_each_price_group() -> None:
    # A map that helps the readable group and hurts the hidden one must show
    # both signs: a pooled improvement can hide a group it makes worse.
    readable = [replace(_inst(f"r{k}", D0, D0, k % 2, 0.5), source="manifold") for k in range(4)]
    hidden = [replace(_inst(f"h{k}", D0, D0, 0, 0.1), source="metaculus") for k in range(4)]
    tests = readable + hidden
    raw = [i.p for i in tests]
    cal = [1.0 if i.outcome else 0.0 for i in readable] + [0.4] * 4
    rows = {e["group"]: e for e in q4._edge("platt", cal, raw, tests)}
    assert rows["readable"]["vs_raw_brier"]["mean"] < 0
    assert rows["hidden"]["vs_raw_brier"]["mean"] > 0
    assert rows["hidden"]["base_rate"] == 0.0
    assert rows["readable"]["base_rate"] == 0.5


# -- source-aware maps ----------------------------------------------------------


def _sourced_corpus() -> list[q4.Instance]:
    sources = ("polymarket", "metaculus", "manifold")
    return [replace(i, source=sources[n % 3]) for n, i in enumerate(_corpus())]


def test_sourced_walk_forward_tests_the_same_instances_in_the_same_order() -> None:
    corpus = _sourced_corpus()
    base = q4.walk_forward(corpus, min_history=40)
    sourced = q4.walk_forward_sourced(corpus, min_history=40)
    assert sourced.fits
    assert [i.key for i in sourced.tests] == [i.key for i in base.tests]
    assert [(f.opened, f.n, f.latest_resolution, f.tests) for f in sourced.fits] == [
        (f.opened, f.n, f.latest_resolution, f.tests) for f in base.fits
    ]
    assert sourced.forecasts["raw"] == [min(max(i.p, 0.01), 0.99) for i in sourced.tests]
    assert set(sourced.forecasts) == {"raw", "platt", "platt-by-source", "platt-partial"}
    assert all(len(v) == len(sourced.tests) for v in sourced.forecasts.values())
    # the pooled arm reproduces M8's Platt exactly
    assert sourced.forecasts["platt"] == base.forecasts["platt"]


def test_sourced_walk_forward_fits_see_only_earlier_outcomes() -> None:
    corpus = _sourced_corpus()
    c = q4.walk_forward_sourced(corpus, min_history=40)
    for f in c.fits:
        assert f.latest_resolution < f.opened
        assert f.n == len(q4.history(corpus, f.opened))
        assert f.params["platt-partial"]["n"] == f.n


def test_sourced_walk_forward_skips_short_histories() -> None:
    c = q4.walk_forward_sourced(_sourced_corpus(), min_history=10_000)
    assert c.tests == []
    assert c.fits == []


def test_sourced_walk_forward_refuses_a_leaking_history(monkeypatch: pytest.MonkeyPatch) -> None:
    def leaky(instances: Sequence[q4.Instance], opened: date) -> list[q4.Instance]:
        return list(instances)

    monkeypatch.setattr(q4, "history", leaky)
    with pytest.raises(LeakageError):
        q4.walk_forward_sourced(_sourced_corpus(), min_history=1)


def test_sourced_walk_forward_refuses_an_unresolved_training_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = [replace(i, known=None) if n == 0 else i for n, i in enumerate(_sourced_corpus())]

    def by_open_day(instances: Sequence[q4.Instance], opened: date) -> list[q4.Instance]:
        return [i for i in instances if i.opened < opened]

    monkeypatch.setattr(q4, "history", by_open_day)
    with pytest.raises(LeakageError):
        q4.walk_forward_sourced(corpus, min_history=1)


def _source_snap(**per_map: tuple[float, float, float, float]) -> dict[str, Any]:
    """Each map: (all vs raw mean, metaculus vs raw high, all vs platt high, ...)."""
    allrows = [
        {
            "calibrator": n,
            "vs_raw_brier": {"mean": v[0], "high": 0.0},
            "vs_platt_brier": {"mean": 0.0, "high": v[2]},
        }
        for n, v in per_map.items()
    ]
    metrows = [
        {"calibrator": n, "vs_raw_brier": {"mean": 0.0, "high": v[1]}, "vs_platt_brier": {}}
        for n, v in per_map.items()
    ]
    base = [{"calibrator": "raw"}, {"calibrator": "platt"}]
    return {
        "by_source": [
            {"group": "all", "rows": base + allrows},
            {"group": "metaculus", "rows": base + metrows},
        ]
    }


def test_source_choice_needs_both_bounds_below_the_margin() -> None:
    # (all-vs-raw mean, metaculus-vs-raw upper, all-vs-platt upper, unused)
    assert q4.source_choice(_source_snap(a=(-0.02, 0.005, 0.005, 0.0))) == "a"
    assert q4.source_choice(_source_snap(a=(-0.02, 0.01, 0.005, 0.0))) is None  # not < 0.01
    assert q4.source_choice(_source_snap(a=(-0.02, 0.005, 0.01, 0.0))) is None
    assert q4.source_choice(_source_snap(a=(-0.02, 0.02, 0.0, 0.0))) is None
    assert q4.source_choice(_source_snap(a=(-0.02, 0.0, 0.03, 0.0))) is None
    assert q4.source_choice(_source_snap()) is None


def test_source_choice_picks_the_lowest_overall_point_estimate_among_passers() -> None:
    snap = _source_snap(
        failing=(-0.09, 0.05, 0.0, 0.0),  # best overall but fails the Metaculus bound
        worse=(-0.01, 0.0, 0.0, 0.0),
        best=(-0.03, 0.0, 0.0, 0.0),
    )
    assert q4.source_choice(snap) == "best"
