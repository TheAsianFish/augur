"""Q4's harness: the time order, the measures, the live choice and the committed snapshot."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from datetime import date, timedelta

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


def test_check_time_order_refuses_a_later_resolution() -> None:
    train = [_inst("x", D0, D0 + timedelta(days=5), 1, 0.3)]
    q4.check_time_order(train, D0 + timedelta(days=6))
    with pytest.raises(LeakageError, match="resolved on or after"):
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
    assert all(f["n"] >= q4.W_MIN for f in snap["walk_forward"]["fits"])
    assert {f["n"] for f in snap["learning_curve"]["fits"]} == {q4.KS[-1]}
