"""M12's statistical arm: base rates, the series rule, both blends and the C4 harness."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date, timedelta

import pytest

from augur import stateval
from augur.distributions import Scaling, validate_cdf
from augur.errors import ConfigError
from augur.predictions import BinaryPrediction, ContinuousPrediction
from augur.q4 import Instance
from augur.replay import blend
from augur.report import REPO
from augur.statarm import (
    MIN_CHANGES,
    Abstain,
    ArmForecast,
    BaseRates,
    base_rate_blend,
    binary_arm,
    changes,
    empirical_quantile,
    horizon,
    load_base_rates,
    primary,
    series_arm,
)


def _scaling(lo: float = 50.0, hi: float = 200.0, n: int = 200) -> Scaling:
    return Scaling(
        range_min=lo,
        range_max=hi,
        zero_point=None,
        open_lower_bound=True,
        open_upper_bound=True,
        inbound_outcome_count=n,
    )


def _weekday_closes(
    days: int, start: float = 100.0, drift: float = 0.0
) -> list[tuple[date, float]]:
    """Closes on weekdays only, with a deterministic zig-zag so changes have spread."""
    out: list[tuple[date, float]] = []
    d = date(2024, 11, 1)
    v = start
    k = 0
    while len(out) < days:
        if d.weekday() < 5:
            v *= math.exp(drift + (0.01 if k % 3 else -0.015))
            out.append((d, v))
            k += 1
        d += timedelta(days=1)
    return out


# -- base rates -----------------------------------------------------------------------


def test_base_rate_is_laplace_smoothed_per_source() -> None:
    rates = BaseRates.fit([("metaculus", 1)] * 5 + [("metaculus", 0)] * 20)
    p, detail = rates.rate("metaculus")
    assert p == pytest.approx(6 / 27)
    assert "5 of 25" in detail


def test_small_source_falls_back_to_the_pooled_rate() -> None:
    rates = BaseRates.fit([("polymarket", 1)] * 30 + [("infer", 0)] * 3)
    p, detail = rates.rate("infer")
    assert p == pytest.approx(31 / 35)
    assert "pooled" in detail


def test_binary_arm_prefers_the_price() -> None:
    rates = BaseRates.fit([("metaculus", 0)] * 30)
    arm = binary_arm(0.7, "metaculus", rates)
    assert isinstance(arm, ArmForecast)
    assert (arm.kind, arm.probability) == ("price", 0.7)
    arm = binary_arm(None, "metaculus", rates)
    assert isinstance(arm, ArmForecast)
    assert arm.kind == "base-rate"
    assert isinstance(binary_arm(None, "metaculus", BaseRates({})), Abstain)


def test_committed_base_rates_load() -> None:
    rates = load_base_rates()
    assert rates.counts["metaculus"][1] >= 20
    bad = REPO / "data" / "nonexistent-base-rates.json"
    with pytest.raises(ConfigError, match="augur statarm"):
        load_base_rates(bad)


# -- the series rule ----------------------------------------------------------------


def test_horizon_follows_the_series_own_rate() -> None:
    weekdays = _weekday_closes(400)
    last = weekdays[-1][0]
    # About 261 weekdays a year: a week ahead is about five observations.
    assert horizon(weekdays, last + timedelta(days=7)) == 5
    daily = [(date(2025, 1, 1) + timedelta(days=k), 1.0) for k in range(400)]
    assert horizon(daily, daily[-1][0] + timedelta(days=7)) == 7
    assert horizon(daily, daily[-1][0] + timedelta(days=0)) == 1


def test_changes_are_overlapping_log_ratios() -> None:
    closes = [(date(2026, 1, 1) + timedelta(days=k), v) for k, v in enumerate([1.0, 2.0, 4.0])]
    assert changes(closes, 1) == pytest.approx([math.log(2), math.log(2)])
    assert changes(closes, 2) == pytest.approx([math.log(4)])


def test_empirical_quantile_interpolates() -> None:
    assert empirical_quantile([0.0, 1.0, 2.0, 3.0], 0.5) == pytest.approx(1.5)
    assert empirical_quantile([0.0, 1.0], 0.0) == 0.0
    assert empirical_quantile([0.0, 1.0], 1.0) == 1.0


def test_series_arm_builds_a_valid_cdf_around_the_last_close() -> None:
    closes = _weekday_closes(500)
    last_day, last = closes[-1]
    scaling = _scaling(last * 0.5, last * 1.5)
    arm = series_arm(closes, last_day + timedelta(days=3), scaling, "TEST")
    assert isinstance(arm, ArmForecast)
    assert arm.kind == "series"
    assert arm.cdf is not None
    assert validate_cdf(list(arm.cdf), scaling) == []
    median = dict(arm.percentiles)[0.4]
    assert last * 0.9 < median < last * 1.1


def test_series_arm_abstains_on_short_history_and_unit_mismatch() -> None:
    short = _weekday_closes(MIN_CHANGES // 2)
    last_day, last = short[-1]
    arm = series_arm(short, last_day + timedelta(days=3), _scaling(last * 0.5, last * 2), "S")
    assert isinstance(arm, Abstain)
    assert "fewer than" in arm.reason
    long = _weekday_closes(500)
    last_day, last = long[-1]
    arm = series_arm(long, last_day + timedelta(days=3), _scaling(last * 10, last * 20), "S")
    assert isinstance(arm, Abstain)
    assert "unit mismatch" in arm.reason
    arm = series_arm(long, last_day, _scaling(last * 0.5, last * 2), "S")
    assert isinstance(arm, Abstain)
    assert isinstance(series_arm([], last_day, _scaling(), "S"), Abstain)


# -- the blends -----------------------------------------------------------------------


def test_primary_blends_only_with_a_price_or_a_series() -> None:
    augur = BinaryPrediction(0.8)
    price = ArmForecast("price", "p", probability=0.2)
    rate = ArmForecast("base-rate", "r", probability=0.2)
    blended, method = primary(augur, price)
    assert isinstance(blended, BinaryPrediction)
    assert blended.probability == pytest.approx(round(blend(0.8, 0.2), 4))
    assert "two-thirds price" in method
    same, method = primary(augur, rate)
    assert same == augur
    assert "unchanged" in method
    same, _ = primary(augur, Abstain("nothing"))
    assert same == augur
    brb, _ = base_rate_blend(augur, rate)
    assert isinstance(brb, BinaryPrediction)
    assert brb.probability == pytest.approx(round(blend(0.8, 0.2), 4))


def test_primary_mixes_cdfs_one_third_augur() -> None:
    closes = _weekday_closes(500)
    last_day, last = closes[-1]
    scaling = _scaling(last * 0.5, last * 1.5)
    arm = series_arm(closes, last_day + timedelta(days=3), scaling, "S")
    assert isinstance(arm, ArmForecast)
    assert arm.cdf is not None
    n = scaling.inbound_outcome_count
    flat = tuple(0.001 + 0.998 * i / n for i in range(n + 1))
    augur = ContinuousPrediction(flat, ())
    mixed, _ = primary(augur, arm, scaling)
    assert isinstance(mixed, ContinuousPrediction)
    assert validate_cdf(list(mixed.cdf), scaling) == []
    k = n // 2
    assert mixed.cdf[k] == pytest.approx(flat[k] / 3 + 2 * arm.cdf[k] / 3, abs=1e-9)


# -- the C4 harness -------------------------------------------------------------------


def _inst(key: str, source: str, opened: date, outcome: int, p: float, price: float) -> Instance:
    return Instance(
        key=key,
        market=f"m:{key}",
        source=source,
        question_set=opened.isoformat(),
        opened=opened,
        resolved=opened + timedelta(days=5),
        known=opened + timedelta(days=5),
        outcome=outcome,
        p=p,
        price=price,
    )


def test_rows_use_only_earlier_outcomes_and_hide_metaculus_prices() -> None:
    d0 = date(2026, 7, 1)
    early = [_inst(f"e{k}", "metaculus", d0, k % 2, 0.5, 0.9) for k in range(30)]
    late_day = d0 + timedelta(days=30)
    late = [
        _inst("m", "metaculus", late_day, 1, 0.6, 0.9),
        _inst("p", "polymarket", late_day, 1, 0.6, 0.9),
    ]
    rs = stateval.rows(early + late, min_history=20)
    assert [r.instance.key for r in rs] == ["m", "p"]
    m, p = rs
    assert m.arm_kind == "base-rate"
    assert m.forecasts["arm"] == pytest.approx(16 / 32)
    assert "price" not in m.forecasts
    assert m.forecasts["candidate"] == m.forecasts["augur"]
    assert p.arm_kind == "price"
    assert p.forecasts["price"] == 0.9
    assert p.forecasts["candidate"] == pytest.approx(round(blend(0.6, 0.9), 4))


def test_rows_never_fit_on_an_outcome_public_after_the_open_day() -> None:
    d0 = date(2026, 7, 1)
    test_day = d0 + timedelta(days=30)
    # Opened earlier, but their outcomes became public only after the test day.
    pending = [
        replace(_inst(f"e{k}", "metaculus", d0, 1, 0.5, 0.5), known=test_day + timedelta(days=1))
        for k in range(30)
    ]
    target = _inst("t", "metaculus", test_day, 1, 0.5, 0.5)
    assert stateval.rows([*pending, target], min_history=1) == []
    public = [replace(i, known=test_day - timedelta(days=1)) for i in pending]
    (row,) = stateval.rows([*public, target], min_history=20)
    assert row.forecasts["arm"] == pytest.approx(31 / 32)


@pytest.mark.skipif(
    not (REPO / stateval.SNAPSHOT).exists() or not (REPO / "data/ledgers/m8-c4.sqlite").exists(),
    reason="augur statarm has not been run, or the C4 ledger is absent (an sdist)",
)
def test_committed_snapshot_matches_a_fresh_run() -> None:
    written = stateval.files(REPO)
    for path, text in written.items():
        assert path.read_text(encoding="utf-8") == text, f"{path} is stale; run augur statarm"
    snap = json.loads(written[REPO / stateval.SNAPSHOT])
    assert [h["id"] for h in snap["hypotheses"]] == ["P34", "P35", "P36", "P37"]
    groups = {g["group"]: g for g in snap["groups"]}
    assert groups["priced"]["n"] + groups["unpriced"]["n"] == groups["all"]["n"]
    assert set(groups["metaculus"]["arm_kinds"]) == {"base-rate"}
