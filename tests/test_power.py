"""The planted-effect power simulation."""

from __future__ import annotations

import pytest

from augur.power import Curve, analytic_n, simulate

SPREAD = [(-1.0) ** i * (0.05 + 0.01 * (i % 7)) for i in range(60)]


def test_no_planted_effect_alarms_about_one_time_in_forty() -> None:
    (curve,) = simulate(SPREAD, [0.0], grid=(50, 200), sims=2000, seed=1)
    assert all(0.01 <= r <= 0.05 for r in curve.rates)


def test_a_large_effect_is_always_found_and_power_grows_with_n() -> None:
    small, large = simulate(SPREAD, [0.005, 0.05], grid=(25, 100, 400), sims=400, seed=2)
    assert large.rates[-1] == 1.0
    assert small.rates[0] < small.rates[1] < small.rates[2]


def test_the_same_seed_gives_the_same_curves() -> None:
    a = simulate(SPREAD, [0.01], grid=(30,), sims=200, seed=3)
    b = simulate(SPREAD, [0.01], grid=(30,), sims=200, seed=3)
    assert a == b


def test_the_simulation_agrees_with_the_textbook_n() -> None:
    sd = (sum(x * x for x in SPREAD) / len(SPREAD) - (sum(SPREAD) / len(SPREAD)) ** 2) ** 0.5
    want = analytic_n(sd, 0.01)
    grid = tuple(sorted({max(2, want // 4), max(2, want // 2), want, want * 2, want * 4}))
    (curve,) = simulate(SPREAD, [0.01], grid=grid, sims=600, seed=4)
    needed = curve.needed()
    assert needed is not None
    assert 0.6 * want <= needed <= 1.6 * want


def test_needed_interpolates_in_log_n_and_says_none_when_never_reached() -> None:
    curve = Curve(0.01, (0.2, 0.6, 1.0), (100, 200, 400))
    assert curve.needed() == 283  # halfway from 200 to 400 in log n: √(200 × 400)
    assert Curve(0.01, (0.1, 0.2), (10, 20)).needed() is None
    assert Curve(0.01, (0.9,), (10,)).needed() == 10


def test_analytic_n() -> None:
    assert analytic_n(0.1, 0.01) == 785  # (2.8016 × 10)² rounded up


def test_needs_two_values() -> None:
    with pytest.raises(ValueError, match="at least two"):
        simulate([0.1], [0.0])
