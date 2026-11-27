"""M18's arms S, S', K and the calendar they read, against PLAN.md's m18-registration."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from augur import fbseason as fs
from augur.fbdataset import DQuestion

FREEZE = datetime(2026, 9, 3, 0, 0, tzinfo=UTC)  # last admitted day: 2 September
DUE = date(2026, 9, 13)
R = date(2026, 9, 20)


def _q(freeze: datetime = FREEZE) -> DQuestion:
    return DQuestion(
        question_set="2026-09-13",
        source="dbnomics",
        id="meteofrance_TEMPERATURE_celsius.07005.D",
        freeze=freeze,
        freeze_value=15.0,
        url="",
        resolutions=((R, 1),),
    )


def test_shift_moves_the_year_and_maps_29_february_to_28() -> None:
    assert fs.shift(date(2026, 9, 13), 1) == date(2025, 9, 13)
    assert fs.shift(date(2028, 2, 29), 1) == date(2027, 2, 28)
    assert fs.shift(date(2028, 2, 29), 4) == date(2024, 2, 29)
    assert fs.shift(date(2028, 2, 29), 8) == date(2020, 2, 29)
    assert fs.shift(date(2032, 2, 29), 3) == date(2029, 2, 28)


def test_window_has_two_w_plus_one_days_centred_on_the_shifted_day() -> None:
    w = fs.window(DUE, 3, 2)
    assert len(w) == 7
    assert w[3] == date(2024, 9, 13)
    assert w[0] == date(2024, 9, 10)
    assert w[-1] == date(2024, 9, 16)
    assert len(fs.window(DUE, 10, 1)) == 21


def test_last_admitted_day_is_the_day_before_the_freeze_date() -> None:
    assert fs.last_admitted_day(FREEZE) == date(2026, 9, 2)
    assert fs.last_admitted_day(datetime(2026, 9, 3, 15, tzinfo=UTC)) == date(2026, 9, 2)


def _full(days: list[date], v: float) -> dict[date, float]:
    return {d: v for d in days}


def test_usable_shifts_needs_full_admission_and_four_of_seven_days() -> None:
    obs: dict[date, float] = {}
    for k in (1, 2):
        obs.update(_full(fs.window(R, 3, k), 20.0))
        obs.update(_full(fs.window(DUE, 3, k), 10.0))
    got = fs.usable_shifts(obs, DUE, R, FREEZE)
    assert [s.k for s in got] == [1, 2]
    assert got[0].m_r == 20.0
    assert got[0].m_d == 10.0

    # k=3 with exactly 4 days in each window is usable, 3 days in one is not
    for d in fs.window(R, 3, 3)[:4]:
        obs[d] = 1.0
    for d in fs.window(DUE, 3, 3)[:3]:
        obs[d] = 1.0
    assert [s.k for s in fs.usable_shifts(obs, DUE, R, FREEZE)] == [1, 2]
    obs[fs.window(DUE, 3, 3)[3]] = 1.0
    assert [s.k for s in fs.usable_shifts(obs, DUE, R, FREEZE)] == [1, 2, 3]


def test_usable_shifts_drops_a_shift_whose_window_ends_after_the_last_admitted_day() -> None:
    # Freeze early: last admitted day is 2025-09-19, so k=1's r window (ends 23 Sep 2025) is out.
    freeze = datetime(2025, 9, 20, tzinfo=UTC)
    obs: dict[date, float] = {}
    for k in (1, 2):
        obs.update(_full(fs.window(R, 3, k), 20.0))
        obs.update(_full(fs.window(DUE, 3, k), 10.0))
    assert [s.k for s in fs.usable_shifts(obs, DUE, R, freeze)] == [2]
    # last admitted day equal to the window's last day keeps it
    freeze_ok = datetime(2025, 9, 24, tzinfo=UTC)
    assert [s.k for s in fs.usable_shifts(obs, DUE, R, freeze_ok)] == [1, 2]
    assert fs.usable_shifts(obs, DUE, R, datetime(2025, 9, 23, tzinfo=UTC)) != []
    assert [s.k for s in fs.usable_shifts(obs, DUE, R, datetime(2025, 9, 23, tzinfo=UTC))] == [2]


def _shifts(*pairs: tuple[float, float]) -> list[fs.Shift]:
    return [fs.Shift(i + 1, r, d) for i, (r, d) in enumerate(pairs)]


def test_arm_s_is_laplace_smoothed_wins_and_ties_are_not_wins() -> None:
    assert fs.arm_s(_shifts((2, 1), (3, 1), (1, 2)), 0.42) == pytest.approx(0.6)
    assert fs.arm_s(_shifts((1, 1)), 0.42) == pytest.approx(1 / 3)  # a tie is not a win
    assert fs.arm_s(_shifts((2, 1)), 0.42) == pytest.approx(2 / 3)


def test_arm_s_falls_back_to_b_without_a_shift() -> None:
    assert fs.arm_s([], 0.42) == 0.42


def _obs_around_r() -> dict[date, float]:
    # r's 10-day windows in 2025 and 2024: values 1..21 on the first, 100 on the second
    obs = {d: float(i + 1) for i, d in enumerate(fs.window(R, 10, 1))}
    obs.update({d: 100.0 for d in fs.window(R, 10, 2)})
    return obs


def test_exceedance_counts_ties_as_not_above() -> None:
    obs = _obs_around_r()
    e = fs.exceedance(obs, R, FREEZE, 11.0)
    assert e == fs.Exceedance(above=10 + 21, n=42)  # 12..21 above in 2025, all of 2024
    tie_only = {fs.window(R, 10, 1)[0]: 5.0}
    assert fs.exceedance(tie_only, R, FREEZE, 5.0) == fs.Exceedance(0, 1)


def test_exceedance_ignores_days_after_the_last_admitted_day() -> None:
    obs = _obs_around_r()
    obs[date(2026, 9, 15)] = 999.0  # a k=0 day: never read
    obs[date(2026, 9, 2)] = 999.0
    e = fs.exceedance(obs, R, FREEZE, 0.0)
    assert e is not None
    assert e.n == 42
    # a freeze that excludes 2025 leaves only 2024
    early = datetime(2025, 1, 1, tzinfo=UTC)
    e2 = fs.exceedance(obs, R, early, 0.0)
    assert e2 == fs.Exceedance(21, 21)


def test_exceedance_is_none_without_a_freeze_value_or_observations() -> None:
    assert fs.exceedance(_obs_around_r(), R, FREEZE, None) is None
    assert fs.exceedance({}, R, FREEZE, 10.0) is None


def test_arm_s_prime_is_laplace_smoothed_and_falls_back_to_b() -> None:
    assert fs.arm_s_prime(fs.Exceedance(3, 8), 0.4) == pytest.approx(0.4)
    assert fs.arm_s_prime(fs.Exceedance(0, 0), 0.4) == pytest.approx(0.5)
    assert fs.arm_s_prime(fs.Exceedance(0, 8), 0.4) == pytest.approx(0.1)
    assert fs.arm_s_prime(None, 0.37) == 0.37
    obs = _obs_around_r()
    assert fs.arm_s_prime(fs.exceedance(obs, R, FREEZE, None), 0.37) == 0.37


def test_station_change_is_the_mean_of_window_mean_differences() -> None:
    assert fs.station_change([]) is None
    assert fs.station_change(_shifts((3, 1), (1, 2))) == pytest.approx(0.5)


def test_arm_k_by_the_sign_of_the_pooled_change() -> None:
    k, pooled = fs.arm_k([1.0, -0.5, None])
    assert k == 0.9
    assert pooled == pytest.approx(0.25)
    k, pooled = fs.arm_k([-1.0, 0.5])
    assert k == 0.1
    assert pooled == pytest.approx(-0.25)
    assert fs.arm_k([1.0, -1.0]) == (0.5, 0.0)
    assert fs.arm_k([None, None]) == (0.5, None)
    assert fs.arm_k([]) == (0.5, None)


def test_needed_days_never_passes_the_last_admitted_day_and_covers_both_windows() -> None:
    q = _q()
    days = fs.needed_days(q, [R, date(2026, 9, 27)])
    assert days
    assert max(days) <= fs.last_admitted_day(q.freeze)
    assert set(fs.window(DUE, 3, 1)) <= days  # the due-date window of the previous year
    assert set(fs.window(R, 10, 1)) <= days
    assert set(fs.window(R, 10, 30)) <= days
    assert not any(d.year == 2026 for d in days)
    # an early freeze cuts days off the window rather than shifting them
    early = _q(datetime(2025, 9, 15, tzinfo=UTC))
    got = fs.needed_days(early, [R])
    assert max(got) == date(2025, 9, 14)
    assert date(2025, 9, 14) in got
    assert date(2025, 9, 15) not in got
    assert fs.needed_days(q, []) == set()
    assert min(days) >= fs.shift(DUE, fs.MAX_SHIFT) - timedelta(days=10)
