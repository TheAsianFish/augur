"""Source-aware calibrators: the penalised solver, the three maps and their params."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from augur.aggregate import CLIP
from augur.calibrate import CalibrationError, Platt, platt_targets
from augur.sourcecal import (
    MIN_PER_SOURCE,
    PRIOR_SD,
    SOURCED,
    PartialPlatt,
    PerSourcePlatt,
    PooledPlatt,
    fit_logistic,
    sourced_from_params,
)


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _history(
    n_by_source: dict[str, int], seed: int = 7
) -> tuple[list[float], list[int], list[str]]:
    """Sources with different base rates and a common over-confidence."""
    rng = random.Random(seed)
    shift = {"polymarket": 0.6, "manifold": 0.3, "metaculus": -1.0}
    ps: list[float] = []
    ys: list[int] = []
    ss: list[str] = []
    for s, n in n_by_source.items():
        for _ in range(n):
            p = rng.uniform(0.03, 0.97)
            truth = _sig(0.5 * _logit(p) + shift.get(s, 0.0))
            ps.append(p)
            ys.append(int(rng.random() < truth))
            ss.append(s)
    return ps, ys, ss


HIST = _history({"polymarket": 60, "manifold": 50, "metaculus": 40})


# -- fit_logistic --------------------------------------------------------------


def test_fit_logistic_matches_the_platt_baseline() -> None:
    ps, ys, _ = HIST
    ref = Platt().fit(ps, ys)
    rows = [[_logit(p), 1.0] for p in ps]
    beta, steps = fit_logistic(rows, platt_targets(ys), [0.0, 0.0])
    assert steps < 30  # Newton, not thousands of descent steps
    assert beta[0] == pytest.approx(ref.a, abs=1e-5)
    assert beta[1] == pytest.approx(ref.b, abs=1e-5)


def test_fit_logistic_gradient_is_zero_at_the_optimum_with_the_penalty() -> None:
    ps, ys, ss = HIST
    names = sorted(set(ss))
    rows = []
    for p, s in zip(ps, ss, strict=True):
        r = [_logit(p), 1.0] + [1.0 if s == n else 0.0 for n in names]
        rows.append(r)
    targets = platt_targets(ys)
    penalty = [0.0, 0.0] + [4.0] * len(names)
    beta, _ = fit_logistic(rows, targets, penalty)
    grad = [penalty[j] * beta[j] for j in range(len(beta))]
    for r, t in zip(rows, targets, strict=True):
        resid = _sig(sum(x * c for x, c in zip(r, beta, strict=True))) - t
        for j in range(len(beta)):
            grad[j] += resid * r[j]
    assert max(abs(g) for g in grad) < 1e-6
    # the penalty is really acting: offsets are not what an unpenalised fit gives
    free, _ = fit_logistic(rows, targets, [0.0] * len(beta))
    assert free != pytest.approx(beta, abs=1e-3)


def test_fit_logistic_is_finite_on_separable_history() -> None:
    rows = [[-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    beta, _ = fit_logistic(rows, [0.0, 0.0, 1.0, 1.0], [0.0, 0.0])
    assert all(math.isfinite(b) for b in beta)


@pytest.mark.parametrize(
    ("rows", "targets", "penalty"),
    [
        ([], [], [0.0]),
        ([[1.0]], [0.5, 0.5], [0.0]),
        ([[1.0, 2.0]], [0.5], [0.0]),
    ],
)
def test_fit_logistic_refuses_a_malformed_problem(
    rows: list[list[float]], targets: list[float], penalty: list[float]
) -> None:
    with pytest.raises(CalibrationError, match="one target per row"):
        fit_logistic(rows, targets, penalty)


# -- PartialPlatt --------------------------------------------------------------


def test_partial_with_a_flat_prior_matches_per_source_intercepts() -> None:
    ps, ys, ss = HIST
    names = sorted(set(ss))
    rows = [
        [_logit(p)] + [1.0 if s == n else 0.0 for n in names] for p, s in zip(ps, ss, strict=True)
    ]
    free, _ = fit_logistic(rows, platt_targets(ys), [0.0] * (1 + len(names)))
    m = PartialPlatt(prior_sd=1e6).fit(ps, ys, ss)
    assert m.a == pytest.approx(free[0], abs=1e-4)
    for k, s in enumerate(names):
        assert m.b + m.offsets[s] == pytest.approx(free[1 + k], abs=1e-4)
    for p in (0.05, 0.4, 0.9):
        for s in names:
            expected = _sig(free[0] * _logit(p) + free[1 + names.index(s)])
            assert m(p, s) == pytest.approx(expected, abs=1e-4)


def test_partial_with_a_tiny_prior_is_the_pooled_platt_fit() -> None:
    ps, ys, ss = HIST
    m = PartialPlatt(prior_sd=1e-4).fit(ps, ys, ss)
    ref = Platt().fit(ps, ys)
    assert all(abs(u) < 1e-3 for u in m.offsets.values())
    for p in (0.05, 0.3, 0.7, 0.95):
        for s in set(ss):
            assert m(p, s) == pytest.approx(ref(p), abs=1e-3)


def test_partial_prior_shrinks_offsets_monotonically() -> None:
    ps, ys, ss = HIST
    sizes = [
        max(abs(u) for u in PartialPlatt(prior_sd=sd).fit(ps, ys, ss).offsets.values())
        for sd in (0.01, 0.5, 100.0)
    ]
    assert sizes[0] < sizes[1] < sizes[2]
    assert PRIOR_SD == 0.5
    assert PartialPlatt().prior_sd == PRIOR_SD


def test_partial_recovers_source_differences() -> None:
    ps, ys, ss = _history({"polymarket": 400, "metaculus": 400}, seed=1)
    m = PartialPlatt().fit(ps, ys, ss)
    assert m(0.5, "polymarket") > m(0.5, "metaculus") + 0.1


def test_partial_unseen_source_gets_the_shared_intercept() -> None:
    ps, ys, ss = HIST
    m = PartialPlatt().fit(ps, ys, ss)
    for p in (0.1, 0.5, 0.8):
        expected = min(max(_sig(m.a * _logit(p) + m.b), CLIP[0]), CLIP[1])
        assert m(p, "unseen") == pytest.approx(expected)


def test_partial_output_stays_within_clip_for_extreme_inputs() -> None:
    ps, ys, ss = _history({"a": 40, "b": 40})
    # a very steep, separable history pushes the map to its limits
    m = PartialPlatt(prior_sd=1e6).fit([0.01, 0.99] * 30, [0, 1] * 30, ["a", "b"] * 30)
    for map_ in (m, PartialPlatt().fit(ps, ys, ss)):
        for p in (0.0, 1.0, 1e-12, 0.5):
            for s in ("a", "b", "unseen"):
                assert CLIP[0] <= map_(p, s) <= CLIP[1]


@pytest.mark.parametrize("calibrator", SOURCED, ids=lambda c: c.name)
def test_params_round_trip_to_identical_outputs(calibrator: Any) -> None:
    ps, ys, ss = _history({"polymarket": 45, "manifold": 35, "metaculus": 12})
    m = calibrator.fit(ps, ys, ss)
    again = sourced_from_params(m.params())
    assert again.params() == m.params()
    assert again.name == m.name == calibrator.name
    for p in (0.0, 0.02, 0.3, 0.77, 1.0):
        for s in ("polymarket", "manifold", "metaculus", "unseen"):
            assert again(p, s) == m(p, s)


# -- PerSourcePlatt ------------------------------------------------------------


def test_per_source_uses_its_own_map_only_from_thirty_outcomes() -> None:
    assert MIN_PER_SOURCE == 30
    ps, ys, ss = _history({"big": 30, "small": 29, "other": 40}, seed=11)
    m = PerSourcePlatt().fit(ps, ys, ss)
    assert set(m.by_source) == {"big", "other"}
    pooled = Platt().fit(ps, ys)
    big_idx = [i for i, s in enumerate(ss) if s == "big"]
    own = Platt().fit([ps[i] for i in big_idx], [ys[i] for i in big_idx])
    for p in (0.1, 0.5, 0.9):
        assert m(p, "small") == pooled(p)
        assert m(p, "unseen") == pooled(p)
        assert m(p, "big") == own(p)
    assert m(0.3, "big") != pooled(0.3)


def test_per_source_min_n_is_configurable() -> None:
    ps, ys, ss = _history({"a": 12, "b": 12})
    assert set(PerSourcePlatt(min_n=10).fit(ps, ys, ss).by_source) == {"a", "b"}
    assert PerSourcePlatt().fit(ps, ys, ss).by_source == {}


def test_pooled_ignores_the_source() -> None:
    ps, ys, ss = HIST
    m = PooledPlatt().fit(ps, ys, ss)
    assert m(0.4, "polymarket") == m(0.4, "metaculus") == Platt().fit(ps, ys)(0.4)


# -- errors --------------------------------------------------------------------


@pytest.mark.parametrize("calibrator", SOURCED, ids=lambda c: c.name)
def test_fit_errors_are_actionable(calibrator: Any) -> None:
    with pytest.raises(CalibrationError, match="one of each per resolved forecast"):
        calibrator.fit([0.2, 0.3], [0], ["a", "a"])
    with pytest.raises(CalibrationError, match="one of each per resolved forecast"):
        calibrator.fit([0.2, 0.3], [0, 1], ["a"])
    with pytest.raises(CalibrationError, match="history is empty"):
        calibrator.fit([], [], [])
    with pytest.raises(CalibrationError, match="not 0 or 1"):
        calibrator.fit([0.2, 0.3], [0, 2], ["a", "a"])
    with pytest.raises(CalibrationError, match="not a finite number in"):
        calibrator.fit([0.2, 1.5], [0, 1], ["a", "a"])
    with pytest.raises(CalibrationError, match="not a finite number in"):
        calibrator.fit([0.2, float("nan")], [0, 1], ["a", "a"])


def test_calling_a_map_with_a_bad_probability_raises() -> None:
    ps, ys, ss = HIST
    for c in SOURCED:
        m = c.fit(ps, ys, ss)
        with pytest.raises(CalibrationError):
            m(1.5, "polymarket")


def test_unknown_or_incomplete_params_raise() -> None:
    with pytest.raises(CalibrationError, match="unknown source-aware calibrator"):
        sourced_from_params({"name": "isotonic"})
    with pytest.raises(CalibrationError, match="unknown source-aware calibrator"):
        sourced_from_params({})
    with pytest.raises(CalibrationError, match="incomplete"):
        sourced_from_params({"name": "platt-partial", "a": 1.0})
    with pytest.raises(CalibrationError, match="incomplete"):
        sourced_from_params({"name": "platt-by-source", "min_n": 30})


def test_fit_logistic_refuses_a_fit_that_does_not_converge() -> None:
    rows = [[float(x), 1.0] for x in range(-3, 4)]
    targets = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    with pytest.raises(CalibrationError, match="did not converge"):
        fit_logistic(rows, targets, [0.0, 0.0], max_iterations=1)
