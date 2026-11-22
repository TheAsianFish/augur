"""The calibrators of :mod:`augur.calibrate`, against the references in ``calibration_oracle``."""

from __future__ import annotations

import itertools
import json
import math
import random
import time
from collections.abc import Sequence

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from augur.aggregate import CLIP, EXTREMIZE_FACTOR, mean_logodds
from augur.calibrate import (
    CALIBRATORS,
    BaseRate,
    CalibrationError,
    Calibrator,
    FixedExtremize,
    Identity,
    Isotonic,
    IsotonicMap,
    Platt,
    PlattMap,
    by_name,
    from_params,
)
from augur.errors import AugurError
from tests.calibration_oracle import (
    clip,
    isotonic_oracle,
    logit,
    platt_hessian_min_eigenvalue,
    platt_oracle,
    pooled,
    sigmoid,
)

GRID = [i / 200 for i in range(201)]

# Small fixed histories: (probabilities, outcomes).
FIXED: list[tuple[list[float], list[int]]] = [
    ([0.1, 0.3, 0.5, 0.7, 0.9], [0, 0, 1, 1, 1]),
    ([0.1, 0.3, 0.5, 0.7, 0.9], [0, 1, 0, 1, 1]),
    ([0.2, 0.2, 0.4, 0.6, 0.6, 0.8], [1, 0, 0, 1, 0, 1]),
    ([0.05, 0.15, 0.25, 0.35, 0.65, 0.95], [0, 0, 1, 0, 1, 0]),
    ([0.9, 0.8, 0.85, 0.6, 0.7, 0.75, 0.95], [1, 0, 1, 0, 1, 1, 1]),
    ([0.3, 0.6], [1, 0]),
]

probability = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
coarse_probability = st.integers(min_value=0, max_value=20).map(lambda k: k / 20)
outcome = st.integers(min_value=0, max_value=1)


def history(
    p: st.SearchStrategy[float], min_size: int, max_size: int
) -> st.SearchStrategy[tuple[list[float], list[int]]]:
    """Equal-length lists of probabilities and outcomes."""
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.tuples(
            st.lists(p, min_size=n, max_size=n), st.lists(outcome, min_size=n, max_size=n)
        )
    )


def miscalibrated(
    seed: int, n: int, mu: float, sd: float, a: float = 0.6, b: float = -0.4
) -> tuple[list[float], list[int]]:
    """Forecasts with log-odds ~ N(mu, sd); the outcome is drawn at sigmoid(a x + b)."""
    rng = random.Random(seed)
    ps: list[float] = []
    ys: list[int] = []
    for _ in range(n):
        p = clip(sigmoid(rng.gauss(mu, sd)))
        ps.append(p)
        ys.append(1 if rng.random() < sigmoid(a * logit(p) + b) else 0)
    return ps, ys


def squared_error(
    fitted: Sequence[float], sums: Sequence[float], weights: Sequence[float]
) -> float:
    """Squared error of per-block fitted values against 0/1 outcomes pooled as (sum, weight)."""
    return math.fsum(
        s * (1.0 - f) ** 2 + (w - s) * f**2 for f, s, w in zip(fitted, sums, weights, strict=True)
    )


def isotonic_fitted_at(fit: IsotonicMap, xs: Sequence[float]) -> list[float]:
    """The fitted block value (unclipped) at each unique point ``xs``."""
    out: list[float] = []
    block = 0
    for x in xs:
        while block + 1 < len(fit.thresholds) and fit.thresholds[block + 1] <= x:
            block += 1
        out.append(fit.values[block])
    return out


# -- registry ----------------------------------------------------------------


def test_registry_order_and_lookup() -> None:
    names = [c.name for c in CALIBRATORS]
    assert names == ["raw", "extremize-1.5", "base-rate", "platt", "isotonic"]
    for calibrator in CALIBRATORS:
        assert by_name(calibrator.name) is calibrator
    with pytest.raises(CalibrationError, match=r"use one of raw, extremize-1\.5"):
        by_name("beta")


def test_calibration_error_is_an_augur_error() -> None:
    assert issubclass(CalibrationError, AugurError)


def test_unknown_params_name() -> None:
    with pytest.raises(CalibrationError, match="unknown calibration map"):
        from_params({"name": "beta"})
    with pytest.raises(CalibrationError, match="lack 'a'"):
        from_params({"name": "platt", "b": 0.0, "iterations": 0, "n": 1})


# -- raw, extremize, base rate -----------------------------------------------


def test_identity_clips_only() -> None:
    fit = Identity().fit([0.4], [1])
    assert fit.name == "raw"
    assert fit(0.0) == 0.01
    assert fit(0.37) == 0.37
    assert fit(1.0) == 0.99
    assert fit.params() == {"name": "raw"}


@pytest.mark.parametrize("p", [0.0, 0.005, 0.01, 0.2, 0.5, 0.73, 0.99, 1.0])
def test_fixed_extremize_matches_mean_logodds(p: float) -> None:
    fit = FixedExtremize().fit([0.5, 0.6], [0, 1])
    assert fit.name == "extremize-1.5"
    assert FixedExtremize().factor == EXTREMIZE_FACTOR
    assert fit(p) == clip(mean_logodds([p], 1.5))
    assert fit(p) == clip(sigmoid(1.5 * logit(clip(p))))


def test_fixed_extremize_values_and_name() -> None:
    assert FixedExtremize().fit([0.5], [0])(0.5) == 0.5
    assert FixedExtremize(2.0).name == "extremize-2"
    assert FixedExtremize(2.0).fit([0.5], [0])(0.75) == pytest.approx(0.9, abs=1e-15)
    assert FixedExtremize(2.0).fit([0.5], [0])(0.999) == 0.99


def test_base_rate_is_laplace_smoothed() -> None:
    fit = BaseRate().fit([0.1, 0.2, 0.3, 0.9], [1, 0, 1, 1])
    assert fit.name == "base-rate"
    assert fit(0.01) == fit(0.99) == 4 / 6
    assert BaseRate().fit([0.5], [0])(0.5) == 1 / 3
    assert BaseRate().fit([0.5] * 500, [1] * 500)(0.5) == 0.99


# -- Platt -------------------------------------------------------------------


@pytest.mark.parametrize(("ps", "ys"), FIXED)
def test_platt_matches_newton_oracle(ps: list[float], ys: list[int]) -> None:
    fit = Platt().fit(ps, ys)
    assert isinstance(fit, PlattMap)
    a, b = platt_oracle(ps, ys)
    assert fit.a == pytest.approx(a, abs=1e-5)
    assert fit.b == pytest.approx(b, abs=1e-5)


@pytest.mark.parametrize(("mu", "sd"), [(0.0, 1.0), (-1.0, 1.0), (1.5, 0.6)])
def test_platt_matches_oracle_on_larger_samples(mu: float, sd: float) -> None:
    ps, ys = miscalibrated(11, 200, mu, sd)
    fit = Platt().fit(ps, ys)
    a, b = platt_oracle(ps, ys)
    assert fit.a == pytest.approx(a, abs=1e-5)
    assert fit.b == pytest.approx(b, abs=1e-5)
    assert fit.iterations < Platt().max_iterations


@settings(max_examples=60, deadline=None)
@given(history(probability, 2, 60))
def test_platt_matches_oracle_property(data: tuple[list[float], list[int]]) -> None:
    ps, ys = data
    xs = [logit(clip(p)) for p in ps]
    assume(max(xs) - min(xs) > 0.1)
    a, b = platt_oracle(ps, ys)
    # Stopping at gradient 1e-9 pins (a, b) to about 1e-9 / (smallest Hessian
    # eigenvalue); compare parameters only where that is well inside 1e-5.
    assume(platt_hessian_min_eigenvalue(a, b, ps) > 1e-3)
    fit = Platt().fit(ps, ys)
    assert fit.a == pytest.approx(a, abs=1e-5)
    assert fit.b == pytest.approx(b, abs=1e-5)


def test_platt_symmetric_data_has_zero_intercept() -> None:
    # Mirror-image history: p and 1 - p with opposite outcomes, so the fit is odd in x.
    ps = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7]
    ys = [0, 1, 0, 1, 1, 0]
    fit = Platt().fit(ps, ys)
    assert abs(fit.b) < 1e-9
    assert fit(0.5) == pytest.approx(0.5, abs=1e-9)


def test_platt_single_point_per_side_by_hand() -> None:
    # x = -c and +c with outcomes 0 and 1: targets 1/3 and 2/3, so b = 0 and
    # sigmoid(a c) = 2/3, i.e. a = log(2) / c.
    ps = [0.2, 0.8]
    fit = Platt().fit(ps, [0, 1])
    assert fit.a == pytest.approx(math.log(2.0) / logit(0.8), abs=1e-8)
    assert fit.b == pytest.approx(0.0, abs=1e-8)
    assert fit(0.8) == pytest.approx(2 / 3, abs=1e-8)


def test_platt_monotone_when_slope_positive() -> None:
    ps, ys = miscalibrated(3, 120, 0.0, 1.2)
    fit = Platt().fit(ps, ys)
    assert fit.a > 0
    outputs = [fit(p) for p in GRID]
    assert all(x <= y for x, y in itertools.pairwise(outputs))


@pytest.mark.parametrize(
    ("ps", "ys"),
    [
        ([0.2, 0.4, 0.6, 0.8], [0, 0, 0, 0]),
        ([0.2, 0.4, 0.6, 0.8], [1, 1, 1, 1]),
        ([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [0, 0, 0, 1, 1, 1]),
        ([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], [0, 0, 0, 1, 1, 1]),
        ([0.0, 1.0], [0, 1]),
    ],
)
def test_platt_finite_on_separable_history(ps: list[float], ys: list[int]) -> None:
    fit = Platt().fit(ps, ys)
    assert math.isfinite(fit.a)
    assert math.isfinite(fit.b)
    assert all(CLIP[0] <= fit(p) <= CLIP[1] for p in GRID)
    a, b = platt_oracle(ps, ys)
    assert fit.a == pytest.approx(a, abs=1e-5)
    assert fit.b == pytest.approx(b, abs=1e-5)


def test_platt_params_shape() -> None:
    fit = Platt().fit(*FIXED[0])
    assert set(fit.params()) == {"name", "a", "b", "iterations", "n"}
    assert fit.params()["n"] == 5


# -- isotonic ----------------------------------------------------------------


@pytest.mark.parametrize(("ps", "ys"), FIXED)
def test_isotonic_matches_max_min_oracle(ps: list[float], ys: list[int]) -> None:
    fit = Isotonic().fit(ps, ys)
    xs, expected = isotonic_oracle(ps, ys)
    assert isotonic_fitted_at(fit, xs) == pytest.approx(expected, abs=1e-12)


@settings(max_examples=80, deadline=None)
@given(st.one_of(history(probability, 1, 40), history(coarse_probability, 1, 40)))
def test_isotonic_matches_oracle_property(data: tuple[list[float], list[int]]) -> None:
    ps, ys = data
    fit = Isotonic().fit(ps, ys)
    xs, expected = isotonic_oracle(ps, ys)
    fitted = isotonic_fitted_at(fit, xs)
    assert fitted == pytest.approx(expected, abs=1e-12)
    assert list(fit.thresholds) == sorted(set(fit.thresholds))
    assert all(x <= y for x, y in itertools.pairwise(fit.values))
    _, sums, weights = pooled(ps, ys)
    best = squared_error(fitted, sums, weights)
    constant = math.fsum(sums) / math.fsum(weights)
    assert best <= squared_error(expected, sums, weights) + 1e-9
    assert best <= squared_error([constant] * len(sums), sums, weights) + 1e-9


def test_isotonic_beats_other_monotone_vectors() -> None:
    ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ys = [0, 1, 0, 0, 1, 1, 0, 1]
    fit = Isotonic().fit(ps, ys)
    xs, sums, weights = pooled(ps, ys)
    best = squared_error(isotonic_fitted_at(fit, xs), sums, weights)
    rng = random.Random(5)
    for _ in range(500):
        candidate = sorted(rng.random() for _ in xs)
        assert best <= squared_error(candidate, sums, weights) + 1e-12


def test_isotonic_output_non_decreasing() -> None:
    ps, ys = miscalibrated(9, 150, 0.0, 1.5)
    fit = Isotonic().fit(ps, ys)
    outputs = [fit(p) for p in GRID]
    assert all(x <= y for x, y in itertools.pairwise(outputs))


def test_isotonic_step_function_lookups() -> None:
    fit = Isotonic().fit([0.2, 0.2, 0.4, 0.6, 0.8], [0, 1, 0, 1, 1])
    # Blocks: {0.2, 0.2, 0.4} -> 1/3, {0.6} -> 1, {0.8} -> 1 (equal values do not violate).
    assert fit.thresholds == (0.2, 0.6, 0.8)
    assert fit.values == (1 / 3, 1.0, 1.0)
    assert fit(0.0) == 1 / 3  # below the first block
    assert fit(0.2) == 1 / 3  # at a threshold
    assert fit(0.59) == 1 / 3  # between blocks
    assert fit(0.6) == 0.99  # next block, clipped
    assert fit(1.0) == 0.99  # above the last block
    assert fit.params() == {
        "name": "isotonic",
        "thresholds": [0.2, 0.6, 0.8],
        "values": [1 / 3, 1.0, 1.0],
        "n": 5,
    }


def test_isotonic_clips_inputs_before_pooling() -> None:
    fit = Isotonic().fit([0.0, 0.005, 0.5], [1, 0, 1])
    assert fit.thresholds == (0.01, 0.5)
    assert fit.values == (0.5, 1.0)


# -- every calibrator --------------------------------------------------------

FITTING_SETS = [*FIXED, miscalibrated(2, 80, 0.3, 1.0), ([0.0, 1.0, 0.5], [0, 1, 1])]


@pytest.mark.parametrize("calibrator", CALIBRATORS, ids=lambda c: c.name)
@pytest.mark.parametrize("data", FITTING_SETS)
def test_outputs_within_clip_and_params_round_trip(
    calibrator: Calibrator, data: tuple[list[float], list[int]]
) -> None:
    fit = calibrator.fit(*data)
    assert fit.name == calibrator.name
    params = fit.params()
    assert params["name"] == calibrator.name
    rebuilt = from_params(json.loads(json.dumps(params)))
    assert rebuilt.params() == params
    for p in [*GRID, 1e-9, 1 - 1e-9]:
        value = fit(p)
        assert CLIP[0] <= value <= CLIP[1]
        assert rebuilt(p) == value


@settings(max_examples=40, deadline=None)
@given(history(probability, 1, 30), st.lists(probability, min_size=1, max_size=20))
def test_outputs_within_clip_property(
    data: tuple[list[float], list[int]], queries: list[float]
) -> None:
    for calibrator in CALIBRATORS:
        fit = calibrator.fit(*data)
        for p in queries:
            assert CLIP[0] <= fit(p) <= CLIP[1]


@pytest.mark.parametrize("calibrator", CALIBRATORS, ids=lambda c: c.name)
def test_fit_rejects_bad_history(calibrator: Calibrator) -> None:
    with pytest.raises(CalibrationError, match="nothing to fit"):
        calibrator.fit([], [])
    with pytest.raises(CalibrationError, match="2 probabilities but 1 outcomes"):
        calibrator.fit([0.2, 0.3], [1])
    with pytest.raises(CalibrationError, match="outcome 2 is not 0 or 1"):
        calibrator.fit([0.2, 0.3], [1, 2])
    with pytest.raises(CalibrationError, match="not a finite number"):
        calibrator.fit([0.2, math.nan], [1, 0])
    with pytest.raises(CalibrationError, match="not a finite number"):
        calibrator.fit([0.2, 1.5], [1, 0])


@pytest.mark.parametrize("calibrator", CALIBRATORS, ids=lambda c: c.name)
def test_fit_is_deterministic(calibrator: Calibrator) -> None:
    ps, ys = miscalibrated(4, 60, 0.0, 1.0)
    assert calibrator.fit(ps, ys).params() == calibrator.fit(ps, ys).params()


# -- contracts for a production-grade fit ------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="needs a second-order (Newton) fit: gradient descent converges linearly",
)
def test_platt_converges_in_few_iterations() -> None:
    # A forecaster whose log-odds are centred well below zero and whose true
    # calibration is sigmoid(0.6 x - 0.4): an ill-conditioned fit.
    ps, ys = miscalibrated(7, 500, -2.0, 0.7)
    fit = Platt().fit(ps, ys)
    a, b = platt_oracle(ps, ys)
    assert fit.params()["iterations"] <= 50
    assert fit.a == pytest.approx(a, abs=1e-10)
    assert fit.b == pytest.approx(b, abs=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="needs a one-pass stack-based PAV: the restarting scan is quadratic",
)
def test_isotonic_worst_case_is_fast() -> None:
    # Increasing forecasts with outcomes all 1 then all 0: every adjacent
    # block pair violates in turn and the fit is one block at 0.5.
    n = 8000
    ps = [(i + 1) / (n + 1) for i in range(n)]
    ys = [1] * (n // 2) + [0] * (n - n // 2)
    start = time.perf_counter()
    fit = Isotonic().fit(ps, ys)
    elapsed = time.perf_counter() - start
    assert fit.values == (0.5,)
    assert elapsed < 0.25
