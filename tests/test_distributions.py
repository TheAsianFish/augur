from __future__ import annotations

import itertools
import math
from typing import Any

import pytest

from augur.distributions import (
    Scaling,
    cdf_from_percentiles,
    max_step,
    min_step,
    mix_cdfs,
    normalise_probabilities,
    quantile,
    validate_cdf,
    validate_probabilities,
)

# Real ``question["scaling"]`` payloads from the bot-testing-area (2026-11-18),
# without the 201-value ``continuous_range``.
NUMERIC_SCALING: dict[str, Any] = {
    "range_min": -35,
    "range_max": 0,
    "zero_point": None,
    "open_upper_bound": True,
    "open_lower_bound": True,
    "inbound_outcome_count": 200,
    "nominal_min": -35,
    "nominal_max": 0,
}
DISCRETE_INT_SCALING: dict[str, Any] = {
    "range_min": -0.5,
    "range_max": 15.5,
    "nominal_min": 0,
    "nominal_max": 15,
    "zero_point": None,
    "open_upper_bound": False,
    "open_lower_bound": False,
    "inbound_outcome_count": 16,
}
DISCRETE_STEP_SCALING: dict[str, Any] = {
    "range_min": 0.095,
    "range_max": 0.805,
    "nominal_min": 0.1,
    "nominal_max": 0.8,
    "zero_point": None,
    "open_upper_bound": True,
    "open_lower_bound": True,
    "inbound_outcome_count": 71,
}
REAL_SCALINGS = [NUMERIC_SCALING, DISCRETE_INT_SCALING, DISCRETE_STEP_SCALING]


def _scaling(
    *,
    lower: bool = True,
    upper: bool = True,
    n: int = 200,
    zero_point: float | None = None,
    range_min: float = 0.0,
    range_max: float = 100.0,
) -> Scaling:
    return Scaling(
        range_min=range_min,
        range_max=range_max,
        zero_point=zero_point,
        open_lower_bound=lower,
        open_upper_bound=upper,
        inbound_outcome_count=n,
    )


def _ramp(scaling: Scaling) -> list[float]:
    """A valid CDF for any scaling: a straight line between legal ends."""
    n = scaling.inbound_outcome_count
    lo = 0.01 if scaling.open_lower_bound else 0.0
    hi = 0.99 if scaling.open_upper_bound else 1.0
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


# Scaling


def test_scaling_reads_each_real_payload() -> None:
    numeric = Scaling.from_question({"scaling": NUMERIC_SCALING})
    assert (numeric.range_min, numeric.range_max, numeric.inbound_outcome_count) == (-35, 0, 200)
    assert numeric.open_lower_bound
    assert numeric.open_upper_bound
    assert not numeric.is_discrete_like
    ints = Scaling.from_question({"scaling": DISCRETE_INT_SCALING})
    assert ints.inbound_outcome_count == 16
    assert ints.is_discrete_like
    assert not ints.open_lower_bound
    assert not ints.open_upper_bound
    step = Scaling.from_question({"scaling": DISCRETE_STEP_SCALING})
    assert step.inbound_outcome_count == 71
    assert step.nominal_min == 0.1


def test_scaling_locations_match_the_payloads() -> None:
    numeric = Scaling.from_question({"scaling": NUMERIC_SCALING})
    expected = [-35 + 35 * i / 200 for i in range(201)]
    assert numeric.locations() == pytest.approx(expected, abs=1e-12)
    ints = Scaling.from_question({"scaling": DISCRETE_INT_SCALING}).locations()
    # The bucket edges sit half way between the integers 0..15.
    assert ints == pytest.approx([k - 0.5 for k in range(17)], abs=1e-12)
    step = Scaling.from_question({"scaling": DISCRETE_STEP_SCALING}).locations()
    assert len(step) == 72
    assert step[1] - step[0] == pytest.approx(0.01)


def test_older_payload_bounds_on_the_question_and_default_count() -> None:
    scaling: dict[str, Any] = {"range_min": 0, "range_max": 10, "zero_point": None}
    question = {"scaling": scaling, "open_lower_bound": False, "open_upper_bound": True}
    s = Scaling.from_question(question)
    assert s.inbound_outcome_count == 200
    assert (s.open_lower_bound, s.open_upper_bound) == (False, True)
    scaling["inbound_outcome_count"] = None
    assert Scaling.from_question(question).inbound_outcome_count == 200


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ({}, "no 'scaling'"),
        ({"scaling": {"range_min": None, "range_max": None}}, "range_min"),
        ({"scaling": {"range_min": 0, "range_max": 1}}, "open_lower_bound"),
        (
            {
                "scaling": {
                    "range_min": 5,
                    "range_max": 1,
                    "open_lower_bound": True,
                    "open_upper_bound": True,
                }
            },
            "must be above",
        ),
        (
            {
                "scaling": {
                    "range_min": 1,
                    "range_max": 5,
                    "zero_point": 2,
                    "open_lower_bound": True,
                    "open_upper_bound": True,
                }
            },
            "zero_point",
        ),
    ],
)
def test_unusable_scaling_raises(question: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Scaling.from_question(question)


def test_linear_round_trip() -> None:
    s = _scaling(range_min=-35, range_max=0)
    for x in [-35.0, -20.5, 0.0, -50.0, 12.0]:
        assert s.to_scaled(s.to_unscaled(x)) == pytest.approx(x, abs=1e-12)
    assert s.to_unscaled(-35) == 0.0
    assert s.to_unscaled(0) == 1.0


@pytest.mark.parametrize("zero_point", [0.0, -50.0, 1e9, 1000.5])
def test_log_round_trip(zero_point: float) -> None:
    s = _scaling(range_min=1.0, range_max=1000.0, zero_point=zero_point)
    assert s.to_scaled(0.0) == pytest.approx(1.0)
    assert s.to_scaled(1.0) == pytest.approx(1000.0)
    for x in [1.0, 2.0, 31.6, 500.0, 1000.0, 1.5]:
        assert s.to_scaled(s.to_unscaled(x)) == pytest.approx(x, rel=1e-9)


def test_log_scale_matches_metaculus_formula() -> None:
    s = _scaling(range_min=1.0, range_max=1000.0, zero_point=0.0)
    ratio = (1000.0 - 0.0) / (1.0 - 0.0)
    for u in [0.0, 0.25, 0.5, 1.0]:
        expected = 1.0 + 999.0 * (ratio**u - 1) / (ratio - 1)
        assert s.to_scaled(u) == pytest.approx(expected, rel=1e-12)
    # Past the zero point there is no position.
    assert s.to_unscaled(-1.0) == -math.inf
    above = _scaling(range_min=1.0, range_max=10.0, zero_point=20.0)
    assert above.to_unscaled(25.0) == math.inf


# validate_cdf


@pytest.mark.parametrize("payload", REAL_SCALINGS)
def test_a_ramp_is_valid_for_each_real_scaling(payload: dict[str, Any]) -> None:
    s = Scaling.from_question({"scaling": payload})
    assert validate_cdf(_ramp(s), s) == []


def test_step_limits() -> None:
    assert min_step(200) == 5e-5
    assert max_step(200) == pytest.approx(0.2)
    assert min_step(16) == 0.000625
    assert max_step(16) == pytest.approx(2.5)


def test_rejects_wrong_length() -> None:
    s = _scaling()
    assert validate_cdf(_ramp(s)[:-1], s) == [
        "CDF must have 201 values (inbound_outcome_count + 1), got 200"
    ]


def test_rejects_non_finite() -> None:
    s = _scaling()
    cdf = _ramp(s)
    cdf[3] = math.nan
    assert validate_cdf(cdf, s) == ["CDF values must all be finite numbers"]


def test_rejects_a_flat_step() -> None:
    s = _scaling()
    cdf = _ramp(s)
    cdf[10] = cdf[9]
    (error,) = validate_cdf(cdf, s)
    assert "at least 5e-05" in error
    assert "index 9 and 10" in error


def test_rejects_a_step_just_under_the_minimum_after_rounding() -> None:
    s = _scaling(lower=False, upper=False)
    cdf = _ramp(s)
    cdf[1] = 4.9e-5  # the first step rounds to 4.9e-05 < 5e-05
    assert len(validate_cdf(cdf, s)) == 1
    cdf[1] = 4.99999999e-5  # within rounding of the minimum: accepted
    assert validate_cdf(cdf, s) == []


def test_rejects_a_too_steep_step() -> None:
    s = _scaling(lower=False, upper=False)
    n = s.inbound_outcome_count
    # 0.25 in one bucket, the rest spread evenly over the others.
    cdf = [0.0]
    for i in range(n):
        cdf.append(cdf[-1] + (0.25 if i == 100 else 0.75 / (n - 1)))
    cdf[-1] = 1.0
    (error,) = validate_cdf(cdf, s)
    assert "at most 0.2" in error
    assert "index 100 and 101" in error


def test_steep_step_allowed_for_few_outcomes() -> None:
    s = Scaling.from_question({"scaling": DISCRETE_INT_SCALING})
    cdf = [0.0, *[0.001 * i for i in range(1, 8)], *[0.99 + 0.001 * i for i in range(8)], 1.0]
    assert len(cdf) == 17
    assert validate_cdf(cdf, s) == []


def test_rejects_each_bound_rule() -> None:
    open_both = _scaling()
    cdf = _ramp(open_both)
    cdf[0] = 0.0009
    assert validate_cdf(cdf, open_both) == [
        "lower bound is open: cdf[0] must be at least 0.001, got 0.0009"
    ]
    cdf = _ramp(open_both)
    cdf[-1] = 0.9995
    assert validate_cdf(cdf, open_both) == [
        "upper bound is open: cdf[-1] must be at most 0.999, got 0.9995"
    ]
    closed_both = _scaling(lower=False, upper=False)
    cdf = _ramp(closed_both)
    cdf[0] = 0.001
    assert validate_cdf(cdf, closed_both) == [
        "lower bound is closed: cdf[0] must be 0.0, got 0.001"
    ]
    cdf = _ramp(closed_both)
    cdf[-1] = 0.999
    assert validate_cdf(cdf, closed_both) == [
        "upper bound is closed: cdf[-1] must be 1.0, got 0.999"
    ]


def test_bounds_accept_their_exact_limits() -> None:
    s = _scaling()
    n = s.inbound_outcome_count
    cdf = [0.001 + 0.998 * i / n for i in range(n + 1)]
    assert validate_cdf(cdf, s) == []
    closed = _scaling(lower=False, upper=False)
    cdf = _ramp(closed)
    cdf[-1] = 1.0 - 1e-12  # rounds to 1.0 at 10 places
    assert validate_cdf(cdf, closed) == []


# mix_cdfs and quantile


def test_mix_is_the_pointwise_weighted_mean() -> None:
    s = _scaling(n=4, lower=False, upper=False)
    a = [0.0, 0.1, 0.2, 0.3, 1.0]
    b = [0.0, 0.7, 0.8, 0.9, 1.0]
    assert mix_cdfs([a, b]) == pytest.approx([0.0, 0.4, 0.5, 0.6, 1.0])
    assert mix_cdfs([a, b], [3, 1]) == pytest.approx([0.0, 0.25, 0.35, 0.45, 1.0])
    assert validate_cdf(mix_cdfs([a, b], [3, 1]), s) == []


@pytest.mark.parametrize(
    ("cdfs", "weights", "message"),
    [
        ([], None, "zero CDFs"),
        ([[0.0, 1.0], [0.0, 0.5, 1.0]], None, "one length"),
        ([[0.0, 1.0]], [1.0, 2.0], "2 weights for 1"),
        ([[0.0, 1.0]], [-1.0], "non-negative"),
        ([[0.0, 1.0]], [0.0], "all be zero"),
    ],
)
def test_mix_rejects_bad_input(
    cdfs: list[list[float]], weights: list[float] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mix_cdfs(cdfs, weights)


def test_quantile_inverts_a_ramp_and_clamps() -> None:
    s = _scaling(lower=False, upper=False)
    cdf = _ramp(s)  # uniform on [0, 100]
    assert quantile(cdf, s, 0.5) == pytest.approx(50.0)
    assert quantile(cdf, s, 0.123) == pytest.approx(12.3)
    open_s = _scaling()
    cdf = _ramp(open_s)  # 0.01 below the range, 0.01 above
    assert quantile(cdf, open_s, 0.005) == 0.0
    assert quantile(cdf, open_s, 0.995) == 100.0
    log_s = _scaling(lower=False, upper=False, range_min=1, range_max=1000, zero_point=0)
    # Uniform in unscaled space is log-uniform: the median is the geometric mean.
    assert quantile(_ramp(log_s), log_s, 0.5) == pytest.approx(math.sqrt(1000))


# cdf_from_percentiles

LEVELS = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95]
SHAPE = [0.12, 0.2, 0.3, 0.42, 0.48, 0.53, 0.65, 0.75, 0.85]  # unscaled, right-skewed


def _percentiles(scaling: Scaling, shape: list[float] = SHAPE) -> list[tuple[float, float]]:
    return [(lv, scaling.to_scaled(u)) for lv, u in zip(LEVELS, shape, strict=True)]


def _round_trip_tolerance(scaling: Scaling) -> float:
    """How far, in unscaled units, a quantile may land from the percentile given.

    The uniform mix moves each CDF value by about 0.01, which at these
    densities is under 0.02 in u; the quantile is linear between grid points,
    so a percentile between them can also be off by up to one bucket.
    """
    return 0.02 + 1.0 / scaling.inbound_outcome_count


def _assert_round_trip(
    cdf: list[float], scaling: Scaling, percentiles: list[tuple[float, float]]
) -> None:
    tol = _round_trip_tolerance(scaling)
    for level, value in percentiles:
        got = scaling.to_unscaled(quantile(cdf, scaling, level))
        assert got == pytest.approx(scaling.to_unscaled(value), abs=tol), (level, value)


@pytest.mark.parametrize("payload", REAL_SCALINGS)
def test_real_scalings_give_valid_round_tripping_cdfs(payload: dict[str, Any]) -> None:
    s = Scaling.from_question({"scaling": payload})
    percentiles = _percentiles(s)
    cdf = cdf_from_percentiles(percentiles, s)
    assert len(cdf) == s.inbound_outcome_count + 1
    assert validate_cdf(cdf, s) == []
    _assert_round_trip(cdf, s, percentiles)


@pytest.mark.parametrize("zero_point", [0.0, -10.0, 2000.0])
def test_log_scale_round_trip(zero_point: float) -> None:
    s = _scaling(range_min=1, range_max=1000, zero_point=zero_point)
    percentiles = _percentiles(s)
    cdf = cdf_from_percentiles(percentiles, s)
    assert validate_cdf(cdf, s) == []
    _assert_round_trip(cdf, s, percentiles)


@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("upper", [True, False])
def test_each_bound_combination(lower: bool, upper: bool) -> None:
    s = _scaling(lower=lower, upper=upper)
    cdf = cdf_from_percentiles(_percentiles(s), s)
    assert validate_cdf(cdf, s) == []
    if lower:
        assert 0.001 <= cdf[0] < 0.05
    else:
        assert cdf[0] == 0.0
    if upper:
        assert 0.95 < cdf[-1] <= 0.999
    else:
        assert cdf[-1] == 1.0
    _assert_round_trip(cdf, s, _percentiles(s))


def test_mass_beyond_a_closed_bound_lands_in_the_edge_bucket() -> None:
    # 20 buckets, so no maximum step gets in the way.
    s = _scaling(lower=False, upper=False, range_max=20, n=20)
    cdf = cdf_from_percentiles([(0.1, -5.0), (0.9, 15.0)], s)
    assert validate_cdf(cdf, s) == []
    assert cdf[0] == 0.0
    # Linear from (-5, 0.1) to (15, 0.9): P(X <= 1) is 0.34, all of it in bucket 1.
    assert cdf[1] == pytest.approx(0.34, abs=0.01)
    # With 200 buckets the edge bucket is capped and the rest spills inwards.
    s = _scaling(lower=False, upper=False, range_max=20)
    cdf = cdf_from_percentiles([(0.1, -5.0), (0.9, 15.0)], s)
    assert validate_cdf(cdf, s) == []
    assert cdf[1] == pytest.approx(0.99 * max_step(200), abs=0.005)
    assert cdf[2] == pytest.approx(0.1 + 0.8 * 5.2 / 20, abs=0.01)


def test_point_mass_is_spread_to_the_steepest_allowed() -> None:
    s = _scaling(range_max=20)
    cdf = cdf_from_percentiles([(0.1, 5.0), (0.5, 5.0), (0.9, 5.0)], s)
    assert validate_cdf(cdf, s) == []
    steps = [b - a for a, b in itertools.pairwise(cdf)]
    # Capped at 99% of the maximum, less the ~1% the uniform mix takes.
    assert max(steps) == pytest.approx(0.99 * max_step(200), abs=0.005)
    # The mass stays around 5 (bucket 50), within a handful of buckets.
    assert quantile(cdf, s, 0.5) == pytest.approx(5.0, abs=0.5)


def test_point_mass_on_a_discrete_question_is_one_bucket() -> None:
    s = Scaling.from_question({"scaling": DISCRETE_INT_SCALING})
    cdf = cdf_from_percentiles([(0.1, 3), (0.5, 3), (0.9, 3)], s)
    assert validate_cdf(cdf, s) == []
    steps = [b - a for a, b in itertools.pairwise(cdf)]
    assert steps.index(max(steps)) == 3  # the bucket (2.5, 3.5]
    assert max(steps) > 0.6


@pytest.mark.parametrize(("lower", "upper"), [(True, True), (False, False), (True, False)])
def test_percentiles_entirely_below_the_range(lower: bool, upper: bool) -> None:
    s = _scaling(lower=lower, upper=upper, range_max=20)
    cdf = cdf_from_percentiles([(0.1, -50.0), (0.5, -45.0), (0.9, -40.0)], s)
    assert validate_cdf(cdf, s) == []
    if lower:
        # The server needs some mass inside; the rest stays below.
        assert cdf[0] >= 0.9
    else:
        assert cdf[0] == 0.0
        assert cdf[10] >= 0.9  # moved into the lowest buckets


@pytest.mark.parametrize(("lower", "upper"), [(True, True), (False, False), (False, True)])
def test_percentiles_entirely_above_the_range(lower: bool, upper: bool) -> None:
    s = _scaling(lower=lower, upper=upper, range_max=20)
    cdf = cdf_from_percentiles([(0.1, 50.0), (0.5, 55.0), (0.9, 60.0)], s)
    assert validate_cdf(cdf, s) == []
    if upper:
        assert cdf[-1] <= 0.1
    else:
        assert cdf[-1] == 1.0
        assert cdf[-11] <= 0.1


def test_non_monotone_answer_is_repaired_by_isotonic_fit() -> None:
    s = _scaling(range_max=20)
    sloppy = cdf_from_percentiles([(0.1, 2.0), (0.5, 8.0), (0.9, 6.0)], s)
    # Pool adjacent violators: 8 and 6 become their mean, 7.
    repaired = cdf_from_percentiles([(0.1, 2.0), (0.5, 7.0), (0.9, 7.0)], s)
    assert validate_cdf(sloppy, s) == []
    assert sloppy == repaired


def test_unsorted_and_duplicate_levels() -> None:
    s = _scaling(range_max=20)
    ordered = cdf_from_percentiles([(0.1, 1.0), (0.5, 5.0), (0.9, 9.0)], s)
    assert cdf_from_percentiles([(0.9, 9.0), (0.1, 1.0), (0.5, 5.0)], s) == ordered
    # Two answers for one level are merged into their mean.
    assert cdf_from_percentiles([(0.5, 4.0), (0.9, 9.0), (0.5, 6.0), (0.1, 1.0)], s) == ordered


def test_values_past_a_log_zero_point_are_usable() -> None:
    s = _scaling(range_min=1, range_max=1000, zero_point=0)
    cdf = cdf_from_percentiles([(0.1, -5.0), (0.5, 30.0), (0.9, 500.0)], s)
    assert validate_cdf(cdf, s) == []


@pytest.mark.parametrize(
    ("percentiles", "message"),
    [
        ([], "no percentiles"),
        ([(0.5, math.nan)], "finite"),
        ([(math.nan, 1.0)], "finite"),
        ([(0.5, math.inf)], "finite"),
        ([(0.0, 1.0)], "outside"),
        ([(1.0, 1.0)], "outside"),
    ],
)
def test_unusable_percentiles_raise(percentiles: list[tuple[float, float]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cdf_from_percentiles(percentiles, _scaling())


def test_mixing_built_cdfs_stays_valid() -> None:
    s = _scaling(lower=False)
    a = cdf_from_percentiles([(0.1, 5.0), (0.5, 5.0), (0.9, 5.0)], s)
    b = cdf_from_percentiles(_percentiles(s), s)
    assert validate_cdf(mix_cdfs([a, b], [0.3, 0.7]), s) == []


# Multiple choice

OPTIONS = ["a", "b", "c"]


def test_normalise_fills_missing_and_clips() -> None:
    out = normalise_probabilities({"a": 5.0, "b": 0.0}, OPTIONS)
    assert list(out) == OPTIONS
    assert out["b"] == out["c"] == pytest.approx(0.001)
    assert out["a"] == pytest.approx(0.998)
    assert validate_probabilities(out, OPTIONS) == []


def test_normalise_keeps_proportions_when_nothing_is_clipped() -> None:
    out = normalise_probabilities({"a": 2.0, "b": 1.0, "c": 1.0}, OPTIONS)
    assert out == pytest.approx({"a": 0.5, "b": 0.25, "c": 0.25})


def test_normalise_treats_nan_as_missing() -> None:
    out = normalise_probabilities({"a": math.nan, "b": 1.0, "c": 1.0}, OPTIONS)
    assert out["a"] == pytest.approx(0.001)
    assert validate_probabilities(out, OPTIONS) == []


@pytest.mark.parametrize(
    ("probabilities", "options", "message"),
    [
        ({"a": 0.0, "b": 0.0}, OPTIONS, "zero or missing"),
        ({"a": math.nan}, OPTIONS, "zero or missing"),
        ({"a": 0.5, "z": 0.5}, OPTIONS, "does not have"),
        ({"a": -0.1, "b": 1.0}, OPTIONS, ">= 0"),
        ({"a": math.inf}, OPTIONS, "finite"),
        ({"a": 1.0}, ["a"], "at least 2"),
        ({"a": 1.0}, ["a", "a"], "distinct"),
    ],
)
def test_normalise_rejects_unusable_input(
    probabilities: dict[str, float], options: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalise_probabilities(probabilities, options)


def test_validate_probabilities_reports_each_rule() -> None:
    # The sum is 1, so only the range rule fails.
    assert validate_probabilities({"a": 0.5, "b": 0.5, "c": 0.0}, OPTIONS) == [
        "every probability must be in [0.001, 0.999]; not {'c': 0.0}"
    ]
    assert validate_probabilities({"a": 0.5, "b": 0.5}, OPTIONS)[0].startswith(
        "every option needs a probability"
    )
    extra = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    assert validate_probabilities(extra, OPTIONS) == [
        "probabilities name options the question does not have: ['d']"
    ]
    assert validate_probabilities({"a": 0.5, "b": 0.3, "c": 0.3}, OPTIONS) == [
        "probabilities must sum to 1, got 1.1"
    ]
    assert validate_probabilities({"a": 0.5, "b": 0.25, "c": 0.25 + 5e-6}, OPTIONS) == []
