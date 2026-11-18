from __future__ import annotations

import math
from typing import Any

import pytest

from augur.distributions import (
    Scaling,
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
