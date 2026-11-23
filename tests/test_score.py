"""Scoring checked against cases computed by hand.

Each expected number below is worked out in the comment beside it from the
formula Metaculus publishes, not from the code under test.
"""

from __future__ import annotations

import math
import random

import pytest

from augur.distributions import Scaling
from augur.score import (
    ABOVE,
    BELOW,
    ScoringError,
    baseline_score,
    brier,
    bucket_index,
    clustered_bootstrap,
    head_to_head,
    log_score,
    outcome_probability,
    paired,
    paired_many,
    peer_score,
    pmf_from_cdf,
    uniform_mass,
)

CLOSED = Scaling(range_min=0, range_max=100, open_lower_bound=False, open_upper_bound=False)
OPEN = Scaling(range_min=0, range_max=100, open_lower_bound=True, open_upper_bound=True)
DISCRETE = Scaling(
    range_min=-0.5,
    range_max=10.5,
    open_lower_bound=False,
    open_upper_bound=False,
    inbound_outcome_count=11,
)


def uniform_cdf(scaling: Scaling) -> list[float]:
    n = scaling.inbound_outcome_count
    low = 0.05 if scaling.open_lower_bound else 0.0
    high = 0.95 if scaling.open_upper_bound else 1.0
    return [low + (high - low) * i / n for i in range(n + 1)]


# -- binary ----------------------------------------------------------------


def test_binary_baseline_score_by_hand() -> None:
    # 100 * (log2(0.8) + 1) = 100 * (1 - 0.321928) = 67.8072
    assert baseline_score("binary", 0.8) == pytest.approx(67.8072, abs=1e-4)
    # 100 * (log2(0.2) + 1) = 100 * (1 - 2.321928) = -132.1928
    assert baseline_score("binary", 0.2) == pytest.approx(-132.1928, abs=1e-4)
    # A 50% forecast is the baseline itself.
    assert baseline_score("binary", 0.5) == pytest.approx(0.0, abs=1e-12)


def test_binary_outcome_probability_brier_and_log() -> None:
    values = {"probability": 0.8}
    assert outcome_probability("binary", values, "yes") == pytest.approx(0.8)
    assert outcome_probability("binary", values, "no") == pytest.approx(0.2)
    assert brier(0.8, 1) == pytest.approx(0.04)
    assert brier(0.8, 0) == pytest.approx(0.64)
    # ln(0.8) = -0.223144
    assert log_score(0.8) == pytest.approx(-0.223144, abs=1e-6)


def test_head_to_head_and_peer_by_hand() -> None:
    # 100 * ln(0.8 / 0.6) = 100 * 0.287682 = 28.7682
    assert head_to_head("binary", 0.8, 0.6) == pytest.approx(28.7682, abs=1e-4)
    assert head_to_head("binary", 0.6, 0.8) == pytest.approx(-28.7682, abs=1e-4)
    # Mean of 100 ln(0.8/0.6) and 100 ln(0.8/0.4) = (28.7682 + 69.3147) / 2 = 49.0415
    assert peer_score("binary", 0.8, [0.6, 0.4]) == pytest.approx(49.0415, abs=1e-4)


def test_binary_resolution_must_be_yes_or_no() -> None:
    with pytest.raises(ScoringError, match="not yes or no"):
        outcome_probability("binary", {"probability": 0.5}, "maybe")


# -- multiple choice -------------------------------------------------------


def test_multiple_choice_baseline_by_hand() -> None:
    values = {"probabilities": {"A": 0.5, "B": 0.3, "C": 0.2}}
    p = outcome_probability("multiple_choice", values, "A")
    assert p == pytest.approx(0.5)
    # 100 * ln(0.5 * 3) / ln(3) = 100 * 0.405465 / 1.098612 = 36.9070
    assert baseline_score("multiple_choice", p, option_count=3) == pytest.approx(36.9070, abs=1e-4)
    # Uniform over three options scores 0.
    assert baseline_score("multiple_choice", 1 / 3, option_count=3) == pytest.approx(0.0)


def test_multiple_choice_unknown_option_is_refused() -> None:
    with pytest.raises(ScoringError, match="not an option"):
        outcome_probability("multiple_choice", {"probabilities": {"A": 1.0}}, "Z")


# -- numeric and discrete --------------------------------------------------


def test_pmf_has_two_tail_buckets() -> None:
    assert pmf_from_cdf([0.1, 0.4, 0.9]) == pytest.approx([0.1, 0.3, 0.5, 0.1])


def test_bucket_index_matches_metaculus_rule() -> None:
    # n = 200: u = 0.5 -> int(0.5 * 200 + 1 - 1e-10) = int(100.9999999999) = 100
    assert bucket_index("50", CLOSED) == 100
    # u = 0 -> max(int(0.9999999999), 1) = 1
    assert bucket_index("0", CLOSED) == 1
    # u = 1 -> n
    assert bucket_index("100", CLOSED) == 200
    assert bucket_index("-3", OPEN) == 0
    assert bucket_index("140", OPEN) == 201
    assert bucket_index(BELOW, OPEN) == 0
    assert bucket_index(ABOVE, OPEN) == 201


def test_uniform_forecast_scores_zero_inside_the_range() -> None:
    for scaling in (CLOSED, OPEN):
        cdf = uniform_cdf(scaling)
        p = outcome_probability("numeric", {"cdf": cdf}, "37.3", scaling=scaling)
        bucket = bucket_index("37.3", scaling)
        assert baseline_score("numeric", p, bucket=bucket, scaling=scaling) == pytest.approx(
            0.0, abs=1e-9
        )


def test_numeric_baseline_by_hand() -> None:
    # Closed bounds, n = 200, so the uniform mass per inner bucket is 1/200.
    # A forecast with mass 0.02 in the outcome's bucket scores
    # 100 * ln(0.02 / 0.005) / 2 = 50 * ln(4) = 69.3147
    assert uniform_mass(100, CLOSED) == pytest.approx(0.005)
    assert baseline_score("numeric", 0.02, bucket=100, scaling=CLOSED) == pytest.approx(
        69.3147, abs=1e-4
    )
    # Open on both sides: 0.9 spread over 200 inner buckets = 0.0045 each, and
    # 0.05 for each tail. Mass 0.1 below the range: 50 * ln(0.1 / 0.05) = 34.6574
    assert uniform_mass(0, OPEN) == pytest.approx(0.05)
    assert uniform_mass(201, OPEN) == pytest.approx(0.05)
    assert uniform_mass(5, OPEN) == pytest.approx(0.0045)
    assert baseline_score("numeric", 0.1, bucket=0, scaling=OPEN) == pytest.approx(
        34.6574, abs=1e-4
    )


def test_numeric_head_to_head_is_halved() -> None:
    # 100 * ln(0.02 / 0.01) / 2 = 50 * ln 2 = 34.6574
    assert head_to_head("numeric", 0.02, 0.01) == pytest.approx(34.6574, abs=1e-4)


def test_discrete_bucket_and_score_by_hand() -> None:
    # Outcomes 0..10, buckets centred on each: 11 inner buckets, closed bounds.
    # The outcome 3 sits at u = 3.5 / 11 = 0.31818, bucket int(3.5 + 1 - 1e-10) = 4.
    assert bucket_index("3", DISCRETE) == 4
    cdf = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]
    p = outcome_probability("discrete", {"cdf": cdf}, "3", scaling=DISCRETE)
    # pmf[4] = cdf[4] - cdf[3] = 0.4 - 0.2 = 0.2
    assert p == pytest.approx(0.2)
    # 100 * ln(0.2 / (1/11)) / 2 = 50 * ln(2.2) = 39.4229
    assert baseline_score("discrete", p, bucket=4, scaling=DISCRETE) == pytest.approx(
        39.4229, abs=1e-4
    )


def test_out_of_range_on_a_closed_bound_is_an_error() -> None:
    with pytest.raises(ScoringError, match="closed bound"):
        uniform_mass(0, CLOSED)
    half = Scaling(range_min=0, range_max=1, open_lower_bound=False, open_upper_bound=True)
    with pytest.raises(ScoringError, match="closed bound"):
        uniform_mass(0, half)
    assert uniform_mass(201, half) == pytest.approx(0.05)


def test_wrong_cdf_length_is_refused() -> None:
    with pytest.raises(ScoringError, match="201 values"):
        outcome_probability("numeric", {"cdf": [0.0, 1.0]}, "5", scaling=CLOSED)


# -- the bootstrap ---------------------------------------------------------


def test_bootstrap_is_seeded_and_brackets_the_mean() -> None:
    values = [float(v) for v in (3, -1, 4, 1, -5, 9, 2, 6, -5, 3, 5, 8)]
    a = clustered_bootstrap(values, resamples=2000, seed=7)
    b = clustered_bootstrap(values, resamples=2000, seed=7)
    assert a == b
    assert a.mean == pytest.approx(sum(values) / len(values))
    assert a.low < a.mean < a.high
    # The standard error of a mean is close to sd / sqrt(n) for independent values.
    n = len(values)
    mu = sum(values) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))
    assert a.se == pytest.approx(sd / math.sqrt(n), rel=0.2)
    # MDE at 80% power, 5% two-sided: (1.95996 + 0.84162) * se = 2.80158 * se
    assert a.mde == pytest.approx(2.80158 * a.se, rel=1e-4)


def test_clusters_widen_the_interval_when_questions_move_together() -> None:
    # Six clusters of five identical values: really six observations, not thirty.
    values = [v for v in (4.0, -2.0, 7.0, 1.0, -3.0, 5.0) for _ in range(5)]
    labels = [i for i in range(6) for _ in range(5)]
    independent = clustered_bootstrap(values, resamples=4000, seed=1)
    clustered = clustered_bootstrap(values, labels, resamples=4000, seed=1)
    assert clustered.clusters == 6
    assert clustered.se > independent.se * 1.8


def test_paired_difference() -> None:
    est = paired([0.1, 0.2, 0.3], [0.0, 0.1, 0.1], resamples=500)
    assert est.mean == pytest.approx((0.1 + 0.1 + 0.2) / 3)
    with pytest.raises(ScoringError, match="same questions"):
        paired([0.1], [0.1, 0.2])


def test_paired_many_is_paired_value_for_value() -> None:
    rng = random.Random(7)
    pairs: list[tuple[list[float], list[float], list[int] | None]] = []
    for k in (2, 3, 12, 12, 30, 1):
        a = [rng.random() for _ in range(k)]
        b = [rng.random() for _ in range(k)]
        # Even sizes get repeated cluster labels (unequal cluster sizes), odd ones none.
        labels = [rng.randrange(max(1, k // 2)) for _ in range(k)] if k % 2 == 0 else None
        pairs.append((a, b, labels))
    many = paired_many(pairs, resamples=700, seed=3)
    one = [paired(a, b, c, resamples=700, seed=3) for a, b, c in pairs]
    assert len(many) == len(one)
    for x, y in zip(many, one, strict=True):
        assert repr(x) == repr(y)  # bit-identical, NaN-safe for the one-cluster case
    with pytest.raises(ScoringError, match="same questions"):
        paired_many([([0.1], [0.1, 0.2], None)])


def test_one_value_has_no_interval() -> None:
    est = clustered_bootstrap([2.5])
    assert est.low == est.high == 2.5
    assert math.isnan(est.mde)
    with pytest.raises(ScoringError, match="nothing to score"):
        clustered_bootstrap([])
