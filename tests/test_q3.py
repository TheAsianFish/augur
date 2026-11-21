"""Q3's scoring pieces: the replacement rule, head-to-head scores and per-member reads."""

from __future__ import annotations

import math

from augur.q3 import NON_INFERIORITY, combined, h2h_scores, replacement_verdict
from augur.score import Estimate


def est(mean: float, low: float, high: float) -> Estimate:
    return Estimate(mean, low, high, 40, 40, 0.01, 0.03, 10_000, 1)


def test_a_variant_with_its_interval_below_zero_replaces_the_default() -> None:
    assert replacement_verdict(est(-0.03, -0.05, -0.001), cheaper=False).startswith("replaces")


def test_a_cheaper_variant_replaces_only_inside_the_margin() -> None:
    inside = est(0.002, -0.004, NON_INFERIORITY - 0.0001)
    outside = est(0.002, -0.004, NON_INFERIORITY + 0.0001)
    assert "cheaper" in replacement_verdict(inside, cheaper=True)
    assert replacement_verdict(outside, cheaper=True) == "default stays"
    # The margin is for cheaper variants only.
    assert replacement_verdict(inside, cheaper=False) == "default stays"


def test_head_to_head_is_metaculus_log_ratio_on_the_outcome() -> None:
    (yes,) = h2h_scores([0.8], [0.4], [1])
    (no,) = h2h_scores([0.8], [0.4], [0])
    assert math.isclose(yes, 100 * math.log(0.8 / 0.4))
    assert math.isclose(no, 100 * math.log(0.2 / 0.6))


def test_head_to_head_clips_like_the_brier_tables() -> None:
    (score,) = h2h_scores([1.0], [0.5], [0])
    assert math.isclose(score, 100 * math.log(0.01 / 0.5))


def test_combined_rounds_as_the_bot_records() -> None:
    assert combined([0.1, 0.2, 0.9]) == 0.2
    assert combined([0.1234, 0.2], "trimmed_mean") == 0.162
