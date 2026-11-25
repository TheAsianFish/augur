from __future__ import annotations

from datetime import UTC, datetime

import pytest

from augur.q2 import DiffRow
from augur.q13 import n_needed, seendate, verdict
from augur.score import Estimate


def _diff(low: float, high: float) -> DiffRow:
    est = Estimate(
        mean=(low + high) / 2,
        low=low,
        high=high,
        n=64,
        clusters=64,
        se=0.01,
        mde=0.03,
        resamples=1000,
        seed=1,
    )
    return DiffRow("G − O", est)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-0.06, -0.001, "moved"),  # entirely below 0
        (-0.05, 0.02, "not the limit"),  # includes 0, lower end above -0.055
        (0.001, 0.04, "not the limit"),  # entirely above 0: news hurt
        (-0.08, 0.01, "unresolved"),  # includes 0 and a Hindcast-sized gain
        (-0.055, 0.01, "unresolved"),  # the bar itself is not "above"
    ],
)
def test_verdict_follows_registration_item_5(low: float, high: float, expected: str) -> None:
    assert verdict(_diff(low, high)) == expected


def test_n_needed_is_the_normal_rule_at_80_percent_power() -> None:
    # ((1.96 + 0.8416) * 0.0765 / 0.02) ** 2 = 114.9: M7's simulated 110 is close.
    assert n_needed(0.0765, 0.02) == 115
    assert n_needed(0.0765, 0.055) == 16
    assert n_needed(0.0, 0.02) is None
    assert n_needed(0.05, 0.0) is None


def test_seendate_is_read_from_the_evidence() -> None:
    detail = "capture=20260601120000 original=https://x.org/a seendate=20260531T081500Z domain=x"
    assert seendate(detail) == datetime(2026, 5, 31, 8, 15, tzinfo=UTC)
    assert seendate("capture=20260601120000 original=https://x.org/a") is None
