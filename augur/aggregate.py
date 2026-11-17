"""Combining member forecasts. M1 has the median for binary questions."""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def median(probabilities: Sequence[float]) -> float:
    if not probabilities:
        raise ValueError("cannot aggregate zero forecasts")
    for p in probabilities:
        if not 0.0 < p < 1.0:
            raise ValueError(f"probability {p} is outside (0, 1)")
    return float(statistics.median(probabilities))
