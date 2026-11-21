"""How many questions it takes to see an improvement: a planted-effect simulation.

Start from the per-question differences of a real paired comparison on the
backtest (variant minus default, one value per question). Centre them, so
the true difference is 0 but the spread, skew and outliers are the real
ones, then plant a known improvement δ. For each n, draw n questions with
replacement many times and count how often the 95% interval of the mean
excludes 0 in the improvement's direction. That share is the power at n;
at δ = 0 it is the false-alarm rate (about 2.5% one-sided), which checks the
simulation itself.

The interval in each draw is the normal one (mean ± 1.96 standard errors),
not the bootstrap the tables use: ten thousand resamples inside each of
thousands of simulated studies is out of reach in pure Python, and at the
n where power is decided (hundreds and up) the two agree closely.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from augur.score import Z_95, Z_POWER_80

GRID = (25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800)
SIMS = 1000
SEED = 20261121
TARGET = 0.8


@dataclass(frozen=True)
class Curve:
    """Detection rate by n for one planted effect."""

    delta: float
    rates: tuple[float, ...]
    """One rate per ``grid`` entry."""
    grid: tuple[int, ...]

    def needed(self, target: float = TARGET) -> int | None:
        """The smallest n (log-interpolated between grid points) detected at ``target``."""
        prev_n, prev_r = 0, 0.0
        for n, r in zip(self.grid, self.rates, strict=True):
            if r >= target:
                if prev_n == 0 or r == prev_r:
                    return n
                # Interpolate in log n: power curves are close to linear there.
                frac = (target - prev_r) / (r - prev_r)
                return round(math.exp(math.log(prev_n) + frac * (math.log(n) - math.log(prev_n))))
            prev_n, prev_r = n, r
        return None


def analytic_n(sd: float, delta: float) -> int:
    """The textbook n for 80% power, two-sided 5%: ((z_.975 + z_.8) · sd / δ)²."""
    return math.ceil(((Z_95 + Z_POWER_80) * sd / delta) ** 2)


def simulate(
    differences: Sequence[float],
    deltas: Sequence[float],
    *,
    grid: Sequence[int] = GRID,
    sims: int = SIMS,
    seed: int = SEED,
) -> list[Curve]:
    """Power curves for each planted improvement ``delta`` (positive = better).

    ``differences`` are oriented so that larger is better for the variant
    (pass default − variant for Brier). The same resampled studies serve
    every delta: planting shifts each study's mean and leaves its spread.
    """
    if len(differences) < 2:
        raise ValueError("a power simulation needs at least two per-question differences")
    centre = statistics.fmean(differences)
    values = [d - centre for d in differences]
    rng = random.Random(seed)
    hits = {d: [0] * len(grid) for d in deltas}
    for i, n in enumerate(grid):
        for _ in range(sims):
            draw = rng.choices(values, k=n)
            mean = math.fsum(draw) / n
            var = math.fsum((x - mean) ** 2 for x in draw) / (n - 1)
            half = Z_95 * math.sqrt(var / n)
            for d in deltas:
                if mean + d - half > 0.0:
                    hits[d][i] += 1
    return [Curve(d, tuple(h / sims for h in hits[d]), tuple(grid)) for d in deltas]
