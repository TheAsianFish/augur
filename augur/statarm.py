"""The statistical arm: a forecast with no model in it (M12).

PLAN.md registers the arm before it was built (the ``m12-registration``
block); this module is that text in code. Per question type:

* **Binary.** The question's own price, where the markets searcher read an
  exact one dated at or before the cutoff. Otherwise the base rate of the
  question's source: the Laplace-smoothed frequency ``(yes + 1) / (n + 2)``
  of that source's outcomes public before the question opened, or of every
  source's outcomes when that source has fewer than :data:`MIN_SOURCE_HISTORY`.
* **Numeric and discrete** about a quoted series
  (:func:`augur.research.series.match_series`): the last admitted close
  times ``exp(c)``, ``c`` drawn from the empirical distribution of the
  series' own log changes over ``h`` observations in its last two years,
  turned into a CDF by :func:`augur.distributions.cdf_from_percentiles`.
* **Everything else** (multiple choice, a question with no series): the arm
  abstains and says why.

Two combinations with Augur's own forecast, both at M5's fixed weight
(:data:`augur.replay.BLEND_WEIGHT`, one-third Augur):

* :func:`primary` — the registered candidate: blended **only where the arm
  has a price or a series**, Augur's forecast unchanged elsewhere;
* :func:`base_rate_blend` — reported, never a candidate: blended wherever
  the arm has any value, base rates included.

Binary blends are in log-odds with both inputs clipped to [0.01, 0.99];
continuous blends are a mixture of CDFs, which keeps every validity rule
Metaculus checks (:func:`augur.distributions.mix_cdfs`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from augur.calibrate import BaseRateMap
from augur.distributions import Scaling, cdf_from_percentiles, mix_cdfs, quantile
from augur.errors import ConfigError
from augur.predictions import (
    PERCENTILE_LEVELS,
    BinaryPrediction,
    ContinuousPrediction,
    Prediction,
)
from augur.replay import BLEND_WEIGHT, blend

MIN_SOURCE_HISTORY = 20
"""A source with fewer resolutions than this falls back to the pooled base rate."""
MIN_CHANGES = 100
"""The fewest h-step changes the series rule needs; fewer and it abstains."""
LOOKBACK_DAYS = 730
"""The series rule reads the changes of the last two years of daily closes."""
RATE_WINDOW_DAYS = 365
"""``h`` is set from how many observations the series had in this many days."""
LEVELS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
"""The levels of the empirical distribution passed to ``cdf_from_percentiles``."""
BASE_RATES_PATH = Path(__file__).resolve().parent / "stat_base_rates.json"
"""Per-source outcome counts from every C4 resolution, written by ``augur statarm``."""

Kind = Literal["price", "base-rate", "series"]
ANCHORS: frozenset[str] = frozenset({"price", "series"})
"""The arm values the primary candidate blends with; a base rate is not one."""


@dataclass(frozen=True)
class ArmForecast:
    """What the arm says, and from what."""

    kind: Kind
    detail: str
    probability: float | None = None
    """Binary arms only."""
    cdf: tuple[float, ...] | None = None
    """Series arms only: a CDF valid for the question's scaling."""
    percentiles: tuple[tuple[float, float], ...] = ()

    def prediction(self) -> Prediction:
        if self.probability is not None:
            return BinaryPrediction(self.probability)
        if self.cdf is None:
            raise ValueError("internal: an arm forecast with neither a probability nor a CDF")
        return ContinuousPrediction(self.cdf, self.percentiles)


@dataclass(frozen=True)
class Abstain:
    """The arm has nothing to say about this question, and why."""

    reason: str


Arm = ArmForecast | Abstain


# -- base rates -----------------------------------------------------------------------


@dataclass(frozen=True)
class BaseRates:
    """Outcome counts per source: ``source -> (yes, n)``."""

    counts: Mapping[str, tuple[int, int]]
    fitted_on: str = ""

    @classmethod
    def fit(cls, outcomes: Iterable[tuple[str, int]], fitted_on: str = "") -> BaseRates:
        counts: dict[str, tuple[int, int]] = {}
        for source, y in outcomes:
            if y not in (0, 1):
                raise ValueError(f"an outcome must be 0 or 1, got {y!r}")
            yes, n = counts.get(source, (0, 0))
            counts[source] = (yes + y, n + 1)
        return cls(dict(sorted(counts.items())), fitted_on)

    def rate(self, source: str) -> tuple[float, str]:
        """The base rate for ``source`` and a line saying what it rests on."""
        yes, n = self.counts.get(source, (0, 0))
        if n >= MIN_SOURCE_HISTORY:
            return BaseRateMap(yes=yes, n=n).rate, f"{source} base rate, {yes} of {n} resolved yes"
        pooled_yes = sum(c[0] for c in self.counts.values())
        pooled_n = sum(c[1] for c in self.counts.values())
        return (
            BaseRateMap(yes=pooled_yes, n=pooled_n).rate,
            f"pooled base rate, {pooled_yes} of {pooled_n} resolved yes "
            f"({source} has {n}, fewer than {MIN_SOURCE_HISTORY})",
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "fitted_on": self.fitted_on,
            "counts": {s: {"yes": c[0], "n": c[1]} for s, c in self.counts.items()},
        }


def load_base_rates(path: Path = BASE_RATES_PATH) -> BaseRates:
    """The committed per-source base rates; a broken file is a configuration error."""
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        counts = {str(s): (int(c["yes"]), int(c["n"])) for s, c in data["counts"].items()}
        return BaseRates(counts, str(data.get("fitted_on", "")))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(
            f"cannot read the statistical arm's base rates {path} ({exc}); re-create them with "
            "augur statarm, or run without the statistical arm"
        ) from None


def binary_arm(price: float | None, source: str, rates: BaseRates, price_detail: str = "") -> Arm:
    """The price where there is one, else the source's base rate."""
    if price is not None:
        if not 0.0 <= price <= 1.0:
            raise ValueError(f"a price must be in [0, 1], got {price}")
        return ArmForecast("price", price_detail or f"price {price:.3f}", probability=price)
    if not rates.counts:
        return Abstain("no price, and no resolved history for a base rate")
    rate, detail = rates.rate(source)
    return ArmForecast("base-rate", detail, probability=rate)


# -- the series rule ------------------------------------------------------------------


def horizon(closes: Sequence[tuple[date, float]], target: date) -> int:
    """Observations expected between the last close and ``target``.

    The series' own observation rate over the last :data:`RATE_WINDOW_DAYS`
    (about 252 a year for a stock, 365 for a coin) times the calendar days to
    ``target``, rounded, and at least 1. The rate is taken over the days the
    window actually spans, so a series younger than the window (a recent
    listing) is not read as one that trades less often.
    """
    last = closes[-1][0]
    window = [d for d, _ in closes if (last - d).days < RATE_WINDOW_DAYS]
    span = (last - window[0]).days
    per_day = (len(window) - 1) / span if span > 0 else 1.0
    return max(1, round((target - last).days * per_day))


def changes(closes: Sequence[tuple[date, float]], h: int) -> list[float]:
    """Every overlapping ``h``-observation log change in the last :data:`LOOKBACK_DAYS`."""
    last = closes[-1][0]
    values = [v for d, v in closes if (last - d).days <= LOOKBACK_DAYS]
    return [math.log(values[i + h] / values[i]) for i in range(len(values) - h)]


def empirical_quantile(sorted_values: Sequence[float], level: float) -> float:
    """Linear interpolation between order statistics (numpy's default rule)."""
    pos = level * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def series_arm(
    closes: Sequence[tuple[date, float]], target: date, scaling: Scaling, label: str
) -> Arm:
    """The series rule: last close times the empirical h-step change distribution.

    ``closes`` must already be the admitted observations (dated at or before
    the cutoff), oldest first; ``label`` names the series in the detail line.
    """
    if not closes:
        return Abstain(f"{label}: no close dated at or before the cutoff")
    last_day, last = closes[-1]
    if target <= last_day:
        return Abstain(f"{label}: the target day {target} is not after the last close {last_day}")
    if not scaling.range_min <= last <= scaling.range_max:
        return Abstain(
            f"{label}: the last close {last:g} is outside the question's range "
            f"[{scaling.range_min:g}, {scaling.range_max:g}] (a unit mismatch)"
        )
    h = horizon(closes, target)
    cs = sorted(changes(closes, h))
    if len(cs) < MIN_CHANGES:
        return Abstain(f"{label}: {len(cs)} past {h}-observation changes, fewer than {MIN_CHANGES}")
    pairs = [(lv, last * math.exp(empirical_quantile(cs, lv))) for lv in LEVELS]
    cdf = cdf_from_percentiles(pairs, scaling)
    shown = tuple((lv, quantile(cdf, scaling, lv)) for lv in PERCENTILE_LEVELS)
    detail = (
        f"{label}: last close {last:g} on {last_day}, {h}-observation log changes to "
        f"{target} from {len(cs)} windows in the last two years "
        f"(5% {math.exp(empirical_quantile(cs, 0.05)) - 1:+.1%}, "
        f"95% {math.exp(empirical_quantile(cs, 0.95)) - 1:+.1%})"
    )
    return ArmForecast("series", detail, cdf=tuple(cdf), percentiles=shown)


# -- the blends -----------------------------------------------------------------------


def _combine(augur: Prediction, arm: ArmForecast, scaling: Scaling | None) -> Prediction:
    if isinstance(augur, BinaryPrediction):
        if arm.probability is None:
            raise ValueError("internal: a binary forecast blended with a non-binary arm")
        return BinaryPrediction(round(blend(augur.probability, arm.probability), 4))
    if isinstance(augur, ContinuousPrediction):
        if arm.cdf is None or scaling is None:
            raise ValueError("internal: a continuous forecast blended without a CDF and scaling")
        mixed = mix_cdfs([augur.cdf, arm.cdf], [BLEND_WEIGHT, 1.0 - BLEND_WEIGHT])
        shown = tuple((lv, quantile(mixed, scaling, lv)) for lv in PERCENTILE_LEVELS)
        return ContinuousPrediction(tuple(mixed), shown)
    raise ValueError("internal: the arm never answers multiple choice")


def primary(augur: Prediction, arm: Arm, scaling: Scaling | None = None) -> tuple[Prediction, str]:
    """The registered candidate: blended only with a price or a series."""
    if isinstance(arm, ArmForecast) and arm.kind in ANCHORS:
        return _combine(augur, arm, scaling), f"one-third Augur, two-thirds {arm.kind}"
    why = arm.reason if isinstance(arm, Abstain) else f"the arm has only a {arm.kind}"
    return augur, f"Augur unchanged ({why})"


def base_rate_blend(
    augur: Prediction, arm: Arm, scaling: Scaling | None = None
) -> tuple[Prediction, str]:
    """Reported, never a candidate: blended with any arm value, base rates included."""
    if isinstance(arm, ArmForecast):
        return _combine(augur, arm, scaling), f"one-third Augur, two-thirds {arm.kind}"
    return augur, f"Augur unchanged ({arm.reason})"
