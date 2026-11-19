"""Proper scores for every question type, and the statistics that compare two forecasters.

The scoring rules are Metaculus's, so a number here is in the tournament's
units:

- every type is scored on ``p``, the probability the forecast put on what
  happened: ``p`` or ``1 - p`` for binary, the option's probability for
  multiple choice, and for numeric and discrete the mass of the CDF bucket
  the outcome fell in (:func:`outcome_probability`);
- **baseline score**: ``100 * ln(p * N) / ln(N)`` for binary (N = 2) and
  multiple choice (N options), so a uniform forecast scores 0; for numeric
  and discrete, ``100 * ln(p / b) / 2`` where ``b`` is the mass a uniform
  forecast puts in that bucket (5% spread over the open tails);
- **head-to-head**: ``100 * ln(p_a / p_b)``, halved for numeric and
  discrete; a **peer score** is the mean head-to-head against every other
  forecaster;
- **Brier** for binary, ``(p_yes - outcome)^2``, and the **log score**
  ``ln(p)`` (natural log) for every type.

Comparisons are paired: both forecasters on the same questions, the
difference taken per question, and the interval from a bootstrap that
resamples clusters of questions (a question by default; questions that
share an opening week or a source when asked), because questions that move
together are not independent. Each interval comes with the minimum
detectable effect: the smallest true difference this many questions would
detect four times in five at the 5% level.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import NormalDist

from augur.distributions import Scaling

BELOW = "below_lower_bound"
ABOVE = "above_upper_bound"
# Outcome mass for a numeric bucket is never exactly zero in a valid CDF, but a
# hand-built or truncated one might be; the log needs a floor.
_FLOOR = 1e-12
TAIL_MASS = 0.05


class ScoringError(ValueError):
    """A forecast or resolution that cannot be scored, with what to fix."""


# -- the probability a forecast put on the outcome ------------------------


def binary_outcome(value: str) -> int:
    text = value.strip().lower()
    if text in ("yes", "1", "true"):
        return 1
    if text in ("no", "0", "false"):
        return 0
    raise ScoringError(f"binary resolution {value!r} is not yes or no")


def pmf_from_cdf(cdf: Sequence[float]) -> list[float]:
    """The bucket masses of a CDF: below the range, each inner bucket, above it."""
    if len(cdf) < 2:
        raise ScoringError("a CDF needs at least two values")
    inner = [b - a for a, b in pairwise(cdf)]
    return [cdf[0], *inner, 1.0 - cdf[-1]]


def bucket_index(value: str, scaling: Scaling) -> int:
    """The bucket of the PMF the resolution falls in, as Metaculus computes it.

    Bucket 0 is below the range, ``n + 1`` above it, and an inner value at
    unscaled position ``u`` is in bucket ``max(1, int(u * n + 1 - 1e-10))``.
    """
    n = scaling.inbound_outcome_count
    if value == BELOW:
        return 0
    if value == ABOVE:
        return n + 1
    try:
        x = float(value)
    except ValueError:
        raise ScoringError(f"numeric resolution {value!r} is not a number") from None
    u = scaling.to_unscaled(x)
    if u < 0:
        return 0
    if u > 1:
        return n + 1
    if u == 1:
        return n
    return max(int(u * n + 1 - 1e-10), 1)


def outcome_probability(
    question_type: str,
    values: Mapping[str, object],
    resolution: str,
    *,
    options: Sequence[str] | None = None,
    scaling: Scaling | None = None,
) -> float:
    """The probability ``values`` (a ledger ``values_json``) put on ``resolution``."""
    if question_type == "binary":
        p = _number(values.get("probability"), "probability")
        return p if binary_outcome(resolution) else 1.0 - p
    if question_type == "multiple_choice":
        probabilities = values.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise ScoringError("a multiple-choice forecast has no 'probabilities'")
        if resolution not in probabilities:
            known = ", ".join(map(str, options or probabilities))
            raise ScoringError(f"resolution {resolution!r} is not an option ({known})")
        return _number(probabilities[resolution], resolution)
    if question_type in ("numeric", "discrete"):
        if scaling is None:
            raise ScoringError("a numeric or discrete question needs its scaling to be scored")
        cdf = values.get("cdf")
        if not isinstance(cdf, Sequence) or len(cdf) != scaling.inbound_outcome_count + 1:
            raise ScoringError(
                f"the CDF must have {scaling.inbound_outcome_count + 1} values for this question"
            )
        pmf = pmf_from_cdf([float(v) for v in cdf])
        return pmf[bucket_index(resolution, scaling)]
    raise ScoringError(f"{question_type} questions are not scored")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScoringError(f"{name} is not a number ({value!r})")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ScoringError(f"{name} = {number} is not a probability")
    return number


def _log(p: float) -> float:
    return math.log(max(p, _FLOOR))


# -- the scores ----------------------------------------------------------


def is_continuous(question_type: str) -> bool:
    return question_type in ("numeric", "discrete")


def log_score(p: float) -> float:
    return _log(p)


def brier(p_yes: float, outcome: int) -> float:
    return (p_yes - outcome) ** 2


def baseline_score(
    question_type: str,
    p: float,
    *,
    option_count: int = 2,
    bucket: int | None = None,
    scaling: Scaling | None = None,
) -> float:
    """Metaculus's baseline score for a forecast that put ``p`` on the outcome."""
    if question_type in ("binary", "multiple_choice"):
        n = 2 if question_type == "binary" else option_count
        if n < 2:
            raise ScoringError("a multiple-choice question needs at least two options")
        return 100.0 * (_log(p) + math.log(n)) / math.log(n)
    if not is_continuous(question_type) or scaling is None or bucket is None:
        raise ScoringError("a numeric baseline score needs the question's scaling and bucket")
    return 100.0 * (_log(p) - math.log(uniform_mass(bucket, scaling))) / 2.0


def uniform_mass(bucket: int, scaling: Scaling) -> float:
    """The mass Metaculus's uniform baseline puts in ``bucket``."""
    n = scaling.inbound_outcome_count
    open_bounds = int(scaling.open_lower_bound) + int(scaling.open_upper_bound)
    if bucket in (0, n + 1):
        if not open_bounds:
            raise ScoringError("an out-of-range outcome on a question with closed bounds")
        return TAIL_MASS / open_bounds
    return (1.0 - TAIL_MASS * open_bounds) / n


def head_to_head(question_type: str, p_a: float, p_b: float) -> float:
    """``100 * ln(p_a / p_b)``, halved for numeric and discrete as Metaculus does."""
    value = 100.0 * (_log(p_a) - _log(p_b))
    return value / 2.0 if is_continuous(question_type) else value


def peer_score(question_type: str, p_self: float, p_others: Sequence[float]) -> float:
    """The mean head-to-head score against every other forecaster on the question."""
    if not p_others:
        raise ScoringError("a peer score needs at least one other forecaster")
    return sum(head_to_head(question_type, p_self, p) for p in p_others) / len(p_others)


# -- comparing two forecasters on the same questions ---------------------

Z_95 = NormalDist().inv_cdf(0.975)
Z_POWER_80 = NormalDist().inv_cdf(0.80)
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261119


@dataclass(frozen=True)
class Estimate:
    """A mean with its bootstrap interval, standard error and detectable effect."""

    mean: float
    low: float
    high: float
    n: int
    clusters: int
    se: float
    mde: float
    resamples: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def fmt(self, digits: int = 2) -> str:
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def clustered_bootstrap(
    values: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Estimate:
    """The mean of ``values`` with a percentile interval from resampling clusters.

    For a paired comparison pass the per-question differences. Each resample
    draws as many clusters as there are, with replacement, and takes the
    mean over every question in the drawn clusters. ``mde`` is
    ``(z_0.975 + z_0.80) * se``: the true mean difference this design would
    find significant (two-sided, 5%) with 80% power.
    """
    if not values:
        raise ScoringError("nothing to score: no question has both forecasts and a resolution")
    if clusters is not None and len(clusters) != len(values):
        raise ScoringError("one cluster label per value is needed")
    labels: Sequence[Hashable] = clusters if clusters is not None else range(len(values))
    groups: dict[Hashable, list[float]] = {}
    for label, value in zip(labels, values, strict=True):
        groups.setdefault(label, []).append(float(value))
    sums = [(sum(g), len(g)) for g in groups.values()]
    k = len(sums)
    mean = sum(values) / len(values)
    if k < 2:
        return Estimate(mean, mean, mean, len(values), k, math.nan, math.nan, 0, seed)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for _ in range(k):
            s, c = sums[rng.randrange(k)]
            total += s
            count += c
        means.append(total / count)
    means.sort()
    alpha = (1.0 - level) / 2.0
    low = means[min(resamples - 1, max(0, math.floor(alpha * resamples)))]
    high = means[min(resamples - 1, max(0, math.ceil((1.0 - alpha) * resamples) - 1))]
    centre = sum(means) / resamples
    se = math.sqrt(sum((m - centre) ** 2 for m in means) / (resamples - 1))
    return Estimate(mean, low, high, len(values), k, se, (Z_95 + Z_POWER_80) * se, resamples, seed)


def paired(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Estimate:
    """``mean(a - b)`` over the same questions, with the clustered interval."""
    if len(a) != len(b):
        raise ScoringError("a paired comparison needs both forecasters on the same questions")
    return clustered_bootstrap(
        [x - y for x, y in zip(a, b, strict=True)], clusters, resamples=resamples, seed=seed
    )
