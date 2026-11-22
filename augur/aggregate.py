"""Combining member forecasts: the rules M7 compares, for every question type.

Each rule is a pure function of the members' forecasts. The names in
:data:`RULES` are the public API (``predictions.aggregate(..., rule=...)``).

Binary, on the members' probabilities:

- ``median``: the median;
- ``trimmed_mean``: drop the ``floor(TRIM * n)`` lowest and highest, then the
  mean of the rest (with ``n < 5`` nothing is dropped, so it is the mean);
- ``mean_logodds``: each probability clipped to ``[0.01, 0.99]``, then the
  sigmoid of the mean log-odds;
- ``extremized_logodds``: ``sigmoid(EXTREMIZE_FACTOR * mean log-odds)``, same
  clipping.

Multiple choice: the binary rule on each option's probabilities (clipped
first for the log-odds rules), then renormalised with
:func:`augur.distributions.normalise_probabilities`.

Numeric and discrete, on CDF vectors (see :func:`combine_cdfs` for why):

- ``median``: the pointwise median of the CDFs;
- ``trimmed_mean``: the pointwise trimmed mean;
- ``mean_logodds``: the pointwise mean of ``logit(cdf)``, through the sigmoid;
- ``extremized_logodds``: *not* extremized; the same pooling as
  ``mean_logodds`` (factor 1), under a method name that says so.

Every combined CDF is checked with :func:`validate_cdf`; one that fails is
replaced by the mean of the CDFs (the mixture, always valid), and the method
name ends in ``_fallback_mean_cdf`` so the substitution is visible.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from augur.distributions import Scaling, mix_cdfs, normalise_probabilities, validate_cdf

RULES: tuple[str, ...] = ("median", "trimmed_mean", "mean_logodds", "extremized_logodds")
EXTREMIZE_FACTOR = 1.5
"""The extremizing factor, the same as the replay evaluation's ``EXTREMIZE``."""
TRIM = 0.2
"""The share of members dropped at each end by ``trimmed_mean``."""
CLIP = (0.01, 0.99)
"""Probabilities are clipped into this range before log-odds (binary, choice)."""
_CDF_EPS = 1e-9
"""CDF values are clipped into ``[eps, 1 - eps]`` before log-odds (only a closed
bound's exact 0 or 1 ever needs it, and those ends are pinned instead)."""
_CDF_DECIMALS = 10


def median(probabilities: Sequence[float]) -> float:
    if not probabilities:
        raise ValueError("cannot aggregate zero forecasts")
    for p in probabilities:
        if not 0.0 < p < 1.0:
            raise ValueError(f"probability {p} is outside (0, 1)")
    return float(statistics.median(probabilities))


def check_rule(rule: str) -> None:
    """``ValueError`` naming the known rules when ``rule`` is not one."""
    if rule not in RULES:
        raise ValueError(f"unknown aggregation rule {rule!r}; use one of {', '.join(RULES)}")


def _clip(p: float, lo: float = CLIP[0], hi: float = CLIP[1]) -> float:
    return min(max(p, lo), hi)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def trimmed_mean(values: Sequence[float], trim: float = TRIM) -> float:
    """The mean after dropping ``floor(trim * n)`` values at each end."""
    if not values:
        raise ValueError("cannot aggregate zero forecasts")
    ordered = sorted(float(v) for v in values)
    k = math.floor(trim * len(ordered))
    kept = ordered[k : len(ordered) - k] if k else ordered
    return math.fsum(kept) / len(kept)


def mean_logodds(probabilities: Sequence[float], factor: float = 1.0) -> float:
    """``sigmoid(factor * mean(logit(clip(p))))``, each ``p`` clipped to :data:`CLIP`."""
    if not probabilities:
        raise ValueError("cannot aggregate zero forecasts")
    return _sigmoid(factor * statistics.fmean(_logit(_clip(p)) for p in probabilities))


def _combine(values: Sequence[float], rule: str) -> float:
    """One rule on a list of probabilities, without the binary range check."""
    if not values:
        raise ValueError("cannot aggregate zero forecasts")
    if rule == "median":
        return float(statistics.median(values))
    if rule == "trimmed_mean":
        return trimmed_mean(values)
    if rule == "mean_logodds":
        return mean_logodds(values)
    if rule == "extremized_logodds":
        return mean_logodds(values, EXTREMIZE_FACTOR)
    check_rule(rule)
    raise AssertionError("unreachable")


def combine_binary(probabilities: Sequence[float], rule: str) -> float:
    """The members' probabilities combined by ``rule`` (unrounded)."""
    check_rule(rule)
    if rule == "median":
        return median(probabilities)
    if not probabilities:
        raise ValueError("cannot aggregate zero forecasts")
    for p in probabilities:
        if not 0.0 < p < 1.0:
            raise ValueError(f"probability {p} is outside (0, 1)")
    return _combine(probabilities, rule)


def combine_choice(
    members: Sequence[Mapping[str, float]], options: Sequence[str], rule: str
) -> dict[str, float]:
    """Each option's probabilities combined by ``rule``, then renormalised to sum to one.

    The log-odds rules clip each option's probability to :data:`CLIP` first,
    and the extremized rule extremizes each option before renormalising.
    """
    check_rule(rule)
    if not members:
        raise ValueError("cannot aggregate zero forecasts")
    per_option = {o: _combine([m[o] for m in members], rule) for o in options}
    return normalise_probabilities(per_option, options)


def _pointwise_cdf(cdfs: Sequence[Sequence[float]], rule: str, scaling: Scaling) -> list[float]:
    n = len(cdfs[0])
    columns = [[float(c[i]) for c in cdfs] for i in range(n)]
    if rule == "median":
        out = [float(statistics.median(col)) for col in columns]
    elif rule == "trimmed_mean":
        out = [trimmed_mean(col) for col in columns]
    else:
        out = [
            _sigmoid(statistics.fmean(_logit(_clip(v, _CDF_EPS, 1.0 - _CDF_EPS)) for v in col))
            for col in columns
        ]
        if not scaling.open_lower_bound:
            out[0] = 0.0
        if not scaling.open_upper_bound:
            out[-1] = 1.0
    return [round(v * 10.0**_CDF_DECIMALS) / 10.0**_CDF_DECIMALS for v in out]


_CDF_METHODS = {
    "median": "median_cdf",
    "trimmed_mean": "trimmed_mean_cdf",
    "mean_logodds": "mean_logodds_cdf",
    "extremized_logodds": "mean_logodds_cdf_unextremized",
}


def combine_cdfs(
    cdfs: Sequence[Sequence[float]], scaling: Scaling, rule: str
) -> tuple[list[float], str]:
    """The members' CDFs combined by ``rule``, and the name of the method used.

    Each rule acts pointwise, on the members' CDF values at each grid point:

    - ``median`` and ``trimmed_mean``: every order statistic of
      non-decreasing functions is non-decreasing, and if each member rises
      by at least ``d`` (at most ``D``) over a step, so does every order
      statistic, hence their median or trimmed mean. Both ends stay within
      their bounds. So these keep every rule :func:`validate_cdf` checks
      (up to rounding to 10 places).
    - ``mean_logodds``: ``sigmoid(mean(logit(F_i(x))))``. Each ``logit(F_i)``
      is increasing, so their mean is, and the result is a monotone function
      between the members' extremes at each point (so an open bound's
      ``[0.001, 0.999]`` holds). A closed bound's exact 0 or 1 has no
      log-odds; those ends are pinned to 0 and 1. The minimum and maximum
      step are *not* guaranteed: a sharp member's jump in log-odds survives
      the pooling, and near 0.5 it can become a step above the server's
      maximum (a point mass pooled with a wide member does it); in a tail the
      pooling behaves like a geometric mean and can flatten a step.
    - ``extremized_logodds``: a CDF is not extremized. Multiplying the
      log-odds of a CDF by a factor sharpens it around wherever it crosses
      0.5, which is the median, a choice of centre that has nothing to do
      with confidence in an outcome; it would also move an open bound's tail
      mass. This rule therefore pools like ``mean_logodds`` (factor 1), and
      the method name, ``mean_logodds_cdf_unextremized``, says so.

    A combined CDF that fails :func:`validate_cdf` is replaced by the mean of
    the CDFs (:func:`mix_cdfs`, valid whenever the members are), and the
    method name gets ``_fallback_mean_cdf`` appended.
    """
    check_rule(rule)
    if not cdfs:
        raise ValueError("cannot aggregate zero forecasts")
    lengths = {len(c) for c in cdfs}
    if len(lengths) != 1:
        raise ValueError(f"CDFs to combine must have one length; got {sorted(lengths)}")
    combined = _pointwise_cdf(cdfs, rule, scaling)
    method = _CDF_METHODS[rule]
    if validate_cdf(combined, scaling):
        return mix_cdfs(cdfs), method + "_fallback_mean_cdf"
    return combined, method


def choice_method(rule: str) -> str:
    """The method name a multiple-choice aggregate records for ``rule``."""
    check_rule(rule)
    return f"{rule}_per_option_normalised"
