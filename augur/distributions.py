"""Numeric, discrete and multiple-choice forecasts in the form Metaculus accepts.

Scaling
-------
A numeric or discrete question has a range ``[range_min, range_max]`` and a
CDF of ``n + 1`` values, ``n = inbound_outcome_count`` (200 for numeric, the
number of outcomes for discrete). Metaculus works in *unscaled* units
``u`` in ``[0, 1]``; ``cdf[i]`` is ``P(outcome <= x(i / n))`` where ``x`` maps
``u`` to the question's own (scaled) units, and ``cdf[0]`` is
``P(outcome < range_min)``. Without a ``zero_point`` the map is linear. With
one, it is logarithmic: ``deriv_ratio = (range_max - zero_point) /
(range_min - zero_point)`` and
``x = range_min + (range_max - range_min) * (deriv_ratio**u - 1) / (deriv_ratio - 1)``
(``utils/the_math/formulas.py`` in github.com/Metaculus/metaculus). The code
here computes the same thing through ``expm1``/``log1p`` of
``deriv_ratio - 1`` so a zero point far from the range (ratio close to 1)
does not lose precision.

Validation
----------
:func:`validate_cdf` mirrors ``continuous_validation`` in Metaculus's
``questions/serializers/common.py``: round the CDF to 10 places and its steps
to 9 (numpy's ``round``: scale, round half to even, unscale); then

* ``len(cdf) == n + 1``;
* every step is at least ``round(0.01 / n, 9)`` (so at least 1% of the mass
  is inside the range, spread over every bucket);
* every step is at most ``0.2 * 200 / n`` (binding only when ``n > 40``);
* an open lower bound needs ``cdf[0] >= 0.001``, a closed one ``cdf[0] == 0``;
  an open upper bound needs ``cdf[-1] <= 0.999``, a closed one ``cdf[-1] == 1``.

A multiple-choice forecast needs every option in ``[0.001, 0.999]`` and a sum
that is ``numpy.isclose`` to 1 (``rtol=1e-5``, ``atol=1e-8``).
:func:`normalise_probabilities` normalises, clips into that range, and then
moves the excess (or deficit) the clipping caused onto the options in
proportion to how far each is from the bound it would cross, so one pass is
exact and nothing is clipped twice.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

DEFAULT_OUTCOME_COUNT = 200
OPEN_BOUND_MIN = 0.001
"""The least ``cdf[0]`` an open lower bound accepts (mass below the range)."""
OPEN_BOUND_MAX = 0.999
"""The most ``cdf[-1]`` an open upper bound accepts."""
PROBABILITY_MIN = 0.001
PROBABILITY_MAX = 0.999
_CDF_DECIMALS = 10
_PMF_DECIMALS = 9
_ISCLOSE_RTOL = 1e-5
_ISCLOSE_ATOL = 1e-8


def _np_round(x: float, decimals: int) -> float:
    """``numpy.round`` for one float: scale, round half to even, unscale."""
    scale = 10.0**decimals
    return round(x * scale) / scale


def min_step(outcome_count: int) -> float:
    """The smallest step between neighbouring CDF values the server accepts."""
    return round(0.01 / outcome_count, _PMF_DECIMALS)


def max_step(outcome_count: int) -> float:
    """The largest step between neighbouring CDF values the server accepts."""
    return 0.2 * DEFAULT_OUTCOME_COUNT / outcome_count


class Scaling(BaseModel):
    """A question's range and bounds, from the API's ``question["scaling"]``."""

    model_config = ConfigDict(frozen=True)

    range_min: float
    range_max: float
    zero_point: float | None = None
    open_lower_bound: bool
    open_upper_bound: bool
    inbound_outcome_count: int = DEFAULT_OUTCOME_COUNT
    nominal_min: float | None = None
    nominal_max: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not (math.isfinite(self.range_min) and math.isfinite(self.range_max)):
            raise ValueError("range_min and range_max must be finite numbers")
        if not self.range_max > self.range_min:
            raise ValueError(
                f"range_max ({self.range_max}) must be above range_min ({self.range_min})"
            )
        if self.inbound_outcome_count < 1:
            raise ValueError(
                f"inbound_outcome_count must be at least 1, got {self.inbound_outcome_count}"
            )
        zp = self.zero_point
        if zp is not None and (not math.isfinite(zp) or self.range_min <= zp <= self.range_max):
            raise ValueError(
                f"zero_point ({zp}) must be a finite number outside "
                f"[{self.range_min}, {self.range_max}]"
            )
        return self

    @classmethod
    def from_question(cls, question: Mapping[str, Any]) -> Scaling:
        """Read the scaling of an API question dict (``post["question"]``).

        The bounds may sit in ``scaling`` or, in older payloads, on the
        question itself; a missing ``inbound_outcome_count`` means 200.
        """
        scaling = question.get("scaling")
        if not isinstance(scaling, Mapping):
            raise ValueError(
                "question has no 'scaling' object; only numeric, discrete and date "
                "questions have a CDF"
            )

        def field(name: str) -> Any:
            value = scaling.get(name)
            return question.get(name) if value is None else value

        range_min, range_max = field("range_min"), field("range_max")
        if range_min is None or range_max is None:
            raise ValueError(
                "question scaling has no range_min/range_max; only numeric, discrete and "
                "date questions have a CDF"
            )
        bounds: dict[str, bool] = {}
        for name in ("open_lower_bound", "open_upper_bound"):
            value = field(name)
            if not isinstance(value, bool):
                raise ValueError(
                    f"question has no boolean {name!r} in 'scaling' or on the question "
                    f"(got {value!r})"
                )
            bounds[name] = value
        count = field("inbound_outcome_count")
        try:
            return cls(
                range_min=range_min,
                range_max=range_max,
                zero_point=field("zero_point"),
                inbound_outcome_count=DEFAULT_OUTCOME_COUNT if count is None else count,
                nominal_min=field("nominal_min"),
                nominal_max=field("nominal_max"),
                **bounds,
            )
        except ValueError as exc:
            raise ValueError(f"question scaling is unusable: {exc}") from exc

    @property
    def is_discrete_like(self) -> bool:
        """Fewer outcomes than a numeric question: a discrete question's buckets."""
        return self.inbound_outcome_count < DEFAULT_OUTCOME_COUNT

    def _log_ratio_minus_one(self) -> float | None:
        """``deriv_ratio - 1``, or None for a linear scale."""
        if self.zero_point is None:
            return None
        dm1 = (self.range_max - self.range_min) / (self.range_min - self.zero_point)
        return dm1 if dm1 != 0.0 else None

    def to_scaled(self, u: float) -> float:
        """The question's units at unscaled ``u`` (0 is range_min, 1 is range_max)."""
        width = self.range_max - self.range_min
        dm1 = self._log_ratio_minus_one()
        if dm1 is None:
            return self.range_min + width * u
        return self.range_min + width * math.expm1(u * math.log1p(dm1)) / dm1

    def to_unscaled(self, x: float) -> float:
        """The unscaled position of ``x``; outside ``[0, 1]`` beyond the range.

        On a log scale a value at or past the zero point has no position; it
        is ``-inf`` (zero point below the range) or ``inf`` (above).
        """
        width = self.range_max - self.range_min
        dm1 = self._log_ratio_minus_one()
        if dm1 is None:
            return (x - self.range_min) / width
        arg = (x - self.range_min) * dm1 / width
        if arg <= -1.0:
            return -math.inf if dm1 > 0 else math.inf
        return math.log1p(arg) / math.log1p(dm1)

    def locations(self) -> list[float]:
        """The ``n + 1`` scaled values the CDF is evaluated at."""
        n = self.inbound_outcome_count
        inner = [self.to_scaled(i / n) for i in range(1, n)]
        return [self.range_min, *inner, self.range_max]


def validate_cdf(cdf: Sequence[float], scaling: Scaling) -> list[str]:
    """The reasons Metaculus would reject ``cdf``; empty when it would accept it."""
    n = scaling.inbound_outcome_count
    values = [float(v) for v in cdf]
    if len(values) != n + 1:
        return [f"CDF must have {n + 1} values (inbound_outcome_count + 1), got {len(values)}"]
    if not all(math.isfinite(v) for v in values):
        return ["CDF values must all be finite numbers"]
    rounded = [_np_round(v, _CDF_DECIMALS) for v in values]
    pmf = [_np_round(b - a, _PMF_DECIMALS) for a, b in pairwise(rounded)]
    errors: list[str] = []
    least, most = min_step(n), max_step(n)
    flat = [i for i, p in enumerate(pmf) if p < least]
    if flat:
        errors.append(
            f"CDF must increase by at least {least} at every step; {len(flat)} steps do "
            f"not (first between index {flat[0]} and {flat[0] + 1}: {pmf[flat[0]]})"
        )
    steep = [i for i, p in enumerate(pmf) if p > most]
    if steep:
        errors.append(
            f"CDF must increase by at most {most} at every step; {len(steep)} steps do "
            f"not (first between index {steep[0]} and {steep[0] + 1}: {pmf[steep[0]]})"
        )
    first, last = rounded[0], rounded[-1]
    if scaling.open_lower_bound:
        if first < OPEN_BOUND_MIN:
            errors.append(
                f"lower bound is open: cdf[0] must be at least {OPEN_BOUND_MIN}, got {first}"
            )
    elif first != 0.0:
        errors.append(f"lower bound is closed: cdf[0] must be 0.0, got {first}")
    if scaling.open_upper_bound:
        if last > OPEN_BOUND_MAX:
            errors.append(
                f"upper bound is open: cdf[-1] must be at most {OPEN_BOUND_MAX}, got {last}"
            )
    elif last != 1.0:
        errors.append(f"upper bound is closed: cdf[-1] must be 1.0, got {last}")
    return errors


def mix_cdfs(
    cdfs: Sequence[Sequence[float]], weights: Sequence[float] | None = None
) -> list[float]:
    """The pointwise weighted mean of ``cdfs`` (weights default to equal).

    Every rule in :func:`validate_cdf` is closed under convex combination, so
    a mixture of CDFs valid for one scaling is valid for it too (up to the
    server's rounding, which the margins in :func:`cdf_from_percentiles`
    cover).
    """
    if not cdfs:
        raise ValueError("cannot mix zero CDFs")
    length = len(cdfs[0])
    if any(len(c) != length for c in cdfs):
        raise ValueError(f"CDFs to mix must have one length; got {sorted({len(c) for c in cdfs})}")
    w = [1.0] * len(cdfs) if weights is None else [float(x) for x in weights]
    if len(w) != len(cdfs):
        raise ValueError(f"got {len(w)} weights for {len(cdfs)} CDFs")
    if not all(math.isfinite(x) and x >= 0.0 for x in w):
        raise ValueError(f"weights must be finite and non-negative, got {w}")
    total = math.fsum(w)
    if total <= 0.0:
        raise ValueError("weights must not all be zero")
    share = [x / total for x in w]
    return [
        _np_round(math.fsum(s * float(c[i]) for s, c in zip(share, cdfs, strict=True)), 10)
        for i in range(length)
    ]


def quantile(cdf: Sequence[float], scaling: Scaling, level: float) -> float:
    """The scaled value at which ``cdf`` reaches ``level``, for comment text.

    Linear between the CDF's points in unscaled space; a level the CDF
    leaves below the range or above it gives range_min or range_max.
    """
    n = scaling.inbound_outcome_count
    if len(cdf) != n + 1:
        raise ValueError(f"CDF must have {n + 1} values, got {len(cdf)}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if level <= cdf[0]:
        return scaling.range_min
    if level >= cdf[-1]:
        return scaling.range_max
    i = bisect_left(cdf, level)  # cdf[i - 1] < level <= cdf[i]
    lo, hi = cdf[i - 1], cdf[i]
    u = (i - 1 + (level - lo) / (hi - lo)) / n
    return scaling.to_scaled(u)


def _check_options(options: Sequence[str]) -> None:
    if len(options) < 2:
        raise ValueError(f"a multiple-choice question needs at least 2 options, got {options}")
    if len(set(options)) != len(options):
        raise ValueError(f"options must be distinct, got {list(options)}")
    if len(options) * PROBABILITY_MIN > 1.0:
        raise ValueError(f"{len(options)} options cannot each get {PROBABILITY_MIN}")


def normalise_probabilities(
    probabilities: Mapping[str, float], options: Sequence[str]
) -> dict[str, float]:
    """Per-option probabilities Metaculus accepts, in the order of ``options``.

    A missing or NaN option counts as zero before normalising and ends at the
    floor, ``PROBABILITY_MIN``. After normalising, each value is clipped into
    ``[PROBABILITY_MIN, PROBABILITY_MAX]``; the clipping's net change is then
    taken from (or given to) the options in proportion to their distance from
    the bound it moves them towards, which keeps every value in range and
    makes the sum 1.
    """
    _check_options(options)
    known = set(options)
    unknown = [k for k in probabilities if k not in known]
    if unknown:
        raise ValueError(f"probabilities name options the question does not have: {unknown}")
    raw: list[float] = []
    for option in options:
        value = float(probabilities.get(option, math.nan))
        if math.isnan(value):
            value = 0.0
        if value < 0.0 or math.isinf(value):
            raise ValueError(f"probability for {option!r} must be finite and >= 0, got {value}")
        raw.append(value)
    total = math.fsum(raw)
    if total <= 0.0:
        raise ValueError("every option's probability is zero or missing; nothing to normalise")
    clipped = [min(max(v / total, PROBABILITY_MIN), PROBABILITY_MAX) for v in raw]
    excess = math.fsum(clipped) - 1.0
    if excess > 0.0:
        room = [v - PROBABILITY_MIN for v in clipped]
    else:
        room = [PROBABILITY_MAX - v for v in clipped]
    room_total = math.fsum(room)
    if excess != 0.0 and room_total > 0.0:
        clipped = [v - excess * r / room_total for v, r in zip(clipped, room, strict=True)]
    final = [min(max(v, PROBABILITY_MIN), PROBABILITY_MAX) for v in clipped]
    return dict(zip(options, final, strict=True))


def validate_probabilities(probabilities: Mapping[str, float], options: Sequence[str]) -> list[str]:
    """The reasons Metaculus would reject a multiple-choice forecast; empty if none."""
    errors: list[str] = []
    known = set(options)
    missing = [o for o in options if o not in probabilities]
    if missing:
        errors.append(f"every option needs a probability; missing {missing}")
    unknown = [k for k in probabilities if k not in known]
    if unknown:
        errors.append(f"probabilities name options the question does not have: {unknown}")
    values = [float(v) for v in probabilities.values()]
    if not all(math.isfinite(v) for v in values):
        return [*errors, "probabilities must all be finite numbers"]
    outside = {
        k: v for k, v in probabilities.items() if not PROBABILITY_MIN <= v <= PROBABILITY_MAX
    }
    if outside:
        errors.append(
            f"every probability must be in [{PROBABILITY_MIN}, {PROBABILITY_MAX}]; not {outside}"
        )
    total = math.fsum(values)
    if abs(total - 1.0) > _ISCLOSE_ATOL + _ISCLOSE_RTOL * 1.0:
        errors.append(f"probabilities must sum to 1, got {total}")
    return errors
