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

From percentiles to a CDF
-------------------------
:func:`cdf_from_percentiles` takes a forecaster's ``(level, value)`` pairs
and returns a CDF that passes :func:`validate_cdf` for any input it does not
reject. The steps, in order:

1. **Unscale.** Each value moves to unscaled ``u``. Everything below is done
   in ``u``, so on a log-scaled question "linear between percentiles" means
   log-linear in the question's units, the shape Metaculus's own slider
   gives. A value past a log scale's zero point (no position) or absurdly far
   outside the range is clamped to ``|u| <= 1e6``.
2. **Repair.** Pairs are sorted by level. A sloppy answer whose values fall
   as the level rises is replaced by its isotonic (least-squares monotone)
   fit, pool-adjacent-violators: each violating run is replaced by its mean,
   the smallest change that makes the values non-decreasing. Pairs with the
   same level are then merged into their mean value. Equal values for
   different levels survive: they are a point mass, a vertical step.
3. **Interpolate.** Between knots the CDF is linear in ``u`` (right-
   continuous at a vertical step, so a point mass is counted at its value).
4. **Tails.** Below the lowest knot ``(u0, p0)`` the CDF is
   ``p0 * exp((u - u0) / s)``, and symmetrically above the highest. The scale
   ``s`` is chosen so the density is continuous at the knot: it is ``p0``
   over the slope of the outermost segment of positive width, so the tail
   decays over about the distance the outer percentiles are apart. It holds
   exactly the mass the forecaster left outside the outer percentile, never
   reaches zero (an open bound naturally gets some mass beyond it), and
   needs no arbitrary end point. With no segment of positive width (one
   percentile, or all values equal) the forecaster stated no spread, and
   ``s`` is one bucket, ``1 / n``.
5. **Bounds.** The CDF is evaluated at the ``n + 1`` grid points. A closed
   bound pins its end to exactly 0 or 1, so mass the forecast put beyond it
   lands in the edge bucket. An open bound clips the CDF into
   ``[0.001, 0.999]`` at that end.
6. **Mass inside.** The minimum step forces at least ``n * min_step = 1%`` of
   the mass inside the range. If ``cdf[-1] - cdf[0]`` is below
   ``_MIN_INBOUND_MASS`` (5%: a forecast almost entirely beyond an open
   bound), the open ends move apart by the shortfall, shared in proportion to
   the room each has before its bound limit, which puts it on the side where
   the forecast's mass is. This is the one step that overrides the forecast
   materially, and only because the server requires it.
7. **Too steep.** Where a bucket holds more than 99% of the maximum step (a
   point mass on a 200-bucket question), the excess is handed to the nearest
   buckets with room, equally to the left and right at each distance,
   filling each to the cap at most. A bucket is never filled past the cap,
   so each move fixes one bucket without breaking another, and the loop ends
   after at most ``n`` moves; the mass stays as close to where it was as the
   rule allows. The ends of the CDF do not move.
8. **Too flat.** The bucket masses are mixed with a uniform spread of the
   same total, ``pmf' = (1 - e) * pmf + e * (cdf[-1] - cdf[0]) / n``, with
   the smallest ``e`` that lifts the emptiest bucket to 1.05 times the minimum
   step plus ``1e-8``. Both terms are within the maximum step, so the mixture
   is too, the ends do not move, and the margin is far larger than the
   ``1e-9`` that rounding to 10 and then 9 places can take off a step. For a
   typical forecast ``e`` is about 1%, so each CDF value moves by at most
   about 0.01.
9. **Round and check.** The CDF is rounded to 10 places and validated; a
   failure there is a bug in this module (the property tests look for one),
   reported as an ``AssertionError``.

A multiple-choice forecast needs every option in ``[0.001, 0.999]`` and a sum
that is ``numpy.isclose`` to 1 (``rtol=1e-5``, ``atol=1e-8``).
:func:`normalise_probabilities` normalises, clips into that range, and then
moves the excess (or deficit) the clipping caused onto the options in
proportion to how far each is from the bound it would cross, so one pass is
exact and nothing is clipped twice.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
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
_U_LIMIT = 1e6
_MIN_TAIL_SCALE = 1e-12
_MIN_INBOUND_MASS = 0.05
_MAX_STEP_SHARE = 0.99
_MIN_STEP_FACTOR = 1.05
_MIN_STEP_MARGIN = 1e-8


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


def _isotonic(values: Sequence[float]) -> list[float]:
    """The least-squares non-decreasing fit (pool adjacent violators)."""
    blocks: list[tuple[float, int]] = []  # (sum, count)
    for v in values:
        total, count = v, 1
        while blocks and blocks[-1][0] / blocks[-1][1] > total / count:
            prev_total, prev_count = blocks.pop()
            total, count = total + prev_total, count + prev_count
        blocks.append((total, count))
    return [total / count for total, count in blocks for _ in range(count)]


def _knots(
    percentiles: Sequence[tuple[float, float]], scaling: Scaling
) -> tuple[list[float], list[float]]:
    """Repaired knots: unscaled values (non-decreasing) and levels (increasing)."""
    if not percentiles:
        raise ValueError("no percentiles to build a CDF from; the forecast needs at least one")
    pairs: list[tuple[float, float]] = []
    for item in percentiles:
        level, value = float(item[0]), float(item[1])
        if not (math.isfinite(level) and math.isfinite(value)):
            raise ValueError(f"percentile ({level}, {value}) is not a pair of finite numbers")
        if not 0.0 < level < 1.0:
            raise ValueError(f"percentile level {level} is outside (0, 1)")
        u = min(max(scaling.to_unscaled(value), -_U_LIMIT), _U_LIMIT)
        pairs.append((level, u))
    pairs.sort()
    fitted = _isotonic([u for _, u in pairs])
    us: list[float] = []
    ps: list[float] = []
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        us.append(math.fsum(fitted[i:j]) / (j - i))
        ps.append(pairs[i][0])
        i = j
    # The mean of a non-decreasing run can drift by an ulp; keep the knots ordered.
    for k in range(1, len(us)):
        us[k] = max(us[k], us[k - 1])
    return us, ps


def _tail_scales(us: Sequence[float], ps: Sequence[float], n: int) -> tuple[float, float]:
    """Exponential tail scales that keep the density continuous at the outer knots."""
    segments = [k for k in range(len(us) - 1) if us[k + 1] > us[k]]
    if not segments:
        return 1.0 / n, 1.0 / n
    lo, hi = segments[0], segments[-1]
    lo_slope = (ps[lo + 1] - ps[lo]) / (us[lo + 1] - us[lo])
    hi_slope = (ps[hi + 1] - ps[hi]) / (us[hi + 1] - us[hi])
    return (
        max(ps[0] / lo_slope, _MIN_TAIL_SCALE),
        max((1.0 - ps[-1]) / hi_slope, _MIN_TAIL_SCALE),
    )


def _evaluate(
    us: Sequence[float], ps: Sequence[float], scales: tuple[float, float], u: float
) -> float:
    """The repaired forecast's CDF at unscaled ``u``."""
    k = bisect_right(us, u) - 1  # the last knot at or below u
    if k < 0:
        return ps[0] * math.exp((u - us[0]) / scales[0])
    if k == len(us) - 1:
        return 1.0 - (1.0 - ps[-1]) * math.exp(-(u - us[-1]) / scales[1])
    return ps[k] + (ps[k + 1] - ps[k]) * (u - us[k]) / (us[k + 1] - us[k])


def _cap_steps(pmf: list[float], cap: float) -> list[float]:
    """Move each bucket's mass above ``cap`` to the nearest buckets with room."""
    n = len(pmf)
    for _ in range(n):
        i = max(range(n), key=pmf.__getitem__)
        if pmf[i] <= cap:
            break
        excess = pmf[i] - cap
        pmf[i] = cap
        for d in range(1, n):
            targets = [j for j in (i - d, i + d) if 0 <= j < n and pmf[j] < cap]
            for j in targets:
                give = min(excess / len(targets), cap - pmf[j])
                pmf[j] += give
                excess -= give
            if excess <= 0.0:
                break
    return pmf


def cdf_from_percentiles(
    percentiles: Sequence[tuple[float, float]], scaling: Scaling
) -> list[float]:
    """A CDF Metaculus accepts, from ``(level, value)`` pairs in the question's units.

    The pairs may be unsorted, tied, outside the range or non-monotone (see
    the module docstring for how each is handled). ``ValueError`` only for an
    empty list, a non-finite number or a level outside (0, 1).
    """
    us, ps = _knots(percentiles, scaling)
    n = scaling.inbound_outcome_count
    scales = _tail_scales(us, ps, n)
    lo_end = OPEN_BOUND_MIN if scaling.open_lower_bound else 0.0
    hi_end = OPEN_BOUND_MAX if scaling.open_upper_bound else 1.0
    cdf = [min(max(_evaluate(us, ps, scales, i / n), lo_end), hi_end) for i in range(n + 1)]
    if not scaling.open_lower_bound:
        cdf[0] = 0.0
    if not scaling.open_upper_bound:
        cdf[-1] = 1.0

    inside = cdf[-1] - cdf[0]
    if inside < _MIN_INBOUND_MASS:
        room_lo = cdf[0] - lo_end
        room_hi = hi_end - cdf[-1]
        shortfall = _MIN_INBOUND_MASS - inside
        cdf[0] -= shortfall * room_lo / (room_lo + room_hi)
        cdf[-1] += shortfall * room_hi / (room_lo + room_hi)
    first, last = cdf[0], cdf[-1]

    pmf = [min(max(b - a, 0.0), 1.0) for a, b in pairwise(cdf)]
    pmf = _cap_steps(pmf, _MAX_STEP_SHARE * max_step(n))
    uniform = (last - first) / n
    target = _MIN_STEP_FACTOR * min_step(n) + _MIN_STEP_MARGIN
    least = min(pmf)
    if least < target:
        e = (target - least) / (uniform - least)
        pmf = [(1.0 - e) * p + e * uniform for p in pmf]

    out = [first]
    for p in pmf:
        out.append(out[-1] + p)
    out[-1] = last
    out = [_np_round(v, _CDF_DECIMALS) for v in out]
    if scaling.open_lower_bound:
        out[0] = max(out[0], OPEN_BOUND_MIN)
    if scaling.open_upper_bound:
        out[-1] = min(out[-1], OPEN_BOUND_MAX)
    errors = validate_cdf(out, scaling)
    if errors:
        raise AssertionError(f"bug: cdf_from_percentiles built an invalid CDF: {errors}")
    return out


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
