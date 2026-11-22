"""Calibrators: maps from a binary forecast probability to a calibrated one.

A calibrator is fitted on resolved history, pairs of (forecast probability,
outcome 0 or 1), and returns a :class:`CalibrationMap` that is then applied to
new forecasts. Five calibrators are defined, in :data:`CALIBRATORS` order:

- ``raw`` (:class:`Identity`): the forecast unchanged.
- ``extremize-<factor>`` (:class:`FixedExtremize`): not fitted;
  ``sigmoid(factor * logit(p))``, the same transform as
  :func:`augur.aggregate.mean_logodds` applied to one forecast.
- ``base-rate`` (:class:`BaseRate`): ignores the forecast and returns the
  Laplace-smoothed frequency of the outcome in the history,
  ``(yes + 1) / (n + 2)``.
- ``platt`` (:class:`Platt`): logistic regression of the outcome on the
  forecast's log-odds, ``q = sigmoid(a * logit(p) + b)``, fitted by maximum
  likelihood with Platt's (1999) smoothed targets so that it stays finite on
  separable history.
- ``isotonic`` (:class:`Isotonic`): the non-decreasing step function of the
  forecast that minimises squared error against the outcomes, found by pool
  adjacent violators.

Every probability is clipped to :data:`augur.aggregate.CLIP` before use, and
every output of every map lies within :data:`~augur.aggregate.CLIP`.

``fit`` sees only the history it is given and nothing else: which resolved
questions may be used to calibrate a given forecast (only those resolved
before it, in a backtest) is decided by the caller. Fitting is deterministic,
and each fitted map's :meth:`~CalibrationMap.params` is JSON-safe and rebuilds
the identical map through :func:`from_params`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from augur.aggregate import CLIP, EXTREMIZE_FACTOR
from augur.errors import AugurError

__all__ = [
    "CALIBRATORS",
    "BaseRate",
    "BaseRateMap",
    "CalibrationError",
    "CalibrationMap",
    "Calibrator",
    "ExtremizeMap",
    "FixedExtremize",
    "Identity",
    "IdentityMap",
    "Isotonic",
    "IsotonicMap",
    "Platt",
    "PlattMap",
    "by_name",
    "from_params",
]

# The sufficient-decrease constant of the Armijo condition.
_ARMIJO_C = 1e-4
# Below this, a backtracking step can no longer change the parameters.
_MIN_STEP = 1e-30
# Ceiling on the step carried from one gradient iteration to the next.
_MAX_STEP = 1e6


class CalibrationError(AugurError):
    """A calibrator was given history it cannot fit, or parameters it cannot rebuild."""


class CalibrationMap(Protocol):
    """A fitted calibration: probability in, calibrated probability within ``CLIP`` out."""

    @property
    def name(self) -> str:
        """The name of the calibrator that produced the map."""
        ...

    def __call__(self, p: float) -> float:
        """The calibrated probability for ``p`` in ``[0, 1]``, within ``CLIP``."""
        ...

    def params(self) -> dict[str, Any]:
        """JSON-safe parameters, including ``name``, that :func:`from_params` rebuilds."""
        ...


class Calibrator(Protocol):
    """Fits a :class:`CalibrationMap` to resolved history."""

    @property
    def name(self) -> str:
        """The calibrator's name, as in :data:`CALIBRATORS` and the fitted map's params."""
        ...

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> CalibrationMap:
        """Fit to ``probabilities[i]`` paired with ``outcomes[i]`` (0 or 1)."""
        ...


# -- numeric helpers ---------------------------------------------------------


def _clip(p: float) -> float:
    return min(max(p, CLIP[0]), CLIP[1])


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _softplus(z: float) -> float:
    """``log(1 + exp(z))`` without overflow."""
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def _softplus_change(z: float, d: float) -> float:
    """``softplus(z + d) - softplus(z)``, accurate when ``d`` is tiny.

    For ``|d| < 1`` this is ``log1p(sigmoid(z) * expm1(d))``, which keeps full
    relative precision as ``d`` goes to zero; otherwise the plain difference.
    """
    if abs(d) < 1.0:
        return math.log1p(_sigmoid(z) * math.expm1(d))
    return _softplus(z + d) - _softplus(z)


def _check_probability(p: float) -> float:
    value = float(p)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CalibrationError(
            f"probability {p!r} is not a finite number in [0, 1]; "
            "pass forecasts as probabilities, not percentages or log-odds"
        )
    return value


def _validate(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> tuple[list[float], list[int]]:
    """The history as clipped probabilities and 0/1 outcomes, or :class:`CalibrationError`."""
    if len(probabilities) != len(outcomes):
        raise CalibrationError(
            f"{len(probabilities)} probabilities but {len(outcomes)} outcomes; "
            "pass one outcome per forecast"
        )
    if not probabilities:
        raise CalibrationError(
            "nothing to fit: the history is empty; fit on at least one resolved forecast"
        )
    clipped = [_clip(_check_probability(p)) for p in probabilities]
    checked: list[int] = []
    for y in outcomes:
        if y not in (0, 1):
            raise CalibrationError(
                f"outcome {y!r} is not 0 or 1; calibrate binary questions only, "
                "resolved yes as 1 and no as 0"
            )
        checked.append(int(y))
    return clipped, checked


def _require(params: Mapping[str, Any], key: str) -> Any:
    if key not in params:
        raise CalibrationError(
            f"calibration params for {params.get('name')!r} lack {key!r}; "
            "pass the dict a fitted map's params() returned"
        )
    return params[key]


# -- raw ---------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityMap:
    """``p -> clip(p)``."""

    name: str = "raw"

    def __call__(self, p: float) -> float:
        return _clip(_check_probability(p))

    def params(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class Identity:
    """Leaves forecasts unchanged apart from clipping."""

    name: str = "raw"

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> IdentityMap:
        _validate(probabilities, outcomes)
        return IdentityMap()


# -- extremize ---------------------------------------------------------------


@dataclass(frozen=True)
class ExtremizeMap:
    """``p -> clip(sigmoid(factor * logit(clip(p))))``."""

    factor: float

    @property
    def name(self) -> str:
        return f"extremize-{self.factor:g}"

    def __call__(self, p: float) -> float:
        return _clip(_sigmoid(self.factor * _logit(_clip(_check_probability(p)))))

    def params(self) -> dict[str, Any]:
        return {"name": self.name, "factor": self.factor}


@dataclass(frozen=True)
class FixedExtremize:
    """Multiplies the forecast's log-odds by a fixed factor; the history is only validated."""

    factor: float = EXTREMIZE_FACTOR

    def __post_init__(self) -> None:
        if not math.isfinite(self.factor):
            raise CalibrationError(
                f"extremizing factor {self.factor!r} is not finite; use a finite factor"
            )

    @property
    def name(self) -> str:
        return f"extremize-{self.factor:g}"

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> ExtremizeMap:
        _validate(probabilities, outcomes)
        return ExtremizeMap(float(self.factor))


# -- base rate ---------------------------------------------------------------


@dataclass(frozen=True)
class BaseRateMap:
    """The constant ``clip((yes + 1) / (n + 2))``, whatever the forecast."""

    yes: int
    n: int
    name: str = "base-rate"

    @property
    def rate(self) -> float:
        """The clipped Laplace-smoothed frequency of the outcome."""
        return _clip((self.yes + 1) / (self.n + 2))

    def __call__(self, p: float) -> float:
        _check_probability(p)
        return self.rate

    def params(self) -> dict[str, Any]:
        return {"name": self.name, "yes": self.yes, "n": self.n, "rate": self.rate}


@dataclass(frozen=True)
class BaseRate:
    """Replaces every forecast with the history's Laplace-smoothed outcome frequency."""

    name: str = "base-rate"

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> BaseRateMap:
        _, ys = _validate(probabilities, outcomes)
        return BaseRateMap(yes=sum(ys), n=len(ys))


# -- Platt -------------------------------------------------------------------


@dataclass(frozen=True)
class PlattMap:
    """``p -> clip(sigmoid(a * logit(clip(p)) + b))``."""

    a: float
    b: float
    iterations: int
    n: int
    name: str = "platt"

    def __call__(self, p: float) -> float:
        x = _logit(_clip(_check_probability(p)))
        return _clip(_sigmoid(self.a * x + self.b))

    def params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "a": self.a,
            "b": self.b,
            "iterations": self.iterations,
            "n": self.n,
        }


def platt_targets(outcomes: Sequence[int]) -> list[float]:
    """Platt's smoothed targets: ``(N+ + 1)/(N+ + 2)`` for a 1, ``1/(N- + 2)`` for a 0."""
    positives = sum(outcomes)
    negatives = len(outcomes) - positives
    hi = (positives + 1) / (positives + 2)
    lo = 1.0 / (negatives + 2)
    return [hi if y == 1 else lo for y in outcomes]


@dataclass(frozen=True)
class Platt:
    """Logistic regression of the outcome on the forecast's log-odds.

    The map is ``q = sigmoid(a * x + b)`` with ``x = logit(clip(p))``. ``(a, b)``
    minimise the mean cross-entropy against Platt's smoothed targets
    (:func:`platt_targets`), which keeps the optimum finite even when the
    history is all one outcome or perfectly separated by ``p``.

    The fit is plain gradient descent from ``a = 1, b = 0`` (the identity
    map), with a backtracking line search under the Armijo condition; the
    step accepted at one iteration, doubled, is the first one tried at the
    next. It stops when the gradient's max-norm falls below ``tolerance`` or
    after ``max_iterations`` iterations. Each iteration costs O(n) for the
    gradient plus O(n) per line-search trial. Gradient descent converges
    only linearly, at a rate set by the condition number of the Hessian, so
    the iteration count grows when the forecasts' log-odds are far from
    centred or the history is nearly separable; on such history it can reach
    ``max_iterations`` before ``tolerance``.
    """

    max_iterations: int = 5000
    tolerance: float = 1e-9
    name: str = "platt"

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> PlattMap:
        ps, ys = _validate(probabilities, outcomes)
        xs = [_logit(p) for p in ps]
        ts = platt_targets(ys)
        n = len(xs)
        a, b = 1.0, 0.0
        step = 1.0
        iterations = 0
        while iterations < self.max_iterations:
            zs = [a * x + b for x in xs]
            residuals = [_sigmoid(z) - t for z, t in zip(zs, ts, strict=True)]
            ga = math.fsum(r * x for r, x in zip(residuals, xs, strict=True)) / n
            gb = math.fsum(residuals) / n
            if max(abs(ga), abs(gb)) < self.tolerance:
                break
            squared = ga * ga + gb * gb
            step = min(step * 2.0, _MAX_STEP)
            while step >= _MIN_STEP:
                da, db = -step * ga, -step * gb
                change = (
                    math.fsum(
                        _softplus_change(z, da * x + db) - t * (da * x + db)
                        for z, x, t in zip(zs, xs, ts, strict=True)
                    )
                    / n
                )
                if change <= -_ARMIJO_C * step * squared:
                    break
                step /= 2.0
            if step < _MIN_STEP:
                break
            a -= step * ga
            b -= step * gb
            iterations += 1
        return PlattMap(a=a, b=b, iterations=iterations, n=n)


# -- isotonic ----------------------------------------------------------------


@dataclass(frozen=True)
class IsotonicMap:
    """A non-decreasing step function.

    ``thresholds[i]`` is the smallest (clipped) forecast in block ``i`` and
    ``values[i]`` that block's mean outcome. An input takes the value of the
    last block whose threshold is at or below it; an input below the first
    threshold takes the first block's value. The output is clipped to ``CLIP``.
    """

    thresholds: tuple[float, ...]
    values: tuple[float, ...]
    n: int
    name: str = "isotonic"

    def __post_init__(self) -> None:
        if not self.thresholds or len(self.thresholds) != len(self.values):
            raise CalibrationError(
                "an isotonic map needs as many values as thresholds, and at least one; "
                "pass the dict a fitted map's params() returned"
            )

    def __call__(self, p: float) -> float:
        x = _clip(_check_probability(p))
        lo, hi = 0, len(self.thresholds)
        # Binary search for the number of thresholds <= x.
        while lo < hi:
            mid = (lo + hi) // 2
            if self.thresholds[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return _clip(self.values[max(lo - 1, 0)])

    def params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "thresholds": list(self.thresholds),
            "values": list(self.values),
            "n": self.n,
        }


@dataclass(frozen=True)
class Isotonic:
    """Isotonic least-squares regression of the outcome on the forecast.

    Forecasts are clipped, then equal forecasts are pooled into one block
    (weight the count, value the mean outcome). Pool adjacent violators then
    scans the blocks from the left for the first adjacent pair whose values
    decrease, merges it into one block (summed weight, weighted mean), and
    restarts the scan from the beginning, until no pair decreases. Each block
    keeps its outcome sum and weight, so its value is an exact ratio rather
    than a running average.

    The result is the least-squares non-decreasing fit. Sorting costs
    O(n log n); the restarting scan costs O(m) per merge over m blocks, so
    O(n^2) in the worst case (for example outcomes all 1 then all 0 over
    increasing forecasts), and little more than a single pass when the
    outcomes are already nearly increasing.
    """

    name: str = "isotonic"

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> IsotonicMap:
        ps, ys = _validate(probabilities, outcomes)
        lows: list[float] = []
        sums: list[float] = []
        weights: list[float] = []
        for p, y in sorted(zip(ps, ys, strict=True)):
            if lows and lows[-1] == p:
                sums[-1] += y
                weights[-1] += 1.0
            else:
                lows.append(p)
                sums.append(float(y))
                weights.append(1.0)
        i = 0
        while i + 1 < len(lows):
            if sums[i] / weights[i] > sums[i + 1] / weights[i + 1]:
                sums[i] += sums[i + 1]
                weights[i] += weights[i + 1]
                del lows[i + 1], sums[i + 1], weights[i + 1]
                i = 0
            else:
                i += 1
        values = tuple(s / w for s, w in zip(sums, weights, strict=True))
        return IsotonicMap(thresholds=tuple(lows), values=values, n=len(ps))


# -- registry ----------------------------------------------------------------

CALIBRATORS: tuple[Calibrator, ...] = (
    Identity(),
    FixedExtremize(),
    BaseRate(),
    Platt(),
    Isotonic(),
)


def by_name(name: str) -> Calibrator:
    """The calibrator in :data:`CALIBRATORS` called ``name``."""
    for calibrator in CALIBRATORS:
        if calibrator.name == name:
            return calibrator
    known = ", ".join(c.name for c in CALIBRATORS)
    raise CalibrationError(f"unknown calibrator {name!r}; use one of {known}")


def from_params(params: Mapping[str, Any]) -> CalibrationMap:
    """Rebuild the fitted map whose :meth:`~CalibrationMap.params` returned ``params``."""
    name = params.get("name")
    if name == "raw":
        return IdentityMap()
    if isinstance(name, str) and name.startswith("extremize-"):
        return ExtremizeMap(float(_require(params, "factor")))
    if name == "base-rate":
        return BaseRateMap(yes=int(_require(params, "yes")), n=int(_require(params, "n")))
    if name == "platt":
        return PlattMap(
            a=float(_require(params, "a")),
            b=float(_require(params, "b")),
            iterations=int(_require(params, "iterations")),
            n=int(_require(params, "n")),
        )
    if name == "isotonic":
        return IsotonicMap(
            thresholds=tuple(float(t) for t in _require(params, "thresholds")),
            values=tuple(float(v) for v in _require(params, "values")),
            n=int(_require(params, "n")),
        )
    raise CalibrationError(
        f"unknown calibration map {name!r}; use params from a map fitted by one of "
        f"{', '.join(c.name for c in CALIBRATORS)}"
    )
