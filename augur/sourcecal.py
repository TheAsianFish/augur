"""Calibrators that know where a question comes from.

M8's Platt map is fitted on every source pooled: Polymarket and Manifold
(base rate near 40%) outnumber Metaculus (near 17%) about five to two, so
the pooled intercept pulls Metaculus questions up toward the pooled rate.
These maps take each forecast's *source* as well as its probability:

- ``platt-by-source`` (:class:`PerSourcePlatt`): one Platt map per source,
  fitted on that source's history alone; a source with fewer than
  :data:`MIN_PER_SOURCE` outcomes in the history gets the pooled map.
- ``platt-partial`` (:class:`PartialPlatt`): one shared slope and a
  per-source intercept, ``q = sigmoid(a * logit(p) + b + u_s)``, fitted by
  penalised maximum likelihood with a normal prior of standard deviation
  :data:`PRIOR_SD` on each offset ``u_s`` around the shared intercept. A
  source with little history stays near the pooled intercept; one with a
  lot moves toward its own. An unseen source gets ``u_s = 0``.

Both keep M8's conventions: forecasts clipped to :data:`augur.aggregate.CLIP`
before use, Platt's smoothed targets (:func:`augur.calibrate.platt_targets`)
so the optimum stays finite, outputs within ``CLIP``, and ``params()`` that
rebuild the identical map through :func:`sourced_from_params`.

:func:`fit_logistic` is the small penalised Newton solver both this module
and the Q6 policies (:mod:`augur.q6`) use: a handful of parameters, so a
dense solve per step costs nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from augur.aggregate import CLIP
from augur.calibrate import CalibrationError, Platt, PlattMap, platt_targets

__all__ = [
    "MIN_PER_SOURCE",
    "PRIOR_SD",
    "SOURCED",
    "PartialPlatt",
    "PartialPlattMap",
    "PerSourcePlatt",
    "PerSourcePlattMap",
    "PooledPlatt",
    "SourcedCalibrator",
    "SourcedMap",
    "fit_logistic",
    "sourced_from_params",
]

PRIOR_SD = 0.5
"""The prior standard deviation of a source's intercept offset, in log-odds (fixed in PLAN.md)."""
MIN_PER_SOURCE = 30
"""Below this many outcomes in the history, per-source Platt falls back to the pooled map."""


def _clip(p: float) -> float:
    value = float(p)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CalibrationError(
            f"probability {p!r} is not a finite number in [0, 1]; "
            "pass forecasts as probabilities, not percentages or log-odds"
        )
    return min(max(value, CLIP[0]), CLIP[1])


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _softplus(z: float) -> float:
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


# -- the solver ------------------------------------------------------------------


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve ``matrix @ x = vector`` by Gaussian elimination with partial pivoting."""
    n = len(vector)
    a = [[*row, vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise CalibrationError(
                "the calibration fit is singular (a feature never varies); "
                "fit on a history where every source has at least one forecast"
            )
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            if f:
                for c in range(col, n + 1):
                    a[r][c] -= f * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - math.fsum(a[r][c] * x[c] for c in range(r + 1, n))) / a[r][r]
    return x


def fit_logistic(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    penalty: Sequence[float],
    start: Sequence[float] | None = None,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> tuple[list[float], int]:
    """Minimise ``Σ softplus(z_i) − t_i z_i + ½ Σ_j penalty_j β_j²``, ``z_i = rows_i · β``.

    Damped Newton: each step solves ``H d = −g`` (``H = Xᵀ W X + diag(penalty)``,
    ``W = σ(z)(1 − σ(z))``) and halves ``d`` until the objective falls. The
    objective is convex, so this converges from any start; it stops when the
    gradient's max-norm is below ``tolerance``. A ridge of ``1e-9`` on every
    coefficient keeps ``H`` invertible on separable history without moving a
    well-posed optimum. Returns the coefficients and the Newton steps taken.
    """
    k = len(penalty)
    if not rows or any(len(r) != k for r in rows) or len(rows) != len(targets):
        raise CalibrationError(
            "fit_logistic needs one target per row and one penalty per column; "
            "pass the history as rows of features"
        )
    ridge = [p + 1e-9 for p in penalty]
    beta = list(start) if start is not None else [0.0] * k

    def objective(b: Sequence[float]) -> float:
        total = math.fsum(
            _softplus(z) - t * z
            for z, t in (
                (math.fsum(x * c for x, c in zip(r, b, strict=True)), t)
                for r, t in zip(rows, targets, strict=True)
            )
        )
        return total + 0.5 * math.fsum(p * c * c for p, c in zip(ridge, b, strict=True))

    current = objective(beta)
    steps = 0
    for steps in range(1, max_iterations + 1):
        grad = [ridge[j] * beta[j] for j in range(k)]
        hess = [[0.0] * k for _ in range(k)]
        for r, t in zip(rows, targets, strict=True):
            z = math.fsum(x * c for x, c in zip(r, beta, strict=True))
            s = _sigmoid(z)
            w = s * (1.0 - s)
            for i in range(k):
                grad[i] += (s - t) * r[i]
                wi = w * r[i]
                if wi:
                    for j in range(i, k):
                        hess[i][j] += wi * r[j]
        if max(abs(g) for g in grad) < tolerance:
            return beta, steps - 1
        for i in range(k):
            hess[i][i] += ridge[i]
            for j in range(i):
                hess[i][j] = hess[j][i]
        direction = _solve(hess, [-g for g in grad])
        scale = 1.0
        while scale > 1e-12:
            trial = [b + scale * d for b, d in zip(beta, direction, strict=True)]
            value = objective(trial)
            if value <= current:
                beta, current = trial, value
                break
            scale /= 2.0
        else:
            return beta, steps
    return beta, steps


# -- the protocol ----------------------------------------------------------------


class SourcedMap(Protocol):
    """A fitted map from a probability and its source to a calibrated probability."""

    @property
    def name(self) -> str: ...

    def __call__(self, p: float, source: str) -> float: ...

    def params(self) -> dict[str, Any]: ...


class SourcedCalibrator(Protocol):
    """Fits a :class:`SourcedMap` to resolved history with each outcome's source."""

    @property
    def name(self) -> str: ...

    def fit(
        self, probabilities: Sequence[float], outcomes: Sequence[int], sources: Sequence[str]
    ) -> SourcedMap: ...


def _validate(
    probabilities: Sequence[float], outcomes: Sequence[int], sources: Sequence[str]
) -> tuple[list[float], list[int], list[str]]:
    if not len(probabilities) == len(outcomes) == len(sources):
        raise CalibrationError(
            f"{len(probabilities)} probabilities, {len(outcomes)} outcomes and "
            f"{len(sources)} sources; pass one of each per resolved forecast"
        )
    if not probabilities:
        raise CalibrationError(
            "nothing to fit: the history is empty; fit on at least one resolved forecast"
        )
    for y in outcomes:
        if y not in (0, 1):
            raise CalibrationError(f"outcome {y!r} is not 0 or 1; calibrate binary questions only")
    return [_clip(p) for p in probabilities], [int(y) for y in outcomes], [str(s) for s in sources]


# -- pooled, for the same interface ----------------------------------------------


@dataclass(frozen=True)
class PooledMap:
    """M8's pooled Platt map behind the sourced interface: the source is ignored."""

    inner: PlattMap
    name: str = "platt"

    def __call__(self, p: float, source: str) -> float:
        return self.inner(p)

    def params(self) -> dict[str, Any]:
        return self.inner.params()


@dataclass(frozen=True)
class PooledPlatt:
    """M8's Platt (:class:`augur.calibrate.Platt`), ignoring the source."""

    name: str = "platt"

    def fit(
        self, probabilities: Sequence[float], outcomes: Sequence[int], sources: Sequence[str]
    ) -> PooledMap:
        ps, ys, _ = _validate(probabilities, outcomes, sources)
        return PooledMap(Platt().fit(ps, ys))


# -- per source ------------------------------------------------------------------


@dataclass(frozen=True)
class PerSourcePlattMap:
    pooled: PlattMap
    by_source: Mapping[str, PlattMap] = field(default_factory=dict)
    min_n: int = MIN_PER_SOURCE
    name: str = "platt-by-source"

    def __call__(self, p: float, source: str) -> float:
        return self.by_source.get(source, self.pooled)(p)

    def params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_n": self.min_n,
            "pooled": self.pooled.params(),
            "by_source": {s: m.params() for s, m in sorted(self.by_source.items())},
        }


@dataclass(frozen=True)
class PerSourcePlatt:
    """A Platt map per source with at least ``min_n`` outcomes; the pooled map otherwise."""

    min_n: int = MIN_PER_SOURCE
    name: str = "platt-by-source"

    def fit(
        self, probabilities: Sequence[float], outcomes: Sequence[int], sources: Sequence[str]
    ) -> PerSourcePlattMap:
        ps, ys, ss = _validate(probabilities, outcomes, sources)
        pooled = Platt().fit(ps, ys)
        own: dict[str, PlattMap] = {}
        for s in sorted(set(ss)):
            idx = [i for i, x in enumerate(ss) if x == s]
            if len(idx) >= self.min_n:
                own[s] = Platt().fit([ps[i] for i in idx], [ys[i] for i in idx])
        return PerSourcePlattMap(pooled, own, self.min_n)


# -- partially pooled ------------------------------------------------------------


@dataclass(frozen=True)
class PartialPlattMap:
    """``q = clip(sigmoid(a * logit(clip(p)) + b + offsets[source]))``; unseen: offset 0."""

    a: float
    b: float
    offsets: Mapping[str, float]
    prior_sd: float
    n: int
    iterations: int
    name: str = "platt-partial"

    def __call__(self, p: float, source: str) -> float:
        x = _logit(_clip(p))
        return min(
            max(_sigmoid(self.a * x + self.b + self.offsets.get(source, 0.0)), CLIP[0]), CLIP[1]
        )

    def params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "a": self.a,
            "b": self.b,
            "offsets": dict(sorted(self.offsets.items())),
            "prior_sd": self.prior_sd,
            "n": self.n,
            "iterations": self.iterations,
        }


@dataclass(frozen=True)
class PartialPlatt:
    """One slope, one intercept and a shrunk offset per source (penalised MLE)."""

    prior_sd: float = PRIOR_SD
    name: str = "platt-partial"

    def fit(
        self, probabilities: Sequence[float], outcomes: Sequence[int], sources: Sequence[str]
    ) -> PartialPlattMap:
        ps, ys, ss = _validate(probabilities, outcomes, sources)
        names = sorted(set(ss))
        col = {s: i for i, s in enumerate(names)}
        rows: list[list[float]] = []
        for p, s in zip(ps, ss, strict=True):
            r = [_logit(p), 1.0] + [0.0] * len(names)
            r[2 + col[s]] = 1.0
            rows.append(r)
        precision = 1.0 / (self.prior_sd * self.prior_sd)
        beta, steps = fit_logistic(
            rows,
            platt_targets(ys),
            [0.0, 0.0] + [precision] * len(names),
            [1.0, 0.0] + [0.0] * len(names),
        )
        return PartialPlattMap(
            a=beta[0],
            b=beta[1],
            offsets={s: beta[2 + col[s]] for s in names},
            prior_sd=self.prior_sd,
            n=len(ps),
            iterations=steps,
        )


SOURCED: tuple[SourcedCalibrator, ...] = (PooledPlatt(), PerSourcePlatt(), PartialPlatt())


def _platt(params: Mapping[str, Any]) -> PlattMap:
    return PlattMap(
        a=float(params["a"]),
        b=float(params["b"]),
        iterations=int(params["iterations"]),
        n=int(params["n"]),
    )


def sourced_from_params(params: Mapping[str, Any]) -> SourcedMap:
    """Rebuild the map whose ``params()`` returned ``params``."""
    try:
        name = params.get("name")
        if name == "platt":
            return PooledMap(_platt(params))
        if name == "platt-by-source":
            return PerSourcePlattMap(
                _platt(params["pooled"]),
                {str(s): _platt(m) for s, m in params["by_source"].items()},
                int(params["min_n"]),
            )
        if name == "platt-partial":
            return PartialPlattMap(
                a=float(params["a"]),
                b=float(params["b"]),
                offsets={str(s): float(v) for s, v in params["offsets"].items()},
                prior_sd=float(params["prior_sd"]),
                n=int(params["n"]),
                iterations=int(params["iterations"]),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(
            f"calibration params for {params.get('name')!r} are incomplete ({exc}); "
            "pass the dict a fitted map's params() returned"
        ) from None
    raise CalibrationError(
        f"unknown source-aware calibrator {params.get('name')!r}; "
        "use platt, platt-by-source or platt-partial"
    )
