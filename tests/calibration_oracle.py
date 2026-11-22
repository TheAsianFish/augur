"""Reference implementations that :mod:`augur.calibrate` is tested against.

Each is written for clarity rather than speed: a second-order Platt fit run
to machine precision, and the closed-form max-min characterisation of the
isotonic regression.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

CLIP = (0.01, 0.99)


def clip(p: float) -> float:
    return min(max(p, CLIP[0]), CLIP[1])


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def softplus(z: float) -> float:
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def smoothed_targets(outcomes: Sequence[int]) -> list[float]:
    """Platt's targets: ``(N+ + 1)/(N+ + 2)`` for a 1 and ``1/(N- + 2)`` for a 0."""
    positives = sum(outcomes)
    negatives = len(outcomes) - positives
    return [
        (positives + 1) / (positives + 2) if y == 1 else 1.0 / (negatives + 2) for y in outcomes
    ]


def platt_objective(a: float, b: float, xs: Sequence[float], ts: Sequence[float]) -> float:
    """Mean cross-entropy of ``sigmoid(a x + b)`` against targets ``t``."""
    return math.fsum(softplus(a * x + b) - t * (a * x + b) for x, t in zip(xs, ts, strict=True)) / (
        len(xs)
    )


def platt_hessian_min_eigenvalue(a: float, b: float, probabilities: Sequence[float]) -> float:
    """The smallest eigenvalue of the mean cross-entropy's Hessian at ``(a, b)``."""
    xs = [logit(clip(p)) for p in probabilities]
    ws = [sigmoid(a * x + b) * (1.0 - sigmoid(a * x + b)) for x in xs]
    n = len(xs)
    haa = math.fsum(w * x * x for w, x in zip(ws, xs, strict=True)) / n
    hab = math.fsum(w * x for w, x in zip(ws, xs, strict=True)) / n
    hbb = math.fsum(ws) / n
    half_trace = (haa + hbb) / 2.0
    return half_trace - math.hypot((haa - hbb) / 2.0, hab)


def platt_oracle(probabilities: Sequence[float], outcomes: Sequence[int]) -> tuple[float, float]:
    """``(a, b)`` by Newton-Raphson from ``(1, 0)``, to gradient max-norm below 1e-13.

    Each Newton step is halved while it raises the objective (beyond rounding).
    Raises ``ValueError`` if the clipped forecasts are all equal, when the
    optimum is a line rather than a point.
    """
    xs = [logit(clip(p)) for p in probabilities]
    if len(set(xs)) < 2:
        raise ValueError("the forecasts do not vary, so (a, b) is not identified")
    ts = smoothed_targets(outcomes)
    n = len(xs)
    a, b = 1.0, 0.0
    for _ in range(100):
        qs = [sigmoid(a * x + b) for x in xs]
        ga = math.fsum((q - t) * x for q, t, x in zip(qs, ts, xs, strict=True)) / n
        gb = math.fsum(q - t for q, t in zip(qs, ts, strict=True)) / n
        if max(abs(ga), abs(gb)) < 1e-13:
            break
        ws = [q * (1.0 - q) for q in qs]
        haa = math.fsum(w * x * x for w, x in zip(ws, xs, strict=True)) / n
        hab = math.fsum(w * x for w, x in zip(ws, xs, strict=True)) / n
        hbb = math.fsum(ws) / n
        # Far from the optimum one group of forecasts can have q(1 - q) near
        # zero, making the Hessian nearly singular; a ridge (Levenberg) keeps
        # the step a descent direction. At the optimum the ridge is zero.
        ridge = 0.0
        while True:
            det = (haa + ridge) * (hbb + ridge) - hab * hab
            if det > 1e-12 * (haa + hbb + 2.0 * ridge) ** 2:
                break
            ridge = max(10.0 * ridge, 1e-8 * (haa + hbb) + 1e-300)
        da = -((hbb + ridge) * ga - hab * gb) / det
        db = -((haa + ridge) * gb - hab * ga) / det
        before = platt_objective(a, b, xs, ts)
        scale = 1.0
        for _ in range(60):
            after = platt_objective(a + scale * da, b + scale * db, xs, ts)
            if after <= before + 1e-15 * abs(before):
                break
            scale /= 2.0
        a += scale * da
        b += scale * db
    return a, b


def pooled(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> tuple[list[float], list[float], list[float]]:
    """Sorted unique clipped forecasts with each one's outcome sum and count."""
    xs: list[float] = []
    sums: list[float] = []
    weights: list[float] = []
    for p, y in sorted(zip((clip(p) for p in probabilities), outcomes, strict=True)):
        if xs and xs[-1] == p:
            sums[-1] += y
            weights[-1] += 1.0
        else:
            xs.append(p)
            sums.append(float(y))
            weights.append(1.0)
    return xs, sums, weights


def isotonic_oracle(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> tuple[list[float], list[float]]:
    """The isotonic fit at each unique clipped forecast, by the max-min formula.

    Returns ``(xs, fitted)`` where ``fitted[i] = max over j <= i of min over
    k >= i of the weighted mean of blocks j..k``. O(m^3) in the number of
    unique forecasts; for small inputs only.
    """
    xs, sums, weights = pooled(probabilities, outcomes)
    m = len(xs)
    fitted: list[float] = []
    for i in range(m):
        best = -math.inf
        for j in range(i + 1):
            worst = math.inf
            for k in range(i, m):
                mean = math.fsum(sums[j : k + 1]) / math.fsum(weights[j : k + 1])
                worst = min(worst, mean)
            best = max(best, worst)
        fitted.append(best)
    return xs, fitted
