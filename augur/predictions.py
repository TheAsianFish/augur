"""What a forecast is for each question type, how members combine, and what is posted.

Binary: one probability. Multiple choice: a probability per option, summing to
one. Numeric and discrete: a CDF over the question's buckets, built from the
percentiles a forecaster states (:mod:`augur.distributions`).

Combining members, per type:

- binary: the median, as the template bot does;
- multiple choice: the median of each option across members, then
  normalised so the options sum to one (a per-option median alone need not);
- numeric and discrete: the mean of the members' CDFs, a mixture. A mixture
  of valid CDFs is itself valid (every rule Metaculus checks is preserved by
  a weighted mean), and it keeps the members' disagreement as spread, where a
  median of percentiles would hide it.

M7 compares these rules against the alternatives; until then they are fixed.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from augur.aggregate import median
from augur.distributions import (
    Scaling,
    mix_cdfs,
    normalise_probabilities,
    quantile,
    validate_cdf,
    validate_probabilities,
)
from augur.questions import Question

# The levels a forecaster states, as the template bot asks for them.
PERCENTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)


@dataclass(frozen=True)
class BinaryPrediction:
    probability: float


@dataclass(frozen=True)
class ChoicePrediction:
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ContinuousPrediction:
    cdf: tuple[float, ...]
    # The percentiles the forecaster stated (members), or the ones read back
    # from the mixed CDF (the aggregate), as (level, value in question units).
    percentiles: tuple[tuple[float, float], ...]


Prediction = BinaryPrediction | ChoicePrediction | ContinuousPrediction


def binary_probability(prediction: Prediction | None) -> float | None:
    """The probability of a binary prediction; ``None`` for every other type."""
    return prediction.probability if isinstance(prediction, BinaryPrediction) else None


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate zero forecasts")
    return float(statistics.median(values))


def aggregate(q: Question, members: Sequence[Prediction]) -> tuple[Prediction, str]:
    """The combined prediction and the name of the rule that made it."""
    if not members:
        raise ValueError("cannot aggregate zero forecasts")
    if q.type == "binary":
        ps = [_binary(m).probability for m in members]
        return BinaryPrediction(round(median(ps), 3)), "median"
    if q.type == "multiple_choice":
        options = _options(q)
        per_option = {o: _median([_choice(m).probabilities[o] for m in members]) for o in options}
        return ChoicePrediction(
            normalise_probabilities(per_option, options)
        ), "median_per_option_normalised"
    scaling = _scaling(q)
    mixed = mix_cdfs([_continuous(m).cdf for m in members])
    errors = validate_cdf(mixed, scaling)
    if errors:
        raise ValueError("internal: the mixed CDF is invalid: " + "; ".join(errors))
    levels = tuple((lv, quantile(mixed, scaling, lv)) for lv in PERCENTILE_LEVELS)
    return ContinuousPrediction(tuple(mixed), levels), "mean_cdf"


def check(q: Question, prediction: Prediction) -> list[str]:
    """Metaculus's validity rules for this prediction; empty when it may be posted."""
    if isinstance(prediction, BinaryPrediction):
        p = prediction.probability
        return [] if 0.001 <= p <= 0.999 else [f"probability {p} is outside [0.001, 0.999]"]
    if isinstance(prediction, ChoicePrediction):
        return validate_probabilities(prediction.probabilities, _options(q))
    return validate_cdf(prediction.cdf, _scaling(q))


def forecast_payload(question_id: int, prediction: Prediction) -> list[dict[str, Any]]:
    """The body of ``POST /api/questions/forecast/`` for one question."""
    body: dict[str, Any] = {
        "question": int(question_id),
        "source": "api",
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }
    if isinstance(prediction, BinaryPrediction):
        body["probability_yes"] = prediction.probability
    elif isinstance(prediction, ChoicePrediction):
        body["probability_yes_per_category"] = dict(prediction.probabilities)
    else:
        body["continuous_cdf"] = list(prediction.cdf)
    return [body]


def values_json(prediction: Prediction) -> dict[str, Any]:
    """The prediction as the ledger stores it."""
    if isinstance(prediction, BinaryPrediction):
        return {"probability": prediction.probability}
    if isinstance(prediction, ChoicePrediction):
        return {"probabilities": dict(prediction.probabilities)}
    return {"percentiles": [list(p) for p in prediction.percentiles], "cdf": list(prediction.cdf)}


def _number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def summary(q: Question, prediction: Prediction) -> str:
    """One line for the console and the comment: ``62%``, ``A 55%, B 45%``, ``median 12``."""
    if isinstance(prediction, BinaryPrediction):
        return f"{prediction.probability * 100:.0f}%"
    if isinstance(prediction, ChoicePrediction):
        return ", ".join(f"{o} {p * 100:.0f}%" for o, p in prediction.probabilities.items())
    unit = f" {q.unit}" if q.unit else ""
    by_level = dict(prediction.percentiles)
    scaling = _scaling(q)
    median = quantile(list(prediction.cdf), scaling, 0.5)
    low = by_level.get(0.1, quantile(list(prediction.cdf), scaling, 0.1))
    high = by_level.get(0.9, quantile(list(prediction.cdf), scaling, 0.9))
    return f"median {_number(median)}{unit} (80% between {_number(low)} and {_number(high)})"


def _binary(p: Prediction) -> BinaryPrediction:
    if not isinstance(p, BinaryPrediction):
        raise TypeError(f"expected a binary prediction, got {type(p).__name__}")
    return p


def _choice(p: Prediction) -> ChoicePrediction:
    if not isinstance(p, ChoicePrediction):
        raise TypeError(f"expected a multiple-choice prediction, got {type(p).__name__}")
    return p


def _continuous(p: Prediction) -> ContinuousPrediction:
    if not isinstance(p, ContinuousPrediction):
        raise TypeError(f"expected a continuous prediction, got {type(p).__name__}")
    return p


def _options(q: Question) -> list[str]:
    if not q.options:
        raise ValueError(f"question {q.question_id} is multiple choice but lists no options")
    return list(q.options)


def _scaling(q: Question) -> Scaling:
    if q.scaling is None:
        raise ValueError(f"question {q.question_id} has no usable scaling")
    return q.scaling
