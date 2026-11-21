"""The aggregation rules of :mod:`augur.aggregate`, for every question type."""

from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from augur.aggregate import (
    EXTREMIZE_FACTOR,
    RULES,
    TRIM,
    _pointwise_cdf,
    combine_binary,
    combine_cdfs,
    combine_choice,
    mean_logodds,
    median,
    trimmed_mean,
)
from augur.distributions import (
    Scaling,
    cdf_from_percentiles,
    mix_cdfs,
    normalise_probabilities,
    quantile,
    validate_cdf,
    validate_probabilities,
)
from augur.predictions import (
    PERCENTILE_LEVELS,
    BinaryPrediction,
    ChoicePrediction,
    ContinuousPrediction,
    Prediction,
    aggregate,
    check,
)
from augur.questions import Question

OPTIONS = ["A", "B", "C"]
_T = datetime(2026, 1, 1, tzinfo=UTC)


def _scaling(*, lower: bool = True, upper: bool = True, n: int = 200) -> Scaling:
    return Scaling(
        range_min=0.0,
        range_max=100.0,
        open_lower_bound=lower,
        open_upper_bound=upper,
        inbound_outcome_count=n,
    )


def _question(kind: str, scaling: Scaling | None = None) -> Question:
    return Question.model_validate(
        {
            "post_id": 1,
            "question_id": 2,
            "title": "t",
            "type": kind,
            "status": "open",
            "url": "https://example.invalid/1",
            "open_time": _T,
            "scheduled_close_time": _T,
            "options": OPTIONS if kind == "multiple_choice" else None,
            "scaling": scaling,
        }
    )


def _member(scaling: Scaling, centre: float, width: float) -> ContinuousPrediction:
    pairs = [(0.1, centre - width), (0.5, centre), (0.9, centre + width)]
    return ContinuousPrediction(tuple(cdf_from_percentiles(pairs, scaling)), tuple(pairs))


# Binary --------------------------------------------------------------------------


def test_binary_rules_by_hand() -> None:
    ps = [0.1, 0.2, 0.9]
    assert combine_binary(ps, "median") == 0.2
    assert combine_binary(ps, "trimmed_mean") == pytest.approx(0.4)  # n < 5: the mean
    # logit 0.1 and 0.9 cancel, so the mean log-odds is logit(0.2) / 3 = -ln(4) / 3.
    assert combine_binary(ps, "mean_logodds") == pytest.approx(1 / (1 + 4 ** (1 / 3)))
    assert combine_binary(ps, "extremized_logodds") == pytest.approx(1 / 3)  # 1/(1 + 4^0.5)


def test_log_odds_rules_clip_to_one_and_ninety_nine_percent() -> None:
    assert combine_binary([0.0001, 0.0001], "mean_logodds") == pytest.approx(0.01)
    assert combine_binary([0.9999], "mean_logodds") == pytest.approx(0.99)


def test_trimmed_mean_with_nine_drops_exactly_one_each_side() -> None:
    assert math.floor(TRIM * 9) == 1
    ps = [0.01, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.99]
    assert trimmed_mean(ps) == pytest.approx(sum(ps[1:-1]) / 7)
    assert combine_binary(ps, "trimmed_mean") == pytest.approx(3.3 / 7)
    assert trimmed_mean([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.25)  # n < 5: nothing dropped


def test_extremizing_moves_away_from_half_and_is_the_log_odds_mean_at_factor_one() -> None:
    for ps in ([0.6, 0.7, 0.8], [0.1, 0.3, 0.45], [0.2, 0.9, 0.95]):
        pooled = combine_binary(ps, "mean_logodds")
        extreme = combine_binary(ps, "extremized_logodds")
        assert abs(extreme - 0.5) > abs(pooled - 0.5)
        assert (extreme - 0.5) * (pooled - 0.5) > 0
        assert extreme == pytest.approx(mean_logodds(ps, EXTREMIZE_FACTOR))
        assert mean_logodds(ps, 1.0) == pytest.approx(pooled)
    assert combine_binary([0.3, 0.7], "extremized_logodds") == pytest.approx(0.5)


def test_binary_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="unknown aggregation rule"):
        combine_binary([0.5], "mean")
    for rule in RULES:
        with pytest.raises(ValueError, match="zero forecasts"):
            combine_binary([], rule)
        with pytest.raises(ValueError, match="outside"):
            combine_binary([0.5, 1.0], rule)
    assert median([0.1, 0.3]) == pytest.approx(0.2)


# Multiple choice -----------------------------------------------------------------

CHOICES = [
    {"A": 0.5, "B": 0.45, "C": 0.05},
    {"A": 0.6, "B": 0.35, "C": 0.05},
    {"A": 0.4, "B": 0.5, "C": 0.1},
]


@pytest.mark.parametrize("rule", RULES)
def test_choice_sums_to_one_and_keeps_order(rule: str) -> None:
    out = combine_choice(CHOICES, OPTIONS, rule)
    assert list(out) == OPTIONS
    assert validate_probabilities(out, OPTIONS) == []
    assert math.fsum(out.values()) == pytest.approx(1.0)
    assert out["A"] > out["B"] > out["C"]


def test_choice_applies_the_binary_rule_per_option_then_renormalises() -> None:
    for rule in RULES:
        per_option = {o: combine_binary([m[o] for m in CHOICES], rule) for o in OPTIONS}
        expected = normalise_probabilities(per_option, OPTIONS)
        assert combine_choice(CHOICES, OPTIONS, rule) == pytest.approx(expected)
    # Extremizing sharpens each option before renormalising: the favourite gains.
    pooled = combine_choice(CHOICES, OPTIONS, "mean_logodds")
    extreme = combine_choice(CHOICES, OPTIONS, "extremized_logodds")
    assert extreme["C"] < pooled["C"]


# Numeric and discrete ------------------------------------------------------------

BOUNDS = [(lo, up) for lo in (True, False) for up in (True, False)]


@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
@pytest.mark.parametrize("n", [200, 16])
@pytest.mark.parametrize("rule", RULES)
def test_continuous_rules_give_valid_cdfs(lower: bool, upper: bool, n: int, rule: str) -> None:
    scaling = _scaling(lower=lower, upper=upper, n=n)
    cdfs = [_member(scaling, c, w).cdf for c, w in ((30, 10), (45, 15), (60, 8), (50, 30))]
    combined, method = combine_cdfs(cdfs, scaling, rule)
    assert validate_cdf(combined, scaling) == []
    assert not method.endswith("_fallback_mean_cdf")
    # Pointwise, every rule sits between the members.
    for i, v in enumerate(combined):
        column = [c[i] for c in cdfs]
        assert min(column) - 1e-9 <= v <= max(column) + 1e-9


def test_continuous_median_and_trimmed_mean_are_pointwise() -> None:
    scaling = _scaling()
    cdfs = [_member(scaling, c, 10).cdf for c in (20, 40, 50, 60, 80)]
    med, method = combine_cdfs(cdfs, scaling, "median")
    assert method == "median_cdf"
    assert med == pytest.approx([statistics.median(col) for col in zip(*cdfs, strict=True)])
    trim, method = combine_cdfs(cdfs, scaling, "trimmed_mean")
    assert method == "trimmed_mean_cdf"
    expected = [sum(sorted(col)[1:4]) / 3 for col in zip(*cdfs, strict=True)]
    assert trim == pytest.approx(expected)


def test_continuous_extremized_is_the_unextremized_log_odds_pool() -> None:
    scaling = _scaling(lower=False)
    cdfs = [_member(scaling, c, 12).cdf for c in (30, 50, 55)]
    pooled, method = combine_cdfs(cdfs, scaling, "mean_logodds")
    assert method == "mean_logodds_cdf"
    assert pooled[0] == 0.0  # a closed bound's end is pinned, not pooled
    same, method = combine_cdfs(cdfs, scaling, "extremized_logodds")
    assert method == "mean_logodds_cdf_unextremized"
    assert same == pooled


def test_an_invalid_combination_falls_back_to_the_mixture_visibly() -> None:
    # A point mass (spread to the steepest step allowed) and a wide member: the
    # log-odds pool keeps most of the point mass's jump in log-odds, and near
    # 0.5 that is a step above the server's maximum, which a mixture never makes.
    scaling = _scaling(lower=False, upper=False)
    cdfs = [_member(scaling, 50, 0.2).cdf, _member(scaling, 52, 40).cdf]
    pooled = _pointwise_cdf(cdfs, "mean_logodds", scaling)
    assert any("at most" in e for e in validate_cdf(pooled, scaling))
    combined, method = combine_cdfs(cdfs, scaling, "mean_logodds")
    assert method == "mean_logodds_cdf_fallback_mean_cdf"
    assert combined == mix_cdfs(cdfs)
    assert validate_cdf(combined, scaling) == []


def test_cdfs_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="one length"):
        combine_cdfs([[0.0, 1.0], [0.0, 0.5, 1.0]], _scaling(n=1), "median")
    with pytest.raises(ValueError, match="zero forecasts"):
        combine_cdfs([], _scaling(), "median")


# predictions.aggregate -----------------------------------------------------------


def _old_aggregate(q: Question, members: list[Prediction]) -> tuple[Prediction, str]:
    """``predictions.aggregate`` as it was before the rules, copied verbatim in effect."""
    if q.type == "binary":
        ps = [m.probability for m in members if isinstance(m, BinaryPrediction)]
        return BinaryPrediction(round(float(statistics.median(ps)), 3)), "median"
    if q.type == "multiple_choice":
        dicts = [m.probabilities for m in members if isinstance(m, ChoicePrediction)]
        per = {o: float(statistics.median([d[o] for d in dicts])) for o in OPTIONS}
        return ChoicePrediction(normalise_probabilities(per, OPTIONS)), (
            "median_per_option_normalised"
        )
    assert q.scaling is not None
    mixed = mix_cdfs([m.cdf for m in members if isinstance(m, ContinuousPrediction)])
    levels = tuple((lv, quantile(mixed, q.scaling, lv)) for lv in PERCENTILE_LEVELS)
    return ContinuousPrediction(tuple(mixed), levels), "mean_cdf"


def test_default_rule_is_unchanged() -> None:
    binary = _question("binary")
    members: list[Prediction] = [BinaryPrediction(p) for p in (0.1, 0.2, 0.9, 0.35)]
    assert aggregate(binary, members) == _old_aggregate(binary, members)
    assert aggregate(binary, members) == (BinaryPrediction(0.275), "median")

    mc = _question("multiple_choice")
    choices: list[Prediction] = [ChoicePrediction(d) for d in CHOICES]
    assert aggregate(mc, choices) == _old_aggregate(mc, choices)
    assert aggregate(mc, choices, rule="median") == _old_aggregate(mc, choices)

    for lower, upper in BOUNDS:
        for n in (200, 16):
            q = _question(
                "numeric" if n == 200 else "discrete", _scaling(lower=lower, upper=upper, n=n)
            )
            assert q.scaling is not None
            conts: list[Prediction] = [_member(q.scaling, c, 9) for c in (25, 50, 70)]
            assert aggregate(q, conts) == _old_aggregate(q, conts)


def test_aggregate_dispatches_and_names_each_rule() -> None:
    binary = _question("binary")
    members: list[Prediction] = [BinaryPrediction(p) for p in (0.1, 0.2, 0.9)]
    assert aggregate(binary, members, rule="trimmed_mean") == (
        BinaryPrediction(0.4),
        "trimmed_mean",
    )
    assert aggregate(binary, members, rule="extremized_logodds") == (
        BinaryPrediction(0.333),
        "extremized_logodds",
    )
    mc = _question("multiple_choice")
    _, method = aggregate(mc, [ChoicePrediction(d) for d in CHOICES], rule="mean_logodds")
    assert method == "mean_logodds_per_option_normalised"
    q = _question("numeric", _scaling())
    assert q.scaling is not None
    conts: list[Prediction] = [_member(q.scaling, c, 9) for c in (25, 50, 70)]
    combined, method = aggregate(q, conts, rule="trimmed_mean")
    assert method == "trimmed_mean_cdf"
    assert check(q, combined) == []
    with pytest.raises(ValueError, match="unknown aggregation rule"):
        aggregate(binary, members, rule="mode")


# Properties ----------------------------------------------------------------------

_prob = st.floats(min_value=0.001, max_value=0.999)


@settings(max_examples=60, deadline=None)
@given(ps=st.lists(_prob, min_size=1, max_size=12), rule=st.sampled_from(RULES))
def test_every_rule_gives_a_valid_binary_prediction(ps: list[float], rule: str) -> None:
    q = _question("binary")
    combined, _ = aggregate(q, [BinaryPrediction(p) for p in ps], rule=rule)
    assert check(q, combined) == []
    assert isinstance(combined, BinaryPrediction)
    if rule != "extremized_logodds":  # the one rule that may leave the members' range
        low, high = min(min(ps), 0.01), max(max(ps), 0.99)
        assert low - 0.0005 <= combined.probability <= high + 0.0005


@settings(max_examples=60, deadline=None)
@given(
    raw=st.lists(st.lists(_prob, min_size=3, max_size=3), min_size=1, max_size=9),
    rule=st.sampled_from(RULES),
)
def test_every_rule_gives_a_valid_choice_prediction(raw: list[list[float]], rule: str) -> None:
    q = _question("multiple_choice")
    members: list[Prediction] = [
        ChoicePrediction(normalise_probabilities(dict(zip(OPTIONS, r, strict=True)), OPTIONS))
        for r in raw
    ]
    combined, _ = aggregate(q, members, rule=rule)
    assert check(q, combined) == []
    assert isinstance(combined, ChoicePrediction)
    assert list(combined.probabilities) == OPTIONS


@settings(max_examples=40, deadline=None)
@given(
    shapes=st.lists(
        st.tuples(st.floats(-50.0, 150.0), st.floats(0.0, 60.0), st.floats(0.0, 60.0)),
        min_size=1,
        max_size=7,
    ),
    lower=st.booleans(),
    upper=st.booleans(),
    n=st.sampled_from([200, 16, 3]),
    rule=st.sampled_from(RULES),
)
def test_every_rule_gives_a_valid_cdf(
    shapes: list[tuple[float, float, float]], lower: bool, upper: bool, n: int, rule: str
) -> None:
    scaling = _scaling(lower=lower, upper=upper, n=n)
    q = _question("numeric" if n == 200 else "discrete", scaling)
    members: list[Prediction] = []
    for centre, below, above in shapes:
        pairs = [(0.1, centre - below), (0.5, centre), (0.9, centre + above)]
        members.append(ContinuousPrediction(tuple(cdf_from_percentiles(pairs, scaling)), ()))
    combined, _ = aggregate(q, members, rule=rule)
    assert check(q, combined) == []
