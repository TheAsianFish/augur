"""Every question type: answers parsed, members combined, payloads Metaculus accepts."""

from __future__ import annotations

from typing import Any

import pytest

from augur.agents.forecaster import parse_choice, parse_percentiles, parse_prediction
from augur.distributions import validate_cdf
from augur.errors import ForecastParseError
from augur.predictions import (
    BinaryPrediction,
    ChoicePrediction,
    ContinuousPrediction,
    aggregate,
    check,
    forecast_payload,
    summary,
)
from augur.questions import Question, question_from_post

from .fakes.metaculus import (
    DISCRETE_DETAIL,
    DISCRETE_STEP_DETAIL,
    MC_DETAIL,
    MINIBENCH_DISCRETE_DETAIL,
    MINIBENCH_MC_DETAIL,
    NUMERIC_DETAIL,
    _choice_errors,
    _continuous_errors,
    load_fixture,
)


def _q(name: str) -> tuple[Question, dict[str, Any]]:
    post = load_fixture(name)
    return question_from_post(post), post["question"]


def test_real_payloads_parse_with_their_scaling() -> None:
    numeric, _ = _q(NUMERIC_DETAIL)
    assert numeric.type == "numeric"
    assert numeric.scaling is not None
    assert numeric.scaling.inbound_outcome_count == 200
    assert numeric.unsupported_reason() is None
    discrete, _ = _q(DISCRETE_DETAIL)
    assert discrete.scaling is not None
    assert discrete.scaling.inbound_outcome_count == 16
    assert not discrete.scaling.open_lower_bound
    assert not discrete.scaling.open_upper_bound
    mc, _ = _q(MC_DETAIL)
    assert mc.options == ["Democratic Party", "Republican Party", "Other"]
    assert mc.unsupported_reason() is None


def test_choice_lines_are_read_per_option_and_normalised() -> None:
    options = ["Democratic Party", "Republican Party", "Other"]
    text = (
        "Some reasoning mentioning Other: 50% in passing is overridden by the final lines.\n"
        "- **Democratic Party**: 48%\nRepublican Party: 47%\nOther: 10%"
    )
    probs = parse_choice(text, options)
    assert list(probs) == options
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["Democratic Party"] == pytest.approx(48 / 105, abs=1e-3)
    with pytest.raises(ForecastParseError, match="Other"):
        parse_choice("Democratic Party: 50%\nRepublican Party: 50%", options)


def test_choice_options_with_punctuation_are_matched_literally() -> None:
    q, _ = _q(MINIBENCH_MC_DETAIL)
    assert q.options is not None
    text = "\n".join(f"{o}: {p}%" for o, p in zip(q.options, (10, 40, 30, 20), strict=True))
    probs = parse_choice(text, q.options)
    assert probs[q.options[1]] == pytest.approx(0.4, abs=1e-3)


def test_percentiles_take_the_last_value_given_and_tolerate_formatting() -> None:
    text = (
        "Percentile 10: 1\n(draft)\n"
        "Percentile 10: 4,000,000\nPercentile 20: **4,500,000**\nPercentile 40: $5.1e6\n"
        "Percentile 60: 5600000\nPercentile 80: 6,200,000.5\nPercentile 90: 7000000"
    )
    stated = parse_percentiles(text)
    assert stated[0] == (0.1, 4_000_000.0)
    assert stated[2] == (0.4, 5_100_000.0)
    assert [lv for lv, _ in stated] == [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    with pytest.raises(ForecastParseError, match="90"):
        parse_percentiles("Percentile 10: 1\nPercentile 20: 2")
    assert parse_percentiles(
        "Percentile 10: −5\n"
        + "\n".join(
            f"Percentile {lv}: {v}" for lv, v in ((20, -4), (40, -3), (60, -2), (80, -1), (90, 0))
        )
    )[0] == (0.1, -5.0)


@pytest.mark.parametrize(
    "name", [NUMERIC_DETAIL, DISCRETE_DETAIL, DISCRETE_STEP_DETAIL, MINIBENCH_DISCRETE_DETAIL]
)
def test_a_parsed_continuous_answer_is_a_cdf_the_server_accepts(name: str) -> None:
    q, raw = _q(name)
    assert q.scaling is not None
    low = q.scaling.nominal_min if q.scaling.nominal_min is not None else q.scaling.range_min
    high = q.scaling.nominal_max if q.scaling.nominal_max is not None else q.scaling.range_max
    lines = "\n".join(
        f"Percentile {lv}: {low + (high - low) * s}"
        for lv, s in zip((10, 20, 40, 60, 80, 90), (0.1, 0.2, 0.4, 0.6, 0.8, 0.9), strict=True)
    )
    prediction = parse_prediction(q, lines)
    assert isinstance(prediction, ContinuousPrediction)
    assert check(q, prediction) == []
    # The fake server's independent transliteration of Metaculus's rules agrees.
    assert _continuous_errors(list(prediction.cdf), raw) == ""
    [body] = forecast_payload(q.question_id, prediction)
    assert body["probability_yes"] is None
    assert len(body["continuous_cdf"]) == len(prediction.cdf)


def test_members_combine_per_type() -> None:
    binary = question_from_post(load_fixture("post_43327_binary_detail.json"))
    combined, method = aggregate(binary, [BinaryPrediction(p) for p in (0.1, 0.2, 0.9)])
    assert combined == BinaryPrediction(0.2)
    assert method == "median"

    mc, raw = _q(MC_DETAIL)
    members = [
        ChoicePrediction({"Democratic Party": a, "Republican Party": b, "Other": c})
        for a, b, c in ((0.5, 0.45, 0.05), (0.6, 0.35, 0.05), (0.4, 0.5, 0.1))
    ]
    combined, method = aggregate(mc, members)
    assert isinstance(combined, ChoicePrediction)
    assert method == "median_per_option_normalised"
    assert sum(combined.probabilities.values()) == pytest.approx(1.0)
    assert _choice_errors(combined.probabilities, raw) == ""

    numeric, raw = _q(NUMERIC_DETAIL)
    preds = [
        parse_prediction(
            numeric,
            "\n".join(
                f"Percentile {lv}: {v + shift}"
                for lv, v in zip(
                    (10, 20, 40, 60, 80, 90), (-30, -25, -20, -15, -10, -5), strict=True
                )
            ),
        )
        for shift in (0, 3, 6)
    ]
    combined, method = aggregate(numeric, preds)
    assert isinstance(combined, ContinuousPrediction)
    assert method == "mean_cdf"
    assert numeric.scaling is not None
    assert validate_cdf(list(combined.cdf), numeric.scaling) == []
    assert _continuous_errors(list(combined.cdf), raw) == ""
    # The mixture's median sits between the members' medians.
    assert -17.5 - 1 < dict(combined.percentiles)[0.6] < -9.5 + 1
    assert "median" in summary(numeric, combined)


def test_check_refuses_what_the_server_would() -> None:
    mc, _ = _q(MC_DETAIL)
    bad = ChoicePrediction({"Democratic Party": 0.9995, "Republican Party": 0.0005, "Other": 0.0})
    assert check(mc, bad)
    binary = question_from_post(load_fixture("post_43327_binary_detail.json"))
    assert check(binary, BinaryPrediction(0.0001))


def test_a_discrete_summary_names_outcomes_not_points_inside_buckets() -> None:
    q, _ = _q(DISCRETE_DETAIL)  # 0 to 15 officials, buckets centred on each count
    lines = "\n".join(
        f"Percentile {lv}: {v}"
        for lv, v in zip((10, 20, 40, 60, 80, 90), (0, 0, 0, 0, 2, 3), strict=True)
    )
    text = summary(q, parse_prediction(q, lines))
    assert text.startswith("median 0 Officials (80% between 0 and ")
    assert "-0." not in text
