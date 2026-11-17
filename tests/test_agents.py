from __future__ import annotations

from datetime import timedelta

import pytest

from augur.agents.forecaster import MemberForecast, parse_probability, reasoning_of
from augur.agents.researcher import ResearchResult, fallback_query, parse_queries
from augur.aggregate import median
from augur.comment import MAX_COMMENT_CHARS, render_comment
from augur.errors import ForecastParseError

from .conftest import T0
from .helpers import binary_question, source


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning.\nProbability: 23%", 0.23),
        ("probability = 7.5 %", 0.075),
        ("**Probability:** 40%", 0.40),
        ("Probability: 10% at first... revised.\nProbability: 30%", 0.30),
        ("Probability: 0%", 0.01),
        ("Probability: 100%", 0.99),
    ],
)
def test_parse_probability_takes_the_last_line_and_clips(text: str, expected: float) -> None:
    assert parse_probability(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["I think it is likely.", "Probability: 140%", "chance: 0.3"])
def test_unparseable_answers_are_errors(text: str) -> None:
    with pytest.raises(ForecastParseError):
        parse_probability(text)


def test_reasoning_drops_the_final_probability_line() -> None:
    assert reasoning_of("Step one.\nStep two.\n\nProbability: 20%\n") == "Step one.\nStep two."


def test_parse_queries_from_json_or_lines() -> None:
    assert parse_queries('Sure: ["brent crude", "opec cut", "brent crude"]', 3) == [
        "brent crude",
        "opec cut",
    ]
    assert parse_queries("1. brent crude price\n2. opec meeting", 3) == [
        "brent crude price",
        "opec meeting",
    ]
    assert parse_queries('["a", "b", "c", "d"]', 2) == ["a", "b"]


def test_fallback_query_keeps_the_content_words() -> None:
    q, _ = binary_question(T0, title="Will the Lions beat the Bears on Thanksgiving?")
    assert fallback_query(q) == "Lions beat Bears Thanksgiving"


def test_question_text_states_the_time_left() -> None:
    from augur.agents.researcher import question_text

    q, _ = binary_question(T0 - timedelta(days=1), hours_open=24 + 44 * 24)
    assert "closes 2026-12-31 18:00 UTC, 44 days from now." in question_text(q, T0)
    soon, _ = binary_question(T0 - timedelta(hours=1), hours_open=4)
    assert "3 hours from now" in question_text(soon, T0)


def test_median() -> None:
    assert median([0.2, 0.9, 0.4]) == pytest.approx(0.4)
    assert median([0.2, 0.4]) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="zero"):
        median([])
    with pytest.raises(ValueError, match="outside"):
        median([0.5, 1.0])


def _members(*pcts: int) -> list[MemberForecast]:
    return [
        MemberForecast(i, p / 100, f"Reasoning of member {i}. " * 3, f"call_{i}")
        for i, p in enumerate(pcts)
    ]


def test_comment_reflects_members_brief_and_dated_sources() -> None:
    s = source(T0 - timedelta(days=4), retrieved_at=T0, url="https://en.wikipedia.org/x")
    research = ResearchResult(queries=["q"], sources=[s], brief="Status quo holds [S1].")
    text = render_comment(
        probability=0.3,
        members=_members(30, 20, 40, 35, 25),
        research=research,
        model="qwen2.5:7b",
        cutoff=T0,
        requested=5,
    )
    assert text.startswith("**Forecast: 30%**, the median of 5 independent forecasts")
    assert "(30%, 20%, 40%, 35%, 25%) by qwen2.5:7b." in text
    assert "only sources dated at or before 2026-11-17 18:00 UTC" in text
    assert "Status quo holds [S1]." in text
    assert "1. (30%) Reasoning of member 0." in text
    assert "- [S1] [Test](https://en.wikipedia.org/x) (wikipedia, revision of 2026-11-13)" in text


def test_comment_says_when_draws_failed_and_stays_short() -> None:
    research = ResearchResult(queries=[], sources=[], brief="x " * 5000)
    members = [MemberForecast(i, 0.5, "word " * 2000, f"c{i}") for i in range(3)]
    text = render_comment(
        probability=0.5,
        members=members,
        research=research,
        model="m",
        cutoff=T0,
        requested=5,
    )
    assert "2 of 5 draws failed" in text
    assert len(text) <= MAX_COMMENT_CHARS
