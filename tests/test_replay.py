"""The replay evaluation: recorded prices, arm (d)'s research, and every arm's score."""

from __future__ import annotations

import math
import sqlite3
import statistics
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from augur.agents.researcher import ResearchResult
from augur.clock import FixedClock
from augur.errors import LeakageError
from augur.forecastbench import FBQuestion
from augur.leakage import scan_ledger
from augur.ledger import Ledger
from augur.replay import (
    BLEND_WEIGHT,
    WITH_PRICES_SUFFIX,
    blend,
    clip,
    logit,
    mean_logodds,
    price_research,
    record_prices,
    recorded_prices,
    score_corpus,
    sigmoid,
)
from augur.research.markets import MarketsSearcher
from augur.research.quotes import MarketQuote
from augur.research.source import DateEvidence, Source
from augur.score import brier

from .fakes.forecastbench import QUESTION_SET, run_backtest, small_set

NOW = datetime(2026, 11, 19, tzinfo=UTC)
TOURNAMENT = f"forecastbench:{QUESTION_SET}"
# small_set: a (manifold, Yes), b (polymarket, No), c (infer, No).
PRICES = {"a": 0.8, "b": 0.3}


class ScriptedMarkets(MarketsSearcher):
    """Exact quotes from a table, dated an hour before the freeze (or after, if asked)."""

    def __init__(self, late: bool = False) -> None:
        super().__init__(httpx.AsyncClient(), FixedClock(NOW))
        self.late = late

    async def exact_quote(
        self, platform: str, market_id: str, as_of: datetime, *, title: str = "", url: str = ""
    ) -> MarketQuote | None:
        if market_id not in PRICES:
            return None
        return MarketQuote(
            platform="manifold" if platform == "manifold" else "polymarket",
            market_id=market_id,
            title=title,
            url=url,
            outcome="Yes",
            probability=PRICES[market_id],
            price_time=as_of - timedelta(hours=1),
            as_of=as_of,
            match="exact",
        )


async def _record(path: Path, questions: list[FBQuestion]) -> None:
    with Ledger.open(path, FixedClock(NOW)) as ledger:
        report = await record_prices(
            questions=questions,
            tournament=TOURNAMENT,
            searcher=ScriptedMarkets(),
            ledger=ledger,
            code_version="test",
            out=lambda _line: None,
        )
    assert report.priced == 2
    assert report.unpriced == {"infer": 1}


def _rows(path: Path, sql: str) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(path)) as conn:
        return [tuple(r) for r in conn.execute(sql)]


def test_the_blend_is_one_third_augur_in_log_odds() -> None:
    assert pytest.approx(1 / 3) == BLEND_WEIGHT
    assert blend(0.5, 0.8) == pytest.approx(sigmoid(2 / 3 * math.log(4)))
    assert blend(0.3, 0.3) == pytest.approx(0.3)
    assert blend(0.0, 1.0) == pytest.approx(sigmoid(logit(0.01) / 3 + 2 * logit(0.99) / 3))
    assert clip(1.0) == 0.99
    assert mean_logodds([0.2, 0.8]) == pytest.approx(0.5)
    assert mean_logodds([0.8, 0.8], 1.5) == pytest.approx(sigmoid(1.5 * logit(0.8)))


async def test_prices_are_recorded_as_admitted_dated_sources(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    path = tmp_path / "prices.sqlite"
    await _record(path, questions)
    prices = recorded_prices(path, TOURNAMENT)
    by_id = {q.fb_id: q.question_id for q in questions}
    assert sorted(prices) == sorted([by_id["a"], by_id["b"]])
    quote = prices[by_id["a"]].quote
    assert quote is not None
    assert quote.probability == 0.8
    assert scan_ledger(path) == []
    outcomes = _rows(path, "SELECT status, reason FROM question_outcomes ORDER BY rowid")
    assert outcomes[2] == ("skipped", "no usable price: infer has no public price API")


async def test_arm_d_shows_the_recorded_price_and_skips_unpriced(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    prices_path = tmp_path / "prices.sqlite"
    await _record(prices_path, questions)
    db = tmp_path / "d.sqlite"
    result = await run_backtest(
        db,
        questions,
        tournament=TOURNAMENT + WITH_PRICES_SUFFIX,
        research_for=price_research(recorded_prices(prices_path, TOURNAMENT)),
    )
    assert (result.forecast, result.skipped) == (2, 1)
    requests = [r for (r,) in _rows(db, "SELECT request_json FROM llm_calls")]
    assert len(requests) == 6
    assert all("Price 80.0% for Yes" in r or "Price 30.0% for Yes" in r for r in requests)
    assert _rows(db, "SELECT count(*) FROM call_sources") == [(6,)]
    assert _rows(db, "SELECT count(*), sum(admitted) FROM source_uses") == [(2, 2)]
    assert scan_ledger(db) == []


async def test_a_late_source_in_the_research_aborts_the_run(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    late = Source.build(
        searcher="markets",
        query="q",
        url="https://example.org",
        title="t",
        excerpt="the answer",
        published_at=questions[0].freeze_datetime + timedelta(days=1),
        retrieved_at=NOW,
        as_of=questions[0].freeze_datetime + timedelta(days=1),
        date_evidence=DateEvidence.REVISION_TIMESTAMP,
        evidence_detail="late",
    )

    def research(_fb: FBQuestion) -> ResearchResult:
        return ResearchResult(queries=[], sources=[late], brief="the answer")

    db = tmp_path / "late.sqlite"
    with pytest.raises(LeakageError):
        await run_backtest(db, questions, research_for=research)
    assert _rows(db, "SELECT count(*) FROM llm_calls") == [(0,)]


async def test_every_arm_scores_on_the_priced_questions(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    prices_path = tmp_path / "prices.sqlite"
    await _record(prices_path, questions)
    b_path = tmp_path / "b.sqlite"
    # Draws 20%, 40%, 70% on every question: median 40%.
    await run_backtest(b_path, questions, percents=(20, 40, 70))
    d_path = tmp_path / "d.sqlite"
    await run_backtest(
        d_path,
        questions,
        percents=(60,),
        tournament=TOURNAMENT + WITH_PRICES_SUFFIX,
        research_for=price_research(recorded_prices(prices_path, TOURNAMENT)),
    )
    score = score_corpus(
        label="test",
        questions=questions,
        tournament=TOURNAMENT,
        forecasts=b_path,
        prices=prices_path,
        with_prices=d_path,
    )
    assert (score.questions, score.priced) == (3, 2)
    assert score.priced_by_platform == {"manifold": (1, 1), "polymarket": (1, 1), "infer": (0, 1)}
    assert score.freeze_match == (2, 2)  # 0.8 and 0.3 are the subset's freeze values
    arms = {a.name: a for a in score.arms}
    ys = [1, 0]
    assert arms["a"].brier.mean == pytest.approx(statistics.fmean([0.04, 0.09]))
    assert arms["b"].brier.mean == pytest.approx(statistics.fmean([0.36, 0.16]))
    c = [blend(0.4, 0.8), blend(0.4, 0.3)]
    assert arms["c"].brier.mean == pytest.approx(
        statistics.fmean(brier(p, y) for p, y in zip(c, ys, strict=True))
    )
    assert arms["d"].brier.mean == pytest.approx(statistics.fmean([0.16, 0.36]))
    diffs = {d.name: d.brier.mean for d in score.diffs}
    assert diffs["b-a"] == pytest.approx(arms["b"].brier.mean - arms["a"].brier.mean)
    assert diffs["d-b"] == pytest.approx(0.0)
    assert score.all_b is not None
    assert score.all_b.brier.n == 3
    settings = {d.name: d.brier.mean for d in score.settings}
    assert score.settings_n == 3
    # One draw (20%) against the median (40%) on outcomes 1, 0, 0.
    one = statistics.fmean([0.64 - 0.36, 0.04 - 0.16, 0.04 - 0.16])
    assert settings["one draw - median of 3"] == pytest.approx(one)
