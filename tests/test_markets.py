"""The ``markets`` searcher against a fake server that sends prices from after the cutoff."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from augur.clock import FixedClock
from augur.errors import SearchError
from augur.leakage import RunMode, admit_source, scan_ledger
from augur.ledger import Ledger
from augur.questions import Question
from augur.ratelimit import RateLimiter
from augur.research.markets import MarketsSearcher, keywords, metaculus_aggregate_at, overlap
from augur.research.quotes import MarketQuote
from augur.research.source import DateEvidence, QuestionSearcher, Source
from tests.fakes.markets import (
    AFTER,
    AS_OF,
    BEFORE,
    KALSHI_TICKER,
    MANIFOLD_ID,
    POLY_CONDITION,
    FakeMarkets,
    FakeMetaculus,
)
from tests.helpers import binary_question, market_source

NOW = datetime(2026, 11, 19, tzinfo=UTC)


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def searcher(
    fake: FakeMarkets, metaculus: FakeMetaculus | None = None
) -> tuple[MarketsSearcher, httpx.AsyncClient]:
    t = FakeTime()
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    return MarketsSearcher(http, FixedClock(NOW), metaculus=metaculus, limiter=limiter), http


@pytest.mark.parametrize(
    ("platform", "market_id"),
    [("manifold", MANIFOLD_ID), ("polymarket", POLY_CONDITION), ("kalshi", KALSHI_TICKER)],
)
async def test_a_price_dated_after_the_cutoff_is_never_read(platform: str, market_id: str) -> None:
    # The fake ignores every time bound and returns a 97% point an hour after AS_OF.
    fake = FakeMarkets(ignore_bounds=True)
    s, http = searcher(fake)
    async with http:
        quote = await s.exact_quote(platform, market_id, AS_OF)
    assert quote is not None
    assert quote.probability == pytest.approx(0.40)
    assert quote.price_time == BEFORE
    assert quote.price_time <= AS_OF
    assert quote.match == "exact"


@pytest.mark.parametrize(
    ("platform", "market_id"),
    [("manifold", MANIFOLD_ID), ("polymarket", POLY_CONDITION), ("kalshi", KALSHI_TICKER)],
)
async def test_only_late_prices_means_no_price(platform: str, market_id: str) -> None:
    s, http = searcher(FakeMarkets(only_late=True))
    async with http:
        assert await s.exact_quote(platform, market_id, AS_OF) is None


async def test_the_request_bounds_are_the_cutoff() -> None:
    fake = FakeMarkets(ignore_bounds=False)
    s, http = searcher(fake)
    async with http:
        await s.exact_quote("manifold", MANIFOLD_ID, AS_OF)
        await s.exact_quote("polymarket", POLY_CONDITION, AS_OF)
    bets = next(r for r in fake.requests if r.url.path.endswith("/bets"))
    assert int(bets.url.params["beforeTime"]) == int(AS_OF.timestamp() * 1000) + 1
    clob = next(r for r in fake.requests if r.url.host == "clob.polymarket.com")
    assert int(clob.url.params["endTs"]) == int(AS_OF.timestamp())


def test_a_quote_dated_after_as_of_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="after the instant it was asked for"):
        MarketQuote(
            platform="kalshi",
            market_id="X",
            title="t",
            url="u",
            outcome="Yes",
            probability=0.5,
            price_time=AFTER,
            as_of=AS_OF,
            match="exact",
        )


async def test_the_metaculus_aggregate_in_force_at_the_cutoff() -> None:
    s, http = searcher(FakeMarkets(), FakeMetaculus())
    async with http:
        quote = await s.metaculus_quote(100, AS_OF)
    assert quote is not None
    assert quote.probability == pytest.approx(0.35)
    assert quote.price_time == BEFORE
    assert quote.platform == "metaculus"


async def test_a_hidden_aggregate_is_no_price() -> None:
    s, http = searcher(FakeMarkets(), FakeMetaculus(hidden=True))
    async with http:
        assert await s.metaculus_quote(100, AS_OF) is None


def test_aggregate_ignores_non_binary_and_entries_starting_later() -> None:
    post = {"question": {"type": "numeric", "aggregations": {}}}
    assert metaculus_aggregate_at(post, AS_OF) is None
    later = {
        "question": {
            "type": "binary",
            "aggregations": {
                "recency_weighted": {
                    "history": [
                        {"start_time": AFTER.timestamp(), "end_time": None, "centers": [0.9]}
                    ]
                }
            },
        }
    }
    assert metaculus_aggregate_at(later, AS_OF) is None


async def test_search_skips_markets_created_after_as_of_and_dates_each_hit() -> None:
    s, http = searcher(FakeMarkets())
    async with http:
        sources = await s.search("test event 2026", AS_OF, 5)
    platforms = sorted(src.quote.platform for src in sources if src.quote)
    assert platforms == ["kalshi", "manifold", "polymarket"]
    for src in sources:
        assert src.quote is not None
        assert src.quote.match == "search"
        assert src.published_at == src.quote.price_time
        assert src.published_at is not None
        assert src.published_at <= AS_OF
        assert "may ask something different" in src.excerpt
        assert admit_source(src, AS_OF, RunMode.BACKTEST).admitted
    assert not any(src.quote and src.quote.market_id == "born-later" for src in sources)


async def test_search_question_puts_the_own_aggregate_first() -> None:
    q, _ = binary_question(AS_OF - timedelta(days=1), title="Will the test event happen in 2026?")
    s, http = searcher(FakeMarkets(), FakeMetaculus())
    async with http:
        sources = await s.search_question(q, AS_OF, 2)
    assert sources[0].quote is not None
    assert sources[0].quote.platform == "metaculus"
    assert sources[0].date_evidence is DateEvidence.AGGREGATE_HISTORY
    assert len(sources) == 3


async def test_a_failing_platform_does_not_sink_the_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.manifold.markets":
            return httpx.Response(403)
        return FakeMarkets()(request)

    t = FakeTime()
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        s = MarketsSearcher(http, FixedClock(NOW), limiter=limiter)
        sources = await s.search("test event 2026", AS_OF, 5)
        with pytest.raises(SearchError, match="HTTP 403"):
            await s.exact_quote("manifold", MANIFOLD_ID, AS_OF)
    assert sorted(src.quote.platform for src in sources if src.quote) == ["kalshi", "polymarket"]


async def test_a_recorded_market_source_scans_clean(tmp_path: Path) -> None:
    s, http = searcher(FakeMarkets())
    async with http:
        quote = await s.exact_quote("polymarket", POLY_CONDITION, AS_OF)
    assert quote is not None
    src = s.to_source(quote, "test")
    path = tmp_path / "l.sqlite"
    with Ledger.open(path, FixedClock(NOW)) as ledger:
        ledger.record_source(src)
    assert scan_ledger(path) == []


def test_keywords_and_overlap() -> None:
    assert keywords("Will the New York Mets win the 2026 World Series?") == [
        "new",
        "york",
        "mets",
        "win",
        "world",
        "series",
    ]
    assert overlap("mets world series", "Will the Mets win the World Series?") == 1.0
    assert overlap("mets world series", "Who wins the Super Bowl?") == 0.0


def test_dates_are_not_content() -> None:
    # The first coverage run matched the Ibovespa to "released on November 26, 2026".
    query = " ".join(keywords("What will the Ibovespa close at on 26 November 2026?"))
    assert query == "ibovespa close"
    assert overlap(query, "Will the next model be released on November 26, 2026?") == 0.0


class QuestionOnly:
    """A question-level searcher: one call per question, never per planned query."""

    name = "markets"

    def __init__(self) -> None:
        self.question_calls = 0
        self.query_calls = 0

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        self.query_calls += 1
        return []

    async def search_question(self, q: Question, as_of: datetime, limit: int) -> list[Source]:
        self.question_calls += 1
        return [market_source(as_of - timedelta(hours=1), as_of=as_of)]


async def test_the_question_level_search_runs_once_and_goes_first(tmp_path: Path) -> None:
    from augur.agents.calls import Budget, CallRecorder
    from augur.agents.researcher import Researcher
    from augur.llm.endpoints import PRESETS, Route
    from augur.llm.router import Router
    from tests.fakes.llm import ScriptedProvider
    from tests.fakes.search import FixedSearcher
    from tests.helpers import free_catalog

    q, _ = binary_question(AS_OF - timedelta(days=1), title="Will the test event happen?")
    clock = FixedClock(AS_OF)
    route = Route(endpoint=PRESETS["ollama"], model="qwen2.5:7b")
    router = Router.single(route, ScriptedProvider(), free_catalog(route), clock)
    markets, fixed = QuestionOnly(), FixedSearcher()
    assert isinstance(markets, QuestionSearcher)
    assert not isinstance(fixed, QuestionSearcher)
    with Ledger.open(tmp_path / "l.sqlite", clock) as ledger:
        run_id = ledger.start_run(
            mode=RunMode.DRY_RUN,
            tournament="t",
            provider="replay",
            model="m",
            settings={},
            code_version="test",
        )
        ledger.record_question(run_id, q, AS_OF, {})
        calls = CallRecorder(router, ledger, clock, Budget(1.0, 1.0))
        researcher = Researcher(
            calls=calls, ledger=ledger, searchers=[fixed, markets], mode=RunMode.DRY_RUN
        )
        result = await researcher.research(run_id, q, AS_OF)
    assert (markets.question_calls, markets.query_calls) == (1, 0)
    assert result.sources[0].searcher == "markets"
    assert len(fixed.calls) >= 1
