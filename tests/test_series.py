"""The ``series`` source: the title rule, and a fake chart server sending closes past the cutoff."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest

from augur.clock import FixedClock
from augur.errors import LeakageError, SearchError
from augur.leakage import Admission, RunMode, admit_source, scan_ledger
from augur.ledger import Ledger
from augur.ratelimit import RateLimiter
from augur.research.series import (
    ALIASES,
    Observation,
    Series,
    SeriesFetcher,
    SeriesMatch,
    match_series,
    observation_dated_at,
    to_source,
)
from augur.research.source import DateEvidence

from .conftest import T0
from .helpers import binary_question

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
NY = -5 * 3600  # gmtoffset of a New York listing in November


# -- the title rule --------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "query", "symbol", "target", "how"),
    [
        (
            "What will Magazine Luiza (MGLU3) close at on Friday, November 27, 2026?",
            "MGLU3",
            None,
            date(2026, 11, 27),
            "ticker",
        ),
        (
            "What will be Grupo Casas Bahia's (BHIA3) closing share price on 27 November 2026?",
            "BHIA3",
            None,
            date(2026, 11, 27),
            "ticker",
        ),
        (
            "What will the Ibovespa close at on 26 November 2026?",
            "ibovespa",
            "^BVSP",
            date(2026, 11, 26),
            "alias",
        ),
        (
            "What will the Nasdaq Composite (^IXIC) close at on Friday, November 27, 2026?",
            "^IXIC",
            "^IXIC",
            date(2026, 11, 27),
            "ticker",
        ),
        (
            "What will SpaceX's (SPCX) closing share price be on Friday, November 27, 2026?",
            "SPCX",
            None,
            date(2026, 11, 27),
            "ticker",
        ),
        (
            "What will Nvidia's (NVDA) closing share price be on Friday, November 27, 2026?",
            "NVDA",
            None,
            date(2026, 11, 27),
            "ticker",
        ),
        (
            "What will Anthropic (ANT) close at on Friday, November 27, 2026?",
            "ANT",
            None,
            date(2026, 11, 27),
            "ticker",
        ),
        (
            "What will Bitcoin (USD) close at on Nov 27, 2026?",
            "bitcoin",
            "BTC-USD",
            date(2026, 11, 27),
            "alias",
        ),
        (
            "What will the Dow Jones Industrial Average close at on 2026-11-27?",
            "dow jones industrial average",
            "^DJI",
            date(2026, 11, 27),
            "alias",
        ),
        (
            "What will EUR/USD close at on 27 Nov 2026?",
            "eur/usd",
            "EURUSD=X",
            date(2026, 11, 27),
            "alias",
        ),
    ],
)
def test_titles_the_rule_reaches(
    title: str, query: str, symbol: str | None, target: date, how: Literal["ticker", "alias"]
) -> None:
    assert match_series(title) == SeriesMatch(query, symbol, target, how)


@pytest.mark.parametrize(
    "title",
    [
        "What value will the Crypto Fear & Greed Index record for November 27, 2026?",
        "By how many percentage points will Okta (OKTA) outperform or underperform the "
        "software ETF IGV between the October 30 and November 13, 2026 market closes?",
        "What will the National Bank of Poland's EUR/PLN reference rate be on 13 November 2026?",
        "What total revenue will Nebius Group report for Q3 2026?",
        "How many drones will Russia launch at Ukraine overnight on 26–27 November 2026?",
        "What will Samolet's BO-P16 bond close at on 13 November 2026, as a percentage of par?",
        "What will Nvidia's (NVDA) % change be on November 27, 2026?",
        "Will Ether be mentioned on November 27, 2026?",
        "What will gold close at on November 26, 2026 and on November 27, 2026?",
        # Outside the registered rule: a bare "price" or "settle" is not a close,
        # and silver, natural gas and Solana are not among the registered aliases.
        "What will Bitcoin's price (USD) be on Nov 27, 2026?",
        "What will WTI crude oil futures settle at on 2026-11-27?",
        "What will silver close at on November 27, 2026?",
        "What will natural gas close at on November 27, 2026?",
        "What will Solana close at on November 27, 2026?",
    ],
)
def test_titles_the_rule_does_not_reach(title: str) -> None:
    assert match_series(title, fallback=date(2026, 11, 27)) is None


def test_the_day_falls_back_only_when_the_title_names_none() -> None:
    title = "What will the S&P 500 close at?"
    assert match_series(title) is None
    found = match_series(title, fallback=date(2026, 11, 30))
    assert found == SeriesMatch("s&p 500", "^GSPC", date(2026, 11, 30), "alias")
    named = match_series("What will the S&P 500 close at on 27 November 2026?", fallback=date.min)
    assert named is not None
    assert named.target == date(2026, 11, 27)


def test_a_ticker_wins_over_an_alias_and_the_longest_alias_wins() -> None:
    found = match_series("What will the Nasdaq Composite (^IXIC) close at on 2026-11-27?")
    assert found is not None
    assert found.how == "ticker"
    longest = match_series("What will the Nasdaq Composite close at on 2026-11-27?")
    assert longest is not None
    assert longest.symbol == "^IXIC"
    assert match_series("Whether the market will close on 2026-11-27?") is None  # not "ether"
    assert all(phrase == phrase.lower() for phrase in ALIASES)


def test_a_dated_at_is_midnight_utc_of_the_next_day() -> None:
    assert observation_dated_at(date(2026, 11, 27)) == datetime(2026, 11, 28, tzinfo=UTC)
    assert observation_dated_at(date(2026, 12, 31)) == datetime(2027, 1, 1, tzinfo=UTC)


# -- a fake chart server ---------------------------------------------------


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _stamp(day: date, offset: int = NY) -> int:
    """A daily bar's timestamp: 09:30 local time on ``day``."""
    local_open = datetime(day.year, day.month, day.day, 9, 30, tzinfo=UTC)
    return int(local_open.timestamp()) - offset


def chart_body(
    points: list[tuple[date, float | None]], *, symbol: str = "NVDA", offset: int = NY
) -> dict[str, Any]:
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD",
                        "symbol": symbol,
                        "gmtoffset": offset,
                        "longName": "NVIDIA Corporation",
                    },
                    "timestamp": [_stamp(d, offset) for d, _ in points],
                    "indicators": {"quote": [{"close": [c for _, c in points]}]},
                }
            ],
            "error": None,
        }
    }


Handler = Callable[[httpx.Request], httpx.Response]


def fetcher(handler: Handler) -> tuple[SeriesFetcher, httpx.AsyncClient]:
    t = FakeTime()
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SeriesFetcher(http, FixedClock(NOW), limiter=limiter), http


def serve(body: dict[str, Any], seen: list[httpx.Request] | None = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=body)

    return handler


D = date(2026, 11, 20)
POINTS: list[tuple[date, float | None]] = [
    (date(2026, 11, 17), 180.5),
    (date(2026, 11, 18), None),
    (date(2026, 11, 19), 182.25),
    (D, 183.0),
    (date(2026, 11, 23), 999.0),  # after every cutoff below: must never be read
]


async def test_points_after_the_cutoff_are_refused_and_null_closes_dropped() -> None:
    seen: list[httpx.Request] = []
    f, http = fetcher(serve(chart_body(POINTS), seen))
    async with http:
        series = await f.fetch("NVDA", observation_dated_at(D))
    assert [o.day for o in series.observations] == [date(2026, 11, 17), date(2026, 11, 19), D]
    assert [o.close for o in series.observations] == [180.5, 182.25, 183.0]
    assert all(o.dated_at <= series.as_of for o in series.observations)
    assert series.refused == 1
    assert series.name == "NVIDIA Corporation"
    assert series.currency == "USD"
    assert series.retrieved_at == NOW
    (request,) = seen
    assert request.url.path == "/v8/finance/chart/NVDA"
    assert request.url.params["range"] == "2y"
    assert request.url.params["interval"] == "1d"
    assert (
        series.url == "https://query1.finance.yahoo.com/v8/finance/chart/NVDA?range=2y&interval=1d"
    )


async def test_a_close_is_admitted_exactly_at_midnight_after_its_day_and_not_a_second_before() -> (
    None
):
    f, http = fetcher(serve(chart_body(POINTS)))
    async with http:
        at = await f.fetch("NVDA", observation_dated_at(D))
        before = await f.fetch("NVDA", observation_dated_at(D) - timedelta(seconds=1))
    assert at.observations[-1].day == D
    assert before.observations[-1].day == date(2026, 11, 19)
    assert before.refused == 2


async def test_the_trading_day_is_the_exchanges_own() -> None:
    # 23:30 UTC on 19 November is 08:30 on 20 November in Tokyo (UTC+9).
    tokyo = 9 * 3600
    stamp = int(datetime(2026, 11, 19, 23, 30, tzinfo=UTC).timestamp())
    body = chart_body([(D, 50000.0)], symbol="^N225", offset=tokyo)
    body["chart"]["result"][0]["timestamp"] = [stamp]
    f, http = fetcher(serve(body))
    async with http:
        series = await f.fetch("^N225", observation_dated_at(D))
    assert [o.day for o in series.observations] == [D]


async def test_nan_and_non_positive_closes_are_dropped() -> None:
    points: list[tuple[date, float | None]] = [
        (date(2026, 11, 17), math.nan),
        (date(2026, 11, 18), 0.0),
        (date(2026, 11, 19), -1.0),
        (D, 5.0),
    ]
    # Python's json writes and reads NaN, so the fake can send one.
    raw = json.dumps(chart_body(points)).encode()
    f, http = fetcher(lambda request: httpx.Response(200, content=raw))
    async with http:
        series = await f.fetch("X", NOW)
    assert [o.close for o in series.observations] == [5.0]


# -- the required one: a series value dated after the cutoff is refused ----


def _ledger_with_question(tmp: Path) -> tuple[Ledger, str, int]:
    ledger = Ledger.open(tmp / "series.sqlite", FixedClock(T0))
    run_id = ledger.start_run(
        mode=RunMode.LIVE,
        tournament="t",
        provider="replay",
        model="qwen2.5:7b",
        settings={},
        code_version="test",
    )
    q, post = binary_question(T0 - timedelta(days=30))
    ledger.record_question(run_id, q, T0, post)
    return ledger, run_id, q.question_id


async def test_a_series_value_dated_after_the_cutoff_is_refused(tmp_path: Path) -> None:
    # The question's cutoff is T0 (17 November 18:00 UTC): the close of the
    # 17th is dated 18 November 00:00 and must not reach anything.
    cutoff = T0
    late_day = date(2026, 11, 17)
    points: list[tuple[date, float | None]] = [
        (date(2026, 11, 13), 101.0),
        (date(2026, 11, 16), 102.0),
        (late_day, 777.0),
        (date(2026, 11, 18), 888.0),
    ]
    f, http = fetcher(serve(chart_body(points)))
    async with http:
        series = await f.fetch("NVDA", cutoff)
        later = await f.fetch("NVDA", NOW)

    # 1. Not in the series, nor in the source built from it.
    assert late_day not in {o.day for o in series.observations}
    assert series.refused == 2
    src = to_source(series, "NVDA")
    assert "777" not in src.excerpt
    assert "888" not in src.excerpt
    table = src.excerpt.splitlines()[1:]
    assert [line.split()[0] for line in table] == ["2026-11-13", "2026-11-16"]
    assert src.published_at == observation_dated_at(date(2026, 11, 16)) <= cutoff
    assert admit_source(src, cutoff, RunMode.LIVE).admitted

    # 2. A source fetched as of a later instant is dated after the cutoff:
    # the gate rejects it in every mode, and the store refuses a forged admission.
    late = to_source(later, "NVDA")
    assert late.published_at is not None
    assert late.published_at > cutoff
    for mode in RunMode:
        decision = admit_source(late, cutoff, mode)
        assert not decision.admitted
        assert "after the cutoff" in decision.reason

    ledger, run_id, qid = _ledger_with_question(tmp_path)
    ledger.record_source(src)
    ledger.record_admission(run_id, qid, admit_source(src, cutoff, RunMode.LIVE))
    ledger.record_source(late)
    with pytest.raises(LeakageError, match="after the question cutoff"):
        ledger.record_admission(run_id, qid, Admission(late.source_id, True, "forged"))
    ledger.record_admission(run_id, qid, admit_source(late, cutoff, RunMode.LIVE))
    ledger.close()

    # 3. The ledger scan reads each series source's own closes: the honest
    # sources pass, and one whose date its closes do not back fails.
    assert scan_ledger(tmp_path / "series.sqlite") == []
    forged = src.model_copy(
        update={"source_id": "src_forged", "published_at": observation_dated_at(late_day)}
    )
    ledger = Ledger.open(tmp_path / "series.sqlite", FixedClock(T0))
    ledger.record_source(forged)
    ledger.close()
    (violation,) = scan_ledger(tmp_path / "series.sqlite")
    assert "src_forged" in violation.detail


def test_the_source_shows_the_last_closes_and_the_dating_rule() -> None:
    days = [date(2026, 10, 1) + timedelta(days=i) for i in range(40)]
    series = Series(
        symbol="^BVSP",
        name="IBOVESPA",
        currency="BRL",
        url="https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP?range=2y&interval=1d",
        observations=tuple(
            Observation(d, 150000.0 + i + 0.25, observation_dated_at(d)) for i, d in enumerate(days)
        ),
        refused=0,
        as_of=NOW,
        retrieved_at=NOW,
    )
    src = to_source(series, "ibovespa", keep=30)
    lines = src.excerpt.splitlines()
    assert "40 closes admitted" in lines[0]
    assert "00:00 UTC of D + 1" in lines[0]
    assert len(lines) == 31
    assert lines[1] == f"{days[10].isoformat()} 150010.25"
    assert lines[-1] == f"{days[-1].isoformat()} 150039.25"
    assert src.searcher == "series"
    assert src.title == "IBOVESPA (^BVSP) daily closes"
    assert src.date_evidence is DateEvidence.PRICE_HISTORY
    assert src.published_at == observation_dated_at(days[-1])
    assert "^BVSP" in src.evidence_detail
    assert days[-1].isoformat() in src.evidence_detail
    with pytest.raises(SearchError, match="abstains"):
        to_source(Series("^BVSP", "IBOVESPA", "BRL", series.url, (), 3, NOW, NOW), "ibovespa")


# -- resolve ---------------------------------------------------------------


def _search_body(*symbols: tuple[str, str]) -> dict[str, Any]:
    return {"quotes": [{"symbol": s, "quoteType": t} for s, t in symbols], "news": []}


async def test_resolve_takes_the_ticker_with_its_exchange_suffix() -> None:
    seen: list[httpx.Request] = []
    body = _search_body(
        ("MGLU3Q.SA", "EQUITY"),
        ("MGLU3F.SA", "EQUITY"),
        ("MGLU3.SA", "EQUITY"),
        ("MGLU3.BA", "EQUITY"),
    )
    f, http = fetcher(serve(body, seen))
    m = match_series("What will Magazine Luiza (MGLU3) close at on Friday, November 27, 2026?")
    assert m is not None
    async with http:
        assert await f.resolve(m) == "MGLU3.SA"
    (request,) = seen
    assert request.url.path == "/v1/finance/search"
    assert request.url.params["q"] == "MGLU3"
    assert request.url.params["quotesCount"] == "10"
    assert request.url.params["newsCount"] == "0"


async def test_resolve_prefers_the_exact_symbol() -> None:
    body = _search_body(("NVDA.TO", "EQUITY"), ("NVDL", "ETF"), ("NVDA", "EQUITY"))
    f, http = fetcher(serve(body))
    async with http:
        assert await f.resolve(SeriesMatch("NVDA", None, D, "ticker")) == "NVDA"


async def test_resolve_finds_nothing_for_an_unknown_ticker() -> None:
    body = _search_body(("ZZZZ1Q.SA", "EQUITY"), ("ZZZZ", "OPTION"))
    f, http = fetcher(serve(body))
    async with http:
        assert await f.resolve(SeriesMatch("ZZZZ", None, D, "ticker")) is None
        empty, http2 = fetcher(serve({"quotes": []}))
        async with http2:
            assert await empty.resolve(SeriesMatch("ZZZZ1", None, D, "ticker")) is None


async def test_an_alias_or_an_index_ticker_needs_no_lookup() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    f, http = fetcher(refuse)
    async with http:
        ibov = match_series("What will the Ibovespa close at on 26 November 2026?")
        ixic = match_series("What will the Nasdaq Composite (^IXIC) close at on 2026-11-27?")
        assert ibov is not None
        assert ixic is not None
        assert await f.resolve(ibov) == "^BVSP"
        assert await f.resolve(ixic) == "^IXIC"


# -- errors and retries ----------------------------------------------------


async def test_a_chart_error_is_a_search_error() -> None:
    body = {
        "chart": {
            "result": None,
            "error": {"code": "Not Found", "description": "No data found, symbol may be delisted"},
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json=body)

    f, http = fetcher(handler)
    async with http:
        with pytest.raises(SearchError, match="symbol may be delisted"):
            await f.fetch("NOPE", NOW)


async def test_a_429_is_retried_through_the_limiter() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(200, json=chart_body(POINTS))

    f, http = fetcher(handler)
    async with http:
        series = await f.fetch("NVDA", NOW)
    assert len(calls) == 2
    assert series.observations[-1].close == 999.0  # dated 24 November 00:00, before NOW


async def test_persistent_failure_says_what_to_do() -> None:
    f, http = fetcher(lambda request: httpx.Response(503))
    async with http:
        with pytest.raises(SearchError, match="after 3 attempts"):
            await f.fetch("NVDA", NOW)
    g, http2 = fetcher(lambda request: httpx.Response(403))
    async with http2:
        with pytest.raises(SearchError, match="HTTP 403"):
            await g.fetch("NVDA", NOW)


# -- review fixes: daylight saving, aliases, coin tickers --------------------------------


async def test_a_summer_bar_fetched_in_winter_keeps_its_own_day() -> None:
    # A futures bar stamped at local midnight in July (04:00 UTC under EDT), fetched
    # when the exchange's offset is EST (-18000): read with today's offset it would be
    # labelled the 14th and dated the 15th 00:00, before that day's settle was known.
    stamp = int(datetime(2026, 7, 15, 4, 0, tzinfo=UTC).timestamp())
    body = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "GC=F",
                        "gmtoffset": -18000,
                        "exchangeTimezoneName": "America/New_York",
                    },
                    "timestamp": [stamp],
                    "indicators": {"quote": [{"close": [2400.0]}]},
                }
            ],
            "error": None,
        }
    }
    f, http = fetcher(serve(body))
    async with http:
        early = await f.fetch("GC=F", datetime(2026, 7, 15, 12, 0, tzinfo=UTC))
        later = await f.fetch("GC=F", datetime(2026, 7, 16, 0, 0, tzinfo=UTC))
    assert early.observations == ()
    assert early.refused == 1
    (obs,) = later.observations
    assert obs.day == date(2026, 7, 15)


async def test_a_bar_is_never_dated_before_the_day_after_its_utc_date() -> None:
    # Without a zone name and with a wrong offset, the UTC date still bounds the dating.
    stamp = int(datetime(2026, 7, 15, 4, 0, tzinfo=UTC).timestamp())
    body = chart_body([(date(2026, 7, 15), 1.0)])
    body["chart"]["result"][0]["timestamp"] = [stamp]
    body["chart"]["result"][0]["meta"]["gmtoffset"] = -6 * 3600
    f, http = fetcher(serve(body))
    async with http:
        series = await f.fetch("X", datetime(2026, 7, 15, 12, 0, tzinfo=UTC))
    assert series.observations == ()


def test_brent_is_not_read_as_wti_and_coin_tickers_are_coins() -> None:
    brent = match_series("What will Brent crude oil close at on December 1, 2026?")
    assert brent is not None
    assert brent.symbol == "BZ=F"
    assert match_series("What will crude oil close at on December 1, 2026?") is None
    btc = match_series("What will Bitcoin (BTC) close at on December 1, 2026?")
    assert btc is not None
    assert btc.symbol == "BTC-USD"
