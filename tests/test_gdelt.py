from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest

from augur.clock import FixedClock
from augur.errors import SearchError
from augur.ratelimit import RateLimiter
from augur.research.gdelt import (
    DEFAULT_DOC_URL,
    LOOKBACK,
    GdeltSearcher,
    ResponseCache,
    gdelt_query,
    parse_seendate,
    request_url,
)
from augur.research.source import DateEvidence
from augur.research.wayback import WaybackSearcher

AS_OF = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
NOW = datetime(2026, 11, 17, 18, 0, tzinfo=UTC)
HEADER = ["timestamp", "original", "statuscode", "mimetype"]
GDELT_HOST = "api.gdeltproject.org"
BACKOFF = (10.0, 30.0)
HTML = (
    "<html><head><title>Rates Story</title></head>"
    "<body><p>The central bank held rates steady in December.</p></body></html>"
)


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def article(
    url: str,
    seendate: str = "20241220T080000Z",
    title: str = "Bank holds rates",
    domain: str | None = None,
) -> dict[str, str]:
    return {
        "url": url,
        "title": title,
        "seendate": seendate,
        "domain": domain or (urlsplit(url).hostname or ""),
        "language": "English",
    }


def raw(stamp: str, original: str) -> str:
    return f"https://web.archive.org/web/{stamp}id_/{original}"


class FakeWeb:
    """GDELT's DOC API, Wayback's CDX index and captures, and the live web; records everything."""

    def __init__(
        self,
        articles: list[dict[str, str]] | None = None,
        captures: dict[str, list[str]] | None = None,
        gdelt_responses: list[httpx.Response] | None = None,
    ) -> None:
        self.articles = articles or []
        self.captures = captures or {}  # original url -> capture stamps
        self.gdelt_responses = gdelt_responses  # served in order; the last repeats
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        if host == GDELT_HOST:
            if self.gdelt_responses is not None:
                i = min(len(self.gdelt_requests()) - 1, len(self.gdelt_responses) - 1)
                return self.gdelt_responses[i]
            return httpx.Response(200, json={"articles": self.articles})
        if request.url.path == "/cdx/search/cdx":
            original = request.url.params["url"]
            stamps = self.captures.get(original, [])
            rows = [[s, original, "200", "text/html"] for s in stamps]
            return httpx.Response(200, json=[HEADER, *rows] if rows else [])
        if host == "web.archive.org":
            for original, stamps in self.captures.items():
                if any(str(request.url) == raw(s, original) for s in stamps):
                    return httpx.Response(200, text=HTML)
            return httpx.Response(404)
        return httpx.Response(200, text="<p>LIVE PAGE</p>")  # must never be reached

    def gdelt_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == GDELT_HOST]

    def wayback_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == "web.archive.org"]

    def cdx_requests(self) -> list[httpx.Request]:
        return [r for r in self.wayback_requests() if r.url.path == "/cdx/search/cdx"]

    def other_hosts(self) -> list[str]:
        return [
            r.url.host for r in self.requests if r.url.host not in (GDELT_HOST, "web.archive.org")
        ]


def make(
    web: Callable[[httpx.Request], httpx.Response],
    *,
    cache: ResponseCache | None = None,
    **kwargs: Any,
) -> tuple[GdeltSearcher, httpx.AsyncClient, FakeTime]:
    t = FakeTime()
    http = httpx.AsyncClient(transport=httpx.MockTransport(web))
    clock = FixedClock(NOW)
    wb_limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    wayback = WaybackSearcher(http, clock, limiter=wb_limiter)
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    kwargs.setdefault("backoff", BACKOFF)
    searcher = GdeltSearcher(http, clock, wayback, cache=cache, limiter=limiter, **kwargs)
    return searcher, http, t


GOOD = "https://news.example.com/story"
QUERY = "central bank rates December"


async def test_article_seen_after_cutoff_is_refused() -> None:
    late = "https://late.example.com/story"
    web = FakeWeb(
        [article(late, seendate="20250101T120001Z")],
        {late: ["20241231000000"]},
    )
    searcher, http, _ = make(web)
    async with http:
        sources = await searcher.search(QUERY, AS_OF, 5)
    assert sources == []
    assert len(web.gdelt_requests()) == 1  # GDELT was asked, and its answer not trusted
    assert web.wayback_requests() == []
    assert web.other_hosts() == []


async def test_article_seen_exactly_at_cutoff_is_admitted() -> None:
    web = FakeWeb([article(GOOD, seendate="20250101T120000Z")], {GOOD: ["20241231000000"]})
    searcher, http, _ = make(web)
    async with http:
        sources = await searcher.search(QUERY, AS_OF, 5)
    assert len(sources) == 1


async def test_article_captured_after_cutoff_is_refused() -> None:
    web = FakeWeb([article(GOOD)], {GOOD: ["20250101120001", "20250301000000"]})
    searcher, http, _ = make(web)
    async with http:
        sources = await searcher.search(QUERY, AS_OF, 5)
    assert sources == []
    assert len(web.cdx_requests()) == 1
    # Neither the later captures nor the live article were ever fetched.
    assert [r for r in web.wayback_requests() if r not in web.cdx_requests()] == []
    assert web.other_hosts() == []
    assert all(r.url.host != "news.example.com" for r in web.requests)


async def test_article_without_any_capture_is_refused() -> None:
    web = FakeWeb([article(GOOD)], {})
    searcher, http, _ = make(web)
    async with http:
        assert await searcher.search(QUERY, AS_OF, 5) == []
    assert web.other_hosts() == []


async def test_good_article_yields_one_dated_source() -> None:
    web = FakeWeb(
        [article(GOOD, seendate="20241220T080000Z", title="Bank holds rates")],
        {GOOD: ["20241230093000", "20250102000000"]},
    )
    searcher, http, _ = make(web)
    async with http:
        sources = await searcher.search(QUERY, AS_OF, 5)
    assert len(sources) == 1
    src = sources[0]
    assert src.searcher == "gdelt"
    assert src.date_evidence is DateEvidence.ARCHIVE_CAPTURE
    assert src.published_at == datetime(2024, 12, 30, 9, 30, tzinfo=UTC)
    assert src.published_at is not None
    assert src.published_at <= AS_OF
    assert src.url == f"https://web.archive.org/web/20241230093000/{GOOD}"
    assert "seendate=20241220T080000Z" in src.evidence_detail
    assert "capture=20241230093000" in src.evidence_detail
    assert "central bank held rates steady" in src.excerpt
    assert src.title == "Rates Story"  # the capture's own title, never GDELT's
    assert src.query == QUERY
    assert src.as_of == AS_OF
    assert src.retrieved_at == NOW
    assert web.other_hosts() == []


async def test_request_parameters() -> None:
    web = FakeWeb([])
    searcher, http, _ = make(web)
    async with http:
        assert await searcher.search("Will the Fed cut rates? (December 2024)", AS_OF, 3) == []
    (req,) = web.gdelt_requests()
    params = req.url.params
    assert f"{req.url.scheme}://{req.url.host}{req.url.path}" == DEFAULT_DOC_URL
    assert params["enddatetime"] == "20250101120000"
    assert params["startdatetime"] == (AS_OF - timedelta(days=90)).strftime("%Y%m%d%H%M%S")
    assert timedelta(days=90) == LOOKBACK
    assert params["mode"] == "artlist"
    assert params["format"] == "json"
    assert params["query"] == "Fed cut rates December 2024 sourcelang:english"
    assert str(req.url) == request_url(DEFAULT_DOC_URL, params["query"], AS_OF)


def test_gdelt_query_normalises() -> None:
    q = gdelt_query("alpha, beta; gamma! delta? epsilon zeta eta theta")
    assert q == "alpha beta gamma delta epsilon zeta sourcelang:english"
    assert gdelt_query("The Fed is up to it") == "Fed sourcelang:english"
    assert gdelt_query("Rates rates RATES") == "Rates sourcelang:english"


async def test_all_stopword_query_sends_no_request() -> None:
    assert gdelt_query("the and for what is it") is None
    web = FakeWeb([article(GOOD)])
    searcher, http, _ = make(web)
    async with http:
        assert await searcher.search("the and for what is it", AS_OF, 5) == []
        assert await searcher.search(QUERY, AS_OF, 0) == []
    assert web.requests == []


async def test_second_identical_search_uses_cache(tmp_path: Path) -> None:
    web = FakeWeb([article(GOOD)], {GOOD: ["20241230093000"]})
    searcher, http, _ = make(web, cache=ResponseCache(tmp_path / "cache"))
    async with http:
        first = await searcher.search(QUERY, AS_OF, 5)
        second = await searcher.search(QUERY, AS_OF, 5)
    assert len(first) == len(second) == 1
    assert len(web.gdelt_requests()) == 1
    assert len(web.cdx_requests()) == 2  # Wayback is not cached here


async def test_cache_file_for_another_request_raises(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    url = request_url(DEFAULT_DOC_URL, gdelt_query(QUERY) or "", AS_OF)
    cache.put(url, "{}", NOW)
    (path,) = tmp_path.glob("*.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["url"] = url + "&other=1"
    path.write_text(json.dumps(record), encoding="utf-8")
    web = FakeWeb([])
    searcher, http, _ = make(web, cache=cache)
    async with http:
        with pytest.raises(SearchError, match="another request"):
            await searcher.search(QUERY, AS_OF, 5)
    assert web.requests == []


async def test_429_then_200_is_retried_with_backoff() -> None:
    ok = httpx.Response(200, json={"articles": [article(GOOD)]})
    web = FakeWeb(
        captures={GOOD: ["20241230093000"]},
        gdelt_responses=[httpx.Response(429), httpx.Response(429), ok],
    )
    searcher, http, t = make(web)
    async with http:
        sources = await searcher.search(QUERY, AS_OF, 5)
    assert len(sources) == 1
    assert len(web.gdelt_requests()) == 3
    assert t.sleeps == list(BACKOFF)


async def test_persistent_429_raises_after_all_attempts() -> None:
    web = FakeWeb(gdelt_responses=[httpx.Response(429)])
    searcher, http, t = make(web)
    async with http:
        with pytest.raises(SearchError, match="HTTP 429"):
            await searcher.search(QUERY, AS_OF, 5)
    assert len(web.gdelt_requests()) == len(BACKOFF) + 1
    assert t.sleeps == list(BACKOFF)


async def test_400_is_not_retried() -> None:
    web = FakeWeb(gdelt_responses=[httpx.Response(400, text="bad")])
    searcher, http, t = make(web)
    async with http:
        with pytest.raises(SearchError, match="HTTP 400"):
            await searcher.search(QUERY, AS_OF, 5)
    assert len(web.gdelt_requests()) == 1
    assert t.sleeps == []


async def test_non_json_200_is_empty_not_an_error() -> None:
    web = FakeWeb(gdelt_responses=[httpx.Response(200, text="The specified phrase is too short.")])
    searcher, http, _ = make(web)
    async with http:
        assert await searcher.search(QUERY, AS_OF, 5) == []
    assert web.wayback_requests() == []


async def test_market_hosts_are_excluded() -> None:
    market = "https://polymarket.com/event/rates"
    web = FakeWeb([article(market)], {market: ["20241230093000"]})
    searcher, http, _ = make(web)
    async with http:
        assert await searcher.search(QUERY, AS_OF, 5) == []
    assert web.wayback_requests() == []


async def test_limit_is_respected() -> None:
    urls = [f"https://site{i}.example.com/a" for i in range(4)]
    web = FakeWeb([article(u) for u in urls], {u: ["20241230093000"] for u in urls})
    searcher, http, _ = make(web)
    async with http:
        sources = await searcher.search(QUERY, AS_OF, 2)
    assert len(sources) == 2
    assert len(web.cdx_requests()) == 2


async def test_at_most_max_lookups_wayback_lookups() -> None:
    urls = [f"https://site{i}.example.com/a" for i in range(6)]
    web = FakeWeb([article(u) for u in urls], {})  # no captures anywhere
    searcher, http, _ = make(web, max_lookups=3)
    async with http:
        assert await searcher.search(QUERY, AS_OF, 10) == []
    assert len(web.cdx_requests()) == 3


async def test_malformed_seendate_and_url_are_skipped() -> None:
    web = FakeWeb(
        [
            article(GOOD, seendate="yesterday"),
            article("ftp://x.example.com/a"),
            article(""),
        ],
        {GOOD: ["20241230093000"]},
    )
    searcher, http, _ = make(web)
    async with http:
        assert await searcher.search(QUERY, AS_OF, 5) == []
    assert web.wayback_requests() == []


def test_parse_seendate() -> None:
    assert parse_seendate("20250101T120000Z") == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    assert parse_seendate(" 20241231T235959Z ") == datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)
    for bad in ["", "2025-01-01T12:00:00Z", "20250101120000", "20251301T120000Z", "abc"]:
        assert parse_seendate(bad) is None


async def test_a_feed_length_gdelt_title_never_reaches_the_source() -> None:
    web = FakeWeb([article(GOOD, title="word " * 5000)], {GOOD: ["20241230093000"]})
    searcher, http, _ = make(web)
    async with http:
        [src] = await searcher.search(QUERY, AS_OF, 5)
    assert src.title == "Rates Story"
    assert "word word" not in src.title + src.excerpt


async def test_an_archive_failure_is_an_error_not_an_empty_answer() -> None:
    def web(request: httpx.Request) -> httpx.Response:
        if request.url.host == GDELT_HOST:
            return httpx.Response(200, json={"articles": [article(GOOD)]})
        return httpx.Response(503)  # the archive is down for every retry

    searcher, http, _ = make(web)
    async with http:
        with pytest.raises(SearchError, match="Wayback lookup"):
            await searcher.search(QUERY, AS_OF, 5)


async def test_no_capture_at_all_is_still_an_empty_answer() -> None:
    web = FakeWeb([article(GOOD)], {})
    searcher, http, _ = make(web)
    async with http:
        assert await searcher.search(QUERY, AS_OF, 5) == []
