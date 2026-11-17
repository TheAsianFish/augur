from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from augur.clock import FixedClock, SystemClock
from augur.errors import SearchError
from augur.ratelimit import RateLimiter
from augur.research.source import DateEvidence
from augur.research.text import html_to_text, truncate_words
from augur.research.wikipedia import USER_AGENT, WikipediaSearcher, mediawiki_timestamp

FIXTURES = Path(__file__).parent / "fixtures" / "wikipedia"
AS_OF = datetime(2024, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 11, 17, 18, 0, tzinfo=UTC)


def fixture_json(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


def parse_reply(html_name: str, revid: int, title: str) -> dict[str, Any]:
    html = (FIXTURES / html_name).read_text(encoding="utf-8")
    return {"parse": {"title": title, "pageid": 1, "revid": revid, "text": html}}


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeWiki:
    """Routes MediaWiki API requests to fixtures and records every request."""

    def __init__(
        self,
        search: str = "search_glp1.json",
        revisions: dict[str, str] | None = None,
        parses: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self.search = search
        self.revisions = revisions or {
            "Semaglutide": "revisions_semaglutide.json",
            "Orforglipron": "revisions_none_before.json",
            "Novo Nordisk": "revisions_late.json",
        }
        self.parses = parses or {"1192893568": ("parse_1192893568.html", "Semaglutide")}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        p = request.url.params
        assert p["format"] == "json"
        assert p["formatversion"] == "2"
        assert p["maxlag"] == "5"
        assert request.headers["User-Agent"] == USER_AGENT
        if p["action"] == "query" and p.get("list") == "search":
            return httpx.Response(200, json=fixture_json(self.search))
        if p["action"] == "query" and p.get("prop") == "revisions":
            name = self.revisions.get(p["titles"], "revisions_missing.json")
            return httpx.Response(200, json=fixture_json(name))
        if p["action"] == "parse":
            html_name, title = self.parses[p["oldid"]]
            return httpx.Response(200, json=parse_reply(html_name, int(p["oldid"]), title))
        return httpx.Response(400, json={"error": {"code": "badvalue"}})

    def of(self, kind: str) -> list[httpx.Request]:
        def match(r: httpx.Request) -> bool:
            p = r.url.params
            return kind in (p.get("action"), p.get("list"), p.get("prop"))

        return [r for r in self.requests if match(r)]


def fast_limiter() -> tuple[RateLimiter, FakeTime]:
    t = FakeTime()
    return RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep), t


def make_searcher(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[WikipediaSearcher, httpx.AsyncClient, FakeTime]:
    limiter, t = fast_limiter()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WikipediaSearcher(http, FixedClock(NOW), limiter=limiter, **kwargs), http, t


async def test_reads_the_revision_current_at_as_of() -> None:
    wiki = FakeWiki()
    searcher, http, _ = make_searcher(wiki)
    async with http:
        sources = await searcher.search("GLP-1 Ozempic", AS_OF, 3)

    search = wiki.of("search")[0].url.params
    assert search["srsearch"] == "GLP-1 Ozempic"
    assert search["srlimit"] == "3"
    revision_requests = wiki.of("revisions")
    assert len(revision_requests) == 3
    for r in revision_requests:
        assert r.url.params["rvstart"] == "2024-01-01T00:00:00Z"
        assert r.url.params["rvdir"] == "older"
        assert r.url.params["rvlimit"] == "1"
        assert r.url.params["rvprop"] == "ids|timestamp"
    parse = wiki.of("parse")
    assert [r.url.params["oldid"] for r in parse] == ["1192893568"]
    assert parse[0].url.params["prop"] == "text"

    # Orforglipron has no revision before as_of; Novo Nordisk's response is dated after it.
    assert [s.title for s in sources] == ["Semaglutide"]
    src = sources[0]
    assert src.searcher == "wikipedia"
    assert src.url == "https://en.wikipedia.org/w/index.php?oldid=1192893568"
    assert src.published_at == datetime(2023, 12, 31, 23, 7, 38, tzinfo=UTC)
    assert src.published_at <= AS_OF
    assert src.as_of == AS_OF
    assert src.retrieved_at == NOW
    assert src.date_evidence is DateEvidence.REVISION_TIMESTAMP
    assert src.evidence_detail == "revid=1192893568 timestamp=2023-12-31T23:07:38Z"


async def test_excerpt_is_prose_without_infobox_references_or_back_matter() -> None:
    searcher, http, _ = make_searcher(FakeWiki())
    async with http:
        (src,) = await searcher.search("GLP-1 Ozempic", AS_OF, 3)
    text = src.excerpt
    assert text.startswith("Semaglutide is an antidiabetic medication used for the treatment")
    assert "type 2 diabetes" in text
    assert "Medical uses" in text
    assert "Oral semaglutide is marketed in three strengths." in text
    assert "kg/m2" in text
    for absent in (
        "Trade names",  # infobox
        "Molar mass",  # infobox
        "[17]",  # reference marker
        "citation needed",
        "redirects here",  # hatnote
        "Anti-diabetic and anti-obesity medication",  # hidden short description
        "mw-parser-output",  # style sheet text
        "Ozempic injection pen",  # figure caption
        "Liraglutide",  # after "See also"
        "Journal of Investigative Medicine",  # references list
        "Tirzepatide",  # navbox
    ):
        assert absent not in text, absent
    assert "  " not in text


async def test_redirect_at_as_of_is_dropped() -> None:
    wiki = FakeWiki(
        revisions={"Semaglutide": "revisions_semaglutide.json"},
        parses={"1192893568": ("parse_redirect.html", "Semaglutide")},
    )
    searcher, http, _ = make_searcher(wiki)
    async with http:
        assert await searcher.search("Ozempic", AS_OF, 3) == []


async def test_excerpt_truncated_at_word_boundary() -> None:
    searcher, http, _ = make_searcher(FakeWiki(), excerpt_chars=60)
    async with http:
        (src,) = await searcher.search("GLP-1 Ozempic", AS_OF, 3)
    assert len(src.excerpt) <= 60
    assert src.excerpt.endswith("…")
    assert src.excerpt[:-1] == src.excerpt[:-1].rstrip()
    assert "Semaglutide is an antidiabetic medication used for the".startswith(src.excerpt[:-1])


async def test_rvstart_is_as_of_in_utc() -> None:
    wiki = FakeWiki()
    searcher, http, _ = make_searcher(wiki)
    as_of = datetime(2024, 1, 1, 5, 30, 15, 999, tzinfo=UTC)
    async with http:
        await searcher.search("x", as_of, 3)
    assert {r.url.params["rvstart"] for r in wiki.of("revisions")} == {"2024-01-01T05:30:15Z"}
    assert mediawiki_timestamp(datetime(2024, 1, 1, 1, tzinfo=UTC)) == "2024-01-01T01:00:00Z"


async def test_naive_as_of_is_refused() -> None:
    searcher, http, _ = make_searcher(FakeWiki())
    async with http:
        with pytest.raises(ValueError, match="naive"):
            await searcher.search("x", datetime(2024, 1, 1), 3)  # noqa: DTZ001


async def test_429_retry_after_is_honoured() -> None:
    wiki = FakeWiki()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return wiki(request)

    searcher, http, t = make_searcher(handler)
    async with http:
        sources = await searcher.search("GLP-1 Ozempic", AS_OF, 3)
    assert [s.title for s in sources] == ["Semaglutide"]
    assert len(wiki.of("search")) == 1
    assert pytest.approx(7.0) in t.sleeps
    assert t.now >= 7.0


async def test_maxlag_error_is_retried() -> None:
    wiki = FakeWiki()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                headers={"Retry-After": "5"},
                json={"error": {"code": "maxlag", "info": "Waiting for a database server"}},
            )
        return wiki(request)

    searcher, http, t = make_searcher(handler)
    async with http:
        sources = await searcher.search("GLP-1 Ozempic", AS_OF, 3)
    assert len(sources) == 1
    assert pytest.approx(5.0) in t.sleeps


async def test_persistent_429_raises_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"})

    searcher, http, _ = make_searcher(handler, max_retries=2)
    async with http:
        with pytest.raises(SearchError, match=r"'q'.*3 attempts.*HTTP 429"):
            await searcher.search("q", AS_OF, 3)


async def test_transport_error_raises_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    searcher, http, _ = make_searcher(handler)
    async with http:
        with pytest.raises(SearchError, match=r"'GLP-1'.*ConnectError.*retry later"):
            await searcher.search("GLP-1", AS_OF, 3)


async def test_non_json_body_raises_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    searcher, http, _ = make_searcher(handler)
    async with http:
        with pytest.raises(SearchError, match="not JSON"):
            await searcher.search("GLP-1", AS_OF, 3)


async def test_parse_api_error_drops_only_that_page() -> None:
    wiki = FakeWiki()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("action") == "parse":
            return httpx.Response(200, json={"error": {"code": "nosuchrevid"}})
        return wiki(request)

    searcher, http, _ = make_searcher(handler)
    async with http:
        assert await searcher.search("GLP-1", AS_OF, 3) == []


def test_html_to_text_stops_at_back_matter_and_keeps_lists() -> None:
    html = (
        "<p>Alpha <b>beta</b>.</p><ul><li>One</li><li>Two</li></ul>"
        '<h2><span class="mw-headline" id="External_links">External links</span></h2>'
        "<p>Gamma</p>"
    )
    assert html_to_text(html) == "Alpha beta.\nOne\nTwo"


def test_truncate_words() -> None:
    assert truncate_words("short", 10) == "short"
    assert truncate_words("one two three four", 10) == "one two…"
    assert len(truncate_words("x" * 50, 10)) == 10


@pytest.mark.integration
async def test_live_wikipedia_sources_are_dated_at_or_before_as_of() -> None:
    as_of = datetime(2024, 1, 1, tzinfo=UTC)
    async with httpx.AsyncClient(timeout=30.0) as http:
        searcher = WikipediaSearcher(http, SystemClock())
        sources = await searcher.search("semaglutide", as_of, 3)
    assert sources, "expected at least one page that existed on 2024-01-01"
    for src in sources:
        assert src.published_at is not None
        assert src.published_at <= as_of
        assert src.date_evidence is DateEvidence.REVISION_TIMESTAMP
        assert "oldid=" in src.url
