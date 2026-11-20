from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from augur.clock import FixedClock
from augur.errors import SearchError
from augur.questions import Question
from augur.ratelimit import RateLimiter
from augur.research.source import DateEvidence, QuestionSearcher
from augur.research.wayback import WaybackSearcher, extract_urls
from augur.research.wikipedia import DEFAULT_RETRY_AFTER, USER_AGENT
from tests.helpers import binary_question

AS_OF = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
NOW = datetime(2026, 11, 17, 18, 0, tzinfo=UTC)
PAGE = "https://example.org/report"
HEADER = ["timestamp", "original", "statuscode", "mimetype"]
HTML = (
    "<html><head><title>The  Report</title><style>p{}</style></head>"
    "<body><script>var x=1;</script><p>Inflation was <b>3.1%</b> in December.</p></body></html>"
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


class FakeArchive:
    """Serves a CDX index and raw captures; records every request."""

    def __init__(
        self,
        rows: list[list[str]] | None,
        pages: dict[str, httpx.Response] | None = None,
    ) -> None:
        self.rows = rows
        self.pages = pages or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers["User-Agent"] == USER_AGENT
        if request.url.path == "/cdx/search/cdx":
            if self.rows is None:
                return httpx.Response(200, text="")
            return httpx.Response(200, json=[HEADER, *self.rows] if self.rows else [])
        key = str(request.url)
        return self.pages.get(key, httpx.Response(404))

    def fetched(self) -> list[str]:
        return [str(r.url) for r in self.requests if r.url.path != "/cdx/search/cdx"]


def row(stamp: str, mime: str = "text/html", original: str = PAGE) -> list[str]:
    return [stamp, original, "200", mime]


def raw(stamp: str, original: str = PAGE) -> str:
    return f"https://web.archive.org/web/{stamp}id_/{original}"


def make_searcher(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[WaybackSearcher, httpx.AsyncClient, FakeTime]:
    t = FakeTime()
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WaybackSearcher(http, FixedClock(NOW), limiter=limiter, **kwargs), http, t


def html_ok(text: str = HTML) -> httpx.Response:
    return httpx.Response(200, text=text)


async def test_newest_capture_at_or_before_as_of_is_chosen_and_dated() -> None:
    archive = FakeArchive(
        [row("20241201000000"), row("20241231080000"), row("20241101000000")],
        {raw("20241231080000"): html_ok()},
    )
    searcher, http, _ = make_searcher(archive)
    async with http:
        src = await searcher.capture(PAGE, AS_OF, "q")

    cdx = archive.requests[0].url.params
    assert cdx["url"] == PAGE
    assert cdx["to"] == "20250101120000"
    assert cdx["output"] == "json"
    assert cdx["filter"] == "statuscode:200"
    assert cdx["limit"] == "-5"
    assert cdx["fl"] == "timestamp,original,statuscode,mimetype"
    assert archive.fetched() == [raw("20241231080000")]
    assert src is not None
    assert src.searcher == "wayback"
    assert src.url == f"https://web.archive.org/web/20241231080000/{PAGE}"
    assert src.published_at == datetime(2024, 12, 31, 8, 0, 0, tzinfo=UTC)
    assert src.date_evidence is DateEvidence.ARCHIVE_CAPTURE
    assert src.evidence_detail == f"capture=20241231080000 original={PAGE}"
    assert src.as_of == AS_OF
    assert src.retrieved_at == NOW
    assert src.title == "The Report"


async def test_capture_later_the_same_day_is_rejected() -> None:
    # The CDX index returned 23:44 for a 12:00 bound: it must never be read.
    archive = FakeArchive(
        [row("20241231000000"), row("20250101234441")],
        {raw("20241231000000"): html_ok(), raw("20250101234441"): html_ok("<p>Leaked later.</p>")},
    )
    searcher, http, _ = make_searcher(archive)
    async with http:
        src = await searcher.capture(PAGE, AS_OF, "q")
    assert src is not None
    assert src.published_at == datetime(2024, 12, 31, tzinfo=UTC)
    assert archive.fetched() == [raw("20241231000000")]
    assert "Leaked" not in src.excerpt


async def test_only_later_captures_means_none() -> None:
    archive = FakeArchive([row("20250101234441")], {raw("20250101234441"): html_ok()})
    searcher, http, _ = make_searcher(archive)
    async with http:
        assert await searcher.capture(PAGE, AS_OF, "q") is None
    assert archive.fetched() == []


async def test_exact_equality_is_admitted() -> None:
    archive = FakeArchive([row("20250101120000")], {raw("20250101120000"): html_ok()})
    searcher, http, _ = make_searcher(archive)
    async with http:
        src = await searcher.capture(PAGE, AS_OF, "q")
    assert src is not None
    assert src.published_at == AS_OF


@pytest.mark.parametrize("rows", [None, []])
async def test_no_captures_is_none(rows: list[list[str]] | None) -> None:
    archive = FakeArchive(rows)
    searcher, http, _ = make_searcher(archive)
    async with http:
        assert await searcher.capture(PAGE, AS_OF, "q") is None
    assert archive.fetched() == []


async def test_non_html_skipped_but_plain_text_kept() -> None:
    archive = FakeArchive(
        [row("20241230000000", "text/plain"), row("20241231000000", "application/pdf")],
        {raw("20241230000000"): httpx.Response(200, text="Plain words here.")},
    )
    searcher, http, _ = make_searcher(archive)
    async with http:
        src = await searcher.capture(PAGE, AS_OF, "q")
    assert src is not None
    assert src.excerpt == "Plain words here."
    assert src.title == PAGE


async def test_redirect_to_later_timestamp_is_dropped() -> None:
    later = "https://web.archive.org/web/20250102000000id_/" + PAGE
    archive = FakeArchive(
        [row("20241231000000")],
        {raw("20241231000000"): httpx.Response(302, headers={"location": later})},
    )
    searcher, http, _ = make_searcher(archive)
    async with http:
        assert await searcher.capture(PAGE, AS_OF, "q") is None
    assert archive.fetched() == [raw("20241231000000")]


async def test_redirect_to_earlier_timestamp_is_followed() -> None:
    earlier = "/web/20241220000000id_/" + PAGE
    archive = FakeArchive(
        [row("20241231000000")],
        {
            raw("20241231000000"): httpx.Response(302, headers={"location": earlier}),
            raw("20241220000000"): html_ok(),
        },
    )
    searcher, http, _ = make_searcher(archive)
    async with http:
        src = await searcher.capture(PAGE, AS_OF, "q")
    assert src is not None
    assert archive.fetched() == [raw("20241231000000"), raw("20241220000000")]


async def test_excerpt_is_text_without_markup_scripts_or_styles() -> None:
    archive = FakeArchive([row("20241231000000")], {raw("20241231000000"): html_ok()})
    searcher, http, _ = make_searcher(archive)
    async with http:
        src = await searcher.capture(PAGE, AS_OF, "q")
    assert src is not None
    assert "Inflation was 3.1% in December." in src.excerpt
    assert not any(bad in src.excerpt for bad in ("<", "var x", "p{}"))


async def test_excerpt_is_truncated() -> None:
    body = "<p>" + "word " * 200 + "</p>"
    archive = FakeArchive([row("20241231000000")], {raw("20241231000000"): html_ok(body)})
    searcher, http, _ = make_searcher(archive, excerpt_chars=50)
    async with http:
        src = await searcher.capture(PAGE, AS_OF, "q")
    assert src is not None
    assert len(src.excerpt) <= 50
    assert src.excerpt.endswith("…")


async def test_empty_capture_text_is_none() -> None:
    archive = FakeArchive(
        [row("20241231000000")], {raw("20241231000000"): html_ok("<script>x</script>")}
    )
    searcher, http, _ = make_searcher(archive)
    async with http:
        assert await searcher.capture(PAGE, AS_OF, "q") is None


async def test_429_is_retried_after_retry_after() -> None:
    calls = {"n": 0}
    archive = FakeArchive([row("20241231000000")], {raw("20241231000000"): html_ok()})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cdx/search/cdx":
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "7"})
        return archive(request)

    searcher, http, t = make_searcher(handler)
    async with http:
        src = await searcher.capture(PAGE, AS_OF, "q")
    assert src is not None
    assert calls["n"] == 2
    assert any(s >= 7 for s in t.sleeps)
    assert DEFAULT_RETRY_AFTER != 7


async def test_persistent_failure_raises_search_error() -> None:
    searcher, http, _ = make_searcher(lambda r: httpx.Response(503))
    async with http:
        with pytest.raises(SearchError, match="retry later"):
            await searcher.capture(PAGE, AS_OF, "q")


async def test_transport_error_is_retried() -> None:
    archive = FakeArchive([row("20241231000000")], {raw("20241231000000"): html_ok()})
    state = {"failed": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["failed"]:
            state["failed"] = True
            raise httpx.ConnectTimeout("slow archive")
        return archive(request)

    searcher, http, _ = make_searcher(handler)
    async with http:
        assert await searcher.capture(PAGE, AS_OF, "q") is not None


async def test_search_with_url_query_returns_capture() -> None:
    archive = FakeArchive([row("20241231000000")], {raw("20241231000000"): html_ok()})
    searcher, http, _ = make_searcher(archive)
    async with http:
        sources = await searcher.search(PAGE, AS_OF, 3)
    assert [s.query for s in sources] == [PAGE]


async def test_search_with_non_url_query_returns_nothing() -> None:
    archive = FakeArchive([row("20241231000000")])
    searcher, http, _ = make_searcher(archive)
    async with http:
        assert await searcher.search("inflation in december", AS_OF, 3) == []
    assert archive.requests == []


def question_with(**text: str) -> Question:
    q, _ = binary_question(datetime(2024, 12, 1, tzinfo=UTC))
    return q.model_copy(update=text)


async def test_search_question_excludes_market_hosts_and_the_question_url() -> None:
    other = "https://data.gov/series/cpi"
    pages = {
        raw("20241231000000", other): html_ok(),
        raw("20241231000000", PAGE): html_ok(),
    }

    class Multi(FakeArchive):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/cdx/search/cdx":
                self.requests.append(request)
                url = request.url.params["url"]
                return httpx.Response(200, json=[HEADER, row("20241231000000", original=url)])
            return super().__call__(request)

    archive = Multi(None, pages)
    q = question_with(
        description=(
            "See https://polymarket.com/event/x and https://www.metaculus.com/questions/1/ "
            f"and https://sub.manifold.markets/a, plus {PAGE}."
        ),
        resolution_criteria=f"Uses ({other}) and https://kalshi.com/m again {PAGE}",
        fine_print="https://www.goodjudgment.io/x https://infer-pub.com/y",
    )
    searcher, http, _ = make_searcher(archive)
    async with http:
        sources = await searcher.search_question(q, AS_OF, 5)

    assert [s.url.split("/", 5)[5] for s in sources] == [PAGE, other]
    asked = [r.url.params["url"] for r in archive.requests if r.url.path == "/cdx/search/cdx"]
    assert asked == [PAGE, other]
    assert q.url not in asked


async def test_search_question_respects_limit_and_skips_failures() -> None:
    a, b, c = "https://a.example/1", "https://b.example/2", "https://c.example/3"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cdx/search/cdx":
            url = request.url.params["url"]
            if url == a:
                return httpx.Response(400)
            return httpx.Response(200, json=[HEADER, row("20241231000000", original=url)])
        return html_ok()

    q = question_with(description=f"{a} {b} {c}")
    searcher, http, _ = make_searcher(handler)
    async with http:
        sources = await searcher.search_question(q, AS_OF, 1)
    assert [s.url.endswith(b) for s in sources] == [True]


def test_is_a_question_searcher() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    assert isinstance(WaybackSearcher(http, FixedClock(NOW)), QuestionSearcher)


def test_extract_urls_strips_punctuation_and_dedupes() -> None:
    text = "Go to https://a.example/x, then (https://b.example/y). Also https://a.example/x."
    assert extract_urls(text) == ["https://a.example/x", "https://b.example/y"]
