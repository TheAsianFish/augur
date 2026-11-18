from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import SecretStr
from structlog.testing import capture_logs

from augur.clock import FixedClock, SystemClock
from augur.config import Settings
from augur.errors import MissingCredentialsError, SearchError
from augur.ratelimit import RateLimiter
from augur.research.asknews import AskNewsSearcher
from augur.research.source import DateEvidence

FIXTURES = Path(__file__).parent / "fixtures" / "asknews"
AS_OF = datetime(2024, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 11, 17, 18, 0, tzinfo=UTC)
CLIENT_ID = "client-id-4f2a9c1b"
CLIENT_SECRET = "very-secret-client-value-8d7e6f5a"
TOKEN_URL = "https://auth.asknews.app/oauth2/token"
SEARCH_URL = "https://api.asknews.app/v1/news/search"


def fixture_json(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


ACCESS_TOKEN = str(fixture_json("token.json")["access_token"])


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeAskNews:
    """Token endpoint and news search, recording every request."""

    def __init__(self, search: str = "search_dicts.json") -> None:
        self.search = search
        self.token_requests: list[httpx.Request] = []
        self.search_requests: list[httpx.Request] = []
        self.search_status: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            self.token_requests.append(request)
            return httpx.Response(200, json=fixture_json("token.json"))
        if str(request.url).startswith(SEARCH_URL):
            self.search_requests.append(request)
            if self.search_status:
                return httpx.Response(self.search_status.pop(0), headers={"Retry-After": "3"})
            assert request.headers["Authorization"] == f"Bearer {ACCESS_TOKEN}"
            return httpx.Response(200, json=fixture_json(self.search))
        return httpx.Response(404)


def make_searcher(
    handler: Callable[[httpx.Request], httpx.Response],
    clock: FixedClock | None = None,
    **kwargs: Any,
) -> tuple[AskNewsSearcher, httpx.AsyncClient, FakeTime]:
    t = FakeTime()
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    searcher = AskNewsSearcher(
        http,
        clock or FixedClock(NOW),
        SecretStr(CLIENT_ID),
        SecretStr(CLIENT_SECRET),
        limiter=limiter,
        **kwargs,
    )
    return searcher, http, t


async def test_token_request_is_client_credentials_with_basic_auth() -> None:
    fake = FakeAskNews()
    searcher, http, _ = make_searcher(fake)
    async with http:
        await searcher.search("Wegovy supply", AS_OF, 5)
    (req,) = fake.token_requests
    assert req.method == "POST"
    assert req.headers["Content-Type"] == "application/x-www-form-urlencoded"
    scheme, _, encoded = req.headers["Authorization"].partition(" ")
    assert scheme == "Basic"
    assert base64.b64decode(encoded).decode() == f"{CLIENT_ID}:{CLIENT_SECRET}"
    body = parse_qs(req.content.decode())
    assert body["grant_type"] == ["client_credentials"]
    assert set(body["scope"][0].split()) == {"offline", "openid", "news"}


async def test_search_params_bound_the_query_at_as_of() -> None:
    fake = FakeAskNews()
    searcher, http, _ = make_searcher(fake)
    async with http:
        await searcher.search("Wegovy supply", AS_OF, 5)
    (req,) = fake.search_requests
    p = req.url.params
    assert req.method == "GET"
    assert p["query"] == "Wegovy supply"
    assert p["n_articles"] == "5"
    assert p["end_timestamp"] == str(int(AS_OF.timestamp())) == "1704067200"
    assert p["return_type"] == "dicts"
    assert p["historical"] == "true"
    assert p["time_filter"] == "crawl_date"
    assert p["method"] == "kw"


async def test_recent_as_of_is_not_historical() -> None:
    fake = FakeAskNews()
    as_of = datetime(2024, 1, 1, 12, tzinfo=UTC)
    searcher, http, _ = make_searcher(fake, clock=FixedClock(datetime(2024, 1, 2, tzinfo=UTC)))
    async with http:
        await searcher.search("q", as_of, 5)
    assert fake.search_requests[0].url.params["historical"] == "false"


async def test_articles_become_dated_sources_and_late_ones_are_dropped() -> None:
    fake = FakeAskNews()
    searcher, http, _ = make_searcher(fake)
    with capture_logs() as logs:
        async with http:
            sources = await searcher.search("Wegovy supply", AS_OF, 5)

    assert [s.title for s in sources] == [
        "Novo Nordisk says Wegovy supply to improve in 2024",
        "The year GLP-1 drugs went mainstream",
    ]
    for src in sources:
        assert src.searcher == "asknews"
        assert src.published_at is not None
        assert src.published_at <= AS_OF
        assert src.published_at.tzinfo is UTC
        assert src.as_of == AS_OF
        assert src.retrieved_at == NOW
        assert src.date_evidence is DateEvidence.PROVIDER_PUBLISHED_BOUNDED
        assert "end_timestamp=1704067200" in src.evidence_detail
    assert sources[0].evidence_detail.startswith("pub_date=2023-12-20T14:32:00+00:00 ")
    # 18:00 at UTC-5 is 23:00 UTC, before the cutoff.
    assert sources[1].published_at == datetime(2023, 12, 31, 23, tzinfo=UTC)
    assert sources[0].url.startswith("https://www.reuters.com/")
    assert sources[0].excerpt.startswith("Novo Nordisk expects supply")

    late = [e for e in logs if e["event"] == "asknews_article_after_as_of"]
    assert len(late) == 1
    assert late[0]["log_level"] == "warning"
    assert late[0]["url"] == "https://example-news.test/2024/01/05/late-article"


async def test_excerpt_is_truncated() -> None:
    searcher, http, _ = make_searcher(FakeAskNews(), excerpt_chars=40)
    async with http:
        sources = await searcher.search("q", AS_OF, 5)
    assert all(len(s.excerpt) <= 40 for s in sources)
    assert sources[0].excerpt.endswith("…")


async def test_token_is_cached_until_shortly_before_expiry() -> None:
    fake = FakeAskNews()
    clock = FixedClock(NOW)
    searcher, http, _ = make_searcher(fake, clock=clock)
    async with http:
        await searcher.search("a", AS_OF, 5)
        clock.advance(3000)
        await searcher.search("b", AS_OF, 5)
        assert len(fake.token_requests) == 1
        clock.advance(560)  # 3560 s: inside the 60 s margin before the 3599 s expiry
        await searcher.search("c", AS_OF, 5)
    assert len(fake.token_requests) == 2
    assert len(fake.search_requests) == 3


async def test_401_refreshes_the_token_once() -> None:
    fake = FakeAskNews()
    fake.search_status = [401]
    searcher, http, _ = make_searcher(fake)
    async with http:
        sources = await searcher.search("q", AS_OF, 5)
    assert len(sources) == 2
    assert len(fake.token_requests) == 2


async def test_429_retry_after_is_honoured() -> None:
    fake = FakeAskNews()
    fake.search_status = [429]
    searcher, http, t = make_searcher(fake)
    async with http:
        sources = await searcher.search("q", AS_OF, 5)
    assert len(sources) == 2
    assert pytest.approx(3.0) in t.sleeps


async def test_missing_credentials_raise() -> None:
    settings = Settings.from_env({"ASKNEWS_CLIENT_ID": CLIENT_ID})
    async with httpx.AsyncClient() as http:
        with pytest.raises(MissingCredentialsError) as info:
            AskNewsSearcher.from_settings(settings, http, FixedClock(NOW))
    assert info.value.variables == ("ASKNEWS_CLIENT_ID", "ASKNEWS_CLIENT_SECRET")
    assert "ASKNEWS_CLIENT_SECRET" in str(info.value)


async def test_from_settings_uses_the_credentials() -> None:
    settings = Settings.from_env({"ASKNEWS_CLIENT_ID": CLIENT_ID, "ASKNEWS_SECRET": CLIENT_SECRET})
    fake = FakeAskNews()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as http:
        searcher = AskNewsSearcher.from_settings(settings, http, FixedClock(NOW))
        assert searcher.name == "asknews"
        assert CLIENT_SECRET not in repr(vars(searcher))


@pytest.mark.parametrize(
    ("token_status", "search_status"),
    [(401, None), (500, None), (None, 403), (None, 500), (None, 401)],
)
async def test_failures_never_reveal_the_secret_or_token(
    token_status: int | None, search_status: int | None
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            if token_status is not None:
                return httpx.Response(token_status, text=f"bad client {CLIENT_SECRET}")
            return httpx.Response(200, json=fixture_json("token.json"))
        assert search_status is not None
        return httpx.Response(search_status, text=f"echo {ACCESS_TOKEN}")

    searcher, http, _ = make_searcher(handler, max_retries=1)
    with capture_logs() as logs:
        async with http:
            with pytest.raises(SearchError) as info:
                await searcher.search("q", AS_OF, 5)
    texts = [str(info.value), repr(info.value), str(info.value.__cause__), repr(logs)]
    for text in texts:
        assert CLIENT_SECRET not in text
        assert ACCESS_TOKEN not in text
    assert info.value.__cause__ is None


async def test_transport_error_never_reveals_the_secret() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"refused {request.headers.get('Authorization', '')}", request=request
        )

    searcher, http, _ = make_searcher(handler)
    with capture_logs() as logs:
        async with http:
            with pytest.raises(SearchError, match="token endpoint") as info:
                await searcher.search("q", AS_OF, 5)
    encoded = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    for text in (str(info.value), repr(logs)):
        assert CLIENT_SECRET not in text
        assert encoded not in text


async def test_search_transport_error_raises_search_error() -> None:
    fake = FakeAskNews()

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return fake(request)
        raise httpx.ReadTimeout(f"timeout {request.headers['Authorization']}", request=request)

    searcher, http, _ = make_searcher(handler)
    with capture_logs() as logs:
        async with http:
            with pytest.raises(SearchError, match=r"'q'.*ReadTimeout") as info:
                await searcher.search("q", AS_OF, 5)
    for text in (str(info.value), repr(logs)):
        assert ACCESS_TOKEN not in text


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.environ.get("ASKNEWS_CLIENT_ID")
        and (os.environ.get("ASKNEWS_CLIENT_SECRET") or os.environ.get("ASKNEWS_SECRET"))
    ),
    reason="needs ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET",
)
async def test_live_asknews_sources_are_dated_at_or_before_as_of() -> None:
    settings = Settings.from_env(os.environ)
    as_of = datetime(2024, 6, 1, tzinfo=UTC)
    async with httpx.AsyncClient(timeout=60.0) as http:
        searcher = AskNewsSearcher.from_settings(settings, http, SystemClock())
        sources = await searcher.search("Federal Reserve interest rates", as_of, 5)
    for src in sources:
        assert src.published_at is not None
        assert src.published_at <= as_of
        assert src.date_evidence is DateEvidence.PROVIDER_PUBLISHED_BOUNDED
