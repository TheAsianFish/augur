from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import SecretStr
from structlog.testing import capture_logs

from augur.clock import FixedClock
from augur.config import Settings
from augur.errors import MetaculusError, MissingCredentialsError
from augur.metaculus import USER_AGENT, MetaculusClient
from augur.ratelimit import RateLimiter
from tests.fakes.metaculus import (
    BINARY_DETAIL,
    BOT_TESTING_NUMERIC,
    FAKE_TOKEN,
    MINIBENCH_NUMERIC,
    FakeMetaculus,
    binary_post,
    group_post,
)

BASE = "https://www.metaculus.com/api"
TOKEN = FAKE_TOKEN
NOW = datetime(2026, 11, 17, 20, 0, tzinfo=UTC)


class FakeTime:
    """A monotonic clock that only moves when the limiter sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake() -> FakeMetaculus:
    server = FakeMetaculus(token=TOKEN, clock=FixedClock(NOW))
    server.add_fixture(BINARY_DETAIL)
    server.add_fixture(MINIBENCH_NUMERIC)
    server.add_fixture(BOT_TESTING_NUMERIC)
    return server


@pytest.fixture
def time() -> FakeTime:
    return FakeTime()


@pytest.fixture
async def http(fake: FakeMetaculus) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=fake.transport) as client:
        yield client


def make_client(
    http: httpx.AsyncClient,
    time: FakeTime,
    token: str | None = TOKEN,
    **kwargs: int,
) -> MetaculusClient:
    limiter = RateLimiter(
        rate_per_second=1.0, burst=2, max_concurrent=2, monotonic=time.monotonic, sleep=time.sleep
    )
    return MetaculusClient(
        BASE, SecretStr(token) if token is not None else None, http, limiter, **kwargs
    )


async def test_listing_paginates_and_strips_my_forecasts(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    for i in range(4):
        fake.add_post(binary_post(90_000 + i, 91_000 + i), tournament="minibench")
    fake.add_post(binary_post(90_100, 91_100, status="closed"), tournament="minibench")
    client = make_client(http, time, page_size=2)

    found = await client.list_open_questions("minibench")

    assert [q.post_id for q, _ in found] == [46616, 90000, 90001, 90002, 90003]
    assert all(q.status == "open" for q, _ in found)
    assert all("my_forecasts" not in post["question"] for _, post in found)
    offsets = [r.url.params["offset"] for r in fake.requests]
    assert offsets == ["0", "2", "4"]
    first = fake.requests[0]
    assert first.url.path == "/api/posts/"
    assert first.url.params["tournaments"] == "minibench"
    assert first.url.params["statuses"] == "open"
    assert first.url.params["forecast_type"] == "binary,multiple_choice,numeric,discrete"
    assert first.url.params["include_description"] == "true"
    assert first.headers["User-Agent"] == USER_AGENT


async def test_listing_stops_at_max_pages(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    for i in range(6):
        fake.add_post(binary_post(90_000 + i, 91_000 + i), tournament="t")
    client = make_client(http, time, page_size=1, max_pages=3)
    found = await client.list_open_questions("t")
    assert len(found) == 3
    assert len(fake.requests) == 3


async def test_group_post_is_skipped_with_a_warning(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    fake.add_post(group_post(95_000), tournament="bot-testing-area")
    client = make_client(http, time)
    with capture_logs() as logs:
        found = await client.list_open_questions("bot-testing-area")
    assert sorted(q.post_id for q, _ in found) == [43327, 46224]
    skipped = [e for e in logs if e["event"] == "metaculus.post_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["post_id"] == 95_000
    assert skipped[0]["log_level"] == "warning"


async def test_detail_parses_my_forecasts(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    client = make_client(http, time)
    question, post = await client.get_question(43327)
    assert (question.post_id, question.question_id, question.type) == (43327, 43332, "binary")
    assert question.url == "https://www.metaculus.com/questions/43327/"
    assert post["question"]["my_forecasts"]["latest"] is None
    assert not question.already_forecast(NOW)

    await client.submit_binary(43332, 0.03)
    again, _ = await client.get_question(43327)
    assert again.my_latest_forecast is not None
    assert again.my_latest_forecast.forecast_values == pytest.approx([0.97, 0.03])
    assert again.my_latest_forecast.start_time == NOW
    assert again.already_forecast(NOW)


async def test_submit_binary_sends_the_exact_payload(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    client = make_client(http, time)
    result = await client.submit_binary(43332, 0.12)
    expected = [
        {
            "question": 43332,
            "source": "api",
            "probability_yes": 0.12,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    ]
    assert result.payload == expected
    assert MetaculusClient.binary_payload(43332, 0.12) == expected
    assert result.status_code == 201
    assert fake.forecasts == expected
    sent = fake.requests[-1]
    assert (sent.method, sent.url.path) == ("POST", "/api/questions/forecast/")
    assert json.loads(sent.content) == expected
    assert sent.headers["Authorization"] == f"Token {TOKEN}"


async def test_submit_to_a_numeric_question_is_rejected_by_the_fake(
    http: httpx.AsyncClient, time: FakeTime
) -> None:
    client = make_client(http, time)
    with pytest.raises(MetaculusError) as info:
        await client.submit_binary(46876, 0.5)
    assert info.value.status_code == 400
    assert "/questions/forecast/" in str(info.value)
    assert "Only continuous_cdf should be provided" in str(info.value)


async def test_private_comment_sends_the_exact_payload(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    client = make_client(http, time)
    result = await client.create_private_comment(43327, "Base rate is low; 3%.")
    expected = {
        "text": "Base rate is low; 3%.",
        "parent": None,
        "included_forecast": True,
        "is_private": True,
        "on_post": 43327,
    }
    assert result.payload == expected
    assert MetaculusClient.comment_payload(43327, "Base rate is low; 3%.") == expected
    assert fake.comments == [expected]
    sent = fake.requests[-1]
    assert (sent.method, sent.url.path) == ("POST", "/api/comments/create/")
    assert json.loads(sent.content) == expected


async def test_429_is_retried_after_retry_after(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    fake.rate_limited = 1
    fake.retry_after = "7"
    client = make_client(http, time)
    post = await client.get_post(43327)
    assert post["id"] == 43327
    assert len(fake.requests) == 2
    assert time.sleeps == [7.0]


async def test_retry_after_defaults_and_caps(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    fake.rate_limited = 2
    fake.retry_after = None
    client = make_client(http, time)
    await client.get_post(43327)
    assert time.sleeps == [5.0, 5.0]

    time.sleeps.clear()
    fake.rate_limited = 1
    fake.retry_after = "3600"
    await client.get_post(43327)
    assert time.sleeps == [120.0]


async def test_persistent_429_raises_after_max_retries(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    fake.rate_limited = 10
    client = make_client(http, time, max_retries=2)
    with pytest.raises(MetaculusError) as info:
        await client.get_post(43327)
    assert info.value.status_code == 429
    assert len(fake.requests) == 3
    assert "GET /posts/43327/" in str(info.value)


async def test_403_names_the_variable_and_hides_the_token(
    http: httpx.AsyncClient, time: FakeTime
) -> None:
    wrong = "wrong-token-abcdef0123456789"
    client = make_client(http, time, token=wrong)
    with capture_logs() as logs, pytest.raises(MetaculusError) as info:
        await client.list_open_questions("minibench")
    message = str(info.value)
    assert info.value.status_code == 403
    assert "METACULUS_TOKEN" in message
    assert "GET /posts/" in message
    assert "403" in message
    assert wrong not in message
    assert TOKEN not in message
    assert wrong not in repr(logs)


async def test_error_body_excerpt_is_bounded_and_redacted(time: FakeTime) -> None:
    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"boom {request.headers['Authorization']} " + "x" * 2000)

    async with httpx.AsyncClient(transport=httpx.MockTransport(echo)) as http:
        client = make_client(http, time)
        with pytest.raises(MetaculusError) as info:
            await client.get_post(1)
    message = str(info.value)
    assert TOKEN not in message
    assert message.count("x") <= 300


async def test_network_error_becomes_metaculus_error(time: FakeTime) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
        client = make_client(http, time)
        with pytest.raises(MetaculusError) as info:
            await client.get_post(1)
    assert info.value.status_code is None
    assert "network" in str(info.value)


async def test_writes_without_token_send_nothing(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    client = make_client(http, time, token=None)
    with pytest.raises(MissingCredentialsError) as info:
        await client.submit_binary(43332, 0.2)
    assert info.value.variables == ("METACULUS_TOKEN",)
    with pytest.raises(MissingCredentialsError):
        await client.create_private_comment(43327, "reasoning")
    assert fake.requests == []
    assert fake.forecasts == []
    assert fake.comments == []


@pytest.mark.parametrize(
    "probability", [0.0, 1.0, 0.0009, 0.9991, -0.5, float("nan"), float("inf")]
)
async def test_invalid_probability_is_refused_before_any_request(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime, probability: float
) -> None:
    client = make_client(http, time)
    with pytest.raises(ValueError, match="outside"):
        await client.submit_binary(43332, probability)
    assert fake.requests == []


async def test_empty_comment_is_refused_before_any_request(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    client = make_client(http, time)
    with pytest.raises(ValueError, match="empty"):
        await client.create_private_comment(43327, "   ")
    assert fake.requests == []


async def test_fake_validates_comment_fields(
    fake: FakeMetaculus, http: httpx.AsyncClient, time: FakeTime
) -> None:
    bad = MetaculusClient.comment_payload(43327, "text") | {"is_private": "yes"}
    response = await http.post(
        f"{BASE}/comments/create/", json=bad, headers={"Authorization": f"Token {TOKEN}"}
    )
    assert response.status_code == 400
    missing = MetaculusClient.comment_payload(1, "text")
    response = await http.post(
        f"{BASE}/comments/create/", json=missing, headers={"Authorization": f"Token {TOKEN}"}
    )
    assert response.status_code == 400
    response = await http.get(f"{BASE}/nowhere/", headers={"Authorization": f"Token {TOKEN}"})
    assert response.status_code == 404
    assert fake.comments == []


async def test_from_settings_uses_base_url_and_token(fake: FakeMetaculus, time: FakeTime) -> None:
    settings = Settings(metaculus_token=SecretStr(TOKEN))
    async with httpx.AsyncClient(transport=fake.transport) as http:
        client = MetaculusClient.from_settings(settings, http)
        assert client.has_token
        post = await client.get_post(43327)
    assert post["id"] == 43327
    assert str(fake.requests[0].url) == f"{BASE}/posts/43327/"
