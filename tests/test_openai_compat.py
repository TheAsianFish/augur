"""The generic OpenAI-compatible provider: request shape, usage, and the refusals it classifies."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from augur.errors import (
    EmptyAnswerError,
    EndpointAuthError,
    FallbackError,
    ModelUnavailableError,
    OutOfCreditError,
    ProviderError,
    RateLimitedError,
)
from augur.llm.endpoints import Endpoint
from augur.llm.openai_compat import OpenAICompatProvider, limiter_for, retry_after
from augur.ratelimit import RateLimiter

from .test_providers import REQ, FakeTime, client, fast_limiter

OR = Endpoint(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1/",
    key_env="OPENROUTER_API_KEY",
    billing="openrouter",
)


def compat_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "served/model",
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0042},
        },
        headers={"x-ratelimit-remaining-requests": "19", "content-type": "application/json"},
    )


def compat(
    handler: httpx.MockTransport | None = None,
    *,
    key: str | None = "sekret",
    time: FakeTime | None = None,
    endpoint: Endpoint = OR,
    retries: int = 3,
) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        client(handler or httpx.MockTransport(compat_ok)),
        endpoint,
        SecretStr(key) if key else None,
        fast_limiter(time),
        max_retries=retries,
    )


async def test_compat_request_and_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return compat_ok(request)

    p = compat(httpx.MockTransport(handler))
    out = await p.complete(REQ)
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer sekret"
    assert json.loads(seen[0].content) == {
        "model": "qwen2.5:7b",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "temperature": 0.3,
        "max_tokens": 50,
        "seed": 7,
    }
    assert (out.text, out.input_tokens, out.output_tokens) == ("answer", 100, 20)
    assert out.cost_usd == 0.0042, "the host's own cost wins"
    assert out.served_model == "served/model"
    assert out.limit_headers == {"x-ratelimit-remaining-requests": "19"}
    assert p.name == "openrouter"


async def test_compat_without_a_key_env_sends_no_authorization() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return compat_ok(request)

    local = Endpoint(name="local", base_url="http://127.0.0.1:8080/v1", billing="free")
    await compat(httpx.MockTransport(handler), key=None, endpoint=local).complete(REQ)
    assert "Authorization" not in seen[0].headers


async def test_compat_missing_key_fails_over_without_a_call() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return compat_ok(request)

    with pytest.raises(EndpointAuthError, match="OPENROUTER_API_KEY"):
        await compat(httpx.MockTransport(handler), key=None).complete(REQ)
    assert calls == 0


async def test_compat_retries_a_short_429() -> None:
    calls = 0
    time = FakeTime()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return compat_ok(request)

    out = await compat(httpx.MockTransport(handler), time=time).complete(REQ)
    assert out.text == "answer"
    assert calls == 2
    assert time.now >= 3.0  # the limiter held the call for Retry-After


async def test_compat_hands_a_long_429_to_the_router() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            json={"error": {"message": "free-models-per-min", "code": 429}},
            headers={"Retry-After": "45", "X-RateLimit-Limit": "20"},
        )

    with pytest.raises(RateLimitedError) as info:
        await compat(httpx.MockTransport(handler)).complete(REQ)
    assert calls == 1, "a wait over MAX_INLINE_WAIT is not spent inside the call"
    assert info.value.retry_after == 45.0
    assert info.value.kind == "rate_limited"
    assert info.value.limit_headers["x-ratelimit-limit"] == "20"
    assert "free-models-per-min" in str(info.value)


@pytest.mark.parametrize("header", ["inf", "nan", "86400", "-5"])
def test_an_unreasonable_retry_after_is_clamped(header: str) -> None:
    resp = httpx.Response(429, headers={"Retry-After": header})
    assert 0 <= retry_after(resp, default=2.0) <= 120.0


def test_openrouter_reset_header_is_read_as_epoch_milliseconds() -> None:
    resp = httpx.Response(429, headers={"X-RateLimit-Reset": "1000030000"})
    assert retry_after(resp, default=2.0, now=1_000_000.0) == 30.0


@pytest.mark.parametrize(
    ("status", "body", "error", "kind"),
    [
        (402, "Insufficient credits", OutOfCreditError, "out_of_credit"),
        (403, "Key limit exceeded (total limit)", OutOfCreditError, "out_of_credit"),
        (401, "No auth credentials found", EndpointAuthError, "auth"),
        (403, "forbidden", EndpointAuthError, "auth"),
        (
            403,
            "x/y:free is only available on agentic harnesses. Try plugging it into a coding agent",
            ModelUnavailableError,
            "unavailable",
        ),
        (404, "No endpoints found for x/y.", ModelUnavailableError, "unavailable"),
        (400, "x/y is not a valid model ID", ModelUnavailableError, "unavailable"),
    ],
)
async def test_compat_classifies_refusals(
    status: int, body: str, error: type[FallbackError], kind: str
) -> None:
    p = compat(httpx.MockTransport(lambda r: httpx.Response(status, text=body + " sekret")))
    with pytest.raises(error) as info:
        await p.complete(REQ)
    assert info.value.kind == kind
    assert info.value.status_code == status
    assert "sekret" not in str(info.value), "an echoed key is redacted"


async def test_compat_a_bare_403_is_read_as_the_key() -> None:
    p = compat(httpx.MockTransport(lambda r: httpx.Response(403)))
    with pytest.raises(EndpointAuthError, match="OPENROUTER_API_KEY"):
        await p.complete(REQ)


async def test_compat_bad_parameters_are_not_a_fallback() -> None:
    p = compat(httpx.MockTransport(lambda r: httpx.Response(400, text="temperature too high")))
    with pytest.raises(ProviderError) as info:
        await p.complete(REQ)
    assert not isinstance(info.value, FallbackError)


async def test_compat_an_error_object_inside_a_200_is_classified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "Insufficient credits", "code": 402}})

    with pytest.raises(OutOfCreditError):
        await compat(httpx.MockTransport(handler)).complete(REQ)


async def test_compat_empty_answer_from_a_reasoning_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": None, "reasoning": "Let me think"},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    with pytest.raises(EmptyAnswerError, match="max_tokens"):
        await compat(httpx.MockTransport(handler)).complete(REQ)


async def test_compat_without_usage_reports_no_tokens_and_no_cost() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    out = await compat(httpx.MockTransport(handler)).complete(REQ)
    # The recorder then charges the worst case (tested in test_router).
    assert (out.input_tokens, out.output_tokens, out.cost_usd) == (None, None, None)


async def test_compat_gives_up_on_5xx_as_unavailable_without_leaking_the_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    with pytest.raises(ModelUnavailableError) as info:
        await compat(httpx.MockTransport(handler)).complete(REQ)
    assert calls == 4
    assert "sekret" not in str(info.value)


async def test_compat_connect_error_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns")

    with pytest.raises(ModelUnavailableError, match="unreachable") as info:
        await compat(httpx.MockTransport(handler)).complete(REQ)
    assert "sekret" not in str(info.value)


def test_limiter_follows_the_endpoint_rpm() -> None:
    assert isinstance(limiter_for(OR), RateLimiter)


async def test_compat_keeps_the_reasoning_trace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Probability: 5%", "reasoning": "UK only."}}]
            },
        )

    out = await compat(httpx.MockTransport(handler)).complete(REQ)
    assert out.reasoning == "UK only."


async def test_compat_a_timeout_is_unavailable_and_passes_the_probe_timeout() -> None:
    seen: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        raise httpx.ReadTimeout("slow upstream", request=request)

    p = OpenAICompatProvider(
        client(httpx.MockTransport(handler)),
        OR,
        SecretStr("sekret"),
        fast_limiter(),
        max_retries=0,
        timeout=30.0,
    )
    with pytest.raises(ModelUnavailableError, match="timed out"):
        await p.complete(REQ)
    assert len(seen) == 1
    assert seen[0] == {"connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}
