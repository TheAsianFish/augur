"""One provider for every OpenAI-compatible host.

OpenRouter (the sponsored key and ``:free`` models), Groq, Gemini's OpenAI
endpoint, GitHub Models and Ollama's ``/v1`` all speak ``POST
{base_url}/chat/completions`` with a Bearer key, so a host is an
:class:`~augur.llm.endpoints.Endpoint` (base URL, the name of the variable
holding its key), not code.

What this class owns is the error taxonomy the router falls back on: a 429
that short retries did not clear is :class:`RateLimitedError`, a spent key is
:class:`OutOfCreditError`, a refused key :class:`EndpointAuthError`, a model
the host does not serve (or a host that keeps failing)
:class:`ModelUnavailableError`. Anything else is a plain ``ProviderError``
and fails only the call that hit it.
"""

from __future__ import annotations

import math
import time
from typing import Any

import httpx
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
from augur.llm.base import LlmRequest, LlmResponse
from augur.llm.endpoints import Endpoint
from augur.ratelimit import RateLimiter

MAX_RETRY_AFTER = 120.0
# A wait longer than this is not spent inside the call: the router moves to
# the next model and comes back to this one after the wait.
MAX_INLINE_WAIT = 20.0
_EXCERPT = 300
_CREDENTIAL_WORDS = ("key", "auth", "credential", "token", "unauthor", "forbidden")


def limiter_for(endpoint: Endpoint) -> RateLimiter:
    """A token bucket at the endpoint's requests-per-minute ceiling."""
    burst = max(1, min(4, int(endpoint.rpm // 10)))
    return RateLimiter(endpoint.rpm / 60.0, burst=burst, max_concurrent=endpoint.max_concurrent)


def limit_headers(resp: httpx.Response) -> dict[str, str]:
    return {
        k.lower(): v
        for k, v in resp.headers.items()
        if k.lower().startswith(("x-ratelimit", "ratelimit")) or k.lower() == "retry-after"
    }


def retry_after(resp: httpx.Response, default: float, now: float | None = None) -> float:
    """The server's wait in seconds, clamped: ``inf``, ``nan`` or a day must not stall a run.

    Reads ``Retry-After`` (seconds), then OpenRouter's ``X-RateLimit-Reset``
    (epoch milliseconds).
    """
    seconds: float | None = None
    try:
        seconds = float(resp.headers["Retry-After"])
    except (KeyError, ValueError):
        try:
            reset_ms = float(resp.headers["X-RateLimit-Reset"])
            seconds = reset_ms / 1000.0 - (time.time() if now is None else now)
        except (KeyError, ValueError):
            seconds = None
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        seconds = default
    return min(seconds, MAX_RETRY_AFTER)


class OpenAICompatProvider:
    def __init__(
        self,
        http: httpx.AsyncClient,
        endpoint: Endpoint,
        key: SecretStr | None,
        limiter: RateLimiter | None = None,
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> None:
        self._timeout = timeout
        self._http = http
        self._endpoint = endpoint
        self._key = key
        self._limiter = limiter or limiter_for(endpoint)
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return self._endpoint.name

    @property
    def endpoint(self) -> Endpoint:
        return self._endpoint

    def _excerpt(self, text: str) -> str:
        text = " ".join(text.split())[:_EXCERPT]
        if self._key is not None:
            secret = self._key.get_secret_value()
            if secret:
                text = text.replace(secret, "[redacted]")
        return text

    async def complete(self, request: LlmRequest) -> LlmResponse:
        endpoint = self._endpoint
        if endpoint.key_env and self._key is None:
            raise EndpointAuthError(
                f"endpoint {endpoint.name} needs {endpoint.key_env} in the environment; "
                f"set it, or drop {endpoint.name}: models from the AUGUR_*_MODELS lists"
            )
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            body["seed"] = request.seed
        if request.reasoning_effort is not None:
            if endpoint.billing == "openrouter" or endpoint.host == "openrouter.ai":
                body["reasoning"] = {"effort": request.reasoning_effort}
            else:
                body["reasoning_effort"] = request.reasoning_effort
        headers = {}
        if self._key is not None:
            headers["Authorization"] = f"Bearer {self._key.get_secret_value()}"
        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        where = f"{endpoint.name} ({endpoint.host})"
        resp: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with self._limiter.acquire():
                    if self._timeout is None:
                        resp = await self._http.post(url, json=body, headers=headers)
                    else:
                        resp = await self._http.post(
                            url, json=body, headers=headers, timeout=self._timeout
                        )
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    continue
                raise ModelUnavailableError(
                    f"{where} timed out on {request.model}; it falls to the next model, "
                    "or retry later"
                ) from exc
            except httpx.HTTPError as exc:
                raise ModelUnavailableError(
                    f"{where} is unreachable ({type(exc).__name__}); check the network or "
                    f"the endpoint's URL"
                ) from exc
            last = attempt == self._max_retries
            if resp.status_code == 429:
                wait = retry_after(resp, default=2.0**attempt)
                if last or wait > MAX_INLINE_WAIT:
                    raise RateLimitedError(
                        f"{where} rate-limited {request.model} (HTTP 429): "
                        f"{self._excerpt(resp.text)}",
                        status_code=429,
                        retry_after=wait,
                        limit_headers=limit_headers(resp),
                    )
                self._limiter.pause(wait)
                continue
            if resp.status_code >= 500 and not last:
                self._limiter.pause(retry_after(resp, default=2.0**attempt))
                continue
            break
        assert resp is not None
        if resp.status_code >= 400:
            raise self._classify(resp, request.model)
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"{where} returned a body that is not JSON; retry, and report it if it persists"
            ) from exc
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # Some hosts answer 200 with an error object.
            code = data["error"].get("code")
            status = code if isinstance(code, int) else 502
            raise self._classify(resp, request.model, status=status)
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content")
            usage = data.get("usage") or {}
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(
                f"{where} returned an unexpected body; retry, and report it if it persists"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            raise EmptyAnswerError(
                f"{request.model} on {endpoint.name} returned no text (finish_reason "
                f"{reason!r}); a reasoning model may need more max_tokens"
            )
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        reported = usage.get("cost")
        served = data.get("model")
        trace = choice["message"].get("reasoning")
        return LlmResponse(
            text=content,
            model=request.model,
            input_tokens=in_tok if isinstance(in_tok, int) else None,
            output_tokens=out_tok if isinstance(out_tok, int) else None,
            cost_usd=max(0.0, float(reported)) if isinstance(reported, int | float) else None,
            served_model=served if isinstance(served, str) else None,
            reasoning=trace if isinstance(trace, str) and trace.strip() else None,
            limit_headers=limit_headers(resp),
        )

    def _classify(
        self, resp: httpx.Response, model: str, status: int | None = None
    ) -> ProviderError:
        status = status or resp.status_code
        text = self._excerpt(resp.text)
        lowered = text.lower()
        where = f"{self._endpoint.name} ({self._endpoint.host})"
        kwargs: dict[str, Any] = {"status_code": status, "limit_headers": limit_headers(resp)}
        error: FallbackError
        if status == 429:
            error = RateLimitedError(
                f"{where} rate-limited {model} (HTTP 429): {text}",
                retry_after=retry_after(resp, default=60.0),
                **kwargs,
            )
        elif status == 402 or (
            status == 403 and any(w in lowered for w in ("credit", "limit", "quota", "balance"))
        ):
            error = OutOfCreditError(
                f"{where} says the key is out of credit for {model} (HTTP {status}): {text}; "
                "the router moves to the next model; never top up a paid balance",
                **kwargs,
            )
        elif status == 401 or (
            status == 403 and (not lowered.strip() or any(w in lowered for w in _CREDENTIAL_WORDS))
        ):
            env = self._endpoint.key_env or "its key"
            error = EndpointAuthError(
                f"{where} refused the credentials (HTTP {status}): {text}; check {env}", **kwargs
            )
        elif status == 403:
            # A gate on the model, not the key (OpenRouter: "only available on
            # agentic harnesses", a moderation or data-policy refusal).
            error = ModelUnavailableError(
                f"{where} refuses {model} to this client (HTTP 403): {text}", **kwargs
            )
        elif status == 404 or (
            status == 400
            and "model" in lowered
            and any(w in lowered for w in ("not a valid", "not found", "does not exist"))
        ):
            error = ModelUnavailableError(
                f"{where} does not serve {model} (HTTP {status}): {text}", **kwargs
            )
        elif status >= 500:
            error = ModelUnavailableError(
                f"{where} kept failing for {model} (HTTP {status}); retry later", **kwargs
            )
        else:
            return ProviderError(
                f"{where} refused the request for {model} (HTTP {status}): {text}; "
                "check the model's parameters"
            )
        return error

    async def aclose(self) -> None:
        return None
