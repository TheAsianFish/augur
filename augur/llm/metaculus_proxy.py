"""Metaculus's sponsored LLM proxy (OpenAI chat-completions format).

Note: on 2026-11-17 ``llm-proxy.metaculus.com`` did not resolve in public DNS,
so this provider is untested against the live service. Override the URL with
``AUGUR_PROXY_BASE_URL`` if Metaculus publishes a new one.
"""

from __future__ import annotations

import math
from typing import Any

import httpx
from pydantic import SecretStr

from augur.errors import ProviderError
from augur.llm.base import LlmRequest, LlmResponse
from augur.llm.registry import priced
from augur.ratelimit import RateLimiter

_MAX_RETRIES = 3


class MetaculusProxyProvider:
    def __init__(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        token: SecretStr,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._limiter = limiter or RateLimiter(2, burst=4, max_concurrent=4)

    @property
    def name(self) -> str:
        return "metaculus_proxy"

    async def complete(self, request: LlmRequest) -> LlmResponse:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            body["seed"] = request.seed
        headers = {"Authorization": f"Token {self._token.get_secret_value()}"}
        url = f"{self._base_url}/chat/completions"
        resp: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with self._limiter.acquire():
                    resp = await self._http.post(url, json=body, headers=headers)
            except httpx.ConnectError as exc:
                raise ProviderError(
                    f"the Metaculus LLM proxy at {self._base_url} is unreachable; check your "
                    "network, or set AUGUR_PROXY_BASE_URL to the proxy's current URL"
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"request to the Metaculus LLM proxy failed ({type(exc).__name__}); retry later"
                ) from exc
            if resp.status_code != 429 and resp.status_code < 500:
                break
            if attempt == _MAX_RETRIES:
                break
            wait = _retry_after(resp, default=2.0**attempt)
            self._limiter.pause(wait)
        assert resp is not None
        if resp.status_code in (401, 403):
            raise ProviderError(
                f"the Metaculus LLM proxy rejected the credentials (HTTP {resp.status_code}); "
                "check METACULUS_TOKEN"
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"the Metaculus LLM proxy returned HTTP {resp.status_code} for {request.model}; "
                "retry later or lower the request rate"
            )
        try:
            data = resp.json()
            text = str(data["choices"][0]["message"]["content"])
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "the Metaculus LLM proxy returned an unexpected body; retry, and report it "
                "if it persists"
            ) from exc
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        info = priced(request.model, self.name)
        if isinstance(in_tok, int) and isinstance(out_tok, int):
            cost = info.cost(in_tok, out_tok)
        else:
            # No usage reported: charge the worst case (about a token per three
            # characters in, max_tokens out) so the cost caps still bind.
            chars = sum(len(m.content) for m in request.messages)
            cost = info.cost(chars // 3 + 1, request.max_tokens)
            in_tok = out_tok = None
        return LlmResponse(
            text=text,
            model=request.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    async def aclose(self) -> None:
        await self._http.aclose()


MAX_RETRY_AFTER = 120.0


def _retry_after(resp: httpx.Response, default: float) -> float:
    """The server's wait, clamped: ``inf``, ``nan`` or a day must not stall the run."""
    try:
        seconds = float(resp.headers["Retry-After"])
    except (KeyError, ValueError):
        seconds = default
    if not math.isfinite(seconds) or seconds < 0:
        seconds = default
    return min(seconds, MAX_RETRY_AFTER)
