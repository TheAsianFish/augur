"""Local Ollama provider. CPU inference, so one generation at a time."""

from __future__ import annotations

from typing import Any

import httpx

from augur.errors import ProviderError
from augur.llm.base import LlmRequest, LlmResponse
from augur.ratelimit import RateLimiter


class OllamaProvider:
    def __init__(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        limiter: RateLimiter | None = None,
        num_ctx: int = 8192,
    ) -> None:
        # Ollama's default window (4096 here) silently drops the start of a longer
        # prompt, so the window is set on every call.
        self._num_ctx = num_ctx
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._limiter = limiter or RateLimiter(10, burst=10, max_concurrent=1)

    @property
    def name(self) -> str:
        return "ollama"

    async def complete(self, request: LlmRequest) -> LlmResponse:
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
            "num_ctx": self._num_ctx,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        body = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": options,
        }
        url = f"{self._base_url}/api/chat"
        try:
            async with self._limiter.acquire():
                resp = await self._http.post(url, json=body)
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Ollama is not reachable at {self._base_url}; start it with `ollama serve` "
                f"and pull the model with `ollama pull {request.model}`"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Ollama at {self._base_url} timed out on {request.model}; use a smaller "
                "model or fewer max tokens, or raise the client read timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Ollama request to {self._base_url} failed ({type(exc).__name__}); "
                "check that `ollama serve` is healthy and retry"
            ) from exc
        if resp.status_code == 404 or "not found" in resp.text.lower():
            raise ProviderError(
                f"model {request.model} is not pulled; run `ollama pull {request.model}`"
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama returned HTTP {resp.status_code} for {request.model}; "
                "check the `ollama serve` log and retry"
            )
        try:
            data = resp.json()
            text = str(data["message"]["content"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                f"Ollama at {self._base_url} returned an unexpected body; upgrade Ollama and retry"
            ) from exc
        return LlmResponse(
            text=text,
            model=request.model,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            cost_usd=0.0,
        )

    async def aclose(self) -> None:
        await self._http.aclose()
