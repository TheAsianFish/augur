"""The ``Provider`` protocol and its request and response types."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Effort = Literal["minimal", "low", "medium", "high", "xhigh"]
EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LlmRequest(BaseModel):
    """One chat completion. Its hash keys the replay cache."""

    model_config = ConfigDict(frozen=True)

    model: str
    messages: tuple[Message, ...]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    max_tokens: int = Field(default=1024, ge=1)
    # How hard a reasoning model should think; ``None`` leaves it to the host.
    reasoning_effort: Effort | None = None

    def request_hash(self) -> str:
        data = self.model_dump(mode="json")
        if data["reasoning_effort"] is None:
            # Left out so hashes (and replay recordings) from before the field existed hold.
            del data["reasoning_effort"]
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    # ``None`` when the host did not say; otherwise its own figure, which wins
    # over a price computed from tokens.
    cost_usd: float | None = None
    # The model the host says answered (a router may serve another).
    served_model: str | None = None
    # A reasoning model's trace, when the host returns it beside the answer.
    reasoning: str | None = None
    # Any rate-limit headers the host sent (``x-ratelimit-*``, ``retry-after``).
    limit_headers: dict[str, str] = Field(default_factory=dict)


class Provider(Protocol):
    """Anything that completes a chat request."""

    @property
    def name(self) -> str: ...

    async def complete(self, request: LlmRequest) -> LlmResponse:
        """Complete ``request``; raise ``ProviderError`` with a next step on failure."""
        ...

    async def aclose(self) -> None: ...
