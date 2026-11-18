"""The ``Provider`` protocol and its request and response types."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


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

    def request_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float = 0.0


class Provider(Protocol):
    """Anything that completes a chat request."""

    @property
    def name(self) -> str: ...

    async def complete(self, request: LlmRequest) -> LlmResponse:
        """Complete ``request``; raise ``ProviderError`` with a next step on failure."""
        ...

    async def aclose(self) -> None: ...
