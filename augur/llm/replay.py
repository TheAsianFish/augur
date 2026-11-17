"""Record and replay model calls, keyed by ``LlmRequest.request_hash()``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from augur.errors import ProviderError, ReplayMissError
from augur.llm.base import LlmRequest, LlmResponse, Provider


class ReplayProvider:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._responses: dict[str, LlmResponse] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ProviderError(
                f"cannot read replay file {path}; record one with AUGUR_RECORD_PATH and a live "
                "provider, or fix AUGUR_REPLAY_PATH"
            ) from exc
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                self._responses[str(row["request_hash"])] = LlmResponse.model_validate(
                    row["response"]
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError(
                    f"{path} line {number} is not a valid recording; re-record the file"
                ) from exc

    @property
    def name(self) -> str:
        return "replay"

    async def complete(self, request: LlmRequest) -> LlmResponse:
        digest = request.request_hash()
        try:
            return self._responses[digest]
        except KeyError:
            raise ReplayMissError(
                f"no recording for model {request.model} request {digest[:12]} in {self._path}; "
                "record it by running with a live provider and AUGUR_RECORD_PATH set"
            ) from None

    async def aclose(self) -> None:
        return None


class RecordingProvider:
    def __init__(self, inner: Provider, path: Path) -> None:
        self._inner = inner
        self._path = path
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._inner.name

    async def complete(self, request: LlmRequest) -> LlmResponse:
        response = await self._inner.complete(request)
        row = {
            "request_hash": request.request_hash(),
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }
        line = json.dumps(row, sort_keys=True) + "\n"
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()
