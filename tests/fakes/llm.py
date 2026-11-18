"""A scripted provider for unit tests: it answers by the role a prompt plays."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from augur.agents.forecaster import FORECASTER_SYSTEM
from augur.agents.researcher import BRIEF_SYSTEM
from augur.errors import ProviderError
from augur.llm.base import LlmRequest, LlmResponse


class ScriptedProvider:
    """Planner → two queries; brief → cites S1; forecaster draw k → ``percents[k]``."""

    def __init__(
        self,
        percents: Sequence[float | None] = (30, 35, 40, 45, 50),
    ) -> None:
        self.percents = list(percents)
        self.requests: list[LlmRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        await asyncio.sleep(0)  # yield like a real HTTP call, so draws interleave
        system = request.messages[0].content
        if system == FORECASTER_SYSTEM:
            seed = request.seed or 0
            pct = self.percents[seed % len(self.percents)]
            if pct is None:
                raise ProviderError(f"scripted failure for draw {seed}")
            text = f"Base rate is low and little time is left (draw {seed}).\nProbability: {pct}%"
        elif system == BRIEF_SYSTEM:
            text = "The status quo holds [S1]."
        else:
            text = '["first query", "second query"]'
        return LlmResponse(
            text=text,
            model=request.model,
            input_tokens=100,
            output_tokens=20,
        )

    async def aclose(self) -> None:
        return None


class RoutedProvider(ScriptedProvider):
    """A scripted endpoint whose models can be made to fail with a given error.

    ``failures[model]`` is raised for every call to that model; ``limit`` makes
    every call after the first ``limit`` raise a 429 (a burst test).
    """

    def __init__(
        self,
        name: str = "openrouter",
        failures: dict[str, ProviderError] | None = None,
        percents: Sequence[float | None] = (30, 35, 40, 45, 50),
    ) -> None:
        super().__init__(percents)
        self._name = name
        self.failures = dict(failures or {})

    @property
    def name(self) -> str:
        return self._name

    def models_called(self, system: str | None = None) -> list[str]:
        return [r.model for r in self.requests if system is None or r.messages[0].content == system]

    async def complete(self, request: LlmRequest) -> LlmResponse:
        failure = self.failures.get(request.model)
        if failure is not None:
            self.requests.append(request)
            raise failure
        return await super().complete(request)
