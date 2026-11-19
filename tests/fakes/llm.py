"""A scripted provider for unit tests: it answers by the role a prompt plays."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence

from augur.agents.forecaster import CHOICE_SYSTEM, FORECASTER_SYSTEM, NUMERIC_SYSTEM
from augur.agents.researcher import BRIEF_SYSTEM
from augur.errors import ProviderError
from augur.llm.base import LlmRequest, LlmResponse


class ScriptedProvider:
    """Planner → two queries; brief → cites S1; forecaster draw k → ``percents[k]``."""

    def __init__(
        self,
        percents: Sequence[float | None] = (30, 35, 40, 45, 50),
        slow_draws: Sequence[int] = (),
    ) -> None:
        self.percents = list(percents)
        self.slow_draws = set(slow_draws)
        """Forecaster draws (by seed) that take ten seconds: past any test's deadline."""
        self.requests: list[LlmRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        await asyncio.sleep(0)  # yield like a real HTTP call, so draws interleave
        system = request.messages[0].content
        user = request.messages[-1].content
        seed = request.seed or 0
        if system in (FORECASTER_SYSTEM, CHOICE_SYSTEM, NUMERIC_SYSTEM):
            if seed in self.slow_draws:
                await asyncio.sleep(10)
            if self.percents[seed % len(self.percents)] is None:
                raise ProviderError(f"scripted failure for draw {seed}")
        if system == CHOICE_SYSTEM:
            text = f"Options weighed (draw {seed}).\n" + _choice_lines(user, seed)
        elif system == NUMERIC_SYSTEM:
            text = f"Trend and tails considered (draw {seed}).\n" + _percentile_lines(user, seed)
        elif system == FORECASTER_SYSTEM:
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


def _choice_lines(user: str, seed: int) -> str:
    found = re.search(r"The options are, in this order: (.*?)\.\n", user)
    assert found, "a multiple-choice prompt lists its options"
    options = found.group(1).split("; ")
    weights = [len(options) - i + seed % 2 for i in range(len(options))]
    total = sum(weights)
    return "\n".join(f"{o}: {100 * w / total:.1f}%" for o, w in zip(options, weights, strict=True))


def _percentile_lines(user: str, seed: int) -> str:
    number = r"(-?[\d,]*\.?\d+)"
    found = re.search(r"range is " + number + r"[^\n]*? to " + number, user)
    assert found, "a numeric prompt states the question's range"
    low, high = (float(g.replace(",", "")) for g in found.groups())
    shares = (0.2, 0.3, 0.45, 0.55, 0.7, 0.8)
    shift = 0.02 * (seed % 3)
    return "\n".join(
        f"Percentile {level}: {low + (high - low) * (s + shift):g}"
        for level, s in zip((10, 20, 40, 60, 80, 90), shares, strict=True)
    )


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
