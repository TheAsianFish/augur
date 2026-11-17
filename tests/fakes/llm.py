"""A scripted provider for unit tests: it answers by the role a prompt plays."""

from __future__ import annotations

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
        *,
        cost_per_call: float = 0.0,
    ) -> None:
        self.percents = list(percents)
        self.cost_per_call = cost_per_call
        self.requests: list[LlmRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
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
            cost_usd=self.cost_per_call,
        )

    async def aclose(self) -> None:
        return None
