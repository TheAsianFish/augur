"""Every model call goes through :class:`CallRecorder`: timed, costed, budgeted, ledgered."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from augur.clock import Clock
from augur.errors import AugurError, ProviderError
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, LlmResponse, Provider
from augur.llm.registry import lookup


class BudgetExceededError(AugurError):
    """A cost cap would be crossed; the run stops spending."""


@dataclass
class Budget:
    """Cost caps per question and per run, in estimated US dollars."""

    max_per_question: float
    max_per_run: float
    spent_run: float = 0.0
    spent_question: dict[int, float] = field(default_factory=dict)

    def check(self, question_id: int) -> None:
        if self.spent_run >= self.max_per_run:
            raise BudgetExceededError(
                f"run budget ${self.max_per_run:.2f} spent; raise AUGUR_MAX_COST_PER_RUN to go on"
            )
        if self.spent_question.get(question_id, 0.0) >= self.max_per_question:
            raise BudgetExceededError(
                f"question {question_id} budget ${self.max_per_question:.2f} spent; "
                "raise AUGUR_MAX_COST_PER_QUESTION to go on"
            )

    def charge(self, question_id: int, cost: float) -> None:
        self.spent_run += cost
        self.spent_question[question_id] = self.spent_question.get(question_id, 0.0) + cost


@dataclass(frozen=True)
class CallContext:
    run_id: str
    question_id: int
    role: str
    member: int | None = None


class CallRecorder:
    def __init__(self, provider: Provider, ledger: Ledger, clock: Clock, budget: Budget) -> None:
        self._provider = provider
        self._ledger = ledger
        self._clock = clock
        self.budget = budget

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def complete(
        self, ctx: CallContext, request: LlmRequest, source_ids: list[str]
    ) -> tuple[LlmResponse, str]:
        """Run ``request`` and ledger it, citing ``source_ids`` (all must be admitted)."""
        self.budget.check(ctx.question_id)
        info = lookup(request.model)
        started = self._clock.now()
        t0 = time.perf_counter()
        response: LlmResponse | None = None
        error: str | None = None
        try:
            response = await self._provider.complete(request)
        except ProviderError as exc:
            error = str(exc)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        cost = 0.0
        if response is not None:
            cost = response.cost_usd or info.cost(response.input_tokens, response.output_tokens)
            self.budget.charge(ctx.question_id, cost)
        call_id = self._ledger.record_call(
            run_id=ctx.run_id,
            question_id=ctx.question_id,
            role=ctx.role,
            member=ctx.member,
            provider=self._provider.name,
            model=request.model,
            model_release_date=info.release_date,
            request_hash=request.request_hash(),
            request=request.model_dump(mode="json"),
            response_text=response.text if response else None,
            input_tokens=response.input_tokens if response else None,
            output_tokens=response.output_tokens if response else None,
            cost_usd=cost,
            latency_ms=latency_ms,
            error=error,
            started_at=started,
            source_ids=source_ids,
        )
        if response is None:
            raise ProviderError(error or "provider failed")
        return response, call_id
