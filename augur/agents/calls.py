"""Every model call goes through :class:`CallRecorder`: timed, costed, budgeted, ledgered."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from augur.clock import Clock
from augur.errors import AugurError, ProviderError
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, LlmResponse, Provider
from augur.llm.registry import ModelInfo, priced


class BudgetExceededError(AugurError):
    """A cost cap would be crossed; the run stops spending."""


@dataclass
class Budget:
    """Cost caps per question and per run, in estimated US dollars.

    A call reserves its worst-case cost before it starts and settles the real
    cost when it ends. Reservation happens with no ``await`` between check and
    update, so concurrent calls on one event loop cannot all pass the check
    and then overspend together.
    """

    max_per_question: float
    max_per_run: float
    spent_run: float = 0.0
    reserved_run: float = 0.0
    spent_question: dict[int, float] = field(default_factory=dict)
    reserved_question: dict[int, float] = field(default_factory=dict)

    def reserve(self, question_id: int, amount: float) -> None:
        if self.spent_run + self.reserved_run + amount > self.max_per_run:
            raise BudgetExceededError(
                f"run budget ${self.max_per_run:.2f} spent; raise AUGUR_MAX_COST_PER_RUN to go on"
            )
        q_total = self.spent_question.get(question_id, 0.0) + self.reserved_question.get(
            question_id, 0.0
        )
        if q_total + amount > self.max_per_question:
            raise BudgetExceededError(
                f"question {question_id} budget ${self.max_per_question:.2f} spent; "
                "raise AUGUR_MAX_COST_PER_QUESTION to go on"
            )
        self.reserved_run += amount
        self.reserved_question[question_id] = self.reserved_question.get(question_id, 0.0) + amount

    def settle(self, question_id: int, reserved: float, actual: float) -> None:
        self.reserved_run -= reserved
        self.reserved_question[question_id] = (
            self.reserved_question.get(question_id, 0.0) - reserved
        )
        self.spent_run += actual
        self.spent_question[question_id] = self.spent_question.get(question_id, 0.0) + actual


def estimate_cost(request: LlmRequest, info: ModelInfo) -> float:
    """Worst case: about one token per three characters in, ``max_tokens`` out."""
    chars = sum(len(m.content) for m in request.messages)
    return info.cost(chars // 3 + 1, request.max_tokens)


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
        info = priced(request.model, self._provider.name)
        reserved = estimate_cost(request, info)
        self.budget.reserve(ctx.question_id, reserved)
        started = self._clock.now()
        t0 = time.perf_counter()
        response: LlmResponse | None = None
        error: str | None = None
        try:
            response = await self._provider.complete(request)
        except ProviderError as exc:
            error = str(exc)
        finally:
            if response is None:
                self.budget.settle(ctx.question_id, reserved, 0.0)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        cost = 0.0
        if response is not None:
            cost = response.cost_usd or info.cost(response.input_tokens, response.output_tokens)
            self.budget.settle(ctx.question_id, reserved, cost)
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
