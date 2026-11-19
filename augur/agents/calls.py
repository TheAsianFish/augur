"""Every model call goes through :class:`CallRecorder`: routed, timed, costed, budgeted, ledgered.

A call names its role (planner, researcher, forecaster); the recorder walks
that role's model chain through the :class:`~augur.llm.router.Router`. Each
attempt is a ledger row, failed ones included, and each move to the next
model is a ``model_switches`` row. In a scored run the forecaster chain has a
floor: a route below the frontier tier is never called, and reaching one
raises :class:`~augur.errors.BelowFloorError` so the question is skipped
rather than answered by a weaker model.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from augur.clock import Clock
from augur.errors import (
    AugurError,
    BelowFloorError,
    EmptyAnswerError,
    FallbackError,
    ModelsExhaustedError,
    ProviderError,
    RateLimitedError,
    UnpricedModelError,
)
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, LlmResponse
from augur.llm.catalog import Price
from augur.llm.endpoints import CALL_ROLE_CHAIN, Route
from augur.llm.router import Router
from augur.log import get_logger

log = get_logger(__name__)

# A route refused for rate after the provider's own short retries rests at
# least this long: an upstream 429 carries no wait, and without a floor every
# later call would spend its retries on the same busy model (M2: 7.8 s each).
MIN_COOL_S = 30.0


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

    def charge_run(self, amount: float) -> None:
        """A cost outside any question (the probe). Never refused: it already happened."""
        self.spent_run += amount


def estimate_cost(request: LlmRequest, price: Price) -> float:
    """Worst case: about one token per three characters in, ``max_tokens`` out."""
    chars = sum(len(m.content) for m in request.messages)
    return price.cost(chars // 3 + 1, request.max_tokens)


@dataclass(frozen=True)
class CallContext:
    run_id: str
    question_id: int
    role: str
    member: int | None = None


@dataclass(frozen=True)
class CallResult:
    response: LlmResponse
    call_id: str
    route: Route


class CallRecorder:
    def __init__(self, router: Router, ledger: Ledger, clock: Clock, budget: Budget) -> None:
        self.router = router
        self._ledger = ledger
        self._clock = clock
        self.budget = budget

    async def complete(
        self, ctx: CallContext, request: LlmRequest, source_ids: list[str]
    ) -> CallResult:
        """Run ``request`` on the first route of its role's chain that answers.

        ``request.model`` is replaced by each route's model in turn. All of
        ``source_ids`` must have been admitted by the gate for this question.
        """
        role = CALL_ROLE_CHAIN.get(ctx.role)
        if role is None:
            raise AugurError(f"internal: no model chain for call role {ctx.role!r}")
        floor = self.router.floor(role)
        moved_from: tuple[Route, FallbackError] | None = None
        skipped: list[str] = []
        last_skipped: tuple[Route, str] | None = None
        for route in self.router.chain(role):
            why_not = self.router.unavailable(route)
            if why_not is not None:
                skipped.append(f"{route.key}: {why_not}")
                last_skipped = (route, self.router.unavailable_kind(route) or "unavailable")
                continue
            if floor is not None and not floor[1](route):
                if moved_from is not None:
                    previous: str | None = moved_from[0].key
                    reason = moved_from[1].kind
                elif last_skipped is not None:
                    previous, reason = last_skipped[0].key, last_skipped[1]
                else:
                    previous, reason = None, "first_in_chain"
                detail = (
                    f"{route.key} is below the {floor[0]} tier, so this scored question is "
                    "skipped instead of answered by it"
                )
                self._switch(ctx, previous, route.key, reason, "refused_by_floor", detail)
                raise BelowFloorError(
                    f"{detail}; earlier routes: "
                    + "; ".join([*skipped, *([str(moved_from[1])] if moved_from else [])])
                )
            if moved_from is not None:
                self._switch(
                    ctx,
                    moved_from[0].key,
                    route.key,
                    moved_from[1].kind,
                    "switched",
                    str(moved_from[1]),
                )
            try:
                return await self._attempt(ctx, route, request, source_ids)
            except FallbackError as exc:
                if isinstance(exc, EmptyAnswerError):
                    pass  # this request only; the route stays in use
                elif isinstance(exc, RateLimitedError):
                    self.router.cool(route, max(exc.retry_after or 0.0, MIN_COOL_S), exc.kind)
                else:
                    self.router.disable(route, f"{exc.kind}: {exc}", exc.kind)
                moved_from = (route, exc)
        if moved_from is not None:
            self._switch(
                ctx, moved_from[0].key, None, moved_from[1].kind, "exhausted", str(moved_from[1])
            )
        tried = [*skipped, *([str(moved_from[1])] if moved_from else [])]
        raise ModelsExhaustedError(
            f"no model in the {role} chain could answer: "
            + "; ".join(tried)
            + "; add models to AUGUR_"
            + role.upper()
            + "_MODELS or retry later"
        )

    def _switch(
        self,
        ctx: CallContext,
        from_route: str | None,
        to_route: str | None,
        reason: str,
        outcome: str,
        detail: str,
    ) -> None:
        log.warning(
            "model switch",
            outcome=outcome,
            from_route=from_route,
            to_route=to_route,
            reason=reason,
            role=ctx.role,
            member=ctx.member,
        )
        self._ledger.record_switch(
            run_id=ctx.run_id,
            question_id=ctx.question_id,
            call_role=ctx.role,
            member=ctx.member,
            from_route=from_route,
            to_route=to_route,
            reason=reason,
            outcome=outcome,
            detail=detail,
        )

    async def _attempt(
        self, ctx: CallContext, route: Route, request: LlmRequest, source_ids: list[str]
    ) -> CallResult:
        price = self.router.price(route)
        if price is None:
            raise UnpricedModelError(
                f"{route.key} has no price in OpenRouter's list and {route.endpoint.name} "
                "bills, so it is not called; pick a listed model, or set "
                f"AUGUR_ENDPOINT_{route.endpoint.name.upper()}_BILLING=free if the host is free"
            )
        request = request.model_copy(
            update={"model": route.model, "reasoning_effort": route.reasoning_effort}
        )
        provider = self.router.provider(route)
        release, release_basis = self.router.release(route)
        reserved = estimate_cost(request, price)
        self.budget.reserve(ctx.question_id, reserved)
        started = self._clock.now()
        t0 = time.perf_counter()
        response: LlmResponse | None = None
        failure: ProviderError | None = None
        cancelled = False
        try:
            response = await provider.complete(request)
        except ProviderError as exc:
            failure = exc
        except asyncio.CancelledError:
            # Stopped at a question's deadline. The host may still bill the
            # request, so it is charged the worst case and ledgered, then the
            # cancellation goes on.
            cancelled = True
            failure = ProviderError("cancelled at the question's deadline; charged the worst case")
        finally:
            if response is None:
                # An empty answer was generated and billed (a reasoning model can
                # spend all of max_tokens thinking), so it is charged the worst
                # case; a refused or failed request cost nothing.
                billed = cancelled or isinstance(failure, EmptyAnswerError)
                self.budget.settle(ctx.question_id, reserved, reserved if billed else 0.0)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        cost = reserved if cancelled or isinstance(failure, EmptyAnswerError) else 0.0
        if response is not None:
            if response.cost_usd is not None:
                cost = response.cost_usd
            elif response.input_tokens is None or response.output_tokens is None:
                # No usage reported: charge the worst case so the caps still bind.
                cost = reserved
            else:
                cost = price.cost(response.input_tokens, response.output_tokens)
            self.budget.settle(ctx.question_id, reserved, cost)
        call_id = self._ledger.record_call(
            run_id=ctx.run_id,
            question_id=ctx.question_id,
            role=ctx.role,
            member=ctx.member,
            provider=provider.name,
            model=route.model,
            model_release_date=release,
            release_basis=release_basis,
            endpoint_host=route.endpoint.host,
            reasoning_effort=route.reasoning_effort,
            reasoning_effort_applied=route.effort_applied,
            served_model=response.served_model if response else None,
            request_hash=request.request_hash(),
            request=request.model_dump(mode="json"),
            response_text=response.text if response else None,
            input_tokens=response.input_tokens if response else None,
            output_tokens=response.output_tokens if response else None,
            cost_usd=cost,
            latency_ms=latency_ms,
            error=str(failure) if failure else None,
            started_at=started,
            source_ids=source_ids,
        )
        if cancelled:
            raise asyncio.CancelledError
        if response is None:
            raise failure or ProviderError("provider failed")
        return CallResult(response, call_id, route)
