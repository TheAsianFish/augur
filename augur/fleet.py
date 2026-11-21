"""The fleet: how many research passes, which forecaster prompts and models, how many at once.

M7 runs several agents per question and several questions at a time. Nothing
here calls a model: the limits that make that safe live where the calls are.
Every request goes through its endpoint's
:class:`~augur.ratelimit.RateLimiter` (a token bucket and a concurrency cap,
one per provider, shared by every question in flight); every call reserves its
worst-case cost against the per-question and per-run caps of
:class:`~augur.agents.calls.Budget` before it starts; and every route, a
member's own included, passes the router's floor in
:class:`~augur.agents.calls.CallRecorder`. This module adds the two pieces
that sit above them:

* :func:`bounded_map`, which runs questions (or anything) with at most
  ``concurrency`` in flight and returns the results in input order; and
* :class:`FleetPlan`, the run's research passes, members, their prompt
  variants and their resolved model chains, recorded and printed with the run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from augur.config import Settings
from augur.errors import ConfigError
from augur.llm.endpoints import Route
from augur.llm.router import Router

MAX_QUESTION_CONCURRENCY = 8


async def bounded_map[T, R](
    items: Sequence[T], worker: Callable[[T], Awaitable[R]], concurrency: int
) -> list[R]:
    """``[await worker(x) for x in items]`` with at most ``concurrency`` running at once.

    ``concurrency`` worker tasks (never more, whatever the length of
    ``items``) take the items in input order, so with ``concurrency=1`` this is
    exactly the sequential loop. The results come back in input order. If a
    worker raises, the others are cancelled and awaited and the first error is
    raised as it is (not wrapped in an ``ExceptionGroup``); if the caller is
    cancelled, every worker is cancelled and awaited before the cancellation
    goes on.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    results: dict[int, R] = {}
    next_index = 0

    async def drain() -> None:
        nonlocal next_index
        while next_index < len(items):
            # No await between the read and the increment: on one event loop
            # no two workers can take the same item.
            index = next_index
            next_index += 1
            results[index] = await worker(items[index])

    tasks = [asyncio.create_task(drain()) for _ in range(min(concurrency, len(items)))]
    if not tasks:
        return []
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    failed = next(
        (t for t in tasks if t in done and not t.cancelled() and t.exception() is not None),
        None,
    )
    if failed is not None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        error = failed.exception()
        assert error is not None
        raise error
    if any(t.cancelled() for t in tasks):
        # A worker's own await was cancelled from elsewhere: its items have no result.
        raise asyncio.CancelledError
    return [results[i] for i in range(len(items))]


@dataclass(frozen=True)
class MemberPlan:
    member: int
    variant: str
    chain: tuple[Route, ...]
    own_chain: bool
    """Whether the chain is the member's own (``AUGUR_MEMBER_MODELS``), not the forecast role's."""


@dataclass(frozen=True)
class FleetPlan:
    """What one question's agents are: research passes, then ``members`` forecasters.

    Member ``m`` takes ``variants[m % len(variants)]`` as its prompt and
    ``member_chains[m % len(member_chains)]`` as its model chain, or the
    forecast role's chain when ``member_chains`` is empty. A member's chain is
    walked by :class:`~augur.agents.calls.CallRecorder` like any other: in a
    scored run a route below the floor is refused for that member, and the
    others still answer.
    """

    passes: int
    members: int
    variants: tuple[str, ...]
    member_chains: tuple[tuple[Route, ...], ...]
    forecast_chain: tuple[Route, ...]
    question_concurrency: int

    @classmethod
    def from_settings(cls, settings: Settings, router: Router) -> FleetPlan:
        member_chains = settings.member_chains()
        for chain in member_chains:
            for route in chain:
                try:
                    router.provider(route)
                except KeyError:
                    raise ConfigError(
                        f"AUGUR_MEMBER_MODELS names {route.key}, but no provider was built for "
                        f"endpoint {route.endpoint.name}; build the router with "
                        "augur.llm.factory.build_router so every member's endpoint has one"
                    ) from None
        return cls(
            passes=settings.research_passes,
            members=settings.forecasters,
            variants=settings.forecaster_variants,
            member_chains=member_chains,
            forecast_chain=router.chain("forecast"),
            question_concurrency=min(settings.question_concurrency, MAX_QUESTION_CONCURRENCY),
        )

    @property
    def chains(self) -> tuple[tuple[Route, ...] | None, ...]:
        """The ``chains`` argument of :func:`~augur.agents.forecaster.draw_forecasts`.

        ``(None,)`` without member chains, so every member uses the forecast
        role's chain exactly as before the fleet.
        """
        return self.member_chains or (None,)

    def member(self, m: int) -> MemberPlan:
        variant = self.variants[m % len(self.variants)]
        if self.member_chains:
            return MemberPlan(m, variant, self.member_chains[m % len(self.member_chains)], True)
        return MemberPlan(m, variant, self.forecast_chain, False)

    def plans(self) -> list[MemberPlan]:
        return [self.member(m) for m in range(self.members)]

    def extra_routes(self) -> list[Route]:
        """Member routes outside the router's role chains (the probe checks these too)."""
        seen: list[Route] = []
        for chain in self.member_chains:
            for route in chain:
                if route not in seen:
                    seen.append(route)
        return seen

    def describe(self) -> str:
        """One line for the run's printout."""
        research = "one research pass" if self.passes == 1 else "two research passes"
        groups: dict[tuple[str, str], int] = {}
        for p in self.plans():
            key = (p.variant, p.chain[0].key if p.own_chain else "")
            groups[key] = groups.get(key, 0) + 1
        members = ", ".join(
            f"{count} {variant}" + (f" on {first}" if first else "")
            for (variant, first), count in groups.items()
        )
        at_once = self.question_concurrency
        return (
            f"fleet: {research}; {self.members} forecasters ({members}); "
            f"{at_once} question{'s' if at_once != 1 else ''} at a time"
        )
