"""The ordered model chain per role, and what this run has learned about each route.

The router does not call models itself: :class:`~augur.agents.calls.CallRecorder`
walks a role's chain through it, so every attempt (including the failed ones)
is costed, budgeted and written to the ledger. The router holds the state that
must be shared by concurrent calls: which routes are disabled for the run
(the probe found them dead, the key is spent) or cooling down after a 429,
and, in a scored run, the floor a route must clear before it may answer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from augur.clock import Clock
from augur.errors import ConfigError
from augur.llm.base import Provider
from augur.llm.catalog import Catalog, Price, price_for, release_bound
from augur.llm.endpoints import ROLES, Role, Route
from augur.log import get_logger

log = get_logger(__name__)

Floor = Callable[[Route], bool]


@dataclass
class _State:
    disabled: str | None = None
    disabled_kind: str = ""
    cooling_until: datetime | None = None
    cooling_reason: str = ""


class Router:
    def __init__(
        self,
        chains: Mapping[Role, Sequence[Route]],
        providers: Mapping[str, Provider],
        catalog: Catalog,
        clock: Clock,
    ) -> None:
        self._chains: dict[Role, tuple[Route, ...]] = {}
        for role in ROLES:
            chain = tuple(chains.get(role, ()))
            if not chain:
                raise ConfigError(f"no models for the {role} role; set AUGUR_{role.upper()}_MODELS")
            self._chains[role] = chain
        missing = {r.endpoint.name for c in self._chains.values() for r in c} - set(providers)
        if missing:
            raise ConfigError(f"no provider for endpoint(s) {', '.join(sorted(missing))}")
        self._providers = dict(providers)
        self.catalog = catalog
        self._clock = clock
        self._state: dict[Route, _State] = {}
        self._floors: dict[Role, tuple[str, Floor]] = {}

    @classmethod
    def single(cls, route: Route, provider: Provider, catalog: Catalog, clock: Clock) -> Router:
        """One route for both roles (tests, and the plain ``AUGUR_PROVIDER``/``AUGUR_MODEL``)."""
        return cls(
            {"research": [route], "forecast": [route]},
            {route.endpoint.name: provider},
            catalog,
            clock,
        )

    # -- configuration -------------------------------------------------------

    def chain(self, role: Role) -> tuple[Route, ...]:
        return self._chains[role]

    def routes(self) -> list[Route]:
        """Every route once, the answering chain first."""
        seen: list[Route] = []
        order: tuple[Role, ...] = ("forecast", "research")
        for role in order:
            for route in self._chains[role]:
                if route not in seen:
                    seen.append(route)
        return seen

    def roles_of(self, route: Route) -> list[Role]:
        return [role for role in ROLES if route in self._chains[role]]

    def provider(self, route: Route) -> Provider:
        return self._providers[route.endpoint.name]

    @property
    def provider_names(self) -> list[str]:
        return sorted({p.name for p in self._providers.values()})

    def price(self, route: Route) -> Price | None:
        return price_for(route, self.catalog)

    def release(self, route: Route) -> tuple[date | None, str]:
        return release_bound(route, self.catalog)

    def set_floor(self, role: Role, name: str, floor: Floor) -> None:
        """From now on a route of ``role`` must pass ``floor`` to be tried."""
        self._floors[role] = (name, floor)

    def floor(self, role: Role) -> tuple[str, Floor] | None:
        return self._floors.get(role)

    # -- what this run has learned ---------------------------------------------

    def disable(self, route: Route, reason: str, kind: str = "unavailable") -> None:
        state = self._state.setdefault(route, _State())
        if state.disabled is None:
            state.disabled = reason
            state.disabled_kind = kind
            log.warning("route disabled for this run", route=route.key, reason=reason)

    def cool(self, route: Route, seconds: float, reason: str) -> None:
        state = self._state.setdefault(route, _State())
        until = self._clock.now() + timedelta(seconds=seconds)
        if state.cooling_until is None or until > state.cooling_until:
            state.cooling_until = until
            state.cooling_reason = reason
            log.warning("route cooling down", route=route.key, seconds=round(seconds, 1))

    def unavailable(self, route: Route) -> str | None:
        """Why ``route`` cannot be tried now, or ``None`` if it can."""
        state = self._state.get(route)
        if state is None:
            return None
        if state.disabled is not None:
            return state.disabled
        if state.cooling_until is not None and self._clock.now() < state.cooling_until:
            return f"cooling down until {state.cooling_until:%H:%M:%S} UTC ({state.cooling_reason})"
        return None

    def unavailable_kind(self, route: Route) -> str:
        """The error kind behind :meth:`unavailable` (``rate_limited``, ``out_of_credit``...)."""
        state = self._state.get(route)
        if state is None:
            return ""
        if state.disabled is not None:
            return state.disabled_kind
        return state.cooling_reason.removeprefix("probe: ")

    def usable(self, role: Role) -> list[Route]:
        """Routes of ``role`` that can be tried now and pass its floor, in order."""
        floor = self._floors.get(role)
        return [
            r
            for r in self._chains[role]
            if self.unavailable(r) is None and (floor is None or floor[1](r))
        ]

    async def aclose(self) -> None:
        for provider in {id(p): p for p in self._providers.values()}.values():
            await provider.aclose()
