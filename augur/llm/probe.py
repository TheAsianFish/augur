"""``augur models probe``: which listed models answer right now, and what their limits are.

Run at the start of every forecast run. Each route in the chains gets one
short request, with no retries, so the first refusal is seen as it is. The
result (answered or why not, latency, listed price and listing date, whether
the route may answer a scored forecast, and any rate-limit headers) is a
``model_probes`` row; a route that did not answer is disabled for the run,
so no question waits on a dead key.

Free tiers do not publish their real limits consistently, so
:func:`measure_limit` measures one: requests sent back to back until the
first refusal (or a cap), recording how many went through, in how long, and
what the refusal looked like. It runs only on routes that cost nothing.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from pydantic import SecretStr

from augur.clock import to_iso
from augur.errors import EmptyAnswerError, FallbackError, ProviderError
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, LlmResponse, Message, Provider
from augur.llm.catalog import Catalog, Price, price_for, release_bound
from augur.llm.endpoints import Endpoint, Route
from augur.llm.registry import in_frontier_tier
from augur.log import get_logger

log = get_logger(__name__)

PROBE_PROMPT = "Reply with the single word: ok"
PROBE_MAX_TOKENS = 64
# A one-word answer slower than this means the route is not fit to forecast
# inside a question's window; the probe gives up on it.
PROBE_TIMEOUT_S = 30.0
ANSWERING = frozenset({"answered", "empty"})


def probe_request(model: str, max_tokens: int = PROBE_MAX_TOKENS) -> LlmRequest:
    return LlmRequest(
        model=model,
        messages=(Message(role="user", content=PROBE_PROMPT),),
        temperature=0.0,
        max_tokens=max_tokens,
    )


def new_session_id() -> str:
    return f"probe_{uuid.uuid4().hex[:16]}"


@dataclass
class ProbeResult:
    route: Route
    roles: list[str]
    outcome: str
    detail: str
    price: Price | None
    listed_on: str | None
    release_basis: str
    in_tier: bool
    http_status: int | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float = 0.0
    served_model: str | None = None
    limit_headers: dict[str, str] = field(default_factory=dict)
    retry_after: float | None = None

    @property
    def answers(self) -> bool:
        return self.outcome in ANSWERING


def _outcome(exc: ProviderError) -> str:
    return exc.kind if isinstance(exc, FallbackError) else "error"


def _cost(response: LlmResponse, price: Price) -> float:
    if response.cost_usd is not None:
        return response.cost_usd
    return price.cost(response.input_tokens, response.output_tokens)


async def probe_routes(
    routes: Sequence[Route],
    roles: Mapping[Route, Sequence[str]],
    providers: Mapping[str, Provider],
    catalog: Catalog,
) -> list[ProbeResult]:
    """One request per route, one route at a time."""
    results: list[ProbeResult] = []
    for route in routes:
        price = price_for(route, catalog)
        release, basis = release_bound(route, catalog)
        result = ProbeResult(
            route=route,
            roles=list(roles.get(route, [])),
            outcome="answered",
            detail="",
            price=price,
            listed_on=release.isoformat() if release else None,
            release_basis=basis,
            in_tier=in_frontier_tier(route),
        )
        results.append(result)
        if price is None:
            result.outcome = "unpriced"
            result.detail = (
                f"{route.endpoint.name} bills and OpenRouter's list has no price for "
                f"{route.model}; not called"
            )
            continue
        provider = providers[route.endpoint.name]
        t0 = time.perf_counter()
        try:
            response = await provider.complete(probe_request(route.model))
        except ProviderError as exc:
            result.latency_ms = int((time.perf_counter() - t0) * 1000)
            result.outcome = _outcome(exc)
            result.detail = str(exc)[:300]
            if isinstance(exc, FallbackError):
                result.http_status = exc.status_code
                result.limit_headers = exc.limit_headers
                result.retry_after = exc.retry_after
            log.warning("probe: route did not answer", route=route.key, outcome=result.outcome)
            continue
        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        result.http_status = 200
        result.input_tokens = response.input_tokens
        result.output_tokens = response.output_tokens
        result.cost_usd = _cost(response, price)
        result.served_model = response.served_model
        result.limit_headers = response.limit_headers
        result.detail = " ".join(response.text.split())[:80]
        log.info("probe: route answered", route=route.key, latency_ms=result.latency_ms)
    return results


# Fields of OpenRouter's ``GET /key`` worth recording. Never the label (it can
# carry part of the key) or account and workspace ids.
_KEY_FIELDS = ("limit", "limit_remaining", "usage", "is_free_tier", "expires_at")


async def openrouter_key_status(
    http: httpx.AsyncClient, endpoint: Endpoint, key: SecretStr | None
) -> dict[str, object] | None:
    """The key's credit limit, spend and free-model daily allowance, or ``None``."""
    if key is None or endpoint.host != "openrouter.ai":
        return None
    try:
        resp = await http.get(
            f"{endpoint.base_url.rstrip('/')}/key",
            headers={"Authorization": f"Bearer {key.get_secret_value()}"},
            timeout=30.0,
        )
        data = resp.json().get("data") if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    if not isinstance(data, dict):
        return None
    status: dict[str, object] = {k: data.get(k) for k in _KEY_FIELDS}
    daily = data.get("free_model_daily_requests")
    if isinstance(daily, dict):
        status["free_model_daily_requests"] = {
            k: daily.get(k) for k in ("used", "limit", "remaining")
        }
    return status


def record_probes(
    ledger: Ledger, results: Sequence[ProbeResult], session_id: str, run_id: str | None
) -> None:
    for r in results:
        ledger.record_probe(
            {
                "run_id": run_id,
                "session_id": session_id,
                "route": r.route.key,
                "endpoint": r.route.endpoint.name,
                "endpoint_host": r.route.endpoint.host,
                "model": r.route.model,
                "roles": ",".join(r.roles),
                "outcome": r.outcome,
                "http_status": r.http_status,
                "latency_ms": r.latency_ms,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "usd_per_million_input": r.price.usd_per_million_input if r.price else None,
                "usd_per_million_output": r.price.usd_per_million_output if r.price else None,
                "price_basis": r.price.basis if r.price else "unpriced",
                "listed_on": r.listed_on,
                "release_basis": r.release_basis,
                "in_tier": int(r.in_tier),
                "served_model": r.served_model,
                "limit_headers_json": json.dumps(r.limit_headers, sort_keys=True),
                "detail": r.detail,
            }
        )


@dataclass
class LimitResult:
    route: Route
    cap: int
    sent: int = 0
    ok: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    first_refusal_at: int | None = None
    refusal_status: int | None = None
    refusal_kind: str | None = None
    refusal_retry_after: float | None = None
    refusal_excerpt: str | None = None
    limit_headers: dict[str, str] = field(default_factory=dict)

    @property
    def requests_per_minute(self) -> float | None:
        """Accepted requests per minute over the burst (a lower bound if never refused)."""
        if self.seconds <= 0:
            return None
        return self.ok * 60.0 / self.seconds


async def measure_limit(
    route: Route, provider: Provider, catalog: Catalog, cap: int
) -> LimitResult:
    """Send up to ``cap`` one-line requests back to back; stop at the first refusal.

    ``provider`` must have no client-side rate limit and no retries, so the
    server's own limit is what stops the burst. Only free routes are measured.
    """
    price = price_for(route, catalog)
    if price is None or price.usd_per_million_input or price.usd_per_million_output:
        raise ProviderError(
            f"{route.key} is not free; limits are measured only on models that cost nothing"
        )
    if route.endpoint.is_local:
        raise ProviderError(f"{route.key} is local; there is no host limit to measure")
    result = LimitResult(route=route, cap=cap)
    t0 = time.perf_counter()
    for i in range(1, cap + 1):
        result.sent = i
        try:
            response = await provider.complete(probe_request(route.model, max_tokens=8))
        except EmptyAnswerError:
            result.ok += 1  # accepted: a reasoning model spent the tokens thinking
            continue
        except ProviderError as exc:
            result.first_refusal_at = i
            result.refusal_kind = _outcome(exc)
            result.refusal_excerpt = str(exc)[:300]
            if isinstance(exc, FallbackError):
                result.refusal_status = exc.status_code
                result.limit_headers = exc.limit_headers
                result.refusal_retry_after = exc.retry_after
            break
        result.ok += 1
        result.input_tokens += response.input_tokens or 0
        result.output_tokens += response.output_tokens or 0
        if response.limit_headers:
            result.limit_headers = response.limit_headers
    result.seconds = round(time.perf_counter() - t0, 3)
    return result


def record_limit(ledger: Ledger, result: LimitResult, session_id: str, started: datetime) -> None:
    ledger.record_limit_test(
        {
            "session_id": session_id,
            "route": result.route.key,
            "endpoint_host": result.route.endpoint.host,
            "requests_cap": result.cap,
            "requests_sent": result.sent,
            "requests_ok": result.ok,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "seconds": result.seconds,
            "first_refusal_at": result.first_refusal_at,
            "refusal_status": result.refusal_status,
            "refusal_kind": result.refusal_kind,
            "refusal_retry_after": result.refusal_retry_after,
            "refusal_excerpt": result.refusal_excerpt,
            "limit_headers_json": json.dumps(result.limit_headers, sort_keys=True),
            "started_at": to_iso(started),
        }
    )
