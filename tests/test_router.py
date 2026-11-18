"""Model chains: fallback, the ledger of switches, the probe, and the floor on who answered."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from augur.agents.calls import Budget, CallContext, CallRecorder
from augur.agents.forecaster import FORECASTER_SYSTEM
from augur.clock import FixedClock
from augur.errors import (
    AugurError,
    EmptyAnswerError,
    ModelsExhaustedError,
    ModelUnavailableError,
    OutOfCreditError,
    ProviderError,
    RateLimitedError,
)
from augur.leakage import RunMode, scan_ledger
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, LlmResponse, Message
from augur.llm.catalog import Catalog
from augur.llm.endpoints import PRESETS, Route
from augur.llm.probe import measure_limit, record_limit
from augur.llm.router import Router
from augur.metaculus import MetaculusClient
from augur.ratelimit import RateLimiter

from .conftest import T0
from .fakes.llm import RoutedProvider
from .fakes.metaculus import FAKE_TOKEN, FakeMetaculus, binary_post
from .helpers import binary_question, catalog_of
from .test_runner import _run

OR = PRESETS["openrouter"]
GPT = Route(endpoint=OR, model="openai/gpt-5.1")
CLAUDE = Route(endpoint=OR, model="anthropic/claude-opus-5.5")
GEMMA = Route(endpoint=OR, model="google/gemma-4-31b-it:free")
NEMOTRON = Route(endpoint=OR, model="nvidia/nemotron-3.5-lightning:free")
CATALOG = catalog_of(
    (GPT.model, 1.25, 10.0),
    (CLAUDE.model, 4.0, 20.0),
    (GEMMA.model, 0.0, 0.0),
    (NEMOTRON.model, 0.0, 0.0),
)


def router(
    forecast: list[Route],
    research: list[Route] | None = None,
    provider: RoutedProvider | None = None,
    catalog: Catalog = CATALOG,
) -> tuple[Router, RoutedProvider]:
    p = provider or RoutedProvider()
    r = Router(
        {"forecast": forecast, "research": research or [GEMMA]},
        {"openrouter": p},
        catalog,
        FixedClock(T0),
    )
    return r, p


def ledger_with_question(tmp_path: Path) -> tuple[Ledger, str, int]:
    led = Ledger.open(tmp_path / "r.sqlite", FixedClock(T0))
    run_id = led.start_run(
        mode=RunMode.DRY_RUN,
        tournament="t",
        provider="openrouter",
        model="m",
        settings={},
        code_version="test",
    )
    q, post = binary_question(T0)
    led.record_question(run_id, q, T0, post)
    return led, run_id, q.question_id


def forecast_request() -> LlmRequest:
    return LlmRequest(
        model="",
        messages=(
            Message(role="system", content=FORECASTER_SYSTEM),
            Message(role="user", content="q"),
        ),
        max_tokens=100,
    )


def switches(led: Ledger) -> list[tuple[str | None, str | None, str, str]]:
    return led.connection.execute(
        "SELECT from_route, to_route, reason, outcome FROM model_switches ORDER BY recorded_at"
    ).fetchall()


# The chain, one call at a time


async def test_a_rate_limited_model_falls_to_the_next_and_cools_down(tmp_path: Path) -> None:
    r, p = router([GPT, CLAUDE])
    p.failures[GPT.model] = RateLimitedError("429", status_code=429, retry_after=45.0)
    led, run_id, qid = ledger_with_question(tmp_path)
    calls = CallRecorder(r, led, FixedClock(T0), Budget(1.0, 1.0))
    ctx = CallContext(run_id, qid, "forecaster", 0)

    done = await calls.complete(ctx, forecast_request(), [])
    assert done.route == CLAUDE
    assert switches(led) == [(GPT.key, CLAUDE.key, "rate_limited", "switched")]
    rows = led.connection.execute(
        "SELECT model, error IS NOT NULL, endpoint_host FROM llm_calls ORDER BY started_at, error"
    ).fetchall()
    assert sorted(rows) == [
        (CLAUDE.model, 0, "openrouter.ai"),
        (GPT.model, 1, "openrouter.ai"),
    ], "the failed attempt is a ledger row too"
    assert "cooling down" in str(r.unavailable(GPT))

    again = await calls.complete(ctx, forecast_request(), [])
    assert again.route == CLAUDE
    assert p.models_called().count(GPT.model) == 1, "a cooling route is not called again"
    assert len(switches(led)) == 1, "a switch is recorded when it happens, not on every call"
    led.close()


async def test_out_of_credit_disables_the_route_for_the_run(tmp_path: Path) -> None:
    r, p = router([GPT, CLAUDE])
    p.failures[GPT.model] = OutOfCreditError("402", status_code=402)
    led, run_id, qid = ledger_with_question(tmp_path)
    calls = CallRecorder(r, led, FixedClock(T0), Budget(1.0, 1.0))
    await calls.complete(CallContext(run_id, qid, "forecaster", 0), forecast_request(), [])
    assert "out_of_credit" in str(r.unavailable(GPT))
    assert switches(led) == [(GPT.key, CLAUDE.key, "out_of_credit", "switched")]
    led.close()


async def test_an_exhausted_chain_is_recorded_and_raised(tmp_path: Path) -> None:
    r, p = router([GPT, CLAUDE])
    p.failures[GPT.model] = OutOfCreditError("402", status_code=402)
    p.failures[CLAUDE.model] = ModelUnavailableError("404", status_code=404)
    led, run_id, qid = ledger_with_question(tmp_path)
    calls = CallRecorder(r, led, FixedClock(T0), Budget(1.0, 1.0))
    with pytest.raises(ModelsExhaustedError, match="AUGUR_FORECAST_MODELS"):
        await calls.complete(CallContext(run_id, qid, "forecaster", 0), forecast_request(), [])
    assert switches(led) == [
        (GPT.key, CLAUDE.key, "out_of_credit", "switched"),
        (CLAUDE.key, None, "unavailable", "exhausted"),
    ]
    led.close()


async def test_an_error_that_is_not_a_fallback_fails_only_the_call(tmp_path: Path) -> None:
    r, p = router([GPT, CLAUDE])
    p.failures[GPT.model] = ProviderError("bad parameters")
    led, run_id, qid = ledger_with_question(tmp_path)
    calls = CallRecorder(r, led, FixedClock(T0), Budget(1.0, 1.0))
    with pytest.raises(ProviderError, match="bad parameters"):
        await calls.complete(CallContext(run_id, qid, "forecaster", 0), forecast_request(), [])
    assert CLAUDE.model not in p.models_called()
    assert r.unavailable(GPT) is None
    led.close()


async def test_an_unpriced_model_on_a_billed_host_is_never_called(tmp_path: Path) -> None:
    unlisted = Route(endpoint=OR, model="vendor/brand-new-model")
    r, p = router([unlisted, GEMMA])
    led, run_id, qid = ledger_with_question(tmp_path)
    budget = Budget(0.01, 0.01)
    calls = CallRecorder(r, led, FixedClock(T0), budget)
    done = await calls.complete(CallContext(run_id, qid, "forecaster", 0), forecast_request(), [])
    assert done.route == GEMMA
    assert unlisted.model not in p.models_called()
    assert switches(led) == [(unlisted.key, GEMMA.key, "unpriced", "switched")]
    assert budget.spent_run == 0.0, "a free model is not priced at frontier rates"
    led.close()


async def test_a_reply_without_usage_is_charged_the_worst_case(tmp_path: Path) -> None:
    class NoUsage(RoutedProvider):
        async def complete(self, request: LlmRequest) -> LlmResponse:
            self.requests.append(request)
            return LlmResponse(text="Probability: 40%", model=request.model)

    r, _ = router([GPT], provider=NoUsage())
    led, run_id, qid = ledger_with_question(tmp_path)
    budget = Budget(1.0, 1.0)
    calls = CallRecorder(r, led, FixedClock(T0), budget)
    await calls.complete(CallContext(run_id, qid, "forecaster", 0), forecast_request(), [])
    assert budget.spent_run > 100 * 10.0 / 1e6, "at least max_tokens at the output price"
    led.close()


async def test_each_call_role_uses_its_own_chain(tmp_path: Path) -> None:
    r, _ = router([GPT], research=[GEMMA])
    led, run_id, qid = ledger_with_question(tmp_path)
    calls = CallRecorder(r, led, FixedClock(T0), Budget(1.0, 1.0))
    planner = await calls.complete(CallContext(run_id, qid, "planner"), forecast_request(), [])
    brief = await calls.complete(CallContext(run_id, qid, "researcher"), forecast_request(), [])
    member = await calls.complete(CallContext(run_id, qid, "forecaster", 0), forecast_request(), [])
    assert (planner.route, brief.route, member.route) == (GEMMA, GEMMA, GPT)
    with pytest.raises(AugurError, match="no model chain"):
        await calls.complete(CallContext(run_id, qid, "judge"), forecast_request(), [])
    led.close()


# The floor in a live run against the fake server


@pytest.fixture
def fake() -> FakeMetaculus:
    server = FakeMetaculus(clock=FixedClock(T0))
    server.add_post(binary_post(500, 501, title="Will the fake event happen?"), "minibench")
    return server


@pytest.fixture
async def http(fake: FakeMetaculus) -> AsyncIterator[httpx.AsyncClient]:
    async with fake.client() as client:
        yield client


def _client(http: httpx.AsyncClient) -> MetaculusClient:
    return MetaculusClient(
        "https://www.metaculus.com/api",
        SecretStr(FAKE_TOKEN),
        http,
        RateLimiter(1000, burst=1000, max_concurrent=4),
    )


async def test_a_fallback_below_the_tier_never_submits(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    """The frontier key runs dry mid-run; the free model behind it must not post to MiniBench."""
    r, p = router([GPT, GEMMA], research=[GEMMA])
    p.failures[GPT.model] = OutOfCreditError("402 Insufficient credits", status_code=402)
    out = io.StringIO()
    report, ledger = await _run(
        tmp_path, client=_client(http), router=r, submit=True, tournament="minibench", out=out
    )
    [q] = [x for x in report.reports if x.question.question_id == 501]
    assert q.status == "skipped"
    assert q.reason.startswith("submit floor:")
    assert fake.forecasts == [], "nothing was submitted"
    assert fake.comments == []
    assert GEMMA.model not in p.models_called(FORECASTER_SYSTEM), "the free model never forecast"
    rows = ledger.connection.execute(
        "SELECT from_route, to_route, reason, outcome FROM model_switches"
    ).fetchall()
    assert rows, "every refusal is in the ledger"
    assert {row[3] for row in rows} == {"refused_by_floor"}
    assert {(row[0], row[1], row[2]) for row in rows} == {(GPT.key, GEMMA.key, "out_of_credit")}
    assert ledger.count("submissions") == 0
    ledger.close()
    assert scan_ledger(tmp_path / "l.sqlite") == []


async def test_the_same_fallback_answers_in_the_practice_area(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    r, p = router([GPT, GEMMA], research=[GEMMA])
    p.failures[GPT.model] = OutOfCreditError("402", status_code=402)
    report, ledger = await _run(
        tmp_path, client=_client(http), router=r, submit=True, tournament="bot-testing-area"
    )
    assert [x.question.question_id for x in report.reports if x.status == "forecast"] == [501]
    assert len(fake.forecasts) == 1
    assert "by google/gemma-4-31b-it:free." in fake.comments[0]["text"]
    outcomes = {row[0] for row in ledger.connection.execute("SELECT outcome FROM model_switches")}
    assert outcomes == {"switched"}
    ledger.close()


async def test_the_last_check_catches_a_below_tier_answer_the_router_let_through(
    tmp_path: Path,
    fake: FakeMetaculus,
    http: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the router's floor were lost, the check on who answered still refuses to post."""
    monkeypatch.setattr(Router, "set_floor", lambda self, role, name, floor: None)
    r, p = router([GPT, GEMMA], research=[GEMMA])
    p.failures[GPT.model] = OutOfCreditError("402", status_code=402)
    report, ledger = await _run(
        tmp_path, client=_client(http), router=r, submit=True, tournament="minibench"
    )
    [q] = [x for x in report.reports if x.question.question_id == 501]
    assert GEMMA.model in p.models_called(FORECASTER_SYSTEM), "the router fell through"
    assert q.status == "skipped"
    assert "below the frontier-2026-11 tier; nothing posted" in q.reason
    assert fake.forecasts == []
    assert fake.comments == []
    refused = ledger.connection.execute(
        "SELECT from_route, reason FROM model_switches WHERE outcome = 'refused_by_floor'"
    ).fetchall()
    assert refused == [(GEMMA.key, "below_floor_at_submit")]
    ledger.close()


async def test_a_frontier_fallback_still_submits(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    r, p = router([GPT, CLAUDE], research=[GEMMA])
    p.failures[GPT.model] = RateLimitedError("429", status_code=429, retry_after=90.0)
    report, ledger = await _run(
        tmp_path, client=_client(http), router=r, submit=True, tournament="minibench"
    )
    assert [x.question.question_id for x in report.reports if x.status == "forecast"] == [501]
    assert len(fake.forecasts) == 1
    assert "by anthropic/claude-opus-5.5." in fake.comments[0]["text"]
    ledger.close()
    assert scan_ledger(tmp_path / "l.sqlite") == []


def test_the_ledger_scan_flags_a_submission_below_the_floor(tmp_path: Path) -> None:
    path = tmp_path / "bad.sqlite"
    led = Ledger.open(path, FixedClock(T0))
    run_id = led.start_run(
        mode=RunMode.LIVE,
        tournament="minibench",
        provider="openrouter",
        model=GEMMA.model,
        settings={},
        code_version="test",
    )
    q, post = binary_question(T0)
    led.record_question(run_id, q, T0, post)
    call_id = led.record_call(
        run_id=run_id,
        question_id=q.question_id,
        role="forecaster",
        member=0,
        provider="openrouter",
        model=GEMMA.model,
        model_release_date=None,
        endpoint_host="openrouter.ai",
        request_hash="h",
        request={},
        response_text="Probability: 40%",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1,
        error=None,
        started_at=T0,
        source_ids=[],
    )
    led.record_forecast(
        run_id=run_id,
        question_id=q.question_id,
        kind="member",
        member=0,
        method="draw",
        probability=0.4,
        reasoning="r",
        call_id=call_id,
    )
    led.record_submission(
        run_id=run_id,
        question_id=q.question_id,
        post_id=q.post_id,
        kind="forecast",
        payload={},
        status="submitted",
        http_status=201,
        response_excerpt=None,
    )
    led.close()
    [violation] = scan_ledger(path)
    assert violation.kind == "below_floor"
    assert GEMMA.model in violation.detail


# The opening probe


async def test_the_probe_disables_a_dead_route_before_any_question(tmp_path: Path) -> None:
    dead = Route(endpoint=OR, model="nvidia/nemotron-3.5-lightning:free")
    r, p = router([dead, GEMMA], research=[GEMMA])
    p.failures[dead.model] = ModelUnavailableError("404 No endpoints found", status_code=404)
    probe_provider = RoutedProvider(failures=dict(p.failures))
    out = io.StringIO()
    report, ledger = await _run(
        tmp_path, client=None, router=r, probe_providers={"openrouter": probe_provider}, out=out
    )
    assert report.count("forecast") >= 1
    assert dead.model not in p.models_called(), "the run never waited on the dead route"
    probes = ledger.connection.execute(
        "SELECT route, outcome, in_tier, roles, run_id FROM model_probes ORDER BY route"
    ).fetchall()
    assert probes == [
        (GEMMA.key, "answered", 0, "research,forecast", report.run_id),
        (dead.key, "unavailable", 0, "forecast", report.run_id),
    ]
    assert "probe:" in out.getvalue()
    assert ledger.count("model_switches") == 0
    ledger.close()


async def test_a_scored_run_skips_everything_when_no_frontier_model_answers_the_probe(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    r, p = router([GPT, GEMMA], research=[GEMMA])
    probe_provider = RoutedProvider(failures={GPT.model: OutOfCreditError("402", status_code=402)})
    report, ledger = await _run(
        tmp_path,
        client=_client(http),
        router=r,
        probe_providers={"openrouter": probe_provider},
        submit=True,
        tournament="minibench",
    )
    [q] = [x for x in report.reports if x.question.question_id == 501]
    assert q.status == "skipped"
    assert "no frontier-2026-11 model answered the probe" in q.reason
    assert p.requests == [], "no question cost a model call"
    assert fake.forecasts == []
    ledger.close()


# Re-forecasting in the practice area


# Measuring a free tier's limit


async def test_measure_limit_stops_at_the_first_refusal(tmp_path: Path) -> None:
    class Limited(RoutedProvider):
        async def complete(self, request: LlmRequest) -> LlmResponse:
            self.requests.append(request)
            if len(self.requests) > 3:
                raise RateLimitedError(
                    "429 free-models-per-min",
                    status_code=429,
                    retry_after=41.0,
                    limit_headers={"x-ratelimit-limit": "20"},
                )
            return LlmResponse(text="ok", model=request.model, input_tokens=9, output_tokens=1)

    provider = Limited()
    result = await measure_limit(GEMMA, provider, CATALOG, cap=10)
    assert (result.sent, result.ok, result.first_refusal_at) == (4, 3, 4)
    assert (result.refusal_status, result.refusal_kind) == (429, "rate_limited")
    assert result.refusal_retry_after == 41.0
    assert result.limit_headers == {"x-ratelimit-limit": "20"}
    assert result.input_tokens == 27
    led = Ledger.open(tmp_path / "l.sqlite", FixedClock(T0))
    record_limit(led, result, "probe_x", T0)
    assert led.count("limit_tests") == 1
    led.close()

    with pytest.raises(ProviderError, match="not free"):
        await measure_limit(GPT, provider, CATALOG, cap=10)
    with pytest.raises(ProviderError, match="local"):
        await measure_limit(
            Route(endpoint=PRESETS["ollama"], model="qwen2.5:7b"), provider, CATALOG, cap=10
        )


async def test_an_empty_answer_moves_this_call_on_but_keeps_the_route(tmp_path: Path) -> None:
    r, p = router([NEMOTRON, GEMMA])
    p.failures[NEMOTRON.model] = EmptyAnswerError("no text (finish_reason 'length')")
    led, run_id, qid = ledger_with_question(tmp_path)
    calls = CallRecorder(r, led, FixedClock(T0), Budget(1.0, 1.0))
    done = await calls.complete(CallContext(run_id, qid, "forecaster", 0), forecast_request(), [])
    assert done.route == GEMMA
    assert r.unavailable(NEMOTRON) is None, "a longer request may still get an answer"
    assert switches(led) == [(NEMOTRON.key, GEMMA.key, "empty", "switched")]
    led.close()
