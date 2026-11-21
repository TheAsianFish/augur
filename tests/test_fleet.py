"""The fleet under load: limits per provider, the per-question budget, members that fail.

Every model request here goes through the real
:class:`~augur.llm.openai_compat.OpenAICompatProvider` (and its own
:class:`~augur.ratelimit.RateLimiter`, built from the endpoint's ``rpm`` and
``max_concurrent``) to a fake OpenAI-compatible host on
``httpx.MockTransport``, which records when each request started and ended.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from augur.agents.forecaster import VARIANTS, system_prompt
from augur.agents.researcher import BRIEF_SYSTEM, SUBQ_SYSTEM
from augur.clock import FixedClock
from augur.config import Settings
from augur.errors import ConfigError
from augur.fleet import FleetPlan, bounded_map
from augur.ledger import Ledger
from augur.llm.catalog import Catalog
from augur.llm.endpoints import PRESETS, Endpoint, Route
from augur.llm.openai_compat import OpenAICompatProvider, limiter_for
from augur.llm.router import Router
from augur.metaculus import MetaculusClient
from augur.questions import question_from_post
from augur.ratelimit import RateLimiter
from augur.runner import ForecastRun, RunReport

from .conftest import T0
from .fakes.llm import RoutedProvider
from .fakes.metaculus import FAKE_TOKEN, FakeMetaculus
from .fakes.metaculus import binary_post as metaculus_post
from .fakes.search import FixedSearcher
from .helpers import binary_post, catalog_of, free_catalog

# -- a fake OpenAI-compatible host ---------------------------------------------

Answer = str | httpx.Response
Forecast = Callable[[str, int, str], Answer]
"""``(question title, seed, model)`` -> the forecaster's answer, or an HTTP error."""


def _default_forecast(title: str, seed: int, model: str) -> Answer:
    return f"Base rates weighed (draw {seed}).\nProbability: {30 + seed}%"


@dataclass
class Hit:
    start: float
    end: float
    model: str
    seed: int | None
    system: str
    user: str


class FakeHost:
    """Answers each prompt by its role; records every request's start and end."""

    def __init__(
        self,
        *,
        latency: float = 0.0,
        forecast: Forecast = _default_forecast,
        usage: bool = True,
        on_request: Callable[[], None] | None = None,
    ) -> None:
        self.latency = latency
        self.forecast = forecast
        self.usage = usage
        self.on_request = on_request
        self.hits: list[Hit] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def handle(self, request: httpx.Request) -> httpx.Response:
        loop = asyncio.get_running_loop()
        start = loop.time()
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.on_request is not None:
                self.on_request()
            await asyncio.sleep(self.latency)
            body = json.loads(request.content)
            system = body["messages"][0]["content"]
            user = body["messages"][-1]["content"]
            answer = self._answer(body, system, user)
        finally:
            self.in_flight -= 1
        self.hits.append(Hit(start, loop.time(), body["model"], body.get("seed"), system, user))
        if isinstance(answer, httpx.Response):
            return answer
        data: dict[str, Any] = {"choices": [{"message": {"content": answer}}]}
        if self.usage:
            data["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
        return httpx.Response(200, json=data)

    def _answer(self, body: dict[str, Any], system: str, user: str) -> Answer:
        if "superforecaster" in system:
            title = user.split("\n", 1)[0].removeprefix("Question: ")
            return self.forecast(title, int(body.get("seed") or 0), body["model"])
        if system == SUBQ_SYSTEM.format(n=3):
            return '[{"sub_question": "What would overturn it?", "query": "counter evidence"}]'
        if system == BRIEF_SYSTEM:
            return "The status quo holds [S1]."
        if system.startswith("You plan web research"):
            return '["first query"]'
        return "ok"

    def forecaster_hits(self) -> list[Hit]:
        return [h for h in self.hits if "superforecaster" in h.system]


def _endpoint(
    rpm: float = 60_000, max_concurrent: int = 8, billing: str = "free", name: str = "fake"
) -> Endpoint:
    return Endpoint.model_validate(
        {
            "name": name,
            "base_url": f"https://{name}.example/v1",
            "billing": billing,
            "rpm": rpm,
            "max_concurrent": max_concurrent,
        }
    )


def _settings(endpoint: Endpoint, **extra: object) -> Settings:
    return Settings.model_validate(
        {
            "provider": "replay",
            "model": "qwen2.5:7b",
            "endpoints": dict(PRESETS) | {endpoint.name: endpoint},
        }
        | extra
    )


def _samples(directory: Path, n: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        post = binary_post(100 + i, 200 + i, opened=T0, title=f"Question number {i}")
        (directory / f"post_{i:02d}.json").write_text(json.dumps(post), encoding="utf-8")
    return directory


@dataclass
class Setup:
    run: ForecastRun
    ledger: Ledger
    host: FakeHost
    out: io.StringIO


def _setup(
    tmp_path: Path,
    host: FakeHost,
    *,
    endpoint: Endpoint | None = None,
    catalog: Catalog | None = None,
    clock: FixedClock | None = None,
    limiter: RateLimiter | None = None,
    http: httpx.AsyncClient,
    **settings: Any,
) -> Setup:
    endpoint = endpoint or _endpoint()
    clock = clock or FixedClock(T0)
    provider = OpenAICompatProvider(http, endpoint, None, limiter=limiter, max_retries=0)
    route = Route(endpoint=endpoint, model="model-a")
    router = Router(
        {"research": [route], "forecast": [route]},
        {endpoint.name: provider},
        catalog or free_catalog(),
        clock,
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.open(tmp_path / "l.sqlite", clock)
    out = io.StringIO()
    run = ForecastRun(
        settings=_settings(endpoint, **settings),
        clock=clock,
        ledger=ledger,
        router=router,
        searchers=[FixedSearcher()],
        client=None,
        out=out,
    )
    return Setup(run, ledger, host, out)


async def _dry_run(s: Setup, samples: Path) -> RunReport:
    return await s.run.run("minibench", submit=False, samples=samples, max_questions=10)


def _outcomes(ledger: Ledger, run_id: str) -> dict[int, tuple[str, str]]:
    rows = ledger.connection.execute(
        "SELECT question_id, status, reason FROM question_outcomes WHERE run_id = ?", (run_id,)
    ).fetchall()
    return {qid: (status, reason) for qid, status, reason in rows}


# -- limits under load ----------------------------------------------------------


def _most_in_window(starts: Sequence[float], window: float) -> int:
    ordered = sorted(starts)
    most, j = 0, 0
    for i, t in enumerate(ordered):
        while ordered[j] < t - window:
            j += 1
        most = max(most, i - j + 1)
    return most


async def test_limits_hold_under_load_across_concurrent_questions(tmp_path: Path) -> None:
    """6 questions x (plan, brief, 5 members), 3 questions at a time, one endpoint."""
    rate, burst, slots = 60.0, 4, 2
    endpoint = _endpoint(rpm=rate * 60, max_concurrent=slots)
    limiter = limiter_for(endpoint)  # what the provider builds for itself
    assert (limiter._rate, int(limiter._burst)) == (rate, burst)
    host = FakeHost(latency=0.02)
    async with httpx.AsyncClient(transport=httpx.MockTransport(host.handle)) as http:
        s = _setup(tmp_path, host, endpoint=endpoint, http=http, question_concurrency=3)
        report = await _dry_run(s, _samples(tmp_path / "samples", 6))

    assert report.count("forecast") == 6
    assert len(host.hits) == 6 * 7
    # The concurrency cap was reached and never passed.
    assert host.max_in_flight == slots
    events = sorted(
        [(h.start, 1) for h in host.hits] + [(h.end, -1) for h in host.hits],
        key=lambda e: (e[0], e[1]),
    )
    flying = 0
    for _, step in events:
        flying += step
        assert flying <= slots
    # The bucket: no window admits more than rate x window + burst requests
    # (plus the event loop's few milliseconds between a token and the request).
    starts = [h.start for h in host.hits]
    jitter = 0.01
    for window in (0.05, 0.1, 0.25, 0.5):
        assert _most_in_window(starts, window) <= rate * (window + jitter) + burst
    # Under load the bucket, not the latency, set the pace.
    span = max(starts) - min(starts)
    assert span >= (len(starts) - burst) / rate * 0.9
    # Questions really ran at the same time: they interleaved on the host.
    order = [h.user.split("\n", 1)[0] for h in host.hits]
    first_done = max(i for i, t in enumerate(order) if t.endswith("number 0"))
    assert any(t.endswith("number 1") for t in order[:first_done])
    rows = _outcomes(s.ledger, report.run_id)
    assert len(rows) == 6
    assert {st for st, _ in rows.values()} == {"forecast"}
    s.ledger.close()


async def test_one_question_at_a_time_keeps_the_old_order(tmp_path: Path) -> None:
    host = FakeHost()
    async with httpx.AsyncClient(transport=httpx.MockTransport(host.handle)) as http:
        s = _setup(tmp_path, host, http=http)
        report = await _dry_run(s, _samples(tmp_path / "samples", 3))
    titles = [h.user.split("\n", 1)[0] for h in host.hits]
    # Each question's requests finish before the next question's start.
    blocks = [t for i, t in enumerate(titles) if i == 0 or titles[i - 1] != t]
    assert blocks == [f"Question: Question number {i}" for i in range(3)]
    assert [r.question.question_id for r in report.reports] == [200, 201, 202]
    assert "fleet: one research pass; 5 forecasters (5 base); 1 question at a time" in (
        s.out.getvalue()
    )
    s.ledger.close()


# -- failed members -------------------------------------------------------------


async def test_a_failed_member_does_not_sink_the_question(tmp_path: Path) -> None:
    """Member 2's own model answers 500; member 1's answer cannot be parsed."""

    def forecast(title: str, seed: int, model: str) -> Answer:
        if model == "model-b":
            return httpx.Response(500, json={"error": {"message": "upstream down"}})
        if seed == 1:
            return "I would rather not say."
        return _default_forecast(title, seed, model)

    host = FakeHost(forecast=forecast)
    async with httpx.AsyncClient(transport=httpx.MockTransport(host.handle)) as http:
        s = _setup(
            tmp_path,
            host,
            http=http,
            member_models="fake:model-a;fake:model-a;fake:model-b",
        )
        report = await _dry_run(s, _samples(tmp_path / "samples", 1))
    [only] = report.reports
    assert only.status == "forecast"
    assert sorted(m.member for m in only.members) == [0, 3, 4]
    assert _outcomes(s.ledger, report.run_id) == {200: ("forecast", "")}
    members = s.ledger.connection.execute(
        "SELECT count(*) FROM forecasts WHERE kind = 'member'"
    ).fetchone()[0]
    assert members == 3
    s.ledger.close()


async def test_more_than_half_failing_fails_that_question_only(tmp_path: Path) -> None:
    def forecast(title: str, seed: int, model: str) -> Answer:
        if title.endswith("number 1") and seed in (0, 2, 4):
            return "No number here."
        if title.endswith("number 2") and seed == 3:
            return httpx.Response(400, json={"error": "bad parameter"})
        return _default_forecast(title, seed, model)

    host = FakeHost(forecast=forecast, latency=0.001)
    async with httpx.AsyncClient(transport=httpx.MockTransport(host.handle)) as http:
        s = _setup(tmp_path, host, http=http, question_concurrency=3)
        report = await _dry_run(s, _samples(tmp_path / "samples", 3))
    rows = _outcomes(s.ledger, report.run_id)
    assert rows[200] == ("forecast", "")
    assert rows[202] == ("forecast", "")
    status, reason = rows[201]
    assert status == "failed"
    assert reason.startswith("only 2 of 5 forecasters answered (need 3)")
    s.ledger.close()


# -- the budget under concurrency -----------------------------------------------


def _priced() -> tuple[Endpoint, Catalog]:
    endpoint = _endpoint(billing="openrouter")
    return endpoint, catalog_of(("model-a", 1.0, 10.0))


def _costs(ledger: Ledger) -> dict[int, float]:
    rows = ledger.connection.execute(
        "SELECT question_id, sum(cost_usd) FROM llm_calls GROUP BY question_id"
    ).fetchall()
    return dict(rows)


async def test_the_question_budget_refuses_concurrent_members_past_the_cap(
    tmp_path: Path,
) -> None:
    endpoint, catalog = _priced()
    cap = 0.15  # the research and two members' worst cases fit; a third does not
    host = FakeHost(usage=False, latency=0.001)  # no usage: every call is charged its worst case
    async with httpx.AsyncClient(transport=httpx.MockTransport(host.handle)) as http:
        s = _setup(
            tmp_path,
            host,
            endpoint=endpoint,
            catalog=catalog,
            http=http,
            question_concurrency=3,
            max_cost_per_question=cap,
            max_cost_per_run=100.0,
        )
        report = await _dry_run(s, _samples(tmp_path / "samples", 6))
    costs = _costs(s.ledger)
    assert costs, "some calls were made"
    for spent in costs.values():
        assert spent <= cap + 1e-12
    by_question: dict[str, int] = {}
    for h in host.forecaster_hits():
        title = h.user.split("\n", 1)[0]
        by_question[title] = by_question.get(title, 0) + 1
    # The members of a question started together; the budget let two through.
    assert set(by_question.values()) == {2}
    rows = _outcomes(s.ledger, report.run_id)
    assert len(rows) == 6
    failed = [r for st, r in rows.values() if st == "failed"]
    assert len(failed) == len(by_question) == 6, "every question hit its own cap"
    assert all("budget $0.15 spent" in r for r in failed)
    # A question's own cap fails that question only: every question was tried.
    s.ledger.close()


async def test_the_run_budget_holds_with_questions_in_parallel(tmp_path: Path) -> None:
    endpoint, catalog = _priced()
    host = FakeHost(usage=False, latency=0.001)
    async with httpx.AsyncClient(transport=httpx.MockTransport(host.handle)) as http:
        s = _setup(
            tmp_path,
            host,
            endpoint=endpoint,
            catalog=catalog,
            http=http,
            question_concurrency=3,
            max_cost_per_question=1.0,
            max_cost_per_run=0.40,
        )
        report = await _dry_run(s, _samples(tmp_path / "samples", 6))
    assert report.cost_usd <= 0.40
    assert sum(_costs(s.ledger).values()) <= 0.40 + 1e-12
    rows = _outcomes(s.ledger, report.run_id)
    assert len(rows) == 6
    assert any(st == "failed" and "run budget" in r for st, r in rows.values())
    assert sum(1 for st, r in rows.values() if st == "skipped" and "run budget" in r) >= 3
    s.ledger.close()


async def test_the_run_deadline_holds_with_questions_in_parallel(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    host = FakeHost(on_request=lambda: clock.advance(60))  # each request takes a minute
    async with httpx.AsyncClient(transport=httpx.MockTransport(host.handle)) as http:
        s = _setup(
            tmp_path,
            host,
            clock=clock,
            http=http,
            question_concurrency=3,
            run_deadline_minutes=10,
        )
        report = await _dry_run(s, _samples(tmp_path / "samples", 6))
    rows = _outcomes(s.ledger, report.run_id)
    assert [rows[q][0] for q in range(200, 206)] == ["forecast"] * 3 + ["missed"] * 3
    assert all("too little time left (0.0 min" in rows[q][1] for q in range(203, 206))
    assert report.coverage()["missed"] == 3
    s.ledger.close()


# -- variants, member chains and the floor ---------------------------------------


async def test_members_carry_their_variant_and_their_own_model(tmp_path: Path) -> None:
    host = FakeHost()
    variants = ("base", "outside", "sceptic")
    async with httpx.AsyncClient(transport=httpx.MockTransport(host.handle)) as http:
        s = _setup(
            tmp_path,
            host,
            http=http,
            forecaster_variants=variants,
            member_models="fake:model-a;fake:model-b",
            research_passes=2,
        )
        provider = s.run._router.provider(s.run._router.chain("forecast")[0])
        s.run._probe_providers = {"fake": provider}
        report = await _dry_run(s, _samples(tmp_path / "samples", 1))
    [only] = report.reports
    assert only.status == "forecast"
    q = only.question
    seen = {h.seed or 0: h for h in host.forecaster_hits()}
    assert sorted(seen) == [0, 1, 2, 3, 4]
    for m, hit in seen.items():
        assert hit.system == system_prompt(q, variants[m % 3])
        assert hit.model == ("model-a", "model-b")[m % 2]
    assert VARIANTS["sceptic"] in seen[2].system
    # Two research passes: the sub-question agent, then the brief rewritten.
    assert sum(1 for h in host.hits if h.system == SUBQ_SYSTEM.format(n=3)) == 1
    assert sum(1 for h in host.hits if h.system == BRIEF_SYSTEM) == 2
    printed = s.out.getvalue()
    assert (
        "fleet: two research passes; 5 forecasters (1 base on fake:model-a, 1 outside on "
        "fake:model-b, 1 sceptic on fake:model-a, 1 base on fake:model-b, 1 outside on "
        "fake:model-a); 1 question at a time"
    ) in printed
    probed = s.ledger.connection.execute(
        "SELECT route, roles FROM model_probes ORDER BY route"
    ).fetchall()
    assert [r for r, _ in probed] == ["fake:model-a", "fake:model-b"]
    assert "member" in probed[1][1]
    settings = json.loads(
        s.ledger.connection.execute("SELECT settings_json FROM runs").fetchone()[0]
    )
    assert settings["member_chains"] == [["fake:model-a"], ["fake:model-b"]]
    assert settings["forecaster_variants"] == list(variants)
    assert settings["research_passes"] == 2
    s.ledger.close()


async def test_a_member_chain_below_the_tier_is_refused_by_the_floor(tmp_path: Path) -> None:
    gpt = Route(endpoint=PRESETS["openrouter"], model="openai/gpt-5.1")
    small = "meta-llama/llama-3.1-8b-instruct"
    fake = FakeMetaculus(clock=FixedClock(T0))
    fake.add_post(metaculus_post(500, 501, title="Will the fake event happen?"), "minibench")
    for keys in fake.tournaments.values():
        keys.add("bot-testing-area")
    clock = FixedClock(T0)
    provider = RoutedProvider("openrouter")
    router = Router.single(
        gpt, provider, catalog_of((gpt.model, 1.25, 10.0), (small, 0.1, 0.1)), clock
    )
    ledger = Ledger.open(tmp_path / "l.sqlite", clock)
    async with fake.client() as http:
        client = MetaculusClient(
            "https://www.metaculus.com/api",
            SecretStr(FAKE_TOKEN),
            http,
            RateLimiter(1000, burst=1000, max_concurrent=4),
        )
        run = ForecastRun(
            settings=Settings.model_validate(
                {
                    "provider": "replay",
                    "model": "qwen2.5:7b",
                    "member_models": f"openrouter:{gpt.model};openrouter:{small}",
                }
            ),
            clock=clock,
            ledger=ledger,
            router=router,
            searchers=[FixedSearcher()],
            client=client,
            out=io.StringIO(),
        )
        report = await run.run("minibench", submit=True, samples=None, max_questions=10)
    [question] = [r for r in report.reports if r.question.question_id == 501]
    assert question.status == "forecast"
    assert question.submitted
    assert sorted(m.member for m in question.members) == [0, 2, 4]
    assert {m.route.model for m in question.members} == {gpt.model}
    assert small not in provider.models_called(), "the floor refuses before the call"
    switches = ledger.connection.execute(
        "SELECT member, to_route, outcome FROM model_switches WHERE question_id = 501 "
        "ORDER BY member"
    ).fetchall()
    assert switches == [
        (1, f"openrouter:{small}", "refused_by_floor"),
        (3, f"openrouter:{small}", "refused_by_floor"),
    ]
    assert len(fake.forecasts) == 1
    ledger.close()


def test_a_member_endpoint_without_a_provider_is_refused() -> None:
    endpoint = _endpoint()
    other = _endpoint(name="other")
    route = Route(endpoint=endpoint, model="model-a")
    router = Router.single(route, RoutedProvider("fake"), free_catalog(), FixedClock(T0))
    settings = Settings.model_validate(
        {
            "provider": "replay",
            "endpoints": dict(PRESETS) | {"fake": endpoint, "other": other},
            "member_models": "other:model-x",
        }
    )
    with pytest.raises(ConfigError, match="no provider was built for endpoint other"):
        FleetPlan.from_settings(settings, router)


def test_the_default_plan_is_the_old_run() -> None:
    route = Route(endpoint=_endpoint(), model="model-a")
    router = Router.single(route, RoutedProvider("fake"), free_catalog(), FixedClock(T0))
    plan = FleetPlan.from_settings(Settings(), router)
    assert (plan.passes, plan.variants, plan.chains, plan.question_concurrency) == (
        1,
        ("base",),
        (None,),
        1,
    )
    assert plan.extra_routes() == []
    assert {p.chain for p in plan.plans()} == {(route,)}


async def test_the_estimate_counts_the_second_pass_and_each_members_model(
    tmp_path: Path,
) -> None:
    endpoint, _ = _priced()
    catalog = catalog_of(("model-a", 1.0, 10.0), ("model-b", 2.0, 20.0))

    async def estimate(name: str, **settings: Any) -> float:
        host = FakeHost()
        async with httpx.AsyncClient(transport=httpx.MockTransport(host.handle)) as http:
            s = _setup(
                tmp_path / name, host, endpoint=endpoint, catalog=catalog, http=http, **settings
            )
            value = s.run._estimate_question_cost(question_from_post(binary_post(1, 2, opened=T0)))
            s.ledger.close()
        return value

    one = await estimate("one")
    assert await estimate("two", research_passes=2) > one
    assert await estimate("mixed", member_models="fake:model-a;fake:model-b") > one
    assert await estimate("variants", forecaster_variants=("base", "sceptic")) > one


# -- bounded_map -------------------------------------------------------------------


async def test_bounded_map_keeps_input_order_and_its_bound() -> None:
    flying = most = 0
    started: list[int] = []

    async def work(i: int) -> int:
        nonlocal flying, most
        started.append(i)
        flying += 1
        most = max(most, flying)
        await asyncio.sleep(0.001 * (7 - i % 7))
        flying -= 1
        return i * i

    assert await bounded_map(list(range(20)), work, 3) == [i * i for i in range(20)]
    assert most == 3
    assert started == list(range(20)), "items are taken in input order"
    most = 0
    started.clear()
    assert await bounded_map(list(range(5)), work, 1) == [0, 1, 4, 9, 16]
    assert most == 1
    assert await bounded_map([], work, 4) == []
    with pytest.raises(ValueError, match="at least 1"):
        await bounded_map([1], work, 0)


async def test_bounded_map_raises_the_first_error_and_cancels_the_rest() -> None:
    cancelled: list[int] = []

    async def work(i: int) -> int:
        try:
            if i == 1:
                await asyncio.sleep(0.001)
                raise KeyError("boom")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(i)
            raise
        return i

    with pytest.raises(KeyError, match="boom"):
        await asyncio.wait_for(bounded_map([0, 1, 2, 3], work, 3), timeout=2)
    assert sorted(cancelled) == [0, 2]


async def test_bounded_map_propagates_cancellation_to_every_worker() -> None:
    cancelled: list[int] = []
    running = asyncio.Event()

    async def work(i: int) -> int:
        try:
            running.set()
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(i)
            raise
        return i

    task = asyncio.create_task(bounded_map(list(range(6)), work, 2))
    await running.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(cancelled) == [0, 1], "the two in flight; the rest never started"


# -- configuration ----------------------------------------------------------------


def test_fleet_settings_parse_with_defaults_that_change_nothing() -> None:
    default = Settings.from_env({})
    assert (default.research_passes, default.forecaster_variants) == (1, ("base",))
    assert (default.member_models, default.question_concurrency) == (None, 1)
    assert default.member_chains() == ()
    public = default.public_dict()
    assert public["member_chains"] == []
    assert public["question_concurrency"] == 1

    s = Settings.from_env(
        {
            "AUGUR_RESEARCH_PASSES": "2",
            "AUGUR_FORECASTER_VARIANTS": "base, Outside,sceptic",
            "AUGUR_MEMBER_MODELS": "openrouter:openai/gpt-5.1,openrouter:x/y; ollama:qwen2.5:7b",
            "AUGUR_QUESTION_CONCURRENCY": "4",
            "OPENROUTER_API_KEY": "sk-or-secret-value",
        }
    )
    assert s.research_passes == 2
    assert s.forecaster_variants == ("base", "outside", "sceptic")
    assert [[r.key for r in c] for c in s.member_chains()] == [
        ["openrouter:openai/gpt-5.1", "openrouter:x/y"],
        ["ollama:qwen2.5:7b"],
    ]
    public = s.public_dict()
    assert public["member_models"] == s.member_models
    endpoints = public["endpoints"]
    assert isinstance(endpoints, dict)
    assert "openrouter" in endpoints
    assert "sk-or-secret-value" not in json.dumps(public)


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"AUGUR_FORECASTER_VARIANTS": "base,optimist"}, "unknown variant.*'optimist'.*sceptic"),
        ({"AUGUR_FORECASTER_VARIANTS": "base,,inside"}, "AUGUR_FORECASTER_VARIANTS"),
        ({"AUGUR_MEMBER_MODELS": "openrouter:a/b;;openrouter:c/d"}, "empty member chain"),
        ({"AUGUR_MEMBER_MODELS": "nowhere:a/b"}, "unknown endpoint 'nowhere'"),
        ({"AUGUR_MEMBER_MODELS": "openrouter"}, "endpoint:model"),
        ({"AUGUR_RESEARCH_PASSES": "3"}, "research_passes"),
        ({"AUGUR_QUESTION_CONCURRENCY": "9"}, "question_concurrency"),
        ({"AUGUR_QUESTION_CONCURRENCY": "0"}, "question_concurrency"),
    ],
)
def test_bad_fleet_settings_fail_with_a_clear_error(env: dict[str, str], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        Settings.from_env(env)


def test_unknown_variant_is_refused_without_the_environment_too() -> None:
    with pytest.raises(ValueError, match="forecaster variant"):
        Settings.model_validate({"forecaster_variants": ("base", "optimist")})
