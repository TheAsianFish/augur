"""``augur forecast`` end to end against the fake Metaculus server."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from augur.clock import FixedClock
from augur.config import Settings
from augur.errors import AugurError, SubmitFloorError
from augur.leakage import RunMode, scan_ledger
from augur.ledger import Ledger
from augur.llm.base import Provider
from augur.llm.catalog import Catalog
from augur.llm.endpoints import PRESETS, Endpoint, Route
from augur.llm.router import Router
from augur.log import configure_logging
from augur.metaculus import MetaculusClient
from augur.ratelimit import RateLimiter
from augur.runner import SAMPLES_DIR, ForecastRun, RunReport

from .conftest import T0
from .fakes.llm import ScriptedProvider
from .fakes.metaculus import FAKE_TOKEN, MINIBENCH_NUMERIC, FakeMetaculus, binary_post
from .fakes.search import FixedSearcher
from .helpers import binary_question, catalog_of, free_catalog


def _fast_limiter() -> RateLimiter:
    return RateLimiter(1000, burst=1000, max_concurrent=4)


@pytest.fixture
def fake() -> FakeMetaculus:
    server = FakeMetaculus(clock=FixedClock(T0))
    server.add_post(binary_post(500, 501, title="Will the fake event happen?"), "minibench")
    server.add_post(binary_post(510, 511, title="A closed one", status="closed"), "minibench")
    server.add_fixture(MINIBENCH_NUMERIC, "minibench")
    for keys in server.tournaments.values():
        keys.add("bot-testing-area")  # submissions go to the practice area (the floor)
    return server


@pytest.fixture
async def http(fake: FakeMetaculus) -> AsyncIterator[httpx.AsyncClient]:
    async with fake.client() as client:
        yield client


def _settings(tmp_path: Path, **extra: object) -> Settings:
    return Settings.model_validate(
        {"provider": "replay", "model": "qwen2.5:7b", "ledger_path": tmp_path / "l.sqlite"} | extra
    )


LOCAL = Route(endpoint=PRESETS["ollama"], model="qwen2.5:7b")
OPENROUTER = PRESETS["openrouter"]


def frontier(model: str = "openai/gpt-5.1") -> Route:
    return Route(endpoint=OPENROUTER, model=model)


async def _run(
    tmp_path: Path,
    *,
    client: MetaculusClient | None,
    provider: Provider | None = None,
    route: Route = LOCAL,
    router: Router | None = None,
    catalog: Catalog | None = None,
    probe_providers: dict[str, Provider] | None = None,
    submit: bool = False,
    samples: Path | None = None,
    max_questions: int = 10,
    out: io.StringIO | None = None,
    searcher: FixedSearcher | None = None,
    tournament: str | None = None,
    **settings: object,
) -> tuple[RunReport, Ledger]:
    clock = FixedClock(T0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.open(tmp_path / "l.sqlite", clock)
    if router is None:
        router = Router.single(
            route, provider or ScriptedProvider(), catalog or free_catalog(route), clock
        )
    run = ForecastRun(
        settings=_settings(tmp_path, **settings),
        clock=clock,
        ledger=ledger,
        router=router,
        searchers=[searcher or FixedSearcher()],
        client=client,
        out=out or io.StringIO(),
        probe_providers=probe_providers,
    )
    slug = tournament or ("bot-testing-area" if submit else "minibench")
    report = await run.run(slug, submit=submit, samples=samples, max_questions=max_questions)
    return report, ledger


def _client(http: httpx.AsyncClient, token: str | None = FAKE_TOKEN) -> MetaculusClient:
    return MetaculusClient(
        "https://www.metaculus.com/api",
        SecretStr(token) if token else None,
        http,
        _fast_limiter(),
    )


async def test_dry_run_on_samples_ledgers_everything_and_prints_the_comment(
    tmp_path: Path,
) -> None:
    out = io.StringIO()
    report, ledger = await _run(tmp_path, client=None, out=out)
    assert report.mode is RunMode.DRY_RUN
    assert report.source.startswith("samples:")
    n_samples = len(list(SAMPLES_DIR.glob("post_*.json")))
    assert report.count("forecast") == n_samples
    [first] = [r for r in report.reports if r.status == "forecast"][:1]
    assert first.probability == pytest.approx(0.40)
    run = report.run_id
    assert ledger.count("questions", run) == n_samples
    # planner + brief + five forecasters per question
    assert ledger.count("llm_calls", run) == 7 * n_samples
    assert ledger.count("forecasts", run) == 6 * n_samples
    assert ledger.count("submissions", run) == 2 * n_samples
    conn = ledger.connection
    statuses = {r[0] for r in conn.execute("SELECT status FROM submissions")}
    assert statuses == {"dry_run"}
    # The late source was stored, judged and rejected, and no call cites it.
    rejected = conn.execute(
        "SELECT count(*) FROM source_uses u JOIN sources s USING (source_id) "
        "WHERE u.admitted = 0 AND s.url = 'https://example.org/the-answer'"
    ).fetchone()[0]
    assert rejected == n_samples
    cited_late = conn.execute(
        "SELECT count(*) FROM call_sources JOIN sources USING (source_id) "
        "WHERE url = 'https://example.org/the-answer'"
    ).fetchone()[0]
    assert cited_late == 0
    text = out.getvalue()
    assert "would POST /api/questions/forecast/" in text
    assert '"probability_yes": 0.4' in text
    assert "| **Forecast: 40%**, the median of 5 independent forecasts" in text
    assert "It resolved Yes" not in text
    ledger.close()
    assert scan_ledger(tmp_path / "l.sqlite") == []


async def test_live_run_submits_once_with_a_private_comment_and_skips_next_time(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    out = io.StringIO()
    report, ledger = await _run(tmp_path, client=_client(http), submit=True, out=out)
    assert report.mode is RunMode.LIVE
    assert "posted this private comment:\n   | **Forecast: 40%**" in out.getvalue()
    forecast = [r for r in report.reports if r.status == "forecast"]
    assert [r.question.question_id for r in forecast] == [501]
    assert forecast[0].submitted
    reasons = {r.question.question_id: r.reason for r in report.reports if r.status == "skipped"}
    assert "not forecast yet" in reasons[46876]
    assert len(fake.forecasts) == 1
    sent = fake.forecasts[0]
    assert sent == {
        "question": 501,
        "source": "api",
        "probability_yes": 0.4,
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }
    assert len(fake.comments) == 1
    comment = fake.comments[0]
    assert comment["is_private"] is True
    assert comment["on_post"] == 500
    assert comment["text"].startswith("**Forecast: 40%**")
    assert "Base rate is low" in comment["text"]
    rows = ledger.connection.execute(
        "SELECT kind, status, http_status FROM submissions ORDER BY kind"
    ).fetchall()
    assert rows == [("comment", "submitted", 201), ("forecast", "submitted", 201)]
    ledger.close()

    second, ledger2 = await _run(tmp_path, client=_client(http), submit=True)
    assert second.count("forecast") == 0
    skipped = {r.question.question_id: r.reason for r in second.reports}
    assert skipped[501] == "already forecast by this account"
    assert len(fake.forecasts) == 1, "a second run must never forecast the same question"
    assert len(fake.comments) == 1
    ledger2.close()


async def test_with_a_token_but_no_submit_the_listing_is_live_and_nothing_is_sent(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    report, ledger = await _run(tmp_path, client=_client(http))
    assert report.source == "metaculus:minibench"
    assert report.count("forecast") == 1
    assert fake.forecasts == []
    assert fake.comments == []
    ledger.close()


async def test_submit_without_a_token_is_refused(tmp_path: Path, http: httpx.AsyncClient) -> None:
    with pytest.raises(AugurError, match="METACULUS_TOKEN"):
        await _run(tmp_path, client=_client(http, token=None), submit=True)


async def test_failed_draws_degrade_the_question_without_aborting_the_run(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider([None, 20, None, 30, 40])
    report, ledger = await _run(tmp_path, client=None, provider=provider)
    first = report.reports[0]
    assert first.status == "forecast"
    assert first.probability == pytest.approx(0.30)
    assert "2 of 5 draws failed" in first.comment
    errors = ledger.connection.execute(
        "SELECT count(*) FROM llm_calls WHERE error IS NOT NULL"
    ).fetchone()[0]
    assert errors >= 2
    ledger.close()

    too_few = ScriptedProvider([None, None, None, 30, 40])
    report2, ledger2 = await _run(tmp_path / "b", client=None, provider=too_few)
    assert report2.reports[0].status == "failed"
    assert "only 2 of 5 forecasters answered" in report2.reports[0].reason
    ledger2.close()


async def test_a_failing_searcher_degrades_research_only(tmp_path: Path) -> None:
    report, ledger = await _run(tmp_path, client=None, searcher=FixedSearcher(fail=True))
    assert report.reports[0].status == "forecast"
    assert "No sources dated before the cutoff" in report.reports[0].comment
    ledger.close()


async def test_the_question_cap(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    fake.add_post(binary_post(520, 521, title="Another open one"), "minibench")
    report, ledger = await _run(tmp_path, client=_client(http), max_questions=1)
    assert report.count("forecast") == 1
    assert any("over the cap of 1" in r.reason for r in report.reports)
    ledger.close()


async def test_no_token_reaches_the_logs_during_a_live_run(
    tmp_path: Path,
    fake: FakeMetaculus,
    http: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METACULUS_TOKEN", FAKE_TOKEN)
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    out = io.StringIO()
    fake.rate_limited = 1
    fake.retry_after = "0.01"
    await _run(tmp_path, client=_client(http), submit=True, out=out)
    logs = stream.getvalue()
    assert logs, "the run should have logged"
    assert FAKE_TOKEN not in logs
    assert FAKE_TOKEN not in out.getvalue()
    events = [json.loads(line) for line in logs.splitlines()]
    assert all("run_id" in e for e in events if e["event"] != "asknews skipped")
    assert any(e.get("question_id") == 501 for e in events)
    ledger_bytes = (tmp_path / "l.sqlite").read_bytes()
    assert FAKE_TOKEN.encode() not in ledger_bytes


def test_code_version_names_this_checkout_not_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from augur.runner import code_version

    monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    assert "ffffffffffff" not in code_version()
    assert code_version().startswith("0.1.0")


async def test_concurrent_draws_cannot_overspend_the_cap(tmp_path: Path) -> None:
    """Five draws start at once; reservations keep the total under the cap."""
    from augur.agents.calls import Budget, CallContext, CallRecorder, estimate_cost
    from augur.llm.base import LlmRequest, Message
    from augur.llm.catalog import price_for

    pricey = frontier("vendor/pricey-model")
    catalog = catalog_of(("vendor/pricey-model", 15.0, 75.0))
    request = LlmRequest(model="", messages=(Message(role="user", content="x" * 300),))
    price = price_for(pricey, catalog)
    assert price is not None
    one = estimate_cost(request.model_copy(update={"model": pricey.model}), price)
    assert one > 0

    report, ledger = await _run(
        tmp_path,
        client=None,
        route=pricey,
        catalog=catalog,
        max_cost_per_question=1.0,
        max_cost_per_run=0.20,
    )
    first = report.reports[0]
    assert first.status == "failed"
    assert "run budget" in first.reason
    assert report.cost_usd <= 0.20
    assert all(r.status != "forecast" for r in report.reports)
    assert ledger.count("submissions") == 0
    ledger.close()

    budget = Budget(max_per_question=one * 2.5, max_per_run=100)
    led = Ledger.open(tmp_path / "c.sqlite", FixedClock(T0))
    run_id = led.start_run(
        mode=RunMode.DRY_RUN,
        tournament="t",
        provider="scripted",
        model=pricey.model,
        settings={},
        code_version="test",
    )
    q, post = binary_question(T0)
    led.record_question(run_id, q, T0, post)
    router = Router.single(pricey, ScriptedProvider(), catalog, FixedClock(T0))
    calls = CallRecorder(router, led, FixedClock(T0), budget)
    ctx = CallContext(run_id, q.question_id, "forecaster", 0)
    results = await asyncio.gather(
        *(calls.complete(ctx, request, []) for _ in range(5)), return_exceptions=True
    )
    assert sum(1 for r in results if not isinstance(r, BaseException)) == 2
    assert budget.spent_question[q.question_id] <= one * 2.5
    led.close()


async def test_a_failed_comment_is_posted_by_the_next_run_and_never_twice(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    fake.unavailable_comments = 1
    first, ledger = await _run(tmp_path, client=_client(http), submit=True)
    [failed] = [r for r in first.reports if r.question.question_id == 501]
    assert failed.status == "failed"
    assert len(fake.forecasts) == 1
    assert fake.comments == []
    comment_posts = [r for r in fake.requests if r.url.path.endswith("/comments/create/")]
    assert len(comment_posts) == 1, "a comment POST answered 503 must not be retried"
    ledger.close()

    second, ledger2 = await _run(tmp_path, client=_client(http), submit=True)
    [repaired] = [r for r in second.reports if r.question.question_id == 501]
    assert "posted the comment an earlier run could not" in repaired.reason
    assert len(fake.forecasts) == 1
    assert len(fake.comments) == 1
    assert fake.comments[0]["text"].startswith("**Forecast: 40%**")
    ledger2.close()

    third, ledger3 = await _run(tmp_path, client=_client(http), submit=True)
    [done] = [r for r in third.reports if r.question.question_id == 501]
    assert done.reason == "already forecast by this account"
    assert len(fake.comments) == 1
    ledger3.close()


async def test_a_forecast_taken_but_unacknowledged_still_gets_its_comment(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    fake.lost_forecast_answers = 100  # every retry is taken, none acknowledged
    first, ledger = await _run(tmp_path, client=_client(http), submit=True)
    [failed] = [r for r in first.reports if r.question.question_id == 501]
    assert failed.status == "failed"
    taken = len(fake.forecasts)
    assert taken >= 1, "the server took the forecast"
    assert fake.comments == []
    ledger.close()
    fake.lost_forecast_answers = 0

    second, ledger2 = await _run(tmp_path, client=_client(http), submit=True)
    [repaired] = [r for r in second.reports if r.question.question_id == 501]
    assert "posted the comment an earlier run could not" in repaired.reason
    assert len(fake.forecasts) == taken, "the repair does not forecast again"
    assert len(fake.comments) == 1
    assert fake.comments[0]["text"].startswith("**Forecast: 40%**")
    ledger2.close()


LOOPBACK_NAMED_OPENROUTER = Endpoint(
    name="openrouter", base_url="http://127.0.0.1:8080/v1", billing="free"
)


@pytest.mark.parametrize(
    "route",
    [
        LOCAL,
        Route(endpoint=OPENROUTER, model="meta-llama/llama-3.1-8b-instruct"),
        Route(endpoint=OPENROUTER, model="google/gemma-4-31b-it:free"),
        # A frontier name is not enough: the host must be one that serves it.
        Route(endpoint=LOOPBACK_NAMED_OPENROUTER, model="openai/gpt-5"),
        Route(endpoint=PRESETS["groq"], model="openai/gpt-5"),
    ],
    ids=["local", "small-hosted", "free-hosted", "loopback-named-openrouter", "groq-host"],
)
@pytest.mark.parametrize("tournament", ["minibench", "futureeval", "32916"])
async def test_the_floor_refuses_live_submission_below_the_frontier_tier(
    tmp_path: Path,
    fake: FakeMetaculus,
    http: httpx.AsyncClient,
    route: Route,
    tournament: str,
) -> None:
    provider = ScriptedProvider()
    with pytest.raises(SubmitFloorError, match="frontier-2026-11"):
        await _run(
            tmp_path,
            client=_client(http),
            provider=provider,
            route=route,
            submit=True,
            tournament=tournament,
        )
    assert provider.requests == [], "refused before any model call"
    assert fake.forecasts == []
    assert fake.comments == []
    ledger = Ledger.open(tmp_path / "l.sqlite", FixedClock(T0))
    assert ledger.count("runs") == 0
    ledger.close()


async def test_the_floor_admits_a_frontier_model_and_the_practice_area(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    gpt = frontier("openai/gpt-5.1")
    report, ledger = await _run(
        tmp_path / "a",
        client=_client(http),
        route=gpt,
        catalog=catalog_of((gpt.model, 1.25, 10.0)),
        submit=True,
        tournament="minibench",
    )
    assert report.count("forecast") == 1
    assert len(fake.forecasts) == 1
    ledger.close()
    assert scan_ledger(tmp_path / "a" / "l.sqlite") == []

    fake.add_post(binary_post(520, 521, title="Practice"), "bot-testing-area")
    small, ledger2 = await _run(tmp_path / "b", client=_client(http), submit=True)
    assert [r.question.question_id for r in small.reports if r.status == "forecast"] == [521]
    ledger2.close()
