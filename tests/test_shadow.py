"""M11's shadow tick: a second tick skips what the shadow ledger holds, with no model call,
and shadow mode can never submit."""

from __future__ import annotations

import importlib.util
import io
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from augur import cli
from augur.baseline import TemplateForecast, questions_to_shadow, run_shadow
from augur.clock import FixedClock
from augur.config import Settings
from augur.errors import AugurError
from augur.export import export_jsonl
from augur.importer import import_exports
from augur.ledger import Ledger
from augur.llm.endpoints import PRESETS, Route
from augur.llm.router import Router
from augur.metaculus import MetaculusClient
from augur.ratelimit import RateLimiter
from augur.runner import ForecastRun, RunReport
from augur.shadow import held_questions, questions_missing_template, templated_questions

from .conftest import T0
from .fakes.llm import ScriptedProvider
from .fakes.metaculus import FAKE_TOKEN, FakeMetaculus, binary_post
from .fakes.search import FixedSearcher
from .helpers import free_catalog

ROUTE = Route(endpoint=PRESETS["openrouter"], model="nvidia/nemotron-3-ultra-550b-a55b:free")


class CountingTemplate:
    """The template's runner shape: one fixed probability, every call counted."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def forecast(self, post_id: int) -> TemplateForecast:
        self.calls.append(post_id)
        return TemplateForecast(detail="fake template", probability=0.3)


def _server() -> FakeMetaculus:
    server = FakeMetaculus(clock=FixedClock(T0))
    server.add_post(binary_post(500, 501, title="First shadow question?"), "minibench")
    server.add_post(binary_post(520, 521, title="Second shadow question?"), "minibench")
    return server


def _client(http: httpx.AsyncClient) -> MetaculusClient:
    return MetaculusClient(
        "https://www.metaculus.com/api",
        SecretStr(FAKE_TOKEN),
        http,
        RateLimiter(1000, burst=1000, max_concurrent=4),
    )


async def _tick(
    workdir: Path,
    http: httpx.AsyncClient,
    provider: ScriptedProvider,
    shadow_ledger: Path,
    template: CountingTemplate,
) -> tuple[RunReport, Path]:
    """One scheduled shadow tick, as the workflow runs it: Augur on a fresh ledger,
    skipping what the shadow ledger holds, then the template on what Augur forecast."""
    clock = FixedClock(T0)
    workdir.mkdir(parents=True)
    path = workdir / "run.sqlite"
    ledger = Ledger.open(path, clock)
    router = Router.single(ROUTE, provider, free_catalog(ROUTE), clock)
    settings = Settings.model_validate(
        {"provider": "replay", "model": "qwen2.5:7b", "ledger_path": path}
    )
    run = ForecastRun(
        settings=settings,
        clock=clock,
        ledger=ledger,
        router=router,
        searchers=[FixedSearcher()],
        client=_client(http),
        out=io.StringIO(),
        probe_providers={"openrouter": provider},
    )
    report = await run.run(
        "minibench",
        submit=False,
        samples=None,
        max_questions=10,
        shadow=held_questions(shadow_ledger),
    )
    questions = questions_to_shadow(ledger) + questions_missing_template(
        shadow_ledger, clock.now(), also_templated=templated_questions(path)
    )
    await run_shadow(ledger, template, questions, clock)
    ledger.close()
    return report, path


def _publish(run_ledger: Path, branch: Path, name: str) -> Path:
    """The push to the ledger branch and the next tick's import, in one step."""
    export_jsonl(run_ledger, branch / "shadow" / "2026-11-30" / f"{name}.jsonl")
    shadow = branch.parent / f"shadow-after-{name}.sqlite"
    import_exports([branch / "shadow"], shadow, FixedClock(T0))
    return shadow


async def test_a_second_tick_skips_every_held_question_with_no_model_call(
    tmp_path: Path,
) -> None:
    server = _server()
    branch = tmp_path / "branch"
    async with server.client() as http:
        first_model = ScriptedProvider()
        first_template = CountingTemplate()
        first, run1 = await _tick(
            tmp_path / "tick1", http, first_model, tmp_path / "none.sqlite", first_template
        )
        assert first.count("forecast") == 2
        assert first_model.requests, "the first tick forecasts"
        assert sorted(first_template.calls) == [500, 520]
        shadow = _publish(run1, branch, "tick1")
        assert held_questions(shadow) == {501, 521}
        assert templated_questions(shadow) == {501, 521}

        second_model = ScriptedProvider()
        second_template = CountingTemplate()
        second, run2 = await _tick(tmp_path / "tick2", http, second_model, shadow, second_template)
    assert second.count("forecast") == 0
    assert {r.reason for r in second.reports} == {"already in the shadow ledger"}
    assert second_model.requests == [], "no model call, not even the probe"
    assert second_template.calls == []
    assert server.forecasts == [], "a shadow tick posts nothing"
    assert server.comments == []
    assert held_questions(run2) == frozenset()


async def test_a_new_question_is_the_only_one_the_second_tick_forecasts(tmp_path: Path) -> None:
    server = _server()
    branch = tmp_path / "branch"
    async with server.client() as http:
        _, run1 = await _tick(
            tmp_path / "tick1",
            http,
            ScriptedProvider(),
            tmp_path / "none.sqlite",
            CountingTemplate(),
        )
        shadow = _publish(run1, branch, "tick1")
        server.add_post(binary_post(540, 541, title="Opened between ticks?"), "minibench")
        template = CountingTemplate()
        second, run2 = await _tick(tmp_path / "tick2", http, ScriptedProvider(), shadow, template)
    assert [r.question.question_id for r in second.reports if r.status == "forecast"] == [541]
    assert template.calls == [540]
    assert held_questions(run2) == {541}


async def test_a_template_cut_off_by_the_timeout_is_caught_up_next_tick(tmp_path: Path) -> None:
    server = _server()
    branch = tmp_path / "branch"
    clock = FixedClock(T0)
    async with server.client() as http:
        # Tick 1's template never ran (the job timed out after the first push).
        run1 = tmp_path / "tick1" / "run.sqlite"
        (tmp_path / "tick1").mkdir()
        ledger = Ledger.open(run1, clock)
        router = Router.single(ROUTE, ScriptedProvider(), free_catalog(ROUTE), clock)
        settings = Settings.model_validate(
            {"provider": "replay", "model": "qwen2.5:7b", "ledger_path": run1}
        )
        run = ForecastRun(
            settings=settings,
            clock=clock,
            ledger=ledger,
            router=router,
            searchers=[FixedSearcher()],
            client=_client(http),
            out=io.StringIO(),
        )
        await run.run("minibench", submit=False, samples=None, max_questions=10, shadow=frozenset())
        ledger.close()
        shadow = _publish(run1, branch, "tick1")
        assert templated_questions(shadow) == frozenset()
        model = ScriptedProvider()
        template = CountingTemplate()
        second, _ = await _tick(tmp_path / "tick2", http, model, shadow, template)
    assert second.count("forecast") == 0
    assert model.requests == []
    assert sorted(template.calls) == [500, 520]


async def test_a_shadow_tick_ignores_what_the_account_posted(tmp_path: Path) -> None:
    server = _server()
    async with server.client() as http:
        client = _client(http)
        # The account posted on question 501 outside the shadow round.
        await client.submit_forecast(
            [
                {
                    "question": 501,
                    "source": "api",
                    "probability_yes": 0.5,
                    "probability_yes_per_category": None,
                    "continuous_cdf": None,
                }
            ]
        )
        report, _ = await _tick(
            tmp_path / "tick",
            http,
            ScriptedProvider(),
            tmp_path / "none.sqlite",
            CountingTemplate(),
        )
    assert sorted(r.question.question_id for r in report.reports if r.status == "forecast") == [
        501,
        521,
    ]


def test_a_closed_question_is_not_caught_up(tmp_path: Path) -> None:
    assert questions_missing_template(tmp_path / "missing.sqlite", T0) == []


async def test_shadow_mode_refuses_submit_in_the_runner(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    ledger = Ledger.open(tmp_path / "l.sqlite", clock)
    run = ForecastRun(
        settings=Settings.model_validate(
            {"provider": "replay", "model": "qwen2.5:7b", "ledger_path": tmp_path / "l.sqlite"}
        ),
        clock=clock,
        ledger=ledger,
        router=Router.single(ROUTE, ScriptedProvider(), free_catalog(ROUTE), clock),
        searchers=[FixedSearcher()],
        client=None,
        out=io.StringIO(),
    )
    with pytest.raises(AugurError, match="never submits"):
        await run.run(
            "bot-testing-area", submit=True, samples=None, max_questions=1, shadow=frozenset()
        )
    with pytest.raises(AugurError, match="reforecast"):
        await run.run(
            "bot-testing-area",
            submit=False,
            samples=None,
            max_questions=1,
            reforecast=True,
            shadow=frozenset(),
        )
    assert ledger.connection.execute("SELECT count(*) FROM runs").fetchone() == (0,)
    ledger.close()


def test_shadow_mode_refuses_submit_on_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["forecast", "--submit", "--shadow-ledger", str(tmp_path / "s.sqlite")])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_the_shadow_ledger_cannot_be_the_run_ledger(tmp_path: Path) -> None:
    path = tmp_path / "same.sqlite"
    with pytest.raises(AugurError, match="must not be the run's own ledger"):
        cli._shadow_held(path, path)
    assert cli._shadow_held(None, path) is None
    assert cli._shadow_held(tmp_path / "missing.sqlite", path) == frozenset()


def test_held_questions_ignore_member_rows(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = tmp_path / "l.sqlite"
    with Ledger.open(path, clock):
        pass
    assert held_questions(path) == frozenset()


def _script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"augur_{name}_script_for_shadow", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def test_the_default_with_no_variable_set_shadows_minibench_and_never_posts(
    tmp_path: Path,
) -> None:
    """M17: a scheduled tick with every repository variable unset, end to end on the fake
    server: the mode step picks a MiniBench shadow tick, the poll finds work, the tick
    posts nothing, and after its export the poll and the second tick skip what it holds."""
    run_mode, poll = _script("run_mode"), _script("poll")
    secrets = {"METACULUS_TOKEN": FAKE_TOKEN, "OPENROUTER_API_KEY": "sk-or-v1-fake"}
    decision = run_mode.decide(
        {"EVENT": "schedule", "REF": "refs/heads/main", "REPO_PRIVATE": "false", **secrets}
    )
    assert (decision.mode, decision.schedule, decision.chain_minutes) == (
        "shadow",
        "minibench",
        "30",
    )
    server = _server()
    branch = tmp_path / "branch"

    def fetch(url: str, headers: Mapping[str, str]) -> Any:
        return server.handle(httpx.Request("GET", url, headers=dict(headers))).json()

    def polled() -> tuple[bool, list[str], list[str]]:
        env = {
            "AUGUR_SCHEDULE": decision.schedule,
            "AUGUR_SHADOW": "true" if decision.mode == "shadow" else "false",
            "AUGUR_SHADOW_LEDGER": str(branch / "shadow"),
            **secrets,
        }
        result: tuple[bool, list[str], list[str]] = poll.poll(
            env, fetch, sleep=lambda _s: None, now=T0
        )
        return result

    assert polled()[:2] == (True, ["minibench"])
    async with server.client() as http:
        _, run1 = await _tick(
            tmp_path / "tick1",
            http,
            ScriptedProvider(),
            tmp_path / "none.sqlite",
            CountingTemplate(),
        )
        shadow = _publish(run1, branch, "tick1")
        has_work, _, lines = polled()
        assert not has_work, lines
        model = ScriptedProvider()
        second, _ = await _tick(tmp_path / "tick2", http, model, shadow, CountingTemplate())
    assert second.count("forecast") == 0
    assert model.requests == []
    assert [r for r in server.requests if r.method != "GET"] == [], "nothing was posted"
    assert server.forecasts == []
    assert server.comments == []
