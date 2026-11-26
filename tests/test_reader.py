"""M16's live arm: the newer free route on the brief Augur's member 0 read (``reader``).

Offline: Augur's tick runs on the scripted provider against the fake Metaculus,
the template is a fixed runner, and the reader's route is a second scripted
provider, so every call the arm makes is counted.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from augur import cli, floor
from augur.agents import forecaster as forecaster_module
from augur.baseline import TemplateForecast, run_shadow
from augur.clock import FixedClock
from augur.config import Settings
from augur.errors import AugurError
from augur.export import export_jsonl
from augur.ledger import Ledger
from augur.llm.endpoints import PRESETS, Route
from augur.llm.router import Router
from augur.metaculus import MetaculusClient
from augur.questions import Question, question_from_post
from augur.ratelimit import RateLimiter
from augur.reader import (
    DRAWS,
    READER,
    READER_BASE,
    READER_ROUTE,
    TEMPLATE_CALLS,
    TOURNAMENT_SUFFIX,
    ReaderResult,
    calls_used,
    run_reader,
)
from augur.runner import ForecastRun
from augur.shadow import held_questions
from augur.statarm import BaseRates
from augur.statlive import LiveStatArm

from .conftest import T0
from .fakes.llm import ScriptedProvider
from .fakes.metaculus import FAKE_TOKEN, FakeMetaculus, binary_post
from .fakes.search import FixedSearcher
from .helpers import free_catalog, round_snapshot

CURRENT = Route(endpoint=PRESETS["openrouter"], model="nvidia/nemotron-3-ultra-550b-a55b:free")
NEWER = Route(endpoint=PRESETS["openrouter"], model=READER_ROUTE.split(":", 1)[1])
RATES = BaseRates.fit([("metaculus", 1)] * 6 + [("metaculus", 0)] * 24, fitted_on="test")
POSTS = (500, 520)
QUESTIONS = (501, 521)
POLL_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "poll.py"


class FixedTemplate:
    """The template's runner shape: one fixed probability."""

    async def forecast(self, post_id: int) -> TemplateForecast:
        return TemplateForecast(detail="fake template", probability=0.3)


def _settings(path: Path) -> Settings:
    return Settings.model_validate(
        {"provider": "replay", "model": "qwen2.5:7b", "ledger_path": path}
    )


def _client(http: httpx.AsyncClient) -> MetaculusClient:
    return MetaculusClient(
        "https://www.metaculus.com/api",
        SecretStr(FAKE_TOKEN),
        http,
        RateLimiter(1000, burst=1000, max_concurrent=4),
    )


def _questions(path: Path) -> dict[int, Question]:
    with closing(sqlite3.connect(path)) as conn:
        rows = conn.execute("SELECT question_id, payload_json FROM questions").fetchall()
    return {int(qid): question_from_post(json.loads(p)) for qid, p in rows}


async def _tick(
    tmp_path: Path, clock: FixedClock, *, templated: tuple[int, ...] = QUESTIONS
) -> Path:
    """Augur (with ``stat`` and ``blend``) on both questions, then the template on
    ``templated`` only, on a fresh run ledger, as the shadow job runs them."""
    server = FakeMetaculus(clock=FixedClock(T0))
    for post_id, qid in zip(POSTS, QUESTIONS, strict=True):
        server.add_post(binary_post(post_id, qid, title=f"Reader question {qid}?"), "minibench")
    path = tmp_path / "run.sqlite"
    async with server.client() as http:
        with Ledger.open(path, clock) as ledger:
            run = ForecastRun(
                settings=_settings(path),
                clock=clock,
                ledger=ledger,
                router=Router.single(CURRENT, ScriptedProvider(), free_catalog(CURRENT), clock),
                searchers=[FixedSearcher()],
                client=_client(http),
                out=io.StringIO(),
                stat_arm=LiveStatArm(rates=RATES, fetcher=None),
            )
            report = await run.run(
                "minibench", submit=False, samples=None, max_questions=10, shadow=frozenset()
            )
            assert report.count("forecast") == 2
            clock.advance(60)
            qs = _questions(path)
            await run_shadow(ledger, FixedTemplate(), [qs[q] for q in templated], clock)
    return path


async def _read(
    path: Path,
    clock: FixedClock,
    provider: ScriptedProvider,
    *,
    shadow: Path | None = None,
    allowance: int = 1000,
    minutes_left: float | None = None,
) -> ReaderResult:
    clock.advance(60)
    router = Router.single(NEWER, provider, free_catalog(NEWER), clock)
    deadline = None if minutes_left is None else clock.now() + timedelta(minutes=minutes_left)
    with Ledger.open(path, clock) as ledger:
        return await run_reader(
            ledger=ledger,
            ledger_path=path,
            shadow_path=shadow,
            router=router,
            route=NEWER,
            settings=_settings(path),
            clock=clock,
            allowance=allowance,
            deadline=deadline,
        )


def _rows(path: Path, sql: str, *args: Any) -> list[sqlite3.Row]:
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, args).fetchall()


def _count(path: Path, table: str) -> int:
    return int(_rows(path, f"SELECT COUNT(*) AS n FROM {table}")[0]["n"])


def _reader_calls(path: Path) -> list[sqlite3.Row]:
    return _rows(
        path,
        "SELECT c.* FROM llm_calls c JOIN runs r ON r.run_id = c.run_id "
        "WHERE r.tournament LIKE ? ORDER BY c.started_at, c.rowid",
        "%" + TOURNAMENT_SUFFIX,
    )


def _references(path: Path, question_id: int) -> dict[str, sqlite3.Row]:
    rows = _rows(
        path, "SELECT * FROM reference_forecasts WHERE question_id = ? ORDER BY rowid", question_id
    )
    return {str(r["forecaster"]): r for r in rows}


# -- served last ------------------------------------------------------------------------


async def test_the_arm_is_served_after_augur_the_template_stat_and_blend(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock, templated=(501,))
    provider = ScriptedProvider(percents=(60, 70, 80))
    result = await _read(path, clock, provider)
    assert result.recorded == 1
    assert result.calls == DRAWS
    # Question 521 has no template row: no reader call for it.
    assert {r["question_id"] for r in _reader_calls(path)} == {501}
    assert len(provider.requests) == DRAWS
    assert READER not in _references(path, 521)
    refs = _references(path, 501)
    assert {"template", "stat", "blend", READER, READER_BASE} <= set(refs)
    for earlier in ("template", "stat", "blend"):
        assert refs[READER]["recorded_at"] > refs[earlier]["recorded_at"]
    # The median of the reader's three draws, and of Augur's own members 0 to 2.
    assert json.loads(refs[READER]["values_json"]) == {"probability": 0.7}
    assert json.loads(refs[READER_BASE]["values_json"]) == {"probability": 0.35}
    assert NEWER.key in refs[READER]["detail"]
    # A second pass reads nothing again.
    again = await _read(path, clock, ScriptedProvider())
    assert again.recorded == 0
    assert len(_reader_calls(path)) == DRAWS


async def test_a_reader_row_in_the_shadow_ledger_is_not_repeated(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    await _read(path, clock, ScriptedProvider())
    shadow = tmp_path / "shadow.sqlite"
    path.rename(shadow)
    # The shadow ledger holds both reader rows: read beside it, no question is read again.
    provider = ScriptedProvider()
    result = await _read(shadow, clock, provider, shadow=shadow)
    assert result.recorded == 0
    assert provider.requests == []


# -- the day's budget -------------------------------------------------------------------


async def test_no_call_once_the_days_free_calls_are_spent(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    used = calls_used([path], T0.date())
    # Augur: planner, brief and five draws a question; the template counted at 15.
    assert used == 2 * 7 + 2 * TEMPLATE_CALLS
    before = _count(path, "llm_calls")
    provider = ScriptedProvider()
    result = await _read(path, clock, provider, allowance=used)
    assert _count(path, "llm_calls") == before
    assert provider.requests == []
    assert result.recorded == 0
    assert any("free calls are spent" in line for line in result.lines)
    assert _references(path, 501).get(READER) is None


async def test_room_for_one_question_draws_exactly_one(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    used = calls_used([path], T0.date())
    provider = ScriptedProvider()
    result = await _read(path, clock, provider, allowance=used + DRAWS)
    assert result.recorded == 1
    assert len(provider.requests) == DRAWS
    assert len(_reader_calls(path)) == DRAWS
    assert sum(READER in _references(path, q) for q in QUESTIONS) == 1
    assert calls_used([path], T0.date()) == used + DRAWS


async def test_no_question_is_started_with_too_little_time_left(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    before = _count(path, "llm_calls")
    provider = ScriptedProvider()
    result = await _read(path, clock, provider, minutes_left=1.0)
    assert provider.requests == []
    assert _count(path, "llm_calls") == before
    assert result.recorded == 0
    assert result.failed == 0
    assert result.skipped == len(QUESTIONS)
    assert all(READER not in _references(path, q) for q in QUESTIONS)


async def test_calls_in_the_shadow_ledger_count_once(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    copy = tmp_path / "copy.sqlite"
    with closing(sqlite3.connect(path)) as src, closing(sqlite3.connect(copy)) as dst:
        src.backup(dst)
    # The same calls and template rows in both ledgers are one day's calls, not two.
    assert calls_used([path, copy], T0.date()) == calls_used([path], T0.date())
    assert calls_used([path], datetime(2026, 11, 18, tzinfo=UTC).date()) == 0


# -- the brief, byte for byte -----------------------------------------------------------


def _augur_member0_user(path: Path, question_id: int) -> tuple[str, str]:
    row = _rows(
        path,
        "SELECT c.request_json FROM llm_calls c JOIN runs r ON r.run_id = c.run_id "
        "WHERE r.tournament = 'minibench' AND c.role = 'forecaster' AND c.member = 0 "
        "AND c.question_id = ?",
        question_id,
    )[0]
    messages = json.loads(row["request_json"])["messages"]
    return messages[0]["content"], messages[-1]["content"]


async def test_the_reader_sees_augurs_request_byte_for_byte(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    provider = ScriptedProvider()
    result = await _read(path, clock, provider)
    assert result.recorded == 2
    assert len(provider.requests) == 2 * DRAWS
    for qid in QUESTIONS:
        system, user = _augur_member0_user(path, qid)
        calls = [r for r in _reader_calls(path) if r["question_id"] == qid]
        assert sorted(int(c["member"]) for c in calls) == list(range(DRAWS))
        for call in calls:
            request = json.loads(call["request_json"])
            assert request["messages"][0]["content"] == system
            assert request["messages"][-1]["content"] == user
            assert request["model"] == NEWER.model
            assert request["seed"] == call["member"]
        # The reader cited the sources Augur's member 0 cited, admitted in its own run.
        cited = _rows(
            path,
            "SELECT cs.source_id FROM call_sources cs JOIN llm_calls c ON c.call_id = "
            "cs.call_id WHERE c.call_id = ?",
            calls[0]["call_id"],
        )
        assert cited
    from augur.leakage import scan_ledger

    assert scan_ledger(path) == []


async def test_a_tampered_brief_is_refused_with_no_call(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP TRIGGER llm_calls_no_update")
        conn.execute(
            "UPDATE llm_calls SET request_json = replace(request_json, 'status quo holds', "
            "'status quo breaks') WHERE role = 'forecaster' AND member = 0"
        )
        conn.commit()
    before = _count(path, "llm_calls")
    provider = ScriptedProvider()
    result = await _read(path, clock, provider)
    assert result.refused == 2
    assert result.recorded == 0
    assert provider.requests == []
    assert _count(path, "llm_calls") == before
    assert all("refused, no call made" in line for line in result.lines[1:])


async def test_a_changed_prompt_is_refused_with_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    changed = forecaster_module.FORECASTER_SYSTEM + " Be brief."
    monkeypatch.setattr(forecaster_module, "FORECASTER_SYSTEM", changed)
    provider = ScriptedProvider()
    result = await _read(path, clock, provider)
    assert result.refused == 2
    assert provider.requests == []
    assert _reader_calls(path) == []


# -- never the floor's candidate --------------------------------------------------------


def test_the_reader_can_never_be_the_floors_candidate(tmp_path: Path) -> None:
    assert READER not in floor.CANDIDATES
    assert READER_BASE not in floor.CANDIDATES
    snap = round_snapshot(list(POSTS))
    for name in (READER, READER_BASE):
        with pytest.raises(AugurError, match="unknown candidate"):
            floor.decide(snap, [tmp_path / "any.sqlite"], name)
        with pytest.raises(SystemExit):
            cli._parser().parse_args(
                ["floor", "--round", "r", "--ledger", "x.sqlite", "--candidate", name]
            )


async def test_reader_rows_change_neither_held_questions_nor_the_floor(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    with Ledger.open(path, clock) as ledger:
        for post_id, qid in zip(POSTS, QUESTIONS, strict=True):
            ledger.record_resolution(
                question_id=qid,
                post_id=post_id,
                question_type="binary",
                status="resolved",
                value="yes",
                resolved_at=T0,
                source="manual",
                evidence="https://example.test/",
            )
    snap = round_snapshot(list(POSTS))
    held = held_questions(path)
    floors = {c: floor.decide(snap, [path], c).json_text() for c in floor.CANDIDATES}
    aggregates = _count(path, "forecasts")
    result = await _read(path, clock, ScriptedProvider(percents=(90, 90, 90)))
    assert result.recorded == 2
    assert _count(path, "forecasts") == aggregates, "the arm never writes a forecast row"
    assert held_questions(path) == held
    assert {c: floor.decide(snap, [path], c).json_text() for c in floor.CANDIDATES} == floors


# -- the switch -------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "false", "1", "yes"])
def test_the_command_is_a_no_op_unless_switched_on(
    value: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if value is None:
        monkeypatch.delenv("AUGUR_SHADOW_READER", raising=False)
    else:
        monkeypatch.setenv("AUGUR_SHADOW_READER", value)
    ledger = tmp_path / "run.sqlite"
    code = cli.main(["reader", "--ledger", str(ledger), "--shadow-ledger", str(ledger)])
    assert code == 0
    assert "the reader arm is off" in capsys.readouterr().out
    assert not ledger.exists()


def test_the_command_has_no_submit() -> None:
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["reader", "--submit"])


def test_switched_on_with_no_run_ledger_does_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AUGUR_SHADOW_READER", "true")
    ledger = tmp_path / "run.sqlite"
    assert cli.main(["reader", "--ledger", str(ledger)]) == 0
    assert "no run ledger" in capsys.readouterr().out
    assert not ledger.exists()


# -- the poll ---------------------------------------------------------------------------


def _poll() -> ModuleType:
    spec = importlib.util.spec_from_file_location("augur_poll_script_reader", POLL_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def test_the_poll_counts_reader_calls_today_but_not_as_augurs(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    path = await _tick(tmp_path, clock)
    tree = tmp_path / "branch" / "shadow"
    export_jsonl(path, tree / "2026-11-17" / "tick-a.jsonl")
    poll = _poll()
    before = poll.read_shadow(str(tree), "2026-11-17")
    await _read(path, clock, ScriptedProvider())
    # The tick's third push replaces its first two under the same name.
    export_jsonl(path, tree / "2026-11-17" / "tick-a.jsonl")
    after = poll.read_shadow(str(tree), "2026-11-17")
    assert after.calls_today == before.calls_today + 2 * DRAWS
    assert before.augur_calls_per_question() == 7.0
    assert after.augur_calls_per_question() == 7.0
    assert after.held == before.held == set(QUESTIONS)
