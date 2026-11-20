"""Q2's arms and the query plan: ``run_forecastbench`` with a retrieval, a gate and a wall clock."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from augur.agents.researcher import PLAN_SYSTEM, ResearchResult
from augur.backtest import (
    PinnedClock,
    Retrieval,
    plan_queries,
    read_queries,
    release_gate,
    run_forecastbench,
    sample,
    write_queries,
)
from augur.clock import FixedClock
from augur.config import Settings
from augur.errors import AugurError
from augur.forecastbench import FBQuestion
from augur.leakage import scan_ledger
from augur.ledger import Ledger
from augur.llm.endpoints import PRESETS, Route
from augur.llm.router import Router
from augur.research.source import DateEvidence, Source

from .fakes.forecastbench import LOCAL, QUESTION_SET, free_local_catalog, small_set
from .fakes.llm import ScriptedProvider
from .helpers import catalog_of

WALL = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
FREEZE = datetime(2025, 3, 1, tzinfo=UTC)
BEFORE_FREEZE = datetime(2025, 2, 20, tzinfo=UTC)
AFTER_FREEZE = datetime(2025, 5, 1, tzinfo=UTC)
TOURNAMENT = f"forecastbench:{QUESTION_SET}"
LATE_TITLE = "Written after the question opened"


class DatedSearcher:
    """Two fixed pages whatever the instant asked for: one before the freeze, one after."""

    name = "dated"

    def __init__(self) -> None:
        self.calls: list[tuple[str, datetime]] = []

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        self.calls.append((query, as_of))
        return [
            self._page("early", "Background before the freeze", BEFORE_FREEZE, as_of),
            self._page("late", LATE_TITLE, AFTER_FREEZE, as_of),
        ][:limit]

    def _page(self, slug: str, title: str, published: datetime, as_of: datetime) -> Source:
        return Source.build(
            searcher=self.name,
            query="q",
            url=f"https://example.org/{slug}",
            title=title,
            excerpt=f"Facts: {slug}.",
            published_at=published,
            retrieved_at=as_of,
            as_of=as_of,
            date_evidence=DateEvidence.REVISION_TIMESTAMP,
            evidence_detail="fixed",
        )


def _rows(path: Path, sql: str, args: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(path)) as conn:
        return [tuple(r) for r in conn.execute(sql, args)]


def _settings() -> Settings:
    return Settings.model_validate({"provider": "replay", "model": "qwen2.5:7b", "forecasters": 3})


async def _run(
    db: Path,
    questions: list[FBQuestion],
    retrieval: Retrieval | None,
    *,
    router: Router | None = None,
    wall_clock: FixedClock | None = None,
    **kwargs: Any,
) -> Any:
    clock = PinnedClock(datetime(2026, 1, 1, tzinfo=UTC))
    r = router or Router.single(
        LOCAL, ScriptedProvider((30,)), free_local_catalog(), FixedClock(clock.now())
    )
    with Ledger.open(db, clock) as ledger:
        return await run_forecastbench(
            questions=questions,
            tournament=TOURNAMENT,
            settings=_settings(),
            router=r,
            ledger=ledger,
            clock=clock,
            code_version="test",
            out=lambda _line: None,
            retrieval=retrieval,
            wall_clock=wall_clock,
            **kwargs,
        )


# -- the open arm: a backtest, cutoff = the freeze instant ------------------------------


async def test_the_open_arm_is_a_backtest_and_never_shows_a_page_dated_after_the_freeze(
    tmp_path: Path,
) -> None:
    *_, questions = small_set(tmp_path)
    searcher = DatedSearcher()
    db = tmp_path / "open.sqlite"
    result = await _run(db, questions[:1], Retrieval([searcher]))
    assert (result.forecast, result.failed) == (1, 0)
    assert _rows(db, "SELECT mode FROM runs") == [("backtest",)]
    [(cutoff,)] = _rows(db, "SELECT cutoff FROM questions")
    assert cutoff == "2025-03-01T00:00:00.000000Z"
    assert {as_of for _, as_of in searcher.calls} == {FREEZE}
    refused = _rows(
        db,
        "SELECT s.title FROM source_uses u JOIN sources s USING (source_id) WHERE u.admitted = 0",
    )
    assert refused == [(LATE_TITLE,)]
    # No model request carries the late page, and none cites it.
    assert not any(LATE_TITLE in r for (r,) in _rows(db, "SELECT request_json FROM llm_calls"))
    shown = _rows(
        db, "SELECT DISTINCT s.title FROM call_sources cs JOIN sources s USING (source_id)"
    )
    assert shown == [("Background before the freeze",)]
    assert scan_ledger(db) == []


async def test_given_queries_skip_the_planner(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    provider = ScriptedProvider((30,))
    router = Router.single(LOCAL, provider, free_local_catalog(), FixedClock(WALL))
    searcher = DatedSearcher()
    plan = {questions[0].question_id: ["given one", "given two"]}
    await _run(
        tmp_path / "a.sqlite", questions[:1], Retrieval([searcher], queries=plan), router=router
    )
    assert sorted({q for q, _ in searcher.calls}) == ["given one", "given two"]
    assert not any(r.messages[0].content == PLAN_SYSTEM.format(n=3) for r in provider.requests)


async def test_a_question_without_a_plan_is_planned_in_the_run(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    searcher = DatedSearcher()
    await _run(tmp_path / "a.sqlite", questions[:1], Retrieval([searcher], queries={}))
    assert sorted({q for q, _ in searcher.calls}) == ["first query", "second query"]


async def test_the_outcome_counts_sources_and_loop_steps(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "loop.sqlite"
    await _run(db, questions[:1], Retrieval([DatedSearcher()], loop_steps=2))
    [(reason,)] = _rows(db, "SELECT reason FROM question_outcomes")
    assert reason.startswith("forecast with 1 source(s), ")
    assert "loop step(s)" in reason
    roles = {r for (r,) in _rows(db, "SELECT DISTINCT role FROM llm_calls")}
    assert {"loop", "forecaster"} <= roles
    assert scan_ledger(db) == []


# -- the now arm: unrestricted, cutoff = the wall clock ---------------------------------


async def test_the_now_arm_is_unrestricted_and_cut_off_at_the_wall_clock(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    searcher = DatedSearcher()
    db = tmp_path / "now.sqlite"
    result = await _run(
        db, questions[:1], Retrieval([searcher], as_of="now"), wall_clock=FixedClock(WALL)
    )
    assert (result.forecast, result.failed) == (1, 0)
    assert _rows(db, "SELECT mode FROM runs") == [("unrestricted",)]
    assert _rows(db, "SELECT cutoff, open_time FROM questions") == [
        ("2026-06-01T12:00:00.000000Z", "2025-03-01T00:00:00.000000Z")
    ]
    assert {as_of for _, as_of in searcher.calls} == {WALL}
    # The page the open arm refuses is dated before the wall clock, so it is read here.
    admitted = _rows(
        db,
        "SELECT s.title FROM source_uses u JOIN sources s USING (source_id) "
        "WHERE u.admitted = 1 ORDER BY s.title",
    )
    assert admitted == [("Background before the freeze",), (LATE_TITLE,)]
    assert any(LATE_TITLE in r for (r,) in _rows(db, "SELECT request_json FROM llm_calls"))
    assert scan_ledger(db) == []


async def test_the_now_arm_needs_the_wall_clock(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    with pytest.raises(AugurError, match="wall clock"):
        await _run(tmp_path / "x.sqlite", questions, Retrieval([DatedSearcher()], as_of="now"))


async def test_retrieval_and_recorded_research_are_exclusive(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)

    def research(_fb: FBQuestion) -> ResearchResult | None:
        return None

    with pytest.raises(AugurError, match="not both"):
        await _run(
            tmp_path / "x.sqlite",
            questions,
            Retrieval([DatedSearcher()]),
            research_for=research,
        )


async def test_arms_resume_separately_in_one_ledger(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arms.sqlite"
    one = questions[:1]
    first = await _run(db, one, Retrieval([DatedSearcher()]))
    assert (first.forecast, first.resumed) == (1, 0)
    now = await _run(
        db, one, Retrieval([DatedSearcher()], as_of="now"), wall_clock=FixedClock(WALL)
    )
    assert (now.forecast, now.resumed) == (1, 0)  # the backtest's aggregate does not count
    again = await _run(
        db, one, Retrieval([DatedSearcher()], as_of="now"), wall_clock=FixedClock(WALL)
    )
    assert (again.forecast, again.resumed) == (0, 1)
    assert scan_ledger(db) == []


def test_the_retrieval_mode_and_label() -> None:
    assert Retrieval([]).mode.value == "backtest"
    assert Retrieval([], as_of="now").mode.value == "unrestricted"
    assert Retrieval([], loop_steps=3, as_of="now").label() == "loop3-asof-now"
    assert Retrieval([]).label() == "onepass-asof-open"


# -- the release gate -------------------------------------------------------------------


def _two_route_router(provider: ScriptedProvider, new: Route) -> Router:
    catalog = catalog_of((new.model, 0.0, 0.0), listed=datetime(2025, 6, 1, tzinfo=UTC))
    return Router(
        {"research": [new, LOCAL], "forecast": [LOCAL]},
        {"openrouter": provider, "ollama": provider},
        catalog,
        FixedClock(WALL),
    )


def test_the_release_gate_names_the_reason() -> None:
    new = Route(endpoint=PRESETS["openrouter"], model="acme/new-model")
    router = _two_route_router(ScriptedProvider(), new)
    gate = release_gate(router, {1: FREEZE})
    assert gate(LOCAL, 1) is None
    reason = gate(new, 1)
    assert reason is not None
    assert "released 2025-06-01" in reason
    unknown = gate(LOCAL, 2)
    assert unknown is not None
    assert "unknown" in unknown
    unlisted = Route(endpoint=PRESETS["openrouter"], model="acme/unlisted")
    unlisted_reason = gate(unlisted, 1)
    assert unlisted_reason is not None
    assert "release date unknown" in unlisted_reason


async def test_a_too_new_fallback_route_never_answers_a_planner_or_brief_call(
    tmp_path: Path,
) -> None:
    *_, questions = small_set(tmp_path)
    new = Route(endpoint=PRESETS["openrouter"], model="acme/new-model")
    provider = ScriptedProvider((30,))
    router = _two_route_router(provider, new)
    db = tmp_path / "gate.sqlite"
    result = await _run(db, questions[:2], Retrieval([DatedSearcher()]), router=router)
    assert (result.forecast, result.failed) == (2, 0)
    assert {r.model for r in provider.requests} == {"qwen2.5:7b"}
    assert _rows(db, "SELECT DISTINCT model FROM llm_calls") == [("qwen2.5:7b",)]
    roles = {r for (r,) in _rows(db, "SELECT DISTINCT role FROM llm_calls")}
    assert {"planner", "researcher", "forecaster"} <= roles
    assert scan_ledger(db) == []


async def test_when_every_research_route_is_too_new_the_question_is_still_forecast(
    tmp_path: Path,
) -> None:
    *_, questions = small_set(tmp_path)
    new = Route(endpoint=PRESETS["openrouter"], model="acme/new-model")
    catalog = catalog_of((new.model, 0.0, 0.0), listed=datetime(2025, 6, 1, tzinfo=UTC))
    provider = ScriptedProvider((30,))
    router = Router(
        {"research": [new], "forecast": [LOCAL]},
        {"openrouter": provider, "ollama": provider},
        catalog,
        FixedClock(WALL),
    )
    db = tmp_path / "gate.sqlite"
    # The planner's failure degrades to the fallback query; the brief's to a plain note.
    result = await _run(db, questions[:1], Retrieval([DatedSearcher()]), router=router)
    assert result.forecast == 1
    assert not any(r.model == "acme/new-model" for r in provider.requests)
    assert scan_ledger(db) == []


# -- the plan, and the sample -----------------------------------------------------------


async def test_plan_queries_records_a_backtest_run_and_returns_each_questions_plan(
    tmp_path: Path,
) -> None:
    *_, questions = small_set(tmp_path)
    provider = ScriptedProvider()
    clock = PinnedClock(datetime(2026, 1, 1, tzinfo=UTC))
    router = Router.single(LOCAL, provider, free_local_catalog(), FixedClock(clock.now()))
    db = tmp_path / "plan.sqlite"
    with Ledger.open(db, clock) as ledger:
        planned = await plan_queries(
            questions=questions,
            tournament="plan:x",
            settings=_settings(),
            router=router,
            ledger=ledger,
            clock=clock,
            code_version="test",
            max_queries=1,
        )
    assert planned == {q.question_id: ["first query"] for q in questions}
    assert _rows(db, "SELECT mode, tournament FROM runs") == [("backtest", "plan:x")]
    assert _rows(db, "SELECT DISTINCT role FROM llm_calls") == [("planner",)]
    assert _rows(db, "SELECT COUNT(*) FROM questions") == [(3,)]
    assert scan_ledger(db) == []


async def test_plan_queries_skips_a_too_new_planner_and_falls_back_to_the_question(
    tmp_path: Path,
) -> None:
    *_, questions = small_set(tmp_path)
    new = Route(endpoint=PRESETS["openrouter"], model="acme/new-model")
    catalog = catalog_of((new.model, 0.0, 0.0), listed=datetime(2025, 6, 1, tzinfo=UTC))
    provider = ScriptedProvider()
    router = Router(
        {"research": [new], "forecast": [LOCAL]},
        {"openrouter": provider, "ollama": provider},
        catalog,
        FixedClock(WALL),
    )
    clock = PinnedClock(datetime(2026, 1, 1, tzinfo=UTC))
    with Ledger.open(tmp_path / "p.sqlite", clock) as ledger:
        planned = await plan_queries(
            questions=questions[:1],
            tournament="plan:x",
            settings=_settings(),
            router=router,
            ledger=ledger,
            clock=clock,
            code_version="test",
        )
    assert provider.requests == []
    [queries] = planned.values()
    assert len(queries) == 1
    assert "event" in queries[0]


def test_queries_round_trip_and_reject_a_foreign_file(tmp_path: Path) -> None:
    planned = {30: ["b b", "naïve query"], 4: ["a"]}
    path = tmp_path / "plan.json"
    write_queries(path, planned)
    assert read_queries(path) == planned
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["4", "30"]  # sorted, stable
    assert "naïve" in path.read_text(encoding="utf-8")  # not escaped
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(AugurError, match="not a query plan"):
        read_queries(path)
    path.write_text('{"1": ["x"], "2": "not a list"}', encoding="utf-8")
    assert read_queries(path) == {1: ["x"]}


def test_sample_is_the_first_n_in_question_id_order(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    by_id = sorted(questions, key=lambda q: q.question_id)
    shuffled = [by_id[2], by_id[0], by_id[1]]
    assert sample(shuffled, None) == by_id
    assert sample(shuffled, 2) == by_id[:2]
    assert sample(shuffled, 0) == []
    assert sample(shuffled, 10) == by_id
    assert sample(list(reversed(by_id)), 2) == sample(by_id, 2)
