"""The bounded research loop, its step parser, and the given-queries path of ``research``."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from augur.agents.calls import Budget, CallRecorder
from augur.agents.researcher import (
    BRIEF_SYSTEM,
    PLAN_SYSTEM,
    Researcher,
    parse_step,
)
from augur.clock import FixedClock
from augur.errors import ProviderError
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, LlmResponse
from augur.llm.endpoints import PRESETS, Route
from augur.llm.router import Router
from augur.questions import Question
from augur.research.source import DateEvidence, Source

from .conftest import T0
from .helpers import binary_question, catalog_of

ROUTE = Route(endpoint=PRESETS["openrouter"], model="google/gemma-4-31b-it:free")
CATALOG = catalog_of((ROUTE.model, 0.0, 0.0))


def step(belief: str, next_query: str | None, unknown: str = "the date") -> str:
    return json.dumps({"belief": belief, "unknown": unknown, "next_query": next_query})


class LoopProvider:
    """Answers loop steps from a script, the planner and the brief with fixed text."""

    name = "openrouter"

    def __init__(self, steps: list[str], fail_on: int | None = None) -> None:
        self.steps = steps
        self.fail_on = fail_on
        self.requests: list[LlmRequest] = []

    def loop_requests(self) -> list[LlmRequest]:
        return [r for r in self.requests if r.messages[0].content.startswith("You research")]

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        system = request.messages[0].content
        if system.startswith("You research"):
            n = len(self.loop_requests())
            if self.fail_on == n:
                raise ProviderError("scripted loop failure")
            text = self.steps[min(n, len(self.steps)) - 1]
        elif system == BRIEF_SYSTEM:
            text = "A one-pass brief [S1]."
        else:
            text = '["planned one", "planned two"]'
        return LlmResponse(text=text, model=request.model, input_tokens=1, output_tokens=1)

    async def aclose(self) -> None:
        return None


class ScriptSearcher:
    """Each query returns one page dated before the cutoff, plus whatever ``extra`` adds."""

    name = "script"

    def __init__(self, extra: list[Source] | None = None, per_query: int = 1) -> None:
        self.extra = extra or []
        self.queries: list[str] = []
        self.per_query = per_query

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        self.queries.append(query)
        pages = [
            page(f"{query} {i}", as_of - timedelta(days=2 + i), as_of)
            for i in range(self.per_query)
        ]
        return pages[:limit] + self.extra


def page(title: str, published: datetime | None, as_of: datetime, **kw: object) -> Source:
    evidence = kw.get("evidence", DateEvidence.REVISION_TIMESTAMP)
    assert isinstance(evidence, DateEvidence)
    return Source.build(
        searcher="script",
        query="q",
        url="https://example.org/" + title.replace(" ", "-"),
        title=title,
        excerpt=f"About {title}.",
        published_at=published,
        retrieved_at=as_of,
        as_of=as_of,
        date_evidence=evidence,
        evidence_detail="fixture",
    )


class Rig:
    def __init__(self, tmp_path: Path, provider: LoopProvider, searcher: ScriptSearcher) -> None:
        self.provider = provider
        self.searcher = searcher
        clock = FixedClock(T0)
        router = Router(
            {"research": [ROUTE], "forecast": [ROUTE]}, {"openrouter": provider}, CATALOG, clock
        )
        self.ledger = Ledger.open(tmp_path / "loop.sqlite", clock)
        self.run_id = self.ledger.start_run(
            mode=RunMode.BACKTEST,
            tournament="t",
            provider="openrouter",
            model="m",
            settings={},
            code_version="test",
        )
        self.q: Question
        self.q, post = binary_question(T0)
        self.ledger.record_question(self.run_id, self.q, T0, post)
        self.researcher = Researcher(
            calls=CallRecorder(router, self.ledger, clock, Budget(1.0, 1.0)),
            ledger=self.ledger,
            searchers=[searcher],
            mode=RunMode.BACKTEST,
        )

    def roles(self) -> list[str]:
        rows = self.ledger.connection.execute("SELECT role FROM llm_calls ORDER BY started_at")
        return [r for (r,) in rows]


MakeRig = Callable[..., Rig]


@pytest.fixture
def make_rig(tmp_path: Path) -> Iterator[MakeRig]:
    rigs: list[Rig] = []

    def build(
        steps: list[str], searcher: ScriptSearcher | None = None, fail_on: int | None = None
    ) -> Rig:
        rig = Rig(tmp_path, LoopProvider(steps, fail_on), searcher or ScriptSearcher())
        rigs.append(rig)
        return rig

    yield build
    for rig in rigs:
        rig.ledger.close()


# -- parse_step --------------------------------------------------------------------------


def test_parse_step_reads_the_object() -> None:
    assert parse_step(step("It holds [S1].", "second query")) == (
        "It holds [S1].",
        "the date",
        "second query",
    )


def test_parse_step_tolerates_prose_and_fences() -> None:
    text = "Sure!\n```json\n" + step("b", "next one") + "\n```\nDone."
    assert parse_step(text) == ("b", "the date", "next one")


@pytest.mark.parametrize("stop", [None, "null", "None", "  ", "", "NULL"])
def test_parse_step_stops_on_a_null_like_next_query(stop: str | None) -> None:
    assert parse_step(step("b", stop))[2] is None


def test_parse_step_without_next_query_key_stops() -> None:
    assert parse_step('{"belief": "b", "unknown": "u"}') == ("b", "u", None)


def test_parse_step_truncates_a_long_query_and_stringifies_numbers() -> None:
    assert len(parse_step(step("b", "x" * 300))[2] or "") == 120
    assert parse_step('{"belief": "b", "next_query": 2025}')[2] == "2025"


def test_parse_step_on_a_list_or_garbage_returns_the_text_and_stops() -> None:
    assert parse_step("  just prose  ") == ("just prose", "", None)
    assert parse_step("{not json}") == ("{not json}", "", None)
    assert parse_step('{"a": 1')[2] is None
    assert parse_step("x" * 3000)[0] == "x" * 1200


# -- the loop ----------------------------------------------------------------------------


async def test_the_loop_runs_at_most_the_given_steps(make_rig: MakeRig) -> None:
    rig = make_rig([step("b1", "second"), step("b2", "third"), step("b3", "fourth")])
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=2)
    assert rig.searcher.queries == ["planned one", "second"]
    assert [s.query for s in result.steps] == ["planned one", "second"]
    assert len(rig.provider.loop_requests()) == 2
    assert result.queries == ["planned one", "second"]
    assert result.steps[-1].next_query == "third"  # asked for, never searched
    assert rig.roles() == ["planner", "loop", "loop"]


async def test_the_brief_is_the_last_belief_with_the_numbered_sources(make_rig: MakeRig) -> None:
    rig = make_rig([step("first view", "second"), step("final view [S2]", None, "the vote")])
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=5)
    assert len(result.steps) == 2
    assert result.brief.startswith("final view [S2]\n\nStill unknown: the vote\n\nSources:\n")
    assert "[S1] planned one 0 (script, dated 2026-11-15)" in result.brief
    assert "[S2] second 0 (script, dated 2026-11-15)" in result.brief
    assert "first view" not in result.brief
    assert [s.title for s in result.sources] == ["planned one 0", "second 0"]
    # The second call saw the first belief and the new source numbered after the old one.
    second = rig.provider.loop_requests()[1].messages[1].content
    assert "first view" in second
    assert "[S2] second 0" in second
    assert "Your notes so far: none yet." in rig.provider.loop_requests()[0].messages[1].content


async def test_a_null_next_query_stops_the_loop(make_rig: MakeRig) -> None:
    rig = make_rig([step("done", None)])
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=5)
    assert len(result.steps) == 1
    assert rig.searcher.queries == ["planned one"]


async def test_a_repeated_query_stops_the_loop_whatever_its_case(make_rig: MakeRig) -> None:
    rig = make_rig([step("b1", "Planned ONE"), step("b2", "never")])
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=5)
    assert len(result.steps) == 1
    assert rig.searcher.queries == ["planned one"]


async def test_a_repeat_of_an_earlier_but_not_the_last_query_also_stops(make_rig: MakeRig) -> None:
    rig = make_rig([step("b1", "second"), step("b2", "planned one"), step("b3", "never")])
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=5)
    assert [s.query for s in result.steps] == ["planned one", "second"]


async def test_an_expired_deadline_starts_no_search(make_rig: MakeRig) -> None:
    rig = make_rig([step("b", "x")])
    result = await rig.researcher.research_loop(
        rig.run_id, rig.q, T0, steps=3, deadline=time.monotonic() - 1
    )
    assert rig.searcher.queries == []
    assert rig.provider.loop_requests() == []
    assert result.steps == []
    assert result.sources == []
    assert result.brief == "No sources dated before the cutoff were found."


async def test_the_deadline_is_checked_before_each_step(make_rig: MakeRig) -> None:
    rig = make_rig([step("b1", "second"), step("b2", "third")])
    deadline = time.monotonic() + 0.2
    original = rig.searcher.search

    async def slow(query: str, as_of: datetime, limit: int) -> list[Source]:
        import asyncio

        found = await original(query, as_of, limit)
        await asyncio.sleep(0.25)  # the first search outlasts the deadline
        return found

    rig.searcher.search = slow  # type: ignore[method-assign]
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=5, deadline=deadline)
    assert len(result.steps) == 1
    assert result.brief.startswith("b1")


async def test_given_queries_skip_the_planner_and_the_first_one_starts(make_rig: MakeRig) -> None:
    rig = make_rig([step("b", None)])
    await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=2, queries=["mine", "other"])
    assert rig.searcher.queries == ["mine"]
    assert "planner" not in rig.roles()


async def test_without_queries_the_planner_runs_once(make_rig: MakeRig) -> None:
    rig = make_rig([step("b", None)])
    await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=2)
    assert rig.roles().count("planner") == 1
    assert any(r.messages[0].content == PLAN_SYSTEM.format(n=3) for r in rig.provider.requests)


async def test_zero_steps_is_a_programming_error(make_rig: MakeRig) -> None:
    rig = make_rig([step("b", None)])
    with pytest.raises(ValueError, match="at least one step"):
        await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=0)


async def test_every_source_is_gated_before_the_model_sees_it(make_rig: MakeRig) -> None:
    extra = [
        page("late page", T0 + timedelta(days=1), T0),
        page("undated page", None, T0),
        page("unproven page", T0 - timedelta(days=1), T0, evidence=DateEvidence.NONE),
    ]
    rig = make_rig([step("b", None)], ScriptSearcher(extra=extra))
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=2)
    [request] = rig.provider.loop_requests()
    shown = request.messages[1].content
    for title in ("late page", "undated page", "unproven page"):
        assert title not in shown
        assert title not in result.brief
    assert "planned one 0" in shown
    assert sorted(s.title for s, _ in result.rejected) == [
        "late page",
        "undated page",
        "unproven page",
    ]
    rows = rig.ledger.connection.execute(
        "SELECT s.title, u.admitted FROM source_uses u JOIN sources s USING (source_id) "
        "ORDER BY s.title"
    ).fetchall()
    assert rows == [
        ("late page", 0),
        ("planned one 0", 1),
        ("undated page", 0),
        ("unproven page", 0),
    ]
    cited = rig.ledger.connection.execute(
        "SELECT s.title FROM call_sources cs JOIN sources s USING (source_id)"
    ).fetchall()
    assert cited == [("planned one 0",)]


async def test_a_step_with_nothing_new_says_so(make_rig: MakeRig) -> None:
    only_late = ScriptSearcher(per_query=0, extra=[page("late", T0 + timedelta(days=1), T0)])
    rig = make_rig([step("nothing", None)], only_late)
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=2)
    assert (
        "no new source dated before the cutoff"
        in rig.provider.loop_requests()[0].messages[1].content
    )
    assert result.sources == []
    assert result.brief == "No sources dated before the cutoff were found."


async def test_per_step_caps_the_new_sources_shown(make_rig: MakeRig) -> None:
    rig = make_rig([step("b", None)], ScriptSearcher(per_query=6))
    rig.researcher._per_query = 6
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=1, per_step=2)
    assert len(result.sources) == 2
    assert result.steps[0].new_sources == 2


async def test_an_empty_belief_falls_back_to_the_one_pass_brief(make_rig: MakeRig) -> None:
    rig = make_rig([json.dumps({"belief": "", "unknown": "", "next_query": None})])
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=2)
    assert result.brief == "A one-pass brief [S1]."
    assert rig.roles()[-1] == "researcher"


async def test_a_failed_step_ends_the_loop_with_an_error_not_an_exception(
    make_rig: MakeRig,
) -> None:
    rig = make_rig([step("b1", "second"), step("b2", None)], fail_on=2)
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=4)
    assert len(result.steps) == 1
    assert any("loop step 2 failed" in e for e in result.errors)
    assert result.brief.startswith("b1")  # the last belief that did arrive


async def test_loop_steps_record_their_call_ids(make_rig: MakeRig) -> None:
    rig = make_rig([step("b1", "second"), step("b2", None)])
    result = await rig.researcher.research_loop(rig.run_id, rig.q, T0, steps=3)
    ids = {r for (r,) in rig.ledger.connection.execute("SELECT call_id FROM llm_calls")}
    assert {s.call_id for s in result.steps} <= ids
    assert None not in {s.call_id for s in result.steps}


# -- one-pass research with given queries ------------------------------------------------


async def test_research_with_given_queries_skips_the_planner(make_rig: MakeRig) -> None:
    rig = make_rig([])
    result = await rig.researcher.research(
        rig.run_id, rig.q, T0, queries=["a b", "c d", "e f", "g h"]
    )
    assert result.queries == ["a b", "c d", "e f"]  # capped at max_queries
    assert sorted(rig.searcher.queries) == ["a b", "c d", "e f"]
    assert rig.roles() == ["researcher"]
    assert result.brief == "A one-pass brief [S1]."


async def test_research_without_queries_plans_them(make_rig: MakeRig) -> None:
    rig = make_rig([])
    result = await rig.researcher.research(rig.run_id, rig.q, T0)
    assert result.queries == ["planned one", "planned two"]
    assert rig.roles() == ["planner", "researcher"]


async def test_an_empty_list_of_queries_is_planned_like_none(make_rig: MakeRig) -> None:
    rig = make_rig([])
    result = await rig.researcher.research(rig.run_id, rig.q, T0, queries=[])
    assert result.queries == ["planned one", "planned two"]
