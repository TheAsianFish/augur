"""M13's arm G: a base arm's evidence plus what the news searchers add."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from augur.agents.researcher import BRIEF_SYSTEM
from augur.backtest import NOTHING_ADDED, Retrieval
from augur.clock import FixedClock
from augur.errors import AugurError, SearchError
from augur.llm.endpoints import Route
from augur.llm.router import Router
from augur.research.source import Source

from .fakes.forecastbench import LOCAL, free_local_catalog, small_set
from .fakes.llm import ScriptedProvider
from .test_backtest_arms import _rows, _run
from .test_research_loop import LoopProvider, Rig, ScriptSearcher, page

CUTOFF = datetime(2025, 3, 1, tzinfo=UTC)
EARLY = CUTOFF - timedelta(days=5)


class FailingSearcher:
    name = "failing"

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        raise SearchError("GDELT refused")


class NoNews:
    name = "empty"

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        return []


class NewsSearcher:
    """One distinct page per query, dated before the cutoff."""

    name = "news"

    def __init__(self, per_query: int = 1) -> None:
        self.per_query = per_query

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        return [page(f"news {query} {i}", as_of - timedelta(days=3), as_of) for i in range(10)][
            : self.per_query
        ]


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    r = Rig(tmp_path, LoopProvider([]), ScriptSearcher(per_query=3))
    yield r
    r.ledger.close()


def titles(sources: list[Source]) -> list[str]:
    return [s.title for s in sources]


# -- Researcher.research_added -----------------------------------------------------------


async def test_base_comes_first_in_order_then_at_most_cap_added(rig: Rig) -> None:
    base = [page(f"base {i}", CUTOFF - timedelta(days=10 + i), CUTOFF) for i in range(2)]
    result, added = await rig.researcher.research_added(
        rig.run_id, rig.q, CUTOFF, base, ["alpha", "beta"], cap=3
    )
    assert titles(result.sources[:2]) == ["base 0", "base 1"]
    assert len(added) == 3
    assert result.sources[2:] == added
    assert len(result.sources) == 5
    assert all(s not in base for s in added)
    assert result.queries == ["alpha", "beta"]
    assert result.brief
    assert rig.provider.requests[-1].messages[0].content == BRIEF_SYSTEM


async def test_brief_cites_base_as_s1_onwards(rig: Rig) -> None:
    base = [page("base 0", EARLY, CUTOFF), page("base 1", EARLY, CUTOFF)]
    await rig.researcher.research_added(rig.run_id, rig.q, CUTOFF, base, ["alpha"], cap=1)
    brief = rig.provider.requests[-1].messages[-1].content
    assert brief.index("base 0") < brief.index("base 1") < brief.index("alpha 0")
    assert "[S1]" in brief
    assert "[S3]" in brief
    assert "[S4]" not in brief


async def test_a_source_the_base_already_holds_is_not_added_again(rig: Rig) -> None:
    dup = page("alpha 0", CUTOFF - timedelta(days=2), CUTOFF)  # what the searcher returns first
    result, added = await rig.researcher.research_added(
        rig.run_id, rig.q, CUTOFF, [dup], ["alpha"], cap=1
    )
    assert dup not in added
    assert titles(result.sources).count("alpha 0") == 1


async def test_nothing_added_makes_no_brief_call(tmp_path: Path) -> None:
    r = Rig(tmp_path, LoopProvider([]), ScriptSearcher(per_query=0))
    base = [page("base 0", EARLY, CUTOFF)]
    result, added = await r.researcher.research_added(r.run_id, r.q, CUTOFF, base, ["alpha"], 3)
    assert added == []
    assert titles(result.sources) == ["base 0"]
    assert r.provider.requests == []
    assert r.roles() == []
    r.ledger.close()


async def test_base_dated_after_the_cutoff_is_refused(rig: Rig) -> None:
    base = [page("fine", EARLY, CUTOFF), page("late", CUTOFF + timedelta(days=1), CUTOFF)]
    with pytest.raises(AugurError, match="refused by the gate"):
        await rig.researcher.research_added(rig.run_id, rig.q, CUTOFF, base, ["alpha"], cap=3)
    assert rig.provider.requests == []


async def test_added_source_dated_after_the_cutoff_is_not_kept(tmp_path: Path) -> None:
    late = page("late news", CUTOFF + timedelta(days=1), CUTOFF)
    r = Rig(tmp_path, LoopProvider([]), ScriptSearcher(per_query=0, extra=[late]))
    result, added = await r.researcher.research_added(r.run_id, r.q, CUTOFF, [], ["alpha"], 3)
    assert added == []
    assert result.sources == []
    assert [s for s, _ in result.rejected] == [late]
    r.ledger.close()


async def test_a_failing_searcher_raises_rather_than_finding_nothing(tmp_path: Path) -> None:
    r = Rig(tmp_path, LoopProvider([]), ScriptSearcher())
    r.researcher._searchers = [FailingSearcher()]
    with pytest.raises(AugurError, match="GDELT refused"):
        await r.researcher.research_added(r.run_id, r.q, CUTOFF, [], ["alpha"], 3)
    assert r.provider.requests == []
    r.ledger.close()


# -- run_forecastbench with Retrieval.base -----------------------------------------------


def test_label_names_the_news_when_a_base_is_set() -> None:
    assert Retrieval([], base={}).label() == "onepass-asof-open+news"
    assert Retrieval([]).label() == "onepass-asof-open"


def router_and_provider() -> tuple[Router, ScriptedProvider]:
    provider = ScriptedProvider((30,))
    route: Route = LOCAL
    return Router.single(route, provider, free_local_catalog(), FixedClock(CUTOFF)), provider


def briefs(provider: ScriptedProvider) -> int:
    return sum(1 for r in provider.requests if r.messages[0].content == BRIEF_SYSTEM)


async def test_nothing_added_skips_the_question_and_writes_no_forecast(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    q = questions[0]
    base = {q.question_id: [page("base 0", EARLY, CUTOFF)]}
    router, provider = router_and_provider()
    db = tmp_path / "g.sqlite"
    retrieval = Retrieval([NoNews()], queries={q.question_id: ["alpha"]}, base=base)
    result = await _run(db, [q], retrieval, router=router)
    assert (result.forecast, result.skipped, result.failed) == (0, 1, 0)
    assert _rows(db, "SELECT status, reason FROM question_outcomes") == [("skipped", NOTHING_ADDED)]
    assert NOTHING_ADDED.startswith("nothing added")
    assert _rows(db, "SELECT COUNT(*) FROM forecasts") == [(0,)]
    assert provider.requests == []  # no brief, no draws


async def test_news_added_writes_a_brief_draws_and_the_aggregate(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    q = questions[0]
    base = {q.question_id: [page("base 0", EARLY, CUTOFF)]}
    router, provider = router_and_provider()
    db = tmp_path / "g.sqlite"
    retrieval = Retrieval(
        [NewsSearcher()], queries={q.question_id: ["alpha"]}, base=base, added_cap=2
    )
    result = await _run(db, [q], retrieval, router=router)
    assert (result.forecast, result.skipped, result.failed) == (1, 0, 0)
    assert briefs(provider) == 1
    assert _rows(db, "SELECT status FROM question_outcomes") == [("forecast",)]
    assert _rows(db, "SELECT kind, COUNT(*) FROM forecasts GROUP BY kind ORDER BY kind") == [
        ("aggregate", 1),
        ("member", 3),
    ]
    [(reason,)] = _rows(db, "SELECT reason FROM question_outcomes")
    assert reason == "forecast with 2 source(s)"


async def test_a_question_without_a_base_entry_is_treated_as_an_empty_base(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    q = questions[0]
    router, provider = router_and_provider()
    db = tmp_path / "g.sqlite"
    retrieval = Retrieval([NewsSearcher()], queries={q.question_id: ["alpha"]}, base={})
    result = await _run(db, [q], retrieval, router=router)
    assert result.forecast == 1
    assert briefs(provider) == 1


async def test_a_failing_searcher_fails_the_question_not_skips_it(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    q = questions[0]
    router, provider = router_and_provider()
    db = tmp_path / "g.sqlite"
    retrieval = Retrieval(
        [FailingSearcher()],
        queries={q.question_id: ["alpha"]},
        base={q.question_id: [page("base 0", EARLY, CUTOFF)]},
    )
    result = await _run(db, [q], retrieval, router=router)
    assert (result.forecast, result.skipped, result.failed) == (0, 0, 1)
    [(status, reason)] = _rows(db, "SELECT status, reason FROM question_outcomes")
    assert status == "failed"
    assert "GDELT refused" in reason
    assert _rows(db, "SELECT COUNT(*) FROM forecasts") == [(0,)]
    assert provider.requests == []


async def test_a_base_source_dated_after_the_cutoff_fails_the_question(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    q = questions[0]
    router, provider = router_and_provider()
    db = tmp_path / "g.sqlite"
    late = page("late base", CUTOFF + timedelta(days=2), CUTOFF)
    retrieval = Retrieval(
        [NewsSearcher()], queries={q.question_id: ["alpha"]}, base={q.question_id: [late]}
    )
    result = await _run(db, [q], retrieval, router=router)
    assert (result.forecast, result.failed) == (0, 1)
    [(status, reason)] = _rows(db, "SELECT status, reason FROM question_outcomes")
    assert status == "failed"
    assert "refused by the gate" in reason
    assert provider.requests == []
