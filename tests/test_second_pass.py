"""The second research pass: sub-questions, their searches, the gate, the rewritten brief."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from augur.agents.calls import Budget, CallRecorder
from augur.agents.researcher import BRIEF_SYSTEM, Researcher, parse_subquestions
from augur.clock import FixedClock
from augur.errors import ProviderError
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, LlmResponse
from augur.llm.router import Router
from augur.research.source import Source

from .conftest import T0
from .helpers import binary_question
from .test_research_loop import CATALOG, ROUTE, ScriptSearcher, page

SUBQ = json.dumps(
    [
        {"sub_question": "Has the vote been scheduled?", "query": "vote schedule"},
        {"sub_question": "What would block it?", "query": "veto threat"},
    ]
)


class PassProvider:
    """Plans two queries, writes briefs, and answers the sub-question agent from a script."""

    name = "openrouter"

    def __init__(self, subq: str = SUBQ, fail_subq: bool = False) -> None:
        self.subq = subq
        self.fail_subq = fail_subq
        self.requests: list[LlmRequest] = []

    def briefs(self) -> list[LlmRequest]:
        return [r for r in self.requests if r.messages[0].content == BRIEF_SYSTEM]

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        system = request.messages[0].content
        if system.startswith("You review research"):
            if self.fail_subq:
                raise ProviderError("scripted failure")
            text = self.subq
        elif system == BRIEF_SYSTEM:
            text = f"Brief number {len(self.briefs())} [S1]."
        else:
            text = '["planned one", "planned two"]'
        return LlmResponse(text=text, model=request.model, input_tokens=1, output_tokens=1)

    async def aclose(self) -> None:
        return None


def rig(
    tmp_path: Path, provider: PassProvider, searcher: ScriptSearcher
) -> tuple[Researcher, Ledger, str]:
    clock = FixedClock(T0)
    router = Router(
        {"research": [ROUTE], "forecast": [ROUTE]}, {"openrouter": provider}, CATALOG, clock
    )
    ledger = Ledger.open(tmp_path / "pass.sqlite", clock)
    run_id = ledger.start_run(
        mode=RunMode.BACKTEST,
        tournament="t",
        provider="openrouter",
        model="m",
        settings={},
        code_version="test",
    )
    q, post = binary_question(T0)
    ledger.record_question(run_id, q, T0, post)
    researcher = Researcher(
        calls=CallRecorder(router, ledger, clock, Budget(1.0, 1.0)),
        ledger=ledger,
        searchers=[searcher],
        mode=RunMode.BACKTEST,
    )
    return researcher, ledger, run_id


async def test_two_passes_search_the_sub_questions_and_rewrite_the_brief(tmp_path: Path) -> None:
    provider = PassProvider()
    searcher = ScriptSearcher()
    researcher, ledger, run_id = rig(tmp_path, provider, searcher)
    q, _ = binary_question(T0)
    result = await researcher.research(run_id, q, T0, passes=2)
    ledger.close()
    assert searcher.queries == ["planned one", "planned two", "vote schedule", "veto threat"]
    assert result.sub_questions == [
        ("Has the vote been scheduled?", "vote schedule"),
        ("What would block it?", "veto threat"),
    ]
    assert len(result.sources) == 4
    assert result.first_brief == "Brief number 1 [S1]."
    assert result.brief == "Brief number 2 [S1]."
    # The final brief is asked to answer the sub-questions, from every admitted source.
    final = provider.briefs()[-1].messages[1].content
    assert "- Has the vote been scheduled?" in final
    assert "[S4]" in final
    # The sub-question agent read the first brief.
    review = [r for r in provider.requests if r.messages[0].content.startswith("You review")]
    assert "Brief number 1" in review[0].messages[1].content


async def test_one_pass_is_unchanged(tmp_path: Path) -> None:
    provider = PassProvider()
    researcher, ledger, run_id = rig(tmp_path, provider, ScriptSearcher())
    q, _ = binary_question(T0)
    result = await researcher.research(run_id, q, T0)
    ledger.close()
    assert len(provider.requests) == 2  # planner and brief
    assert result.sub_questions == []
    assert result.first_brief == ""


async def test_a_late_source_from_the_second_pass_never_reaches_the_brief(tmp_path: Path) -> None:
    late = page("from the future", T0 + timedelta(days=1), T0)
    provider = PassProvider()
    searcher = ScriptSearcher(extra=[late])
    researcher, ledger, run_id = rig(tmp_path, provider, searcher)
    q, _ = binary_question(T0)
    result = await researcher.research(run_id, q, T0, passes=2)
    shown: list[Source] = result.sources
    assert late.source_id not in {s.source_id for s in shown}
    assert late.source_id in {s.source_id for s, _ in result.rejected}
    assert all("from the future" not in r.messages[1].content for r in provider.briefs())
    ledger.close()


async def test_a_failed_second_pass_keeps_the_first_brief(tmp_path: Path) -> None:
    provider = PassProvider(fail_subq=True)
    researcher, ledger, run_id = rig(tmp_path, provider, ScriptSearcher())
    q, _ = binary_question(T0)
    result = await researcher.research(run_id, q, T0, passes=2)
    ledger.close()
    assert result.brief == "Brief number 1 [S1]."
    assert any("second pass failed" in e for e in result.errors)


async def test_repeated_queries_are_dropped_and_nothing_new_keeps_the_brief(
    tmp_path: Path,
) -> None:
    provider = PassProvider(subq='["Planned One", "planned two"]')
    searcher = ScriptSearcher()
    researcher, ledger, run_id = rig(tmp_path, provider, searcher)
    q, _ = binary_question(T0)
    result = await researcher.research(run_id, q, T0, passes=2)
    ledger.close()
    assert searcher.queries == ["planned one", "planned two"]
    assert result.brief == "Brief number 1 [S1]."
    assert len(provider.briefs()) == 1


def test_parse_subquestions_reads_objects_strings_and_prose() -> None:
    assert parse_subquestions("Here:\n" + SUBQ + "\nthanks", 3) == [
        ("Has the vote been scheduled?", "vote schedule"),
        ("What would block it?", "veto threat"),
    ]
    assert parse_subquestions('["a b", "A B", 7, {"query": ""}]', 3) == [("", "a b")]
    assert parse_subquestions("no json here", 3) == []
    assert len(parse_subquestions(json.dumps([{"query": f"q{i}"} for i in range(9)]), 3)) == 3


@pytest.mark.parametrize("passes", [0, 3])
async def test_only_one_or_two_passes(tmp_path: Path, passes: int) -> None:
    researcher, ledger, run_id = rig(tmp_path, PassProvider(), ScriptSearcher())
    q, _ = binary_question(T0)
    with pytest.raises(ValueError, match="one or two"):
        await researcher.research(run_id, q, T0, passes=passes)
    ledger.close()
