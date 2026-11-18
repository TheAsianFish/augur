"""One researcher: plan queries, search as of the cutoff, gate, write a cited brief.

Order matters and is the whole point: every source a searcher returns is
written to the ledger, then the gate decides, and only admitted sources are
put in front of a model. The brief cites them as ``[S1]``, ``[S2]``...
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from augur.agents.calls import CallContext, CallRecorder
from augur.clock import to_iso
from augur.errors import AugurError, ProviderError
from augur.leakage import Admission, RunMode, admit_source
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, Message
from augur.log import get_logger
from augur.questions import Question
from augur.research.source import Searcher, Source

log = get_logger(__name__)

PLAN_SYSTEM = (
    "You plan web research for a forecaster. Reply with a JSON array of at most "
    "{n} short search queries (2 to 6 words each) for an encyclopedia and a news "
    "search engine, and nothing else."
)
BRIEF_SYSTEM = (
    "You are a research assistant for a forecaster. Using only the numbered sources "
    "below, write a factual brief of at most 250 words on what bears on the question: "
    "the current state of affairs, relevant base rates, and scheduled events before the "
    "question resolves. Cite sources as [S1], [S2]. If the sources say little that is "
    "relevant, say so plainly. Do not give a probability."
)


@dataclass
class ResearchResult:
    queries: list[str]
    sources: list[Source]
    """Admitted sources shown to the brief writer, in citation order (S1 first)."""
    rejected: list[tuple[Source, Admission]] = field(default_factory=list)
    brief: str = ""
    brief_call_id: str | None = None
    errors: list[str] = field(default_factory=list)

    def label(self, source_id: str) -> str:
        for i, s in enumerate(self.sources, start=1):
            if s.source_id == source_id:
                return f"S{i}"
        raise KeyError(source_id)


def _time_left(cutoff: datetime, q: Question) -> str:
    """Small models miscount days between two dates, so the prompt states it."""
    seconds = max(0.0, (q.scheduled_close_time - cutoff).total_seconds())
    if seconds < 2 * 86400:
        hours = round(seconds / 3600)
        return f"{hours} hour" + ("" if hours == 1 else "s")
    return f"{round(seconds / 86400)} days"


def question_text(q: Question, cutoff: datetime) -> str:
    parts = [
        f"Question: {q.title}",
        f"Today is {cutoff:%Y-%m-%d %H:%M} UTC. The question closes "
        f"{q.scheduled_close_time:%Y-%m-%d %H:%M} UTC, {_time_left(cutoff, q)} from now.",
    ]
    if q.description:
        parts.append(f"Background:\n{q.description.strip()}")
    if q.resolution_criteria:
        parts.append(f"Resolution criteria:\n{q.resolution_criteria.strip()}")
    if q.fine_print:
        parts.append(f"Fine print:\n{q.fine_print.strip()}")
    return "\n\n".join(parts)


def parse_queries(text: str, limit: int) -> list[str]:
    """Pull a list of queries out of a model reply; tolerate prose around the JSON."""
    match = re.search(r"\[.*?\]", text, re.S)
    queries: list[str] = []
    if match:
        try:
            raw = json.loads(match.group(0))
            queries = [str(x).strip() for x in raw if isinstance(x, str | int | float)]
        except json.JSONDecodeError:
            queries = []
    if not queries:
        queries = [
            line.strip(" -*\"'0123456789.") for line in text.splitlines() if line.strip(" -*")
        ]
    seen: list[str] = []
    for query in queries:
        if query and query.lower() not in (s.lower() for s in seen) and len(query) <= 120:
            seen.append(query)
    return seen[:limit]


def fallback_query(q: Question) -> str:
    words = re.sub(r"[^\w\s'-]", " ", q.title).split()
    stop = {"will", "the", "a", "an", "of", "by", "on", "in", "be", "to", "at", "before", "than"}
    return " ".join(w for w in words if w.lower() not in stop)[:100]


class Researcher:
    def __init__(
        self,
        *,
        calls: CallRecorder,
        ledger: Ledger,
        searchers: list[Searcher],
        mode: RunMode,
        max_queries: int = 3,
        per_query: int = 2,
        max_sources: int = 6,
    ) -> None:
        self._calls = calls
        self._ledger = ledger
        self._searchers = searchers
        self._mode = mode
        self._max_queries = max_queries
        self._per_query = per_query
        self._max_sources = max_sources

    async def research(self, run_id: str, q: Question, cutoff: datetime) -> ResearchResult:
        result = ResearchResult(queries=[], sources=[])
        result.queries = await self._plan(run_id, q, cutoff, result)
        found = await self._search_all(result.queries, cutoff, result)
        for source in found:
            # Stored before any model can see it; then the gate decides.
            self._ledger.record_source(source)
            decision = admit_source(source, cutoff, self._mode)
            self._ledger.record_admission(run_id, q.question_id, decision)
            if decision.admitted:
                if len(result.sources) < self._max_sources:
                    result.sources.append(source)
            else:
                result.rejected.append((source, decision))
                log.warning(
                    "source rejected",
                    source_id=source.source_id,
                    url=source.url,
                    reason=decision.reason,
                )
        await self._brief(run_id, q, cutoff, result)
        return result

    async def _plan(
        self, run_id: str, q: Question, cutoff: datetime, result: ResearchResult
    ) -> list[str]:
        request = LlmRequest(
            model="",  # the router sets each route's model
            messages=(
                Message(role="system", content=PLAN_SYSTEM.format(n=self._max_queries)),
                Message(role="user", content=question_text(q, cutoff)),
            ),
            temperature=0.0,
            seed=0,
            max_tokens=600,  # a reasoning model emptied 200 on thinking (M2)
        )
        ctx = CallContext(run_id=run_id, question_id=q.question_id, role="planner")
        queries: list[str] = []
        try:
            done = await self._calls.complete(ctx, request, source_ids=[])
            queries = parse_queries(done.response.text, self._max_queries)
        except ProviderError as exc:
            result.errors.append(f"query planning failed: {exc}")
        if not queries:
            queries = [fallback_query(q)]
        return queries

    async def _search_all(
        self, queries: list[str], cutoff: datetime, result: ResearchResult
    ) -> list[Source]:
        async def one(searcher: Searcher, query: str) -> list[Source]:
            try:
                return await searcher.search(query, cutoff, self._per_query)
            except AugurError as exc:
                result.errors.append(f"{searcher.name} failed for {query!r}: {exc}")
                log.warning("searcher failed", searcher=searcher.name, query=query, error=str(exc))
                return []

        # Bounded: at most len(searchers) x max_queries tasks, each behind its searcher's limiter.
        batches = await asyncio.gather(
            *(one(searcher, query) for query in queries for searcher in self._searchers)
        )
        seen: set[str] = set()
        ordered: list[Source] = []
        # Interleave so that each query's best hit comes before any query's second hit.
        for rank in range(max((len(b) for b in batches), default=0)):
            for batch in batches:
                if rank < len(batch) and batch[rank].source_id not in seen:
                    seen.add(batch[rank].source_id)
                    ordered.append(batch[rank])
        return ordered

    async def _brief(
        self, run_id: str, q: Question, cutoff: datetime, result: ResearchResult
    ) -> None:
        if not result.sources:
            result.brief = "No sources dated before the cutoff were found."
            return
        blocks = [
            f"[S{i}] {s.title} ({s.searcher}, dated {to_iso(s.published_at)[:10]})\n{s.excerpt}"
            if s.published_at
            else f"[S{i}] {s.title} ({s.searcher}, undated)\n{s.excerpt}"
            for i, s in enumerate(result.sources, start=1)
        ]
        request = LlmRequest(
            model="",  # the router sets each route's model
            messages=(
                Message(role="system", content=BRIEF_SYSTEM),
                Message(
                    role="user",
                    content=question_text(q, cutoff) + "\n\nSources:\n\n" + "\n\n".join(blocks),
                ),
            ),
            temperature=0.0,
            seed=0,
            max_tokens=1000,
        )
        ctx = CallContext(run_id=run_id, question_id=q.question_id, role="researcher")
        try:
            done = await self._calls.complete(
                ctx, request, source_ids=[s.source_id for s in result.sources]
            )
            result.brief = done.response.text.strip()
            result.brief_call_id = done.call_id
        except ProviderError as exc:
            result.errors.append(f"brief failed: {exc}")
            result.brief = "The research brief could not be written."
