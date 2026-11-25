"""One researcher: plan queries, search as of the cutoff, gate, write a cited brief.

Order matters and is the whole point: every source a searcher returns is
written to the ledger, then the gate decides, and only admitted sources are
put in front of a model. The brief cites them as ``[S1]``, ``[S2]``...
"""

from __future__ import annotations

import asyncio
import json
import re
import time
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
from augur.research.source import QuestionSearcher, Searcher, Source

log = get_logger(__name__)

# Reasoning models spend part of these before answering: a free one emptied a
# 200-token plan on thinking (M2).
PLAN_MAX_TOKENS = 2000
BRIEF_MAX_TOKENS = 4000

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


SUBQ_MAX_TOKENS = 3000
SUBQ_SYSTEM = (
    "You review research for a forecaster. You get the question and a brief written "
    "from the first round of searching. Find what is missing: write at most {n} "
    "sub-questions whose answers would most change the forecast, especially evidence "
    "that would overturn what the brief suggests, each with one search query (2 to 6 "
    "words) for an encyclopedia that could answer it. Reply with a JSON array of "
    'objects and nothing else: [{{"sub_question": "...", "query": "..."}}].'
)


def parse_subquestions(text: str, limit: int) -> list[tuple[str, str]]:
    """``(sub_question, query)`` pairs from a second-pass reply; prose around the JSON is fine."""
    match = re.search(r"\[.*\]", text, re.S)
    pairs: list[tuple[str, str]] = []
    if match:
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            raw = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, dict):
                sub = str(item.get("sub_question") or "").strip()
                query = str(item.get("query") or "").strip()
            elif isinstance(item, str):
                sub, query = "", item.strip()
            else:
                continue
            if query and len(query) <= 120 and query.lower() not in (p[1].lower() for p in pairs):
                pairs.append((sub[:300], query))
    return pairs[:limit]


LOOP_MAX_TOKENS = 3000
LOOP_SYSTEM = (
    "You research a forecasting question one search at a time. You get the question, "
    "your notes so far and the new numbered sources from the last search. Reply with "
    "one JSON object and nothing else: "
    '{{"belief": "at most 120 words: what all the evidence so far says about how the '
    'question will resolve, citing sources as [S1], [S2]", '
    '"unknown": "the one thing still unknown that would most change the forecast", '
    '"next_query": "a 2 to 6 word search query that would find it, or null to stop"}}. '
    "Use only the sources and the question; do not give a probability. "
    "You have {left} search(es) left."
)


@dataclass(frozen=True)
class LoopStep:
    """One step of the research loop: what was searched, and what the model wrote."""

    query: str
    new_sources: int
    belief: str
    unknown: str
    next_query: str | None
    call_id: str | None


def parse_step(text: str) -> tuple[str, str, str | None]:
    """``(belief, unknown, next_query)`` from a loop reply; tolerate prose around the JSON."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            raw = None
        if isinstance(raw, dict):
            belief = str(raw.get("belief") or "").strip()
            unknown = str(raw.get("unknown") or "").strip()
            nxt = raw.get("next_query")
            query = str(nxt).strip() if isinstance(nxt, str | int | float) else ""
            if query.lower() in ("", "null", "none"):
                return belief, unknown, None
            return belief, unknown, query[:120]
    return text.strip()[:1200], "", None


@dataclass
class ResearchResult:
    queries: list[str]
    sources: list[Source]
    """Admitted sources shown to the brief writer, in citation order (S1 first)."""
    rejected: list[tuple[Source, Admission]] = field(default_factory=list)
    brief: str = ""
    brief_call_id: str | None = None
    errors: list[str] = field(default_factory=list)
    steps: list[LoopStep] = field(default_factory=list)
    """The research loop's steps, in order (empty for the one-pass researcher)."""
    sub_questions: list[tuple[str, str]] = field(default_factory=list)
    """The second pass's ``(sub_question, query)`` pairs (empty for one pass)."""
    first_brief: str = ""
    """With two passes, the first pass's brief (``brief`` is the final one)."""

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


def _source_block(i: int, s: Source) -> str:
    if s.published_at:
        return f"[S{i}] {s.title} ({s.searcher}, dated {to_iso(s.published_at)[:10]})\n{s.excerpt}"
    return f"[S{i}] {s.title} ({s.searcher}, undated)\n{s.excerpt}"


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

    async def research(
        self,
        run_id: str,
        q: Question,
        cutoff: datetime,
        queries: list[str] | None = None,
        passes: int = 1,
    ) -> ResearchResult:
        """Plan (unless ``queries`` are given), search, gate, write the brief.

        With ``passes=2`` a second research agent then reads that brief,
        writes the sub-questions whose answers would most change the
        forecast (evidence against the brief's lean first), searches for
        them, and the brief is rewritten from every admitted source.
        """
        if passes not in (1, 2):
            raise ValueError("research runs one or two passes")
        result = ResearchResult(queries=[], sources=[])
        result.queries = (
            list(queries[: self._max_queries])
            if queries
            else await self._plan(run_id, q, cutoff, result)
        )
        found = await self._search_question(q, cutoff, result)
        found += await self._search_all(result.queries, cutoff, result)
        self._gate(run_id, q, cutoff, found, result, self._max_sources)
        await self._brief(run_id, q, cutoff, result)
        if passes == 2:
            await self.second_pass(run_id, q, cutoff, result)
        return result

    async def research_added(
        self,
        run_id: str,
        q: Question,
        cutoff: datetime,
        base: list[Source],
        queries: list[str],
        cap: int,
    ) -> tuple[ResearchResult, list[Source]]:
        """M13's arm G: ``base`` (another arm's evidence) plus what the searchers add.

        ``base`` goes through the gate first, unchanged and in order; then each
        query is searched and up to ``cap`` new sources are admitted after it.
        Returns the research and the added sources; the brief is written only
        when something was added. A searcher that fails raises, so a question whose search did
        not happen is never mistaken for one that found nothing.
        """
        result = ResearchResult(queries=list(queries[: self._max_queries]), sources=[])
        self._gate(run_id, q, cutoff, list(base), result, len(base))
        if len(result.sources) != len(base):
            raise AugurError(
                f"question {q.question_id}: {len(base) - len(result.sources)} base source(s) "
                "were refused by the gate; the base arm's evidence must be dated at the cutoff"
            )
        found = await self._search_all(result.queries, cutoff, result)
        if result.errors:
            raise AugurError("; ".join(result.errors))
        added = self._gate(run_id, q, cutoff, found, result, cap)
        if added:
            await self._brief(run_id, q, cutoff, result)
            if result.errors:
                raise AugurError("; ".join(result.errors))
        return result, added

    async def second_pass(
        self, run_id: str, q: Question, cutoff: datetime, result: ResearchResult
    ) -> None:
        """Deepen a finished first pass in place; ``result.brief`` changes only on success."""
        request = LlmRequest(
            model="",  # the router sets each route's model
            messages=(
                Message(role="system", content=SUBQ_SYSTEM.format(n=self._max_queries)),
                Message(
                    role="user",
                    content=question_text(q, cutoff) + "\n\nFirst brief:\n" + result.brief,
                ),
            ),
            temperature=0.0,
            seed=0,
            max_tokens=SUBQ_MAX_TOKENS,
        )
        ctx = CallContext(run_id=run_id, question_id=q.question_id, role="subquestions")
        try:
            done = await self._calls.complete(
                ctx, request, source_ids=[s.source_id for s in result.sources]
            )
        except ProviderError as exc:
            result.errors.append(f"second pass failed: {exc}")
            return
        seen = {x.lower() for x in result.queries}
        pairs = [
            p
            for p in parse_subquestions(done.response.text, self._max_queries)
            if p[1].lower() not in seen
        ]
        result.sub_questions = pairs
        if not pairs:
            result.errors.append("second pass wrote no new query; the first brief stands")
            return
        result.queries.extend(p[1] for p in pairs)
        found = await self._search_all([p[1] for p in pairs], cutoff, result)
        new = self._gate(run_id, q, cutoff, found, result, self._max_sources)
        if not new:
            # Nothing new to read: the first brief already covers every source.
            return
        first, first_call = result.brief, result.brief_call_id
        result.first_brief = first
        failed = len(result.errors)
        await self._brief(run_id, q, cutoff, result, sub_questions=[p[0] for p in pairs if p[0]])
        if len(result.errors) > failed:
            # The rewrite failed: the first brief is still a good brief.
            result.brief, result.brief_call_id = first, first_call
            result.errors.append("the rewritten brief failed; the first brief stands")

    def _gate(
        self,
        run_id: str,
        q: Question,
        cutoff: datetime,
        found: list[Source],
        result: ResearchResult,
        cap: int,
    ) -> list[Source]:
        """Store each source, let the gate decide, keep up to ``cap`` new admitted ones."""
        shown = {s.source_id for s in result.sources}
        kept: list[Source] = []
        for source in found:
            if source.source_id in shown:
                continue
            # Stored before any model can see it; then the gate decides.
            self._ledger.record_source(source)
            decision = admit_source(source, cutoff, self._mode)
            self._ledger.record_admission(run_id, q.question_id, decision)
            if decision.admitted:
                if len(kept) < cap:
                    kept.append(source)
                    shown.add(source.source_id)
            else:
                result.rejected.append((source, decision))
                log.warning(
                    "source rejected",
                    source_id=source.source_id,
                    url=source.url,
                    reason=decision.reason,
                )
        result.sources.extend(kept)
        return kept

    async def research_loop(
        self,
        run_id: str,
        q: Question,
        cutoff: datetime,
        *,
        steps: int,
        queries: list[str] | None = None,
        per_step: int = 4,
        deadline: float | None = None,
    ) -> ResearchResult:
        """Search, read, write down a belief and what is unknown, pick the next query.

        Bounded: at most ``steps`` searches, and none started after
        ``deadline`` (a ``time.monotonic()`` instant). The first query is the
        planner's (or the first of ``queries``); the question searchers run
        once, before it. Every source passes the gate before the model sees
        it. The brief the forecasters read is the last belief and the open
        question, citing the sources by number.
        """
        if steps < 1:
            raise ValueError("the research loop needs at least one step")
        result = ResearchResult(queries=[], sources=[])
        planned = list(queries) if queries else await self._plan(run_id, q, cutoff, result)
        query: str | None = planned[0] if planned else fallback_query(q)
        pending = await self._search_question(q, cutoff, result)
        belief = unknown = ""
        for step in range(steps):
            if query is None or (deadline is not None and time.monotonic() >= deadline):
                break
            result.queries.append(query)
            pending += await self._search_all([query], cutoff, result)
            first = len(result.sources)
            new = self._gate(run_id, q, cutoff, pending, result, per_step)
            pending = []
            blocks = [_source_block(first + i, s) for i, s in enumerate(new, start=1)] or [
                "(this search found no new source dated before the cutoff)"
            ]
            notes = (
                f"Your notes so far:\n{belief}\nStill unknown: {unknown}"
                if belief
                else "Your notes so far: none yet."
            )
            request = LlmRequest(
                model="",  # the router sets each route's model
                messages=(
                    Message(role="system", content=LOOP_SYSTEM.format(left=steps - step - 1)),
                    Message(
                        role="user",
                        content=question_text(q, cutoff)
                        + f"\n\n{notes}\n\nSearch {step + 1}: {query!r}. New sources:\n\n"
                        + "\n\n".join(blocks),
                    ),
                ),
                temperature=0.0,
                seed=0,
                max_tokens=LOOP_MAX_TOKENS,
            )
            ctx = CallContext(run_id=run_id, question_id=q.question_id, role="loop")
            try:
                done = await self._calls.complete(
                    ctx, request, source_ids=[s.source_id for s in result.sources]
                )
            except ProviderError as exc:
                result.errors.append(f"loop step {step + 1} failed: {exc}")
                break
            new_belief, new_unknown, nxt = parse_step(done.response.text)
            belief = new_belief or belief
            unknown = new_unknown or unknown
            result.steps.append(
                LoopStep(query, len(new), new_belief, new_unknown, nxt, done.call_id)
            )
            # A repeated query would find the same pages: stop instead.
            seen = {x.lower() for x in result.queries}
            query = nxt if nxt and nxt.lower() not in seen else None
        if not result.sources:
            result.brief = "No sources dated before the cutoff were found."
        elif belief:
            result.brief = (
                f"{belief}\n\nStill unknown: {unknown or 'not stated'}\n\nSources:\n"
                + "\n".join(
                    f"[S{i}] {s.title} ({s.searcher}, "
                    f"dated {to_iso(s.published_at)[:10] if s.published_at else 'undated'})"
                    for i, s in enumerate(result.sources, start=1)
                )
            )
        else:
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
            max_tokens=PLAN_MAX_TOKENS,
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

    async def _search_question(
        self, q: Question, cutoff: datetime, result: ResearchResult
    ) -> list[Source]:
        """Searchers that look up the question itself (its market price) go first, once."""
        found: list[Source] = []
        for searcher in self._searchers:
            if not isinstance(searcher, QuestionSearcher):
                continue
            try:
                found += await searcher.search_question(q, cutoff, self._per_query)
            except AugurError as exc:
                result.errors.append(f"{searcher.name} failed for the question: {exc}")
                log.warning("searcher failed", searcher=searcher.name, error=str(exc))
        return found

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
        by_query = [s for s in self._searchers if not isinstance(s, QuestionSearcher)]
        batches = await asyncio.gather(
            *(one(searcher, query) for query in queries for searcher in by_query)
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
        self,
        run_id: str,
        q: Question,
        cutoff: datetime,
        result: ResearchResult,
        sub_questions: list[str] | None = None,
    ) -> None:
        if not result.sources:
            result.brief = "No sources dated before the cutoff were found."
            return
        blocks = [_source_block(i, s) for i, s in enumerate(result.sources, start=1)]
        request = LlmRequest(
            model="",  # the router sets each route's model
            messages=(
                Message(role="system", content=BRIEF_SYSTEM),
                Message(
                    role="user",
                    content=question_text(q, cutoff)
                    + (
                        "\n\nAnswer these sub-questions where the sources allow:\n"
                        + "\n".join(f"- {x}" for x in sub_questions)
                        if sub_questions
                        else ""
                    )
                    + "\n\nSources:\n\n"
                    + "\n\n".join(blocks),
                ),
            ),
            temperature=0.0,
            seed=0,
            max_tokens=BRIEF_MAX_TOKENS,
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
