"""Backtests: the bot's own forecasting code on resolved questions, clock pinned.

M4 runs the tokenless ForecastBench number: the forecaster draws and the
median from :mod:`augur.agents.forecaster`, with no retrieval, on each
eligible question as of its freeze date. Every call, forecast and outcome is
a ledger row in a ``backtest`` run, so ``tests/test_leakage.py`` checks the
model's release date against each question, and a run can be resumed: a
question that already has an aggregate in the ledger is not forecast again.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from augur.agents.calls import Budget, CallRecorder
from augur.agents.forecaster import draw_forecasts
from augur.agents.researcher import Researcher, ResearchResult
from augur.clock import Clock, ensure_utc
from augur.config import Settings
from augur.errors import AugurError, LeakageError
from augur.forecastbench import FBQuestion
from augur.leakage import RunMode, admit_model, admit_source
from augur.ledger import Ledger
from augur.llm.endpoints import Route
from augur.llm.router import Router
from augur.log import get_logger
from augur.predictions import BinaryPrediction, aggregate, values_json
from augur.research.source import Searcher

log = get_logger(__name__)

NO_RESEARCH_BRIEF = (
    "No research was done for this question: forecast from the question text and what "
    "you already know."
)


class PinnedClock:
    """A clock a backtest moves to each question's cutoff before forecasting it."""

    def __init__(self, instant: datetime) -> None:
        self._instant = ensure_utc(instant)

    def now(self) -> datetime:
        return self._instant

    def set(self, instant: datetime) -> None:
        self._instant = ensure_utc(instant)


@dataclass(frozen=True)
class Retrieval:
    """How a backtest arm researches each question (Q2's arms and the research loop).

    ``as_of="open"`` reads every source as of the question's open (freeze)
    instant: a backtest. ``as_of="now"`` reads them as of the moment the run
    reaches the question: Q2's leaky arm, recorded as an ``unrestricted``
    run whose cutoff is that moment, never as a backtest. ``queries`` are the
    planned queries per question (shared by every arm, so the arms differ
    only in the instant or the loop); a question without them is planned
    in the run. ``loop_steps`` above 0 runs the bounded research loop
    instead of the one-pass brief.
    """

    searchers: list[Searcher]
    as_of: Literal["open", "now"] = "open"
    queries: dict[int, list[str]] | None = None
    loop_steps: int = 0
    max_queries: int = 3
    per_query: int = 2
    max_sources: int = 6
    loop_deadline_s: float = 600.0

    @property
    def mode(self) -> RunMode:
        return RunMode.UNRESTRICTED if self.as_of == "now" else RunMode.BACKTEST

    def label(self) -> str:
        how = f"loop{self.loop_steps}" if self.loop_steps else "onepass"
        return f"{how}-asof-{self.as_of}"


def release_gate(router: Router, opened: dict[int, datetime]) -> Callable[[Route, int], str | None]:
    """A model gate: no route answers a question unless it was released before it opened.

    The run checks the answering model once; this checks every fallback and
    every role (planner, brief, loop) at the moment it would be called.
    """

    def gate(route: Route, question_id: int) -> str | None:
        opened_at = opened.get(question_id)
        if opened_at is None:
            return "question open time unknown to the release gate"
        release, _ = router.release(route)
        decision = admit_model(release, opened_at, RunMode.BACKTEST)
        return None if decision.admitted else decision.reason

    return gate


def sample(questions: Sequence[FBQuestion], n: int | None) -> list[FBQuestion]:
    """The first ``n`` questions in question-id order: a fixed pseudo-random sample.

    The id is a hash of the source and the ForecastBench id, so the order is
    unrelated to topic, platform or outcome, and anyone re-running gets the
    same questions.
    """
    ordered = sorted(questions, key=lambda q: q.question_id)
    return ordered if n is None else ordered[:n]


@dataclass(frozen=True)
class BacktestResult:
    run_id: str
    forecast: int
    resumed: int
    skipped: int
    failed: int


def done_questions(ledger: Ledger, tournament: str, mode: RunMode = RunMode.BACKTEST) -> set[int]:
    """Questions with an aggregate forecast in an earlier run of ``tournament``."""
    rows = ledger.connection.execute(
        "SELECT DISTINCT f.question_id FROM forecasts f JOIN runs r ON r.run_id = f.run_id "
        "WHERE r.mode = ? AND r.tournament = ? AND f.kind = 'aggregate'",
        (mode.value, tournament),
    )
    return {int(r[0]) for r in rows}


async def run_forecastbench(
    *,
    questions: Sequence[FBQuestion],
    tournament: str,
    settings: Settings,
    router: Router,
    ledger: Ledger,
    clock: PinnedClock,
    code_version: str,
    out: Callable[[str], None] = print,
    research_for: Callable[[FBQuestion], ResearchResult | None] | None = None,
    extra_settings: dict[str, object] | None = None,
    retrieval: Retrieval | None = None,
    wall_clock: Clock | None = None,
) -> BacktestResult:
    """Forecast each question once, as of its freeze date.

    With no ``research_for`` there is no retrieval. With it (the replay
    evaluation's arm (d)), each question's sources come from it, every one
    passes the leakage gate before a model sees it, and a question it has no
    research for is skipped. With ``retrieval``, the bot's researcher runs
    on each question (one pass or the loop) as of its open instant, or, for
    Q2's leaky arm, as of the wall clock (``wall_clock``) in an
    ``unrestricted`` run.
    """
    if retrieval is not None and research_for is not None:
        raise AugurError("internal: give recorded research or a retrieval, not both")
    mode = retrieval.mode if retrieval is not None else RunMode.BACKTEST
    if mode is RunMode.UNRESTRICTED and wall_clock is None:
        raise AugurError("internal: the unrestricted arm needs the wall clock")
    done = done_questions(ledger, tournament, mode)
    todo = [q for q in questions if q.question_id not in done]
    if not todo:
        out(f"all {len(questions)} questions already forecast in this ledger")
    answering = router.chain("forecast")[0]
    run_id = ledger.start_run(
        mode=mode,
        tournament=tournament,
        provider=router.provider(answering).name,
        model=answering.model,
        settings=settings.public_dict()
        | {
            "retrieval": "none" if retrieval is None else retrieval.label(),
            "searchers": [] if retrieval is None else [s.name for s in retrieval.searchers],
            "questions": len(questions),
            "resumed": len(done),
        }
        | (extra_settings or {}),
        code_version=code_version,
    )
    # The run row records when the run really started; only then is the clock pinned.
    if todo:
        clock.set(todo[0].freeze_datetime)
    # A local model costs nothing; the caps still bind if a hosted one is configured.
    budget = Budget(settings.max_cost_per_question, settings.max_cost_per_run)
    opened = {q.question_id: q.freeze_datetime for q in questions}
    calls = CallRecorder(router, ledger, clock, budget, model_gate=release_gate(router, opened))
    researcher = (
        None
        if retrieval is None
        else Researcher(
            calls=calls,
            ledger=ledger,
            searchers=retrieval.searchers,
            mode=mode,
            max_queries=retrieval.max_queries,
            per_query=retrieval.per_query,
            max_sources=retrieval.max_sources,
        )
    )
    release, _basis = router.release(answering)
    n = settings.forecasters
    forecast = skipped = failed = 0
    for i, fb in enumerate(todo, start=1):
        q = fb.to_question()
        if mode is RunMode.UNRESTRICTED and wall_clock is not None:
            # The leaky arm: the question as it was posed, researched as of now.
            cutoff = wall_clock.now()
        else:
            cutoff = fb.freeze_datetime
        clock.set(cutoff)
        ledger.record_question(run_id, q, cutoff, fb.model_dump(mode="json", exclude={"outcome"}))
        started = time.monotonic()
        gate = admit_model(release, q.open_time, mode)
        if not gate.admitted:
            skipped += 1
            _outcome(ledger, run_id, fb, "skipped", gate.reason, started)
            continue
        if researcher is not None and retrieval is not None:
            planned = (retrieval.queries or {}).get(q.question_id)
            try:
                if retrieval.loop_steps:
                    research = await researcher.research_loop(
                        run_id,
                        q,
                        cutoff,
                        steps=retrieval.loop_steps,
                        queries=planned,
                        deadline=time.monotonic() + retrieval.loop_deadline_s,
                    )
                else:
                    research = await researcher.research(run_id, q, cutoff, queries=planned)
            except AugurError as exc:
                failed += 1
                _outcome(ledger, run_id, fb, "failed", f"research: {exc}"[:300], started)
                out(f"[{i}/{len(todo)}] {fb.source}:{fb.fb_id} research failed: {exc}")
                continue
        elif research_for is None:
            research = ResearchResult(queries=[], sources=[], brief=NO_RESEARCH_BRIEF)
        else:
            found = research_for(fb)
            if found is None:
                skipped += 1
                _outcome(ledger, run_id, fb, "skipped", "no research for this question", started)
                continue
            research = found
            for source in research.sources:
                ledger.record_source(source)  # before any model sees it; a no-op if present
                admission = admit_source(source, cutoff, RunMode.BACKTEST)
                ledger.record_admission(run_id, q.question_id, admission)
                if not admission.admitted:
                    raise LeakageError(
                        f"{fb.fb_id}: source {source.source_id} refused ({admission.reason}); "
                        "the research for a backtest must be dated at or before its cutoff"
                    )
        try:
            members, errors = await draw_forecasts(
                calls=calls,
                run_id=run_id,
                q=q,
                cutoff=cutoff,
                research=research,
                n=n,
                concurrency=n,
            )
            if len(members) < math.ceil(n / 2):
                raise AugurError(f"only {len(members)} of {n} draws parsed: {'; '.join(errors)}")
            for m in members:
                ledger.record_forecast(
                    run_id=run_id,
                    question_id=q.question_id,
                    kind="member",
                    member=m.member,
                    method="draw",
                    probability=m.probability,
                    reasoning=m.reasoning,
                    call_id=m.call_id,
                )
            prediction, method = aggregate(q, [m.prediction for m in members])
            if not isinstance(prediction, BinaryPrediction):
                raise AugurError("a binary question combined to a non-binary forecast")
            ledger.record_forecast(
                run_id=run_id,
                question_id=q.question_id,
                kind="aggregate",
                member=None,
                method=method,
                probability=prediction.probability,
                reasoning=f"{method} of {len(members)} members",
                call_id=None,
                values=values_json(prediction),
            )
        except AugurError as exc:
            failed += 1
            log.warning("backtest question failed", question=fb.fb_id, error=str(exc))
            _outcome(ledger, run_id, fb, "failed", str(exc)[:300], started)
            out(f"[{i}/{len(todo)}] {fb.source}:{fb.fb_id} failed: {exc}")
            continue
        forecast += 1
        how = (
            "no retrieval"
            if research_for is None and researcher is None
            else f"{len(research.sources)} source(s)"
            + (f", {len(research.steps)} loop step(s)" if research.steps else "")
        )
        _outcome(ledger, run_id, fb, "forecast", f"forecast with {how}", started)
        out(
            f"[{i}/{len(todo)}] {fb.source}:{fb.fb_id} p={prediction.probability:.2f} "
            f"outcome={fb.outcome} ({time.monotonic() - started:.0f} s)"
        )
    return BacktestResult(run_id, forecast, len(done), skipped, failed)


def _outcome(
    ledger: Ledger, run_id: str, fb: FBQuestion, status: str, reason: str, started: float
) -> None:
    ledger.record_outcome(
        run_id=run_id,
        question_id=fb.question_id,
        post_id=fb.question_id,
        question_type="binary",
        status=status,
        reason=reason,
        submitted=False,
        closes_at=None,
        minutes_left=None,
        cost_usd=0.0,
        seconds=time.monotonic() - started,
    )


PLAN_SUFFIX = "+plan"


async def plan_queries(
    *,
    questions: Sequence[FBQuestion],
    tournament: str,
    settings: Settings,
    router: Router,
    ledger: Ledger,
    clock: PinnedClock,
    code_version: str,
    max_queries: int = 3,
) -> dict[int, list[str]]:
    """Each question's search queries, planned by the research model as of its open instant.

    Every arm of Q2 and the loop reads the same plan, so the arms differ
    only in when the searches read the web (or in the loop). The planner
    calls are a backtest run of their own, so the release gate and the
    leakage scan cover them.
    """
    answering = router.chain("research")[0]
    run_id = ledger.start_run(
        mode=RunMode.BACKTEST,
        tournament=tournament,
        provider=router.provider(answering).name,
        model=answering.model,
        settings=settings.public_dict() | {"retrieval": "plan only", "questions": len(questions)},
        code_version=code_version,
    )
    budget = Budget(settings.max_cost_per_question, settings.max_cost_per_run)
    opened = {q.question_id: q.freeze_datetime for q in questions}
    calls = CallRecorder(router, ledger, clock, budget, model_gate=release_gate(router, opened))
    planner = Researcher(
        calls=calls, ledger=ledger, searchers=[], mode=RunMode.BACKTEST, max_queries=max_queries
    )
    planned: dict[int, list[str]] = {}
    for fb in questions:
        clock.set(fb.freeze_datetime)
        q = fb.to_question()
        ledger.record_question(
            run_id, q, fb.freeze_datetime, fb.model_dump(mode="json", exclude={"outcome"})
        )
        # With no searchers, research() is the planner call and nothing else.
        result = await planner.research(run_id, q, fb.freeze_datetime)
        planned[q.question_id] = result.queries
    return planned


def write_queries(path: Path, planned: dict[int, list[str]]) -> None:
    rows = {str(k): v for k, v in sorted(planned.items())}
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def read_queries(path: Path) -> dict[int, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AugurError(f"{path} is not a query plan; write one with --plan-out")
    return {int(k): [str(x) for x in v] for k, v in raw.items() if isinstance(v, list)}
