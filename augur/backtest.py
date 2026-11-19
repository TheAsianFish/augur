"""Backtests: the bot's own forecasting code on resolved questions, clock pinned.

M4 runs the tokenless ForecastBench number: the forecaster draws and the
median from :mod:`augur.agents.forecaster`, with no retrieval, on each
eligible question as of its freeze date. Every call, forecast and outcome is
a ledger row in a ``backtest`` run, so ``tests/test_leakage.py`` checks the
model's release date against each question, and a run can be resumed: a
question that already has an aggregate in the ledger is not forecast again.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from augur.agents.calls import Budget, CallRecorder
from augur.agents.forecaster import draw_forecasts
from augur.agents.researcher import ResearchResult
from augur.clock import ensure_utc
from augur.config import Settings
from augur.errors import AugurError
from augur.forecastbench import FBQuestion
from augur.leakage import RunMode, admit_model
from augur.ledger import Ledger
from augur.llm.router import Router
from augur.log import get_logger
from augur.predictions import BinaryPrediction, aggregate, values_json

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
class BacktestResult:
    run_id: str
    forecast: int
    resumed: int
    skipped: int
    failed: int


def done_questions(ledger: Ledger, tournament: str) -> set[int]:
    """Questions with an aggregate forecast in an earlier backtest run of ``tournament``."""
    rows = ledger.connection.execute(
        "SELECT DISTINCT f.question_id FROM forecasts f JOIN runs r ON r.run_id = f.run_id "
        "WHERE r.mode = 'backtest' AND r.tournament = ? AND f.kind = 'aggregate'",
        (tournament,),
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
) -> BacktestResult:
    """Forecast each question once, as of its freeze date, with no retrieval."""
    done = done_questions(ledger, tournament)
    todo = [q for q in questions if q.question_id not in done]
    if not todo:
        out(f"all {len(questions)} questions already forecast in this ledger")
    first = todo[0].freeze_datetime if todo else questions[0].freeze_datetime
    clock.set(first)
    answering = router.chain("forecast")[0]
    run_id = ledger.start_run(
        mode=RunMode.BACKTEST,
        tournament=tournament,
        provider=router.provider(answering).name,
        model=answering.model,
        settings=settings.public_dict()
        | {"retrieval": "none", "questions": len(questions), "resumed": len(done)},
        code_version=code_version,
    )
    # A local model costs nothing; the caps still bind if a hosted one is configured.
    budget = Budget(settings.max_cost_per_question, settings.max_cost_per_run)
    calls = CallRecorder(router, ledger, clock, budget)
    release, _basis = router.release(answering)
    n = settings.forecasters
    forecast = skipped = failed = 0
    for i, fb in enumerate(todo, start=1):
        q = fb.to_question()
        clock.set(fb.freeze_datetime)
        cutoff = fb.freeze_datetime
        ledger.record_question(run_id, q, cutoff, fb.model_dump(mode="json", exclude={"outcome"}))
        started = time.monotonic()
        gate = admit_model(release, q.open_time, RunMode.BACKTEST)
        if not gate.admitted:
            skipped += 1
            _outcome(ledger, run_id, fb, "skipped", gate.reason, started)
            continue
        research = ResearchResult(queries=[], sources=[], brief=NO_RESEARCH_BRIEF)
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
        _outcome(ledger, run_id, fb, "forecast", "forecast with no retrieval", started)
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
