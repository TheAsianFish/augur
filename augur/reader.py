"""M16's live arm: the newer free route reads the brief Augur's member 0 read.

PLAN.md's ``m16-registration`` item 4. On each question of a shadow tick that
holds Augur's aggregate and the template's row, the forecaster request
Augur's member 0 saw is rebuilt from the tick's ledger (the recorded brief and
the admitted sources it cited) and refused unless it hashes to the recorded
one once only the model is swapped. Then members 0 to 2 are drawn on the
newer route alone, combined as Augur combines draws, and written as a
``reader`` reference row; beside it a ``reader-base`` row holds the same
combination of Augur's own members 0 to 2 from that tick, before calibration
or any blend (the registered comparator).

The arm is served last: after Augur, the template, ``stat`` and ``blend``,
and only while the day's free calls (the shadow exports' calls today, the
tick's own, and the template's counted at :data:`TEMPLATE_CALLS` each) plus
:data:`DRAWS` stay within ``AUGUR_SHADOW_DAILY_CALLS``. Its calls go in a run
of their own whose tournament ends in :data:`TOURNAMENT_SUFFIX`, so the poll
can tell them apart. It never writes to ``forecasts``: the shadow ledger's
held questions and the floor read Augur's aggregates there, and a reader row
is never the floor's candidate (:data:`augur.floor.CANDIDATES`).

Off unless ``AUGUR_SHADOW_READER`` is ``true``; it has no ``--submit``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from augur.agents.calls import Budget, CallRecorder
from augur.agents.forecaster import draw_forecasts, forecaster_request
from augur.clock import Clock, SystemClock, parse_iso
from augur.errors import AugurError
from augur.leakage import RunMode, admit_source
from augur.ledger import Ledger
from augur.log import get_logger
from augur.predictions import (
    BinaryPrediction,
    ChoicePrediction,
    ContinuousPrediction,
    Prediction,
    aggregate,
    values_json,
)
from augur.questions import FORECAST_TYPES, Question, question_from_post
from augur.redraw import Recorded, recorded_research

if TYPE_CHECKING:
    from augur.config import Settings
    from augur.llm.endpoints import Route
    from augur.llm.router import Router

log = get_logger("augur.reader")

READER = "reader"
READER_BASE = "reader-base"
READER_ROUTE = "openrouter:nvidia/nemotron-3.5-super-vl:free"
"""The newer route (OpenRouter listing 2026-10-15), the only route of its chain."""
DRAWS = 3
"""Members 0 to 2."""
TEMPLATE_CALLS = 15
"""A template question's calls, which never reach the ledger (``scripts/poll.py``)."""
DAILY_CALLS = 1000
"""``AUGUR_SHADOW_DAILY_CALLS``'s default: OpenRouter's daily allowance for free models."""
TOURNAMENT_SUFFIX = "+reader"
SOURCE = "augur reader (M16), unpublished"
SWITCH_ENV = "AUGUR_SHADOW_READER"
MODELS_ENV = "AUGUR_READER_MODELS"


@dataclass(frozen=True)
class Candidate:
    """A question of the tick the arm may read, with what Augur recorded for it."""

    question: Question
    payload: dict[str, Any]
    cutoff: datetime
    run_id: str
    tournament: str


@dataclass
class ReaderResult:
    recorded: int = 0
    skipped: int = 0
    refused: int = 0
    failed: int = 0
    calls: int = 0
    lines: list[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.lines.append(line)


# -- reading the ledgers ------------------------------------------------------------


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def calls_used(
    paths: Iterable[Path | None], day: date, template_calls: int = TEMPLATE_CALLS
) -> int:
    """The day's free calls: distinct model calls started on ``day`` (UTC) in any of
    the ledgers, plus ``template_calls`` for each distinct template row recorded that day."""
    prefix = day.isoformat() + "%"
    calls: set[str] = set()
    templates: set[str] = set()
    for path in paths:
        if path is None or not path.exists():
            continue
        with closing(_ro(path)) as conn:
            if _has_table(conn, "llm_calls"):
                calls.update(
                    str(r[0])
                    for r in conn.execute(
                        "SELECT call_id FROM llm_calls WHERE started_at LIKE ?", (prefix,)
                    )
                )
            if _has_table(conn, "reference_forecasts"):
                templates.update(
                    str(r[0])
                    for r in conn.execute(
                        "SELECT reference_id FROM reference_forecasts "
                        "WHERE forecaster = 'template' AND recorded_at LIKE ?",
                        (prefix,),
                    )
                )
    return len(calls) + template_calls * len(templates)


def _questions_with(path: Path | None, forecaster: str) -> set[int]:
    if path is None or not path.exists():
        return set()
    with closing(_ro(path)) as conn:
        if not _has_table(conn, "reference_forecasts"):
            return set()
        rows = conn.execute(
            "SELECT DISTINCT question_id FROM reference_forecasts WHERE forecaster = ?",
            (forecaster,),
        ).fetchall()
    return {int(r[0]) for r in rows}


def candidates(
    ledger: Ledger,
    ledger_path: Path,
    shadow_path: Path | None,
    now: datetime,
    margin: timedelta,
) -> list[Candidate]:
    """The tick's questions with Augur's aggregate and a template row and no reader row
    (in either ledger), still open ``margin`` before their close, soonest-closing first."""
    templated = _questions_with(ledger_path, "template") | _questions_with(shadow_path, "template")
    read = _questions_with(ledger_path, READER) | _questions_with(shadow_path, READER)
    rows = ledger.connection.execute(
        "SELECT q.question_id, q.payload_json, q.cutoff, q.run_id, r.tournament "
        "FROM questions q JOIN runs r ON r.run_id = q.run_id "
        "WHERE r.tournament NOT LIKE ? AND EXISTS (SELECT 1 FROM forecasts f "
        "WHERE f.run_id = q.run_id AND f.question_id = q.question_id AND f.kind = 'aggregate') "
        "ORDER BY q.recorded_at, q.rowid",
        ("%" + TOURNAMENT_SUFFIX,),
    ).fetchall()
    latest: dict[int, tuple[str, str, str, str]] = {}
    for qid, payload, cutoff, run_id, tournament in rows:
        latest[int(qid)] = (str(payload), str(cutoff), str(run_id), str(tournament))
    found: list[Candidate] = []
    for qid, (payload, cutoff, run_id, tournament) in latest.items():
        if qid not in templated or qid in read:
            continue
        try:
            post = json.loads(payload)
            q = question_from_post(post)
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("reader_unparsable_payload", question_id=qid, error=str(exc))
            continue
        if q.scheduled_close_time - margin <= now:
            continue
        found.append(Candidate(q, post, parse_iso(cutoff), run_id, tournament))
    found.sort(key=lambda c: (c.question.scheduled_close_time, c.question.question_id))
    return found


def prediction_from_values(question_type: str, values: Mapping[str, Any]) -> Prediction:
    """A forecast back from the ledger's ``values_json`` (:func:`values_json`'s inverse)."""
    if question_type == "binary":
        return BinaryPrediction(float(values["probability"]))
    if question_type == "multiple_choice":
        return ChoicePrediction({str(k): float(v) for k, v in values["probabilities"].items()})
    return ContinuousPrediction(
        tuple(float(x) for x in values["cdf"]),
        tuple((float(level), float(v)) for level, v in values["percentiles"]),
    )


def augur_members(ledger: Ledger, run_id: str, q: Question) -> list[Prediction]:
    """Augur's own members 0 to ``DRAWS - 1`` of ``run_id`` on ``q``, by member."""
    rows = ledger.connection.execute(
        "SELECT member, probability, values_json FROM forecasts WHERE run_id = ? "
        "AND question_id = ? AND kind = 'member' AND member BETWEEN 0 AND ? ORDER BY member",
        (run_id, q.question_id, DRAWS - 1),
    ).fetchall()
    found: dict[int, Prediction] = {}
    for member, probability, raw in rows:
        values = json.loads(raw) if raw else {"probability": probability}
        found.setdefault(int(member), prediction_from_values(q.type, values))
    missing = [m for m in range(DRAWS) if m not in found]
    if missing:
        raise AugurError(
            f"Augur's member(s) {', '.join(map(str, missing))} are not in run {run_id}, so "
            "there is no like-for-like comparator"
        )
    return [found[m] for m in range(DRAWS)]


def check_rebuilt(q: Question, cutoff: datetime, recorded: Recorded) -> None:
    """Refuse unless Augur's member-0 request, rebuilt from the ledger, hashes to the
    recorded one once its model and reasoning effort are put back."""
    rebuilt = forecaster_request(q, cutoff, recorded.research, member=0)
    rebuilt = rebuilt.model_copy(
        update={
            "model": recorded.request["model"],
            "reasoning_effort": recorded.request.get("reasoning_effort"),
        }
    )
    if rebuilt.request_hash() != recorded.request_hash:
        raise AugurError(
            f"post {q.post_id}: the rebuilt forecaster request does not match the one Augur's "
            "member 0 recorded, so the reader would not see what Augur saw; was the prompt or "
            "the ledger changed since the tick ran?"
        )


# -- the arm --------------------------------------------------------------------------


async def run_reader(
    *,
    ledger: Ledger,
    ledger_path: Path,
    shadow_path: Path | None,
    router: Router,
    route: Route,
    settings: Settings,
    clock: Clock,
    allowance: int,
    template_calls: int = TEMPLATE_CALLS,
    deadline: datetime | None = None,
    code_version: str = "",
) -> ReaderResult:
    """Read Augur's brief with ``route`` on each candidate the day's calls allow."""
    result = ReaderResult()
    margin = timedelta(minutes=settings.close_margin_minutes)
    todo = candidates(ledger, ledger_path, shadow_path, clock.now(), margin)
    result.add(f"reader ({route.key}) on {len(todo)} question(s)")
    recorded: dict[str, dict[int, Recorded]] = {}
    runs: dict[str, str] = {}
    calls = CallRecorder(
        router,
        ledger,
        clock,
        Budget(settings.max_cost_per_question, settings.max_cost_per_run),
    )
    spent = False
    for c in todo:
        q = c.question
        bound = log.bind(question_id=q.question_id, post_id=q.post_id)
        if spent:
            result.skipped += 1
            result.add(f"post {q.post_id}: skipped, the day's free calls are spent")
            continue
        if deadline is not None and clock.now() >= deadline:
            result.skipped += 1
            result.add(f"post {q.post_id}: skipped, past the run's deadline")
            continue
        if q.type not in FORECAST_TYPES:
            result.skipped += 1
            result.add(f"post {q.post_id}: skipped, {q.type} questions are not compared")
            continue
        # Refusals first: none of them costs a call.
        try:
            if c.tournament not in recorded:
                recorded[c.tournament] = recorded_research(ledger_path, c.tournament)
            found = recorded[c.tournament].get(q.question_id)
            if found is None:
                raise AugurError(f"post {q.post_id}: no recorded member-0 request to read from")
            check_rebuilt(q, c.cutoff, found)
            base_members = augur_members(ledger, c.run_id, q)
        except AugurError as exc:
            result.refused += 1
            bound.warning("reader_refused", error=str(exc))
            result.add(f"post {q.post_id}: refused, no call made: {exc}")
            continue
        left = (q.scheduled_close_time - margin - clock.now()).total_seconds()
        if deadline is not None:
            left = min(left, (deadline - clock.now()).total_seconds())
        if left < settings.min_forecast_minutes * 60:
            # As the runner does: a question too close to its close or to the run's
            # deadline is not started, so no draw is cut off after spending a call.
            result.skipped += 1
            result.add(
                f"post {q.post_id}: skipped, {max(left, 0.0) / 60:.1f} min left "
                f"(under the {settings.min_forecast_minutes:g} a forecast needs); no call"
            )
            continue
        used = calls_used((shadow_path, ledger_path), clock.now().date(), template_calls)
        if used + DRAWS > allowance:
            spent = True
            result.skipped += 1
            result.add(
                f"post {q.post_id}: skipped, the day's free calls are spent ({used} used + "
                f"{DRAWS} would pass AUGUR_SHADOW_DAILY_CALLS={allowance}); no further call"
            )
            continue
        research = found.research
        try:
            run_id = runs.get(c.tournament)
            if run_id is None:
                run_id = ledger.start_run(
                    mode=RunMode.DRY_RUN,
                    tournament=c.tournament + TOURNAMENT_SUFFIX,
                    provider=router.provider(route).name,
                    model=route.model,
                    settings=settings.public_dict()
                    | {
                        "arm": READER,
                        "route": route.key,
                        "draws": DRAWS,
                        "daily_calls": allowance,
                        "reads": c.tournament,
                    },
                    code_version=code_version,
                )
                runs[c.tournament] = run_id
            ledger.record_question(run_id, q, c.cutoff, c.payload)
            for source in research.sources:
                ledger.record_source(source)  # a no-op: the tick recorded it
                admission = admit_source(source, c.cutoff, RunMode.DRY_RUN)
                ledger.record_admission(run_id, q.question_id, admission)
                if not admission.admitted:
                    raise AugurError(f"source {source.source_id} refused ({admission.reason})")
            left = (q.scheduled_close_time - margin - clock.now()).total_seconds()
            if deadline is not None:
                left = min(left, (deadline - clock.now()).total_seconds())
            before = _calls_in(ledger, run_id, q.question_id)
            try:
                members, errors = await draw_forecasts(
                    calls=calls,
                    run_id=run_id,
                    q=q,
                    cutoff=c.cutoff,
                    research=research,
                    n=DRAWS,
                    concurrency=DRAWS,
                    timeout_s=max(1.0, left),
                    first_member=0,
                    chains=[[route]],
                )
            finally:
                result.calls += _calls_in(ledger, run_id, q.question_id) - before
            if len(members) < DRAWS:
                raise AugurError(
                    f"only {len(members)} of {DRAWS} draws answered: " + "; ".join(errors[:3])
                )
            members.sort(key=lambda m: m.member)
            mine, method = aggregate(q, [m.prediction for m in members])
            theirs, base_method = aggregate(q, base_members)
        except Exception as exc:  # one question's failure degrades only that question
            result.failed += 1
            reason = f"{type(exc).__name__}: {exc}"[:400]
            bound.warning("reader_failed", error=reason)
            result.add(f"post {q.post_id}: failed, {reason}")
            continue
        ledger.record_reference(
            question_id=q.question_id,
            post_id=q.post_id,
            question_type=q.type,
            forecaster=READER,
            detail=(
                f"{route.key}: {method} of members 0-{DRAWS - 1} ({DRAWS} draws) on the "
                f"forecaster request of Augur's run {c.run_id}"
            ),
            values=values_json(mine),
            as_of=c.cutoff,
            source=SOURCE,
        )
        ledger.record_reference(
            question_id=q.question_id,
            post_id=q.post_id,
            question_type=q.type,
            forecaster=READER_BASE,
            detail=(
                f"Augur's own members 0-{DRAWS - 1} of run {c.run_id}, {base_method}, before "
                "calibration or any blend"
            ),
            values=values_json(theirs),
            as_of=c.cutoff,
            source=SOURCE,
        )
        result.recorded += 1
        bound.info("reader_recorded", question_type=q.type)
        result.add(
            f"post {q.post_id} ({q.type}): reader {_summary(mine)}, Augur {_summary(theirs)}"
        )
    return result


def _calls_in(ledger: Ledger, run_id: str, question_id: int) -> int:
    row = ledger.connection.execute(
        "SELECT COUNT(*) FROM llm_calls WHERE run_id = ? AND question_id = ?",
        (run_id, question_id),
    ).fetchone()
    return int(row[0])


def _summary(prediction: Prediction) -> str:
    if isinstance(prediction, BinaryPrediction):
        return f"{prediction.probability * 100:.1f}%"
    if isinstance(prediction, ChoicePrediction):
        return ", ".join(f"{o} {p * 100:.1f}%" for o, p in prediction.probabilities.items())
    return ", ".join(f"p{lv * 100:g}={v:.4g}" for lv, v in prediction.percentiles)


# -- command line ---------------------------------------------------------------------


def add_parser(subparsers: Any) -> None:
    """Register ``augur reader``."""
    p = subparsers.add_parser(
        "reader",
        help="M16's live arm: the newer free route on Augur's recorded brief (never posted)",
        description=(
            "On each question of a shadow tick with Augur's aggregate and the template's row, "
            "rebuild the forecaster request Augur's member 0 saw, refuse unless it matches the "
            "recorded one but for the model, draw members 0 to 2 on the newer free route and "
            "record a reader reference row beside reader-base (Augur's own members 0 to 2). "
            f"Off unless {SWITCH_ENV}=true; never posts and is never the floor's candidate."
        ),
    )
    p.add_argument("--ledger", type=Path, help="the tick's run ledger (default AUGUR_LEDGER)")
    p.add_argument(
        "--shadow-ledger",
        type=Path,
        metavar="PATH",
        help="the shadow ledger: its reader rows are not repeated, its calls today count",
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        metavar="M",
        help="start no question after this many minutes (the scheduled job's deadline)",
    )


def _exists(path: Path) -> bool:
    return path.exists()


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    if raw:
        print(f"warning: {name}={raw!r} is not a positive whole number; using {default}")
    return default


def reader_settings(settings: Settings, env: Mapping[str, str]) -> Settings:
    """``settings`` with the reader's route as the only chain of every role:
    ``AUGUR_READER_MODELS`` if set, else :data:`READER_ROUTE`. The arm draws on the
    chain's first route alone."""
    text = env.get(MODELS_ENV, "").strip() or READER_ROUTE
    return settings.model_copy(
        update={"forecast_models": text, "research_models": text, "member_models": None}
    )


async def main(args: argparse.Namespace, settings: Settings) -> int:
    env = os.environ
    if env.get(SWITCH_ENV, "").strip().lower() != "true":
        print(f"the reader arm is off ({SWITCH_ENV} is not 'true'); nothing read, no call")
        return 0
    ledger_path: Path = args.ledger or settings.ledger_path
    if not _exists(ledger_path):
        print(f"no run ledger at {ledger_path}: nothing for the reader")
        return 0
    from augur.llm.catalog import fetch_catalog, load_snapshot
    from augur.llm.factory import build_router, default_http_client
    from augur.runner import code_version

    clock = SystemClock()
    allowance = _positive_int(env, "AUGUR_SHADOW_DAILY_CALLS", DAILY_CALLS)
    template_calls = _positive_int(env, "AUGUR_SHADOW_TEMPLATE_CALLS", TEMPLATE_CALLS)
    own = reader_settings(settings, env)
    route = own.chains()["forecast"][0]
    deadline = (
        clock.now() + timedelta(minutes=args.max_minutes) if args.max_minutes is not None else None
    )
    async with default_http_client() as http:
        catalog = load_snapshot() if own.is_replay else await fetch_catalog(http, clock.now())
        router = build_router(own, http, catalog, clock)
        try:
            with Ledger.open(ledger_path, clock) as ledger:
                result = await run_reader(
                    ledger=ledger,
                    ledger_path=ledger_path,
                    shadow_path=args.shadow_ledger,
                    router=router,
                    route=route,
                    settings=own,
                    clock=clock,
                    allowance=allowance,
                    template_calls=template_calls,
                    deadline=deadline,
                    code_version=code_version(),
                )
        finally:
            await router.aclose()
    for line in result.lines:
        print(line)
    print(
        f"ledger: {ledger_path}: {result.recorded} recorded, {result.skipped} skipped, "
        f"{result.refused} refused, {result.failed} failed; {result.calls} call(s)"
    )
    return 1 if result.failed else 0
