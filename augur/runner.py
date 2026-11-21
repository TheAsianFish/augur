"""``augur forecast``: list (or load) questions, skip, cap, forecast, submit or print.

A live round leaves each question open for a few hours, so every question
gets a deadline (its close minus a margin, and never past the run's own
deadline): research and the forecaster draws stop there, and the best
ensemble finished by then is posted. Questions are taken soonest-closing
first. Every question the run saw ends with one outcome row in the ledger
(forecast, missed, skipped or failed, and why): the run's coverage report.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, TextIO

import structlog

import augur
from augur.agents.calls import (
    MIN_COOL_S,
    Budget,
    BudgetExceededError,
    CallRecorder,
    QuestionBudgetExceededError,
    estimate_cost,
)
from augur.agents.forecaster import MemberForecast, draw_forecasts, forecaster_request
from augur.agents.researcher import (
    BRIEF_MAX_TOKENS,
    PLAN_MAX_TOKENS,
    SUBQ_MAX_TOKENS,
    Researcher,
    ResearchResult,
)
from augur.clock import Clock
from augur.comment import render_comment
from augur.config import Settings
from augur.errors import AugurError, BelowFloorError, MetaculusError, SubmitFloorError
from augur.fleet import FleetPlan, bounded_map
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.llm.base import Provider
from augur.llm.endpoints import Route
from augur.llm.probe import ProbeResult, new_session_id, probe_routes, record_probes
from augur.llm.registry import FRONTIER_TIER_NAME, PRACTICE_TOURNAMENTS, in_frontier_tier
from augur.llm.router import Router
from augur.log import get_logger
from augur.metaculus import MetaculusClient
from augur.predictions import (
    BinaryPrediction,
    Prediction,
    aggregate,
    binary_probability,
    check,
    forecast_payload,
    summary,
    values_json,
)
from augur.questions import Question, question_from_post
from augur.replay import blend, exact_price
from augur.research.source import Searcher

log = get_logger(__name__)

TOURNAMENTS: dict[str, str] = {
    "minibench": "minibench",
    "futureeval": "33121",
    "fall-2026": "33121",
    "bot-testing-area": "bot-testing-area",
}
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples" / "minibench"

Status = Literal["forecast", "missed", "skipped", "failed"]
STATUSES: tuple[Status, ...] = ("forecast", "missed", "skipped", "failed")
# Research may use at most this share of a question's time; the draws get the rest.
RESEARCH_SHARE = 0.5
# Kept back from a question's deadline for the comparison, the re-read and the posts.
SUBMIT_RESERVE_S = 30.0
# Rough size of the research brief and the question text, for the cost estimate.
ESTIMATED_BRIEF_CHARS = 2500


@dataclass
class QuestionReport:
    question: Question
    status: Status
    reason: str = ""
    prediction: Prediction | None = None
    members: list[MemberForecast] = field(default_factory=list)
    comment: str = ""
    submitted: bool = False
    minutes_left: float | None = None
    seconds: float | None = None
    cost_usd: float = 0.0

    @property
    def probability(self) -> float | None:
        """The binary forecast's probability (``None`` for other types)."""
        return binary_probability(self.prediction)


@dataclass
class RunReport:
    run_id: str
    mode: RunMode
    source: str
    reports: list[QuestionReport]
    cost_usd: float

    def count(self, status: Status) -> int:
        return sum(1 for r in self.reports if r.status == status)

    def coverage(self) -> dict[str, Any]:
        """Questions forecast in time, missed, skipped and failed, with the reasons."""
        reasons: dict[str, dict[str, int]] = {s: {} for s in STATUSES if s != "forecast"}
        for r in self.reports:
            if r.status != "forecast":
                key = _reason_key(r.reason)
                reasons[r.status][key] = reasons[r.status].get(key, 0) + 1
        return {
            "run_id": self.run_id,
            "mode": self.mode.value,
            "source": self.source,
            "listed": len(self.reports),
            **{s: self.count(s) for s in STATUSES},
            "submitted": sum(1 for r in self.reports if r.submitted),
            "cost_usd": round(self.cost_usd, 6),
            "reasons": reasons,
            "questions": [
                {
                    "question_id": r.question.question_id,
                    "post_id": r.question.post_id,
                    "type": r.question.type,
                    "status": r.status,
                    "submitted": r.submitted,
                    "reason": r.reason,
                    "minutes_left": None if r.minutes_left is None else round(r.minutes_left, 1),
                    "seconds": None if r.seconds is None else round(r.seconds, 1),
                }
                for r in self.reports
            ],
        }


def _reason_key(reason: str) -> str:
    """Reasons grouped for the summary: the text before any detail in parentheses."""
    head = reason.split(" (", 1)[0].split(";", 1)[0]
    return re.sub(r"\d+(?:\.\d+)?", "N", head).strip() or "unspecified"


def load_samples(directory: Path) -> list[tuple[Question, dict[str, Any]]]:
    """Committed sample posts (``post_*.json``), for runs without a token."""
    paths = sorted(directory.glob("post_*.json"))
    if not paths:
        raise AugurError(f"no sample questions in {directory}; pass --samples with a directory")
    out = []
    for path in paths:
        post = json.loads(path.read_text(encoding="utf-8"))
        out.append((question_from_post(post), post))
    return out


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _git(root: Path, *args: str) -> str:
    # Fixed arguments, no shell, no user input.
    done = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return done.stdout.strip()


def code_version() -> str:
    """The package version plus this checkout's commit (``.dirty`` with local changes)."""
    root = Path(__file__).resolve().parent.parent
    sha = ""
    if (root / ".git").exists():
        try:
            sha = _git(root, "rev-parse", "--short=12", "HEAD")
            if _git(root, "status", "--porcelain", "--untracked-files=no"):
                sha += ".dirty"
        except (OSError, subprocess.SubprocessError):
            sha = ""
    return f"{augur.__version__}+{sha}" if sha else augur.__version__


class ForecastRun:
    def __init__(
        self,
        *,
        settings: Settings,
        clock: Clock,
        ledger: Ledger,
        router: Router,
        searchers: Sequence[Searcher],
        client: MetaculusClient | None,
        out: TextIO,
        probe_providers: Mapping[str, Provider] | None = None,
        blend_exact_price: bool = False,
    ) -> None:
        self._settings = settings
        # M5's blend: a binary forecast is combined, one third to two thirds in
        # log-odds, with the question's own community prediction when research found it.
        self._blend_exact_price = blend_exact_price
        self._clock = clock
        self._ledger = ledger
        self._router = router
        # Research passes, the members' prompt variants and model chains, and
        # how many questions run at once (M7); the defaults are the old run.
        self._fleet = FleetPlan.from_settings(settings, router)
        # Providers without retries for the opening probe; ``None`` skips it.
        self._probe_providers = probe_providers
        self._reforecast = False
        self._scored = False
        self._searchers = list(searchers)
        self._run_deadline = clock.now() + timedelta(minutes=settings.run_deadline_minutes)
        self._client = client
        self._out = out

    async def run(
        self,
        tournament: str,
        *,
        submit: bool,
        samples: Path | None,
        max_questions: int,
        reforecast: bool = False,
    ) -> RunReport:
        slug = TOURNAMENTS.get(tournament, tournament)
        if submit and (self._client is None or not self._client.has_token):
            raise AugurError("--submit needs METACULUS_TOKEN; without it run a dry run")
        if submit and samples is not None:
            raise AugurError("--submit forecasts live questions only; drop --samples")
        if reforecast and slug not in PRACTICE_TOURNAMENTS:
            raise AugurError(
                f"--reforecast is for {', '.join(sorted(PRACTICE_TOURNAMENTS))} only; a scored "
                "tournament is forecast once per question"
            )
        self._reforecast = reforecast
        scored = submit and slug not in PRACTICE_TOURNAMENTS
        self._scored = scored
        if scored:
            self._check_submit_floor(slug)
        mode = RunMode.LIVE if submit else RunMode.DRY_RUN
        use_samples = samples is not None or self._client is None or not self._client.has_token
        source = (
            f"samples:{_display(samples or SAMPLES_DIR)}" if use_samples else f"metaculus:{slug}"
        )
        answering = self._router.chain("forecast")[0]
        catalog = self._router.catalog
        run_id = self._ledger.start_run(
            mode=mode,
            tournament=slug,
            provider=self._router.provider(answering).name,
            model=answering.model,
            settings=self._settings.public_dict()
            | {
                "question_source": source,
                "scored": scored,
                "floor": FRONTIER_TIER_NAME if scored else None,
                "reforecast": reforecast,
                "catalog": f"{catalog.origin} {catalog.fetched_at:%Y-%m-%d}",
            },
            code_version=code_version(),
        )
        structlog.contextvars.bind_contextvars(run_id=run_id)
        log.info("run started", mode=mode.value, source=source)
        chains = "; ".join(
            f"{role}: {', '.join(r.key for r in self._router.chain(role))}"
            for role in ("forecast", "research")
        )
        self._print(f"run {run_id} ({mode.value}) on {source}")
        self._print(f"models: {chains}")
        self._print(self._fleet.describe())
        budget = Budget(self._settings.max_cost_per_question, self._settings.max_cost_per_run)
        calls = CallRecorder(self._router, self._ledger, self._clock, budget)
        if scored:
            self._router.set_floor("forecast", FRONTIER_TIER_NAME, in_frontier_tier)
        if self._probe_providers is not None:
            budget.charge_run(await self._probe(run_id))
        no_model = ""
        if not self._router.usable("forecast"):
            no_model = (
                f"no {FRONTIER_TIER_NAME} model answered the probe, so nothing is forecast"
                if scored
                else "no forecasting model answered the probe"
            )

        run_started = self._clock.now()
        self._run_deadline = run_started + timedelta(minutes=self._settings.run_deadline_minutes)
        candidates = await self._candidates(slug, samples if use_samples else None, use_samples)
        # Soonest-closing first: under a cap or a deadline, the question that
        # closes next is the one that cannot wait for the next run.
        candidates.sort(key=lambda c: (c[0].scheduled_close_time, c[0].question_id))
        reports: list[QuestionReport] = []
        chosen: list[tuple[Question, dict[str, Any]]] = []
        for q, post in candidates:
            status, reason = self._skip_reason(q, live_listing=not use_samples)
            if reason is None and len(chosen) >= max_questions:
                status, reason = "skipped", f"over the cap of {max_questions} questions per run"
            if reason:
                skipped = QuestionReport(
                    q, status, reason, minutes_left=self._minutes_left(q, live=not use_samples)
                )
                if mode is RunMode.LIVE and q.already_forecast(self._clock.now()):
                    await self._repair_comment(run_id, q, post, skipped)
                reports.append(skipped)
                log.info("question skipped", question_id=q.question_id, reason=skipped.reason)
            else:
                chosen.append((q, post))
        if chosen:
            each = self._estimate_question_cost(chosen[0][0])
            self._print(
                f"estimated worst case ${each:.4f} a question, ${each * len(chosen):.4f} for "
                f"{len(chosen)} (caps ${self._settings.max_cost_per_question:.2f} a question, "
                f"${self._settings.max_cost_per_run:.2f} a run)"
            )

        # Shared by the questions in flight. A run-budget refusal stops the run:
        # questions already running finish (their own calls are refused by the
        # same budget), none starts after it. A question's own cap fails only it.
        stop_reason = ""

        async def handle(item: tuple[Question, dict[str, Any]]) -> QuestionReport:
            nonlocal stop_reason
            q, post = item
            if stop_reason or no_model:
                return QuestionReport(q, "skipped", stop_reason or no_model)
            # Taken when the question starts, not when the run listed it: a
            # question waiting for a slot can reach its deadline, or the run's.
            budget_s = self._time_budget_s(q, live=not use_samples)
            if budget_s is not None and budget_s < self._settings.min_forecast_minutes * 60:
                left = self._minutes_left(q, live=True)
                return QuestionReport(
                    q,
                    "missed",
                    f"too little time left ({budget_s / 60:.1f} min before its deadline, "
                    f"under the {self._settings.min_forecast_minutes:g} a forecast needs)",
                    minutes_left=left,
                )
            # Each question runs in its own task, so this binding is its own.
            structlog.contextvars.bind_contextvars(question_id=q.question_id)
            t0 = time.perf_counter()
            spent_before = budget.spent_question.get(q.question_id, 0.0)
            try:
                outcome = await self._forecast(run_id, mode, calls, q, post, budget_s)
            except BelowFloorError as exc:
                log.warning("question skipped by the floor", question_id=q.question_id)
                outcome = QuestionReport(q, "skipped", f"submit floor: {exc}")
            except QuestionBudgetExceededError as exc:
                # This question's cap: it fails without an ensemble cut short,
                # and the others go on with their own budgets.
                outcome = QuestionReport(q, "failed", str(exc))
            except BudgetExceededError as exc:
                stop_reason = stop_reason or str(exc)
                outcome = QuestionReport(q, "failed", str(exc))
            except AugurError as exc:
                log.error("question failed", question_id=q.question_id, error=str(exc))
                outcome = QuestionReport(q, "failed", str(exc))
            outcome.seconds = time.perf_counter() - t0
            outcome.cost_usd = budget.spent_question.get(q.question_id, 0.0) - spent_before
            outcome.minutes_left = self._minutes_left(q, live=not use_samples)
            return outcome

        reports.extend(await bounded_map(chosen, handle, self._fleet.question_concurrency))
        structlog.contextvars.unbind_contextvars("question_id")
        report = RunReport(run_id, mode, source, reports, budget.spent_run)
        for r in reports:
            self._ledger.record_outcome(
                run_id=run_id,
                question_id=r.question.question_id,
                post_id=r.question.post_id,
                question_type=r.question.type,
                status=r.status,
                reason=r.reason,
                submitted=r.submitted,
                closes_at=r.question.scheduled_close_time,
                minutes_left=r.minutes_left,
                cost_usd=r.cost_usd,
                seconds=r.seconds,
            )
        log.info(
            "run finished",
            forecast=report.count("forecast"),
            missed=report.count("missed"),
            skipped=report.count("skipped"),
            failed=report.count("failed"),
            cost_usd=round(report.cost_usd, 6),
        )
        structlog.contextvars.unbind_contextvars("run_id")
        submitted = sum(1 for r in reports if r.submitted)
        self._print(
            f"coverage: {len(reports)} listed; {report.count('forecast')} forecast in time "
            f"({submitted} submitted), {report.count('missed')} missed, "
            f"{report.count('skipped')} skipped, {report.count('failed')} failed; "
            f"estimated cost ${report.cost_usd:.4f}"
        )
        for r in reports:
            if r.status != "forecast":
                self._print(f"  {r.status}: {r.question.url} {r.reason}")
        return report

    def _minutes_left(self, q: Question, *, live: bool) -> float | None:
        if not live:
            return None
        return (q.scheduled_close_time - self._clock.now()).total_seconds() / 60

    def _time_budget_s(self, q: Question, *, live: bool) -> float | None:
        """Seconds until this question's deadline: its close less the margin, and
        never past the run's deadline. ``None`` for committed samples (no window)."""
        now = self._clock.now()
        run_left = (self._run_deadline - now).total_seconds()
        if not live:
            return max(0.0, run_left)
        margin = timedelta(minutes=self._settings.close_margin_minutes)
        close_left = (q.scheduled_close_time - margin - now).total_seconds()
        return max(0.0, min(close_left, run_left))

    def _estimate_question_cost(self, q: Question) -> float:
        """Worst case for one question on the first route of each chain: the
        planner and the brief at their token caps, with two research passes the
        sub-question call and a second brief, and every forecaster draw on its
        own variant and its own chain's first route."""
        total = 0.0
        research = self._router.chain("research")[0]
        research_price = self._router.price(research)
        question_tokens = len(q.title) // 3 + 200
        if research_price is not None:
            total += research_price.cost(question_tokens, PLAN_MAX_TOKENS)
            total += research_price.cost(4000, BRIEF_MAX_TOKENS)
            if self._fleet.passes == 2:
                brief_tokens = ESTIMATED_BRIEF_CHARS // 3
                total += research_price.cost(question_tokens + brief_tokens, SUBQ_MAX_TOKENS)
                # The rewritten brief reads up to twice the sources.
                total += research_price.cost(8000, BRIEF_MAX_TOKENS)
        brief = ResearchResult(queries=[], sources=[], brief="x" * ESTIMATED_BRIEF_CHARS)
        for member in self._fleet.plans():
            price = self._router.price(member.chain[0])
            if price is not None:
                request = forecaster_request(
                    q, self._clock.now(), brief, member.member, member.variant
                )
                total += estimate_cost(request, price)
        return total

    async def _candidates(
        self, slug: str, samples: Path | None, use_samples: bool
    ) -> list[tuple[Question, dict[str, Any]]]:
        if use_samples:
            return load_samples(samples or SAMPLES_DIR)
        client = self._live_client()
        listed = await client.list_open_questions(slug)
        detailed: list[tuple[Question, dict[str, Any]]] = []
        now = self._clock.now()
        for q, post in listed:
            if q.unsupported_reason() is not None or not q.is_open(now):
                detailed.append((q, post))
                continue
            # Only the detail carries my_forecasts, which the skip rule needs.
            try:
                detailed.append(await client.get_question(q.post_id))
            except MetaculusError as exc:
                log.warning("detail fetch failed", post_id=q.post_id, error=str(exc))
        return detailed

    def _skip_reason(self, q: Question, *, live_listing: bool) -> tuple[Status, str | None]:
        now = self._clock.now()
        unsupported = q.unsupported_reason()
        if unsupported:
            return "skipped", unsupported
        if live_listing and not q.is_open(now):
            if now >= q.scheduled_close_time:
                return "missed", "closed before this run reached it"
            return "skipped", "not open now"
        if live_listing and q.already_forecast(now) and not self._reforecast:
            return "skipped", "already forecast by this account"
        return "forecast", None

    async def _forecast(
        self,
        run_id: str,
        mode: RunMode,
        calls: CallRecorder,
        q: Question,
        post: dict[str, Any],
        budget_s: float | None = None,
    ) -> QuestionReport:
        cutoff = self._clock.now()
        started = time.monotonic()
        self._ledger.record_question(run_id, q, cutoff, post)
        researcher = Researcher(
            calls=calls,
            ledger=self._ledger,
            searchers=self._searchers,
            mode=mode,
        )
        research_s = None if budget_s is None else budget_s * RESEARCH_SHARE
        first: ResearchResult | None = None
        try:
            async with asyncio.timeout(research_s):
                first = await researcher.research(run_id, q, cutoff)
                if self._fleet.passes == 2:
                    await researcher.second_pass(run_id, q, cutoff, first)
        except TimeoutError:
            log.warning("research stopped at its deadline", seconds=research_s)
            if first is not None:
                # The deadline fell in the second pass: the first brief is complete and stands.
                first.errors.append("second pass stopped at its deadline; the first brief stands")
        research = (
            first
            if first is not None
            else ResearchResult(
                queries=[],
                sources=[],
                brief=(
                    "Research did not finish within its share of this question's time, so "
                    "the forecasters saw no brief."
                ),
                errors=["research stopped at its deadline"],
            )
        )
        n = self._settings.forecasters
        draws_s = (
            None
            if budget_s is None
            else max(1.0, budget_s - (time.monotonic() - started) - SUBMIT_RESERVE_S)
        )
        members, errors = await draw_forecasts(
            calls=calls,
            run_id=run_id,
            q=q,
            cutoff=cutoff,
            research=research,
            n=n,
            concurrency=n,
            timeout_s=draws_s,
            variants=self._fleet.variants,
            chains=self._fleet.chains,
        )
        needed = math.ceil(n / 2)
        if len(members) < needed:
            raise AugurError(
                f"only {len(members)} of {n} forecasters answered (need {needed}): "
                + "; ".join(errors[:3])
            )
        for m in members:
            self._ledger.record_forecast(
                run_id=run_id,
                question_id=q.question_id,
                kind="member",
                member=m.member,
                method="draw",
                probability=binary_probability(m.prediction),
                reasoning=m.reasoning,
                call_id=m.call_id,
                question_type=q.type,
                values=values_json(m.prediction),
            )
        try:
            prediction, method = aggregate(q, [m.prediction for m in members])
        except ValueError as exc:
            raise AugurError(f"could not combine the members ({exc})") from None
        own_price = exact_price(research) if self._blend_exact_price else None
        if not isinstance(prediction, BinaryPrediction):
            own_price = None
        if own_price is not None and isinstance(prediction, BinaryPrediction):
            prediction = BinaryPrediction(blend(prediction.probability, own_price.probability))
            method = f"blend of the {method} (1/3) and the community prediction (2/3)"
        problems = check(q, prediction)
        if problems:
            raise AugurError(
                "the combined forecast breaks Metaculus's rules, so it is not posted: "
                + "; ".join(problems)
            )
        self._ledger.record_forecast(
            run_id=run_id,
            question_id=q.question_id,
            kind="aggregate",
            member=None,
            method=method,
            probability=binary_probability(prediction),
            reasoning=f"{method} of {len(members)} members",
            call_id=None,
            question_type=q.type,
            values=values_json(prediction),
        )
        comment = render_comment(
            q=q,
            prediction=prediction,
            members=members,
            research=research,
            cutoff=cutoff,
            requested=n,
            blended_with=own_price,
        )
        report = QuestionReport(q, "forecast", prediction=prediction, members=members)
        report.comment = comment
        await self._deliver(run_id, mode, q, report, research)
        return report

    async def _deliver(
        self,
        run_id: str,
        mode: RunMode,
        q: Question,
        report: QuestionReport,
        research: ResearchResult,
    ) -> None:
        if report.prediction is None:
            raise AugurError("internal: delivering a question with no forecast")
        payload = forecast_payload(q.question_id, report.prediction)
        comment_payload = MetaculusClient.comment_payload(q.post_id, report.comment)
        spread = " | ".join(summary(q, m.prediction) for m in report.members)
        self._print("")
        self._print(f"== {q.title}")
        self._print(f"   {q.url} ({q.type})")
        self._print(
            f"   sources: {len(research.sources)} admitted, {len(research.rejected)} rejected; "
            f"members: {spread}"
        )
        self._print(f"   combined: {summary(q, report.prediction)}")
        if mode is not RunMode.LIVE:
            for kind, body in (("forecast", payload), ("comment", comment_payload)):
                self._ledger.record_submission(
                    run_id=run_id,
                    question_id=q.question_id,
                    post_id=q.post_id,
                    kind=kind,
                    payload=body,
                    status="dry_run",
                    http_status=None,
                    response_excerpt=None,
                )
            self._print("   would POST /api/questions/forecast/ (dry run, not sent):")
            shown = json.dumps(payload)
            self._print("   " + (shown if len(shown) <= 400 else shown[:400] + " …"))
            self._print("   would POST /api/comments/create/ as a private comment (dry run):")
            for line in report.comment.splitlines():
                self._print("   | " + line)
            return

        below = self._below_floor(report)
        if below:
            # The router never lets a scored question fall below the tier; this
            # is the last check, on the models that actually answered.
            for route in below:
                self._ledger.record_switch(
                    run_id=run_id,
                    question_id=q.question_id,
                    call_role="forecaster",
                    member=None,
                    from_route=route,
                    to_route=None,
                    reason="below_floor_at_submit",
                    outcome="refused_by_floor",
                    detail=f"{route} answered but is below the {FRONTIER_TIER_NAME} tier",
                )
            report.status = "skipped"
            report.reason = (
                f"submit floor: {', '.join(below)} answered, below the {FRONTIER_TIER_NAME} "
                "tier; nothing posted"
            )
            self._print(f"   not submitted: {report.reason}")
            return
        client = self._live_client()
        # Re-read just before writing: another run may have forecast meanwhile.
        fresh, _ = await client.get_question(q.post_id)
        if fresh.already_forecast(self._clock.now()) and not self._reforecast:
            report.status = "skipped"
            report.reason = "already forecast by this account (checked again before submitting)"
            return
        try:
            await self._submit(run_id, q, "forecast", payload, report)
        except MetaculusError:
            # The server may have taken the forecast and failed to answer. Keep
            # the comment as unsent so a later run that sees the forecast
            # standing posts this text, the reasoning behind it.
            self._ledger.record_submission(
                run_id=run_id,
                question_id=q.question_id,
                post_id=q.post_id,
                kind="comment",
                payload=comment_payload,
                status="failed",
                http_status=None,
                response_excerpt="not sent: the forecast's own post failed",
            )
            raise
        await self._submit(run_id, q, "comment", comment_payload, report)
        report.submitted = True
        self._print(f"   submitted {summary(q, report.prediction)}; posted this private comment:")
        for line in report.comment.splitlines():
            self._print("   | " + line)

    async def _submit(
        self,
        run_id: str,
        q: Question,
        kind: Literal["forecast", "comment"],
        payload: dict[str, Any] | list[dict[str, Any]],
        report: QuestionReport,
    ) -> None:
        client = self._live_client()
        try:
            if kind == "forecast":
                if not isinstance(payload, list):
                    raise AugurError("internal: a forecast payload is a list")
                result = await client.submit_forecast(payload)
            else:
                result = await client.create_private_comment(q.post_id, report.comment)
        except MetaculusError as exc:
            self._ledger.record_submission(
                run_id=run_id,
                question_id=q.question_id,
                post_id=q.post_id,
                kind=kind,
                payload=payload,
                status="failed",
                http_status=exc.status_code,
                response_excerpt=str(exc)[:300],
            )
            raise
        self._ledger.record_submission(
            run_id=run_id,
            question_id=q.question_id,
            post_id=q.post_id,
            kind=kind,
            payload=result.payload,
            status="submitted",
            http_status=result.status_code,
            response_excerpt=result.response_excerpt,
        )

    async def _repair_comment(
        self, run_id: str, q: Question, post: dict[str, Any], report: QuestionReport
    ) -> None:
        """Post the comment an earlier run failed to post after its forecast went in."""
        text = self._ledger.unposted_comment(q.question_id)
        if text is None:
            return
        self._ledger.record_question(run_id, q, self._clock.now(), post)
        report.comment = text
        try:
            await self._submit(
                run_id, q, "comment", MetaculusClient.comment_payload(q.post_id, text), report
            )
        except MetaculusError as exc:
            report.reason += f"; its missing comment failed again ({exc})"
            return
        report.reason += "; posted the comment an earlier run could not"

    def _check_submit_floor(self, slug: str) -> None:
        """Refuse before any model call if the first answering model is below the tier.

        Later routes in the chain are fallbacks: a scored question that would
        fall to one below the tier is skipped when it happens (the router's
        floor), and :meth:`_below_floor` checks the models that answered.
        """
        first = self._router.chain("forecast")[0]
        if in_frontier_tier(first):
            return
        log.warning("submit refused by the floor", tournament=slug, route=first.key)
        raise SubmitFloorError(
            f"--submit to {slug} needs a hosted model in the {FRONTIER_TIER_NAME} tier first "
            f"in AUGUR_FORECAST_MODELS, and {first.key} ({first.endpoint.host}) is not one; a "
            "weak forecast costs peer points that skipping does not. Drop --submit for a dry "
            f"run, or submit to {', '.join(sorted(PRACTICE_TOURNAMENTS))} to test the live path."
        )

    def _below_floor(self, report: QuestionReport) -> list[str]:
        """Answering routes below the tier, when the run is scored (else nothing)."""
        if not self._scored:
            return []
        return sorted({m.route.key for m in report.members if not in_frontier_tier(m.route)})

    async def _probe(self, run_id: str) -> float:
        """Probe every route once; disable the ones that do not answer. Returns the cost."""
        assert self._probe_providers is not None
        routes = self._router.routes()
        roles: dict[Route, list[str]] = {r: list(self._router.roles_of(r)) for r in routes}
        for route in self._fleet.extra_routes():
            if route not in roles:
                routes.append(route)
            roles[route] = [*roles.get(route, []), "member"]
        results = await probe_routes(routes, roles, self._probe_providers, self._router.catalog)
        record_probes(self._ledger, results, new_session_id(), run_id)
        for r in results:
            if r.answers:
                continue
            if r.outcome == "rate_limited":
                self._router.cool(
                    r.route, max(r.retry_after or 0.0, MIN_COOL_S), "probe: rate_limited"
                )
            else:
                self._router.disable(r.route, f"probe: {r.outcome}", r.outcome)
        self._print(format_probe(results))
        return sum(r.cost_usd for r in results)

    def _live_client(self) -> MetaculusClient:
        if self._client is None:
            raise AugurError("no Metaculus client; set METACULUS_TOKEN for live listing")
        return self._client

    def _print(self, text: str) -> None:
        print(text, file=self._out, flush=True)


def format_probe(results: Sequence[ProbeResult]) -> str:
    """The probe as a short table: one line per route."""
    lines = ["probe:"]
    for r in results:
        price = (
            f"${r.price.usd_per_million_input:g}/${r.price.usd_per_million_output:g} per M"
            if r.price
            else "unpriced"
        )
        latency = f"{r.latency_ms} ms" if r.latency_ms is not None else "-"
        tier = "frontier" if r.in_tier else "below tier"
        lines.append(
            f"  {r.route.key:<48} {r.outcome:<13} {latency:>8}  {price:<22} "
            f"listed {r.listed_on or '?'}  {tier}  [{'+'.join(r.roles)}]"
        )
    return "\n".join(lines)
