"""``augur forecast``: list (or load) questions, skip, cap, forecast, submit or print."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

import structlog

import augur
from augur.agents.calls import Budget, BudgetExceededError, CallRecorder
from augur.agents.forecaster import MemberForecast, draw_forecasts
from augur.agents.researcher import Researcher, ResearchResult
from augur.aggregate import median
from augur.clock import Clock
from augur.comment import render_comment
from augur.config import Settings
from augur.errors import AugurError, BelowFloorError, MetaculusError, SubmitFloorError
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.llm.registry import FRONTIER_TIER_NAME, PRACTICE_TOURNAMENTS, in_frontier_tier
from augur.llm.router import Router
from augur.log import get_logger
from augur.metaculus import MetaculusClient
from augur.questions import Question, question_from_post
from augur.research.source import Searcher

log = get_logger(__name__)

TOURNAMENTS: dict[str, str] = {
    "minibench": "minibench",
    "futureeval": "33121",
    "fall-2026": "33121",
    "bot-testing-area": "bot-testing-area",
}
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples" / "minibench"

Status = Literal["forecast", "skipped", "failed"]


@dataclass
class QuestionReport:
    question: Question
    status: Status
    reason: str = ""
    probability: float | None = None
    members: list[MemberForecast] = field(default_factory=list)
    comment: str = ""
    submitted: bool = False


@dataclass
class RunReport:
    run_id: str
    mode: RunMode
    source: str
    reports: list[QuestionReport]
    cost_usd: float

    def count(self, status: Status) -> int:
        return sum(1 for r in self.reports if r.status == status)


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
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._ledger = ledger
        self._router = router
        self._searchers = list(searchers)
        self._client = client
        self._out = out

    async def run(
        self,
        tournament: str,
        *,
        submit: bool,
        samples: Path | None,
        max_questions: int,
    ) -> RunReport:
        slug = TOURNAMENTS.get(tournament, tournament)
        if submit and (self._client is None or not self._client.has_token):
            raise AugurError("--submit needs METACULUS_TOKEN; without it run a dry run")
        if submit and samples is not None:
            raise AugurError("--submit forecasts live questions only; drop --samples")
        scored = submit and slug not in PRACTICE_TOURNAMENTS
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
        budget = Budget(self._settings.max_cost_per_question, self._settings.max_cost_per_run)
        calls = CallRecorder(self._router, self._ledger, self._clock, budget)
        if scored:
            self._router.set_floor("forecast", FRONTIER_TIER_NAME, in_frontier_tier)

        candidates = await self._candidates(slug, samples if use_samples else None, use_samples)
        reports: list[QuestionReport] = []
        chosen: list[tuple[Question, dict[str, Any]]] = []
        for q, post in candidates:
            reason = self._skip_reason(q, live_listing=not use_samples)
            if reason is None and len(chosen) >= max_questions:
                reason = f"over the cap of {max_questions} questions per run"
            if reason:
                skipped = QuestionReport(q, "skipped", reason)
                if mode is RunMode.LIVE and q.already_forecast(self._clock.now()):
                    await self._repair_comment(run_id, q, post, skipped)
                reports.append(skipped)
                log.info("question skipped", question_id=q.question_id, reason=skipped.reason)
            else:
                chosen.append((q, post))

        stop_reason = ""
        for q, post in chosen:
            if stop_reason:
                reports.append(QuestionReport(q, "skipped", stop_reason))
                continue
            structlog.contextvars.bind_contextvars(question_id=q.question_id)
            try:
                reports.append(await self._forecast(run_id, mode, calls, q, post))
            except BelowFloorError as exc:
                log.warning("question skipped by the floor", question_id=q.question_id)
                reports.append(QuestionReport(q, "skipped", f"submit floor: {exc}"))
            except BudgetExceededError as exc:
                stop_reason = str(exc)
                reports.append(QuestionReport(q, "failed", stop_reason))
            except AugurError as exc:
                log.error("question failed", question_id=q.question_id, error=str(exc))
                reports.append(QuestionReport(q, "failed", str(exc)))
        structlog.contextvars.unbind_contextvars("question_id")
        report = RunReport(run_id, mode, source, reports, budget.spent_run)
        log.info(
            "run finished",
            forecast=report.count("forecast"),
            skipped=report.count("skipped"),
            failed=report.count("failed"),
            cost_usd=round(report.cost_usd, 6),
        )
        structlog.contextvars.unbind_contextvars("run_id")
        self._print(
            f"done: {report.count('forecast')} forecast, {report.count('skipped')} skipped, "
            f"{report.count('failed')} failed; estimated cost ${report.cost_usd:.4f}"
        )
        for r in reports:
            if r.status != "forecast":
                self._print(f"  {r.status}: {r.question.url} {r.reason}")
        return report

    async def _candidates(
        self, slug: str, samples: Path | None, use_samples: bool
    ) -> list[tuple[Question, dict[str, Any]]]:
        if use_samples:
            return load_samples(samples or SAMPLES_DIR)
        client = self._live_client()
        listed = await client.list_open_questions(slug)
        detailed: list[tuple[Question, dict[str, Any]]] = []
        for q, post in listed:
            if q.type != "binary":
                detailed.append((q, post))
                continue
            # Only the detail carries my_forecasts, which the skip rule needs.
            try:
                detailed.append(await client.get_question(q.post_id))
            except MetaculusError as exc:
                log.warning("detail fetch failed", post_id=q.post_id, error=str(exc))
        return detailed

    def _skip_reason(self, q: Question, *, live_listing: bool) -> str | None:
        now = self._clock.now()
        if q.type != "binary":
            return f"{q.type} questions are not forecast yet (binary only until M2)"
        if live_listing and not q.is_open(now):
            return "not open now"
        if live_listing and q.already_forecast(now):
            return "already forecast by this account"
        return None

    async def _forecast(
        self,
        run_id: str,
        mode: RunMode,
        calls: CallRecorder,
        q: Question,
        post: dict[str, Any],
    ) -> QuestionReport:
        cutoff = self._clock.now()
        self._ledger.record_question(run_id, q, cutoff, post)
        researcher = Researcher(
            calls=calls,
            ledger=self._ledger,
            searchers=self._searchers,
            mode=mode,
        )
        research = await researcher.research(run_id, q, cutoff)
        n = self._settings.forecasters
        members, errors = await draw_forecasts(
            calls=calls,
            run_id=run_id,
            q=q,
            cutoff=cutoff,
            research=research,
            n=n,
            concurrency=n,
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
                probability=m.probability,
                reasoning=m.reasoning,
                call_id=m.call_id,
            )
        probability = round(median([m.probability for m in members]), 3)
        self._ledger.record_forecast(
            run_id=run_id,
            question_id=q.question_id,
            kind="aggregate",
            member=None,
            method="median",
            probability=probability,
            reasoning=f"median of {len(members)} members",
            call_id=None,
        )
        comment = render_comment(
            probability=probability,
            members=members,
            research=research,
            cutoff=cutoff,
            requested=n,
        )
        report = QuestionReport(q, "forecast", probability=probability, members=members)
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
        if report.probability is None:
            raise AugurError("internal: delivering a question with no forecast")
        forecast_payload = MetaculusClient.binary_payload(q.question_id, report.probability)
        comment_payload = MetaculusClient.comment_payload(q.post_id, report.comment)
        spread = " ".join(f"{m.probability * 100:.0f}%" for m in report.members)
        self._print("")
        self._print(f"== {q.title}")
        self._print(f"   {q.url}")
        self._print(
            f"   sources: {len(research.sources)} admitted, {len(research.rejected)} rejected; "
            f"members: {spread}; median: {report.probability * 100:.1f}%"
        )
        if mode is not RunMode.LIVE:
            for kind, payload in (("forecast", forecast_payload), ("comment", comment_payload)):
                self._ledger.record_submission(
                    run_id=run_id,
                    question_id=q.question_id,
                    post_id=q.post_id,
                    kind=kind,
                    payload=payload,
                    status="dry_run",
                    http_status=None,
                    response_excerpt=None,
                )
            self._print("   would POST /api/questions/forecast/ (dry run, not sent):")
            self._print("   " + json.dumps(forecast_payload))
            self._print("   would POST /api/comments/create/ as a private comment (dry run):")
            for line in report.comment.splitlines():
                self._print("   | " + line)
            return

        client = self._live_client()
        # Re-read just before writing: another run may have forecast meanwhile.
        fresh, _ = await client.get_question(q.post_id)
        if fresh.already_forecast(self._clock.now()):
            report.status = "skipped"
            report.reason = "already forecast by this account (checked again before submitting)"
            return
        try:
            await self._submit(run_id, q, "forecast", forecast_payload, report)
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
        self._print(f"   submitted {report.probability * 100:.1f}%; posted this private comment:")
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
                if report.probability is None:
                    raise AugurError("internal: submitting a question with no forecast")
                result = await client.submit_binary(q.question_id, report.probability)
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
        floor).
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

    def _live_client(self) -> MetaculusClient:
        if self._client is None:
            raise AugurError("no Metaculus client; set METACULUS_TOKEN for live listing")
        return self._client

    def _print(self, text: str) -> None:
        print(text, file=self._out, flush=True)
