"""The shadow template baseline: Metaculus's own template bot, run unpublished.

Q1 asks whether Augur beats the template bot on the same questions with the
same model. To make that like for like, this runs the real ``TemplateBot``
from Metaculus's ``forecasting-tools`` package (the optional ``baseline``
extra), never a copy of its prompts, on questions Augur forecast, with the
model of Augur's first forecast route. Nothing is posted to Metaculus: the
bot is built with ``publish_reports_to_metaculus=False``, and its forecast is
written to the ledger as a ``reference_forecasts`` row with
``forecaster='template'``.

Every ``forecasting_tools`` import lives inside :func:`build_template_runner`
and the runner it returns, so the default install and the unit tests do not
need the package. Tests drive :func:`run_shadow` with any object that has the
:class:`TemplateRunner` shape.

Research: the template uses AskNews when its keys are set. Without them its
own fallback is Perplexity through OpenRouter, a paid model, so the shadow
run uses no research instead and says so in each row's ``detail``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from augur.clock import Clock, SystemClock
from augur.distributions import validate_cdf, validate_probabilities
from augur.errors import AugurError, ConfigError, MissingCredentialsError
from augur.ledger import Ledger
from augur.log import get_logger
from augur.questions import FORECAST_TYPES, Question, question_from_post

if TYPE_CHECKING:
    from augur.config import Settings
    from augur.llm.endpoints import Route

log = get_logger("augur.baseline")

INSTALL_HINT = "install the baseline extra: uv sync --extra baseline"
SOURCE = "forecasting-tools TemplateBot, unpublished"
# The template's own settings for a run (its ``__main__`` block and the
# example ``GeneralLlm`` in its docstring): one research report, five
# predictions, aggregated by the template.
RESEARCH_REPORTS = 1
PREDICTIONS_PER_REPORT = 5
TEMPERATURE = 0.3
TIMEOUT_S = 160
ALLOWED_TRIES = 2


class BaselineUnavailableError(AugurError):
    """The ``baseline`` extra (``forecasting-tools``) is not installed."""

    def __init__(self, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        super().__init__(f"the shadow template needs forecasting-tools; {INSTALL_HINT}{suffix}")


class TemplateFailedError(AugurError):
    """The template bot returned no usable forecast for one question."""


@dataclass(frozen=True)
class TemplateForecast:
    """What the template forecast, in plain values, before Augur validates it.

    Exactly one of ``probability`` (binary), ``probabilities`` (multiple
    choice) or ``cdf`` (numeric, discrete) is set. ``percentiles`` are the
    ``(level, value)`` pairs the template's forecasters stated, when it has them.
    """

    detail: str
    probability: float | None = None
    probabilities: dict[str, float] | None = None
    cdf: list[float] | None = None
    percentiles: list[tuple[float, float]] = field(default_factory=list)


class TemplateRunner(Protocol):
    async def forecast(self, post_id: int) -> TemplateForecast: ...


@dataclass(frozen=True)
class ShadowResult:
    recorded: int = 0
    skipped: int = 0
    failed: int = 0
    lines: tuple[str, ...] = ()


# -- conversion -------------------------------------------------------------


def from_template_prediction(prediction: object, detail: str) -> TemplateForecast:
    """Read a ``forecasting-tools`` report's ``prediction`` into plain values.

    Duck-typed, so it needs no import: a float is binary; an object with
    ``predicted_options`` is a ``PredictedOptionList``; one with ``get_cdf``
    is a ``NumericDistribution``, whose CDF is the one the template would post
    (its ``publish_report_to_metaculus`` sends ``[p.percentile for p in get_cdf()]``).
    """
    if isinstance(prediction, bool):
        raise TemplateFailedError(f"the template returned a bool ({prediction}), not a forecast")
    if isinstance(prediction, int | float):
        return TemplateForecast(detail=detail, probability=float(prediction))
    options = getattr(prediction, "predicted_options", None)
    if options is not None:
        return TemplateForecast(
            detail=detail,
            probabilities={str(o.option_name): float(o.probability) for o in options},
        )
    get_cdf = getattr(prediction, "get_cdf", None)
    if callable(get_cdf):
        declared = getattr(prediction, "declared_percentiles", None) or []
        return TemplateForecast(
            detail=detail,
            cdf=[float(p.percentile) for p in get_cdf()],
            percentiles=[(float(p.percentile), float(p.value)) for p in declared],
        )
    raise TemplateFailedError(
        f"the template returned a {type(prediction).__name__}, which Augur cannot read"
    )


def template_values(question: Question, forecast: TemplateForecast) -> dict[str, Any]:
    """The forecast in the shape of :func:`augur.predictions.values_json`, checked
    against Metaculus's rules for the question. Raises :class:`TemplateFailedError`."""
    qid = question.question_id
    if question.type == "binary":
        p = forecast.probability
        if p is None:
            raise TemplateFailedError(f"question {qid} is binary; the template gave no probability")
        if not 0.0 <= p <= 1.0:
            raise TemplateFailedError(f"question {qid}: template probability {p} is not in [0, 1]")
        return {"probability": p}
    if question.type == "multiple_choice":
        probs = forecast.probabilities
        if probs is None:
            raise TemplateFailedError(f"question {qid} is multiple choice; the template gave none")
        errors = validate_probabilities(probs, question.options or [])
        if errors:
            raise TemplateFailedError(f"question {qid}: template forecast invalid: {errors}")
        return {"probabilities": {o: probs[o] for o in question.options or []}}
    if question.type in ("numeric", "discrete"):
        if forecast.cdf is None:
            raise TemplateFailedError(
                f"question {qid} is {question.type}; the template gave no CDF"
            )
        if question.scaling is None:
            raise TemplateFailedError(f"question {qid} has no usable scaling")
        errors = validate_cdf(forecast.cdf, question.scaling)
        if errors:
            raise TemplateFailedError(f"question {qid}: template CDF invalid: {errors}")
        return {
            "cdf": list(forecast.cdf),
            "percentiles": [[level, value] for level, value in forecast.percentiles],
        }
    raise TemplateFailedError(f"question {qid}: {question.type} questions are not compared")


# -- the batch --------------------------------------------------------------


def has_template_row(ledger: Ledger, question_id: int) -> bool:
    row = ledger.connection.execute(
        "SELECT 1 FROM reference_forecasts WHERE question_id = ? AND forecaster = 'template' "
        "LIMIT 1",
        (question_id,),
    ).fetchone()
    return row is not None


def questions_to_shadow(ledger: Ledger) -> list[Question]:
    """Questions with an Augur aggregate forecast and no template row, oldest first,
    each parsed from the latest post payload the ledger holds for it."""
    rows = ledger.connection.execute(
        "SELECT q.question_id, q.payload_json FROM questions q "
        "WHERE EXISTS (SELECT 1 FROM forecasts f WHERE f.question_id = q.question_id "
        "AND f.kind = 'aggregate') "
        "AND NOT EXISTS (SELECT 1 FROM reference_forecasts r "
        "WHERE r.question_id = q.question_id AND r.forecaster = 'template') "
        "ORDER BY q.recorded_at, q.rowid"
    ).fetchall()
    latest: dict[int, str] = {}
    for question_id, payload in rows:
        latest.pop(int(question_id), None)
        latest[int(question_id)] = str(payload)
    found: list[Question] = []
    for question_id, payload in latest.items():
        try:
            found.append(question_from_post(json.loads(payload)))
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("baseline_unparsable_payload", question_id=question_id, error=str(exc))
    return found


async def run_shadow(
    ledger: Ledger, runner: TemplateRunner, questions: Iterable[Question], clock: Clock
) -> ShadowResult:
    """Run the template on each question in turn and record its forecast.

    A question that already has a template row, or is of a type Augur does not
    forecast, is skipped. One question's failure is logged and counted; the
    batch continues.
    """
    recorded = skipped = failed = 0
    lines: list[str] = []
    for q in questions:
        bound = log.bind(question_id=q.question_id, post_id=q.post_id)
        if has_template_row(ledger, q.question_id):
            skipped += 1
            lines.append(f"post {q.post_id}: skipped, a template forecast is already recorded")
            continue
        if q.type not in FORECAST_TYPES:
            skipped += 1
            lines.append(f"post {q.post_id}: skipped, {q.type} questions are not compared")
            continue
        try:
            forecast = await runner.forecast(q.post_id)
            values = template_values(q, forecast)
        except Exception as exc:
            failed += 1
            reason = f"{type(exc).__name__}: {exc}"[:400]
            bound.warning("baseline_failed", error=reason)
            lines.append(f"post {q.post_id}: failed, {reason}")
            continue
        ledger.record_reference(
            question_id=q.question_id,
            post_id=q.post_id,
            question_type=q.type,
            forecaster="template",
            detail=forecast.detail,
            values=values,
            as_of=clock.now(),
            source=SOURCE,
        )
        recorded += 1
        bound.info("baseline_recorded", question_type=q.type)
        lines.append(f"post {q.post_id} ({q.type}): {_summary(values)}")
    return ShadowResult(recorded, skipped, failed, tuple(lines))


def _summary(values: Mapping[str, Any]) -> str:
    if "probability" in values:
        return f"template {values['probability'] * 100:.1f}%"
    if "probabilities" in values:
        parts = (f"{o} {p * 100:.1f}%" for o, p in values["probabilities"].items())
        return "template " + ", ".join(parts)
    stated = ", ".join(f"p{level * 100:g}={value:.4g}" for level, value in values["percentiles"])
    return f"template CDF of {len(values['cdf'])} points; stated {stated or 'none'}"


# -- the real runner, behind the optional extra -------------------------------


def forecasting_tools_version() -> str:
    try:
        return metadata.version("forecasting-tools")
    except metadata.PackageNotFoundError:
        raise BaselineUnavailableError("forecasting-tools is not installed") from None


def litellm_model(route: Route) -> tuple[str, dict[str, Any]]:
    """The litellm model string and extra arguments for ``route``, with no key in them."""
    endpoint = route.endpoint
    extra: dict[str, Any] = {}
    if route.reasoning_effort is not None and endpoint.api != "ollama":
        extra["reasoning_effort"] = route.reasoning_effort
    if endpoint.host == "openrouter.ai":
        return f"openrouter/{route.model}", extra
    if endpoint.api == "ollama":
        return f"ollama_chat/{route.model}", {**extra, "api_base": endpoint.base_url}
    return f"openai/{route.model}", {**extra, "base_url": endpoint.base_url}


def research_mode(settings: Settings) -> tuple[str, str]:
    """``(researcher, description)``: AskNews when its keys are set, else none."""
    if settings.has_asknews:
        return "asknews/news-summaries", "asknews/news-summaries"
    return (
        "no_research",
        "none (no AskNews keys; the template's own fallback, Perplexity via OpenRouter, is paid)",
    )


class ForecastingToolsRunner:
    """The real ``TemplateBot``, one question at a time, never published."""

    def __init__(self, bot: Any, client: Any, detail: str, cost_limit: float) -> None:
        self._bot = bot
        self._client = client
        self._detail = detail
        self._cost_limit = cost_limit

    async def forecast(self, post_id: int) -> TemplateForecast:
        from forecasting_tools import MonetaryCostManager

        question = await asyncio.to_thread(self._client.get_question_by_post_id, int(post_id))
        with MonetaryCostManager(self._cost_limit):
            reports = await self._bot.forecast_questions([question], return_exceptions=True)
        if len(reports) != 1:
            raise TemplateFailedError(f"the template returned {len(reports)} reports for one")
        report = reports[0]
        if isinstance(report, BaseException):
            raise TemplateFailedError(f"{type(report).__name__}: {report}"[:400]) from report
        return from_template_prediction(report.prediction, self._detail)


def build_template_runner(settings: Settings, route: Route | None = None) -> ForecastingToolsRunner:
    """The ``TemplateBot`` with Augur's first forecast route for every role it calls.

    Every role is set explicitly: left unset, the template defaults to paid
    models (Claude through OpenRouter, Perplexity for research).
    """
    try:
        from forecasting_tools import GeneralLlm, TemplateBot
        from forecasting_tools.helpers.metaculus_client import (
            MetaculusClient as FtMetaculusClient,
        )
    except ImportError as exc:
        raise BaselineUnavailableError(str(exc)) from None
    version = forecasting_tools_version()
    if settings.metaculus_token is None:
        raise MissingCredentialsError("the shadow template", ("METACULUS_TOKEN",))
    route = route or settings.chains()["forecast"][0]
    model, extra = litellm_model(route)
    key_env = route.endpoint.key_env
    if key_env:
        key = settings.keys.get(key_env)
        if key is None:
            raise MissingCredentialsError(f"the route {route.key}", (key_env,))
        extra["api_key"] = key.get_secret_value()
    llm = GeneralLlm(
        model=model,
        temperature=TEMPERATURE,
        timeout=TIMEOUT_S,
        allowed_tries=ALLOWED_TRIES,
        **extra,
    )
    researcher, research_detail = research_mode(settings)
    if settings.has_asknews and settings.asknews_client_secret is not None:
        # The template reads ASKNEWS_SECRET; Augur also accepts ASKNEWS_CLIENT_SECRET.
        os.environ.setdefault("ASKNEWS_SECRET", settings.asknews_client_secret.get_secret_value())
    client = FtMetaculusClient(
        base_url=settings.metaculus_base_url,
        token=settings.metaculus_token.get_secret_value(),
    )
    bot = TemplateBot(
        research_reports_per_question=RESEARCH_REPORTS,
        predictions_per_research_report=PREDICTIONS_PER_REPORT,
        use_research_summary_to_forecast=False,
        # The summary only decorates the published comment, and nothing is published.
        enable_summarize_research=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        llms={"default": llm, "summarizer": llm, "parser": llm, "researcher": researcher},
        metaculus_client=client,
    )
    if bot.publish_reports_to_metaculus:
        raise ConfigError("internal: the shadow template must never publish")
    detail = (
        f"forecasting-tools {version} {type(bot).__mro__[1].__name__} (TemplateBot); "
        f"model {model} (route {route.key}); research {research_detail}; "
        f"{RESEARCH_REPORTS}x{PREDICTIONS_PER_REPORT} predictions, temperature {TEMPERATURE}; "
        "unpublished"
    )
    return ForecastingToolsRunner(bot, client, detail, settings.max_cost_per_question)


# -- command line -----------------------------------------------------------


def add_parser(subparsers: Any) -> None:
    """Register ``augur baseline``."""
    p = subparsers.add_parser(
        "baseline",
        help="run Metaculus's template bot unpublished on questions Augur forecast",
        description=(
            "Run the real TemplateBot from forecasting-tools (the baseline extra) on the same "
            "questions with the model of the first forecast route, and record its forecast in "
            "the ledger as a reference forecast. Nothing is posted to Metaculus. By default, "
            "every question in the ledger with an Augur aggregate forecast and no template row."
        ),
    )
    p.add_argument("--ledger", type=Path, help="override AUGUR_LEDGER")
    p.add_argument(
        "--post",
        type=int,
        action="append",
        default=[],
        metavar="ID",
        help="run on this post (fetched from Metaculus) instead; repeatable",
    )


async def _fetch(settings: Settings, post_ids: Sequence[int]) -> list[Question]:
    import httpx

    from augur.metaculus import MetaculusClient

    async with httpx.AsyncClient(timeout=30) as http:
        client = MetaculusClient.from_settings(settings, http)
        return [(await client.get_question(pid))[0] for pid in post_ids]


async def main(args: argparse.Namespace, settings: Settings) -> int:
    clock = SystemClock()
    runner = build_template_runner(settings)
    ledger_path = args.ledger or settings.ledger_path
    with Ledger.open(ledger_path, clock) as ledger:
        if args.post:
            questions = await _fetch(settings, args.post)
        else:
            questions = questions_to_shadow(ledger)
        print(f"shadow template on {len(questions)} question(s)")
        result = await run_shadow(ledger, runner, questions, clock)
    for line in result.lines:
        print(line)
    print(
        f"ledger: {ledger_path}: {result.recorded} recorded, {result.skipped} skipped, "
        f"{result.failed} failed"
    )
    return 1 if result.failed else 0
