"""Forecaster draws for every question type, and the parsers for their answers.

Each type has its own instruction for the final lines of the answer and its
own parser: a probability (binary), one probability per option (multiple
choice), or six percentiles (numeric and discrete) that
:mod:`augur.distributions` turns into the CDF Metaculus accepts. An answer
the parser cannot read drops that draw, never the question.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from datetime import datetime

from augur.agents.calls import BudgetExceededError, CallContext, CallRecorder
from augur.agents.researcher import ResearchResult, question_text
from augur.distributions import cdf_from_percentiles, normalise_probabilities
from augur.errors import AugurError, BelowFloorError, ForecastParseError
from augur.llm.base import LlmRequest, LlmResponse, Message
from augur.llm.endpoints import Route
from augur.log import get_logger
from augur.predictions import (
    PERCENTILE_LEVELS,
    BinaryPrediction,
    ChoicePrediction,
    ContinuousPrediction,
    Prediction,
)
from augur.questions import CONTINUOUS_TYPES, Question

log = get_logger(__name__)

MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99

FORECASTER_SYSTEM = (
    "You are a careful superforecaster. Think step by step: (a) the time left until the "
    "question resolves; (b) the status quo outcome if nothing changes; (c) a base rate "
    "from comparable past events; (d) what the research says that should move you from "
    "the base rate; (e) a brief scenario for Yes and one for No. Weight the status quo, "
    "since the world changes slowly most of the time. Keep it under 250 words. "
    "End with a final line exactly in the form: Probability: ZZ%"
)
CHOICE_SYSTEM = (
    "You are a careful superforecaster. Think step by step: (a) the time left until the "
    "question resolves; (b) the status quo outcome if nothing changes; (c) base rates from "
    "comparable past events; (d) what the research says that should move you; (e) leave "
    "some probability on the options you think unlikely, since surprises happen. Keep it "
    "under 250 words. End with one line per option, in the order given, exactly in the "
    "form 'Option name: ZZ%', with the percentages summing to 100."
)
NUMERIC_SYSTEM = (
    "You are a careful superforecaster. Think step by step: (a) the time left until the "
    "question resolves; (b) the outcome if current trends continue; (c) the outcome "
    "experts and markets expect, if known; (d) an unexpected scenario that gives a low "
    "outcome and one that gives a high outcome. Set wide intervals: good forecasters are "
    "humble about unknown unknowns. Keep it under 250 words. End with exactly these six "
    "lines, each value a plain number in the question's units, increasing:\n"
    "Percentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\nPercentile 60: XX\n"
    "Percentile 80: XX\nPercentile 90: XX"
)

_PROBABILITY = re.compile(r"probability\s*[:=]?\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*%", re.I)
_NUMBER = r"[-\u2212]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
_PERCENTILE = re.compile(
    r"percentile\s*(\d{1,2})\s*\**\s*[:=]\s*\**\s*[$\u20ac\u00a3]?\s*(" + _NUMBER + r")",
    re.I,
)


@dataclass(frozen=True)
class MemberForecast:
    member: int
    prediction: Prediction
    reasoning: str
    call_id: str
    route: Route

    @property
    def probability(self) -> float:
        """A binary member's probability (a type error for the other types)."""
        if not isinstance(self.prediction, BinaryPrediction):
            raise TypeError("only a binary member has a single probability")
        return self.prediction.probability


def parse_probability(text: str) -> float:
    """The last ``Probability: NN%`` in ``text``, clipped to [1%, 99%]."""
    matches = _PROBABILITY.findall(text)
    if not matches:
        raise ForecastParseError("no 'Probability: NN%' line in the answer; the draw is dropped")
    value = float(matches[-1])
    if not 0.0 <= value <= 100.0:
        raise ForecastParseError(f"probability {value}% is outside 0-100%; the draw is dropped")
    return min(MAX_PROBABILITY, max(MIN_PROBABILITY, value / 100.0))


def parse_choice(text: str, options: list[str]) -> dict[str, float]:
    """The last ``Option: NN%`` line for each option, normalised to sum to one."""
    found: dict[str, float] = {}
    for option in options:
        pattern = re.compile(
            r"^[\s*_#>\-\d.)]*[\"']?" + re.escape(option) + r"[\"']?\s*\**\s*[:=\-\u2013]\s*\**\s*"
            r"(\d{1,3}(?:\.\d+)?)\s*%",
            re.I | re.M,
        )
        matches = pattern.findall(text)
        if not matches:
            raise ForecastParseError(
                f"no 'Option: NN%' line for option {option!r}; the draw is dropped"
            )
        found[option] = float(matches[-1]) / 100.0
    if sum(found.values()) <= 0:
        raise ForecastParseError("every option was given 0%; the draw is dropped")
    try:
        return normalise_probabilities(found, options)
    except ValueError as exc:
        raise ForecastParseError(f"option probabilities unusable ({exc}); dropped") from None


def _to_float(raw: str) -> float:
    return float(raw.replace(",", "").replace("\u2212", "-"))


def parse_percentiles(text: str) -> list[tuple[float, float]]:
    """The last value given for each of the six percentiles, as (level, value)."""
    values: dict[int, float] = {}
    for level, raw in _PERCENTILE.findall(text):
        try:
            values[int(level)] = _to_float(raw)
        except ValueError:
            continue
    wanted = [round(lv * 100) for lv in PERCENTILE_LEVELS]
    missing = [lv for lv in wanted if lv not in values]
    if missing:
        raise ForecastParseError(
            "no 'Percentile NN: value' line for "
            + ", ".join(str(m) for m in missing)
            + "; the draw is dropped"
        )
    return [(lv / 100.0, values[lv]) for lv in wanted]


def parse_prediction(q: Question, text: str) -> Prediction:
    """The prediction a forecaster's answer states, for ``q``'s type."""
    if q.type == "binary":
        return BinaryPrediction(parse_probability(text))
    if q.type == "multiple_choice":
        return ChoicePrediction(parse_choice(text, list(q.options or [])))
    if q.type in CONTINUOUS_TYPES and q.scaling is not None:
        stated = parse_percentiles(text)
        try:
            cdf = cdf_from_percentiles(stated, q.scaling)
        except ValueError as exc:
            raise ForecastParseError(f"percentiles unusable ({exc}); the draw is dropped") from None
        return ContinuousPrediction(tuple(cdf), tuple(stated))
    raise ForecastParseError(f"{q.type} questions cannot be parsed")


def _is_answer_line(line: str) -> bool:
    return bool(_PROBABILITY.search(line) or _PERCENTILE.search(line))


def reasoning_of(text: str, options: list[str] | None = None) -> str:
    """The answer without its final probability, option or percentile lines."""
    lines = text.strip().splitlines()
    option_line = (
        re.compile(
            r"^[\s*_#>\-\d.)]*[\"']?(?:"
            + "|".join(re.escape(o) for o in options)
            + r")[\"']?\s*\**\s*[:=\-\u2013]\s*\**\s*\d{1,3}(?:\.\d+)?\s*%",
            re.I,
        )
        if options
        else None
    )
    while lines and (
        not lines[-1].strip()
        or _is_answer_line(lines[-1])
        or (option_line is not None and option_line.search(lines[-1]))
    ):
        lines.pop()
    return "\n".join(lines).strip()


def member_reasoning(response: LlmResponse, options: list[str] | None = None) -> str:
    """The stated reasoning; if the answer was only a number, the model's own trace."""
    stated = reasoning_of(response.text, options)
    if stated or not response.reasoning:
        return stated
    return "(from the model's reasoning trace) " + response.reasoning.strip()


def _bound(value: float | None, unit: str) -> str:
    if value is None:
        return "?"
    shown = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"
    return f"{shown} {unit}".strip()


def answer_format(q: Question) -> str:
    """What the question asks for, in the terms the answer lines must use."""
    if q.type == "multiple_choice":
        return "The options are, in this order: " + "; ".join(q.options or []) + "."
    if q.type in CONTINUOUS_TYPES and q.scaling is not None:
        s = q.scaling
        low = s.nominal_min if s.nominal_min is not None else s.range_min
        high = s.nominal_max if s.nominal_max is not None else s.range_max
        parts = [
            f"The answer is a number in {q.unit or 'the units of the question'}. "
            f"The question's range is {_bound(low, q.unit)} to {_bound(high, q.unit)}."
        ]
        parts.append(
            "Outcomes below the lower end are possible."
            if s.open_lower_bound
            else f"The outcome cannot be below {_bound(low, q.unit)}."
        )
        parts.append(
            "Outcomes above the upper end are possible."
            if s.open_upper_bound
            else f"The outcome cannot be above {_bound(high, q.unit)}."
        )
        if q.type == "discrete":
            parts.append("The outcome takes discrete values within that range.")
        return " ".join(parts)
    return ""


def system_prompt(q: Question) -> str:
    if q.type == "multiple_choice":
        return CHOICE_SYSTEM
    if q.type in CONTINUOUS_TYPES:
        return NUMERIC_SYSTEM
    return FORECASTER_SYSTEM


def forecaster_request(
    q: Question, cutoff: datetime, research: ResearchResult, member: int
) -> LlmRequest:
    shape = answer_format(q)
    return LlmRequest(
        model="",  # the router sets each route's model
        messages=(
            Message(role="system", content=system_prompt(q)),
            Message(
                role="user",
                content=question_text(q, cutoff)
                + (f"\n\n{shape}" if shape else "")
                + "\n\nResearch brief:\n"
                + research.brief,
            ),
        ),
        temperature=0.7,
        seed=member,
        # Reasoning models spend part of this before the answer: at 600, two of
        # five draws from a free reasoning model stopped before their
        # probability line (M2 bot-testing-area run).
        max_tokens=1500,
    )


async def draw_forecasts(
    *,
    calls: CallRecorder,
    run_id: str,
    q: Question,
    cutoff: datetime,
    research: ResearchResult,
    n: int,
    concurrency: int,
    timeout_s: float | None = None,
) -> tuple[list[MemberForecast], list[str]]:
    """``n`` independent draws (distinct seeds), at most ``concurrency`` at a time.

    With ``timeout_s``, draws still running when it expires are cancelled and
    the ones that finished are returned: near a question's deadline the best
    ensemble available is worth more than a complete one posted too late.
    """
    slots = asyncio.Semaphore(concurrency)
    errors: list[str] = []
    refused: list[BudgetExceededError] = []
    floored: list[BelowFloorError] = []
    cited = [s.source_id for s in research.sources]
    options = list(q.options or []) if q.type == "multiple_choice" else None

    async def one(member: int) -> MemberForecast | None:
        async with slots:
            request = forecaster_request(q, cutoff, research, member)
            ctx = CallContext(run_id, q.question_id, "forecaster", member)
            try:
                done = await calls.complete(ctx, request, source_ids=cited)
                prediction = parse_prediction(q, done.response.text)
            except BudgetExceededError as exc:
                refused.append(exc)
                return None
            except BelowFloorError as exc:
                # Like a failed draw: the others may still have come from the tier.
                floored.append(exc)
                errors.append(f"member {member}: {exc}")
                return None
            except AugurError as exc:
                errors.append(f"member {member}: {exc}")
                log.warning("forecaster failed", member=member, error=str(exc))
                return None
            return MemberForecast(
                member,
                prediction,
                member_reasoning(done.response, options),
                done.call_id,
                done.route,
            )

    tasks = [asyncio.create_task(one(m)) for m in range(n)]
    try:
        _, pending = await asyncio.wait(tasks, timeout=timeout_s)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        errors.append(f"{len(pending)} of {n} draws stopped at the question's deadline")
        log.warning("draws stopped at the deadline", stopped=len(pending), of=n)
    for t in tasks:
        if not t.cancelled() and (bug := t.exception()) is not None:
            raise bug  # an exception one() does not handle is a bug, not a failed draw
    if refused:
        # Never submit an ensemble the budget cut short; the run stops here.
        raise refused[0]
    members = [r for t in tasks if not t.cancelled() and (r := t.result()) is not None]
    if floored and len(members) < math.ceil(n / 2):
        # Too few draws from the tier: the question is skipped, not failed.
        raise floored[0]
    return members, errors
