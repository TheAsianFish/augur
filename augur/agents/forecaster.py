"""Forecaster draws for binary questions, and the parser for their answers."""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from datetime import datetime

from augur.agents.calls import BudgetExceededError, CallContext, CallRecorder
from augur.agents.researcher import ResearchResult, question_text
from augur.errors import AugurError, BelowFloorError, ForecastParseError
from augur.llm.base import LlmRequest, LlmResponse, Message
from augur.llm.endpoints import Route
from augur.log import get_logger
from augur.questions import Question

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

_PROBABILITY = re.compile(r"probability\s*[:=]?\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*%", re.I)


@dataclass(frozen=True)
class MemberForecast:
    member: int
    probability: float
    reasoning: str
    call_id: str
    route: Route


def parse_probability(text: str) -> float:
    """The last ``Probability: NN%`` in ``text``, clipped to [1%, 99%]."""
    matches = _PROBABILITY.findall(text)
    if not matches:
        raise ForecastParseError("no 'Probability: NN%' line in the answer; the draw is dropped")
    value = float(matches[-1])
    if not 0.0 <= value <= 100.0:
        raise ForecastParseError(f"probability {value}% is outside 0-100%; the draw is dropped")
    return min(MAX_PROBABILITY, max(MIN_PROBABILITY, value / 100.0))


def reasoning_of(text: str) -> str:
    """The answer without its final probability line."""
    lines = text.strip().splitlines()
    while lines and (not lines[-1].strip() or _PROBABILITY.search(lines[-1])):
        lines.pop()
    return "\n".join(lines).strip()


def member_reasoning(response: LlmResponse) -> str:
    """The stated reasoning; if the answer was only a number, the model's own trace."""
    stated = reasoning_of(response.text)
    if stated or not response.reasoning:
        return stated
    return "(from the model's reasoning trace) " + response.reasoning.strip()


def forecaster_request(
    q: Question, cutoff: datetime, research: ResearchResult, member: int
) -> LlmRequest:
    return LlmRequest(
        model="",  # the router sets each route's model
        messages=(
            Message(role="system", content=FORECASTER_SYSTEM),
            Message(
                role="user",
                content=question_text(q, cutoff) + "\n\nResearch brief:\n" + research.brief,
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
) -> tuple[list[MemberForecast], list[str]]:
    """``n`` independent draws (distinct seeds), at most ``concurrency`` at a time."""
    slots = asyncio.Semaphore(concurrency)
    errors: list[str] = []
    refused: list[BudgetExceededError] = []
    floored: list[BelowFloorError] = []
    cited = [s.source_id for s in research.sources]

    async def one(member: int) -> MemberForecast | None:
        async with slots:
            request = forecaster_request(q, cutoff, research, member)
            ctx = CallContext(run_id, q.question_id, "forecaster", member)
            try:
                done = await calls.complete(ctx, request, source_ids=cited)
                probability = parse_probability(done.response.text)
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
                member, probability, member_reasoning(done.response), done.call_id, done.route
            )

    results = await asyncio.gather(*(one(m) for m in range(n)))
    if refused:
        # Never submit an ensemble the budget cut short; the run stops here.
        raise refused[0]
    members = [r for r in results if r is not None]
    if floored and len(members) < math.ceil(n / 2):
        # Too few draws from the tier: the question is skipped, not failed.
        raise floored[0]
    return members, errors
