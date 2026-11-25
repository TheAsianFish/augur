"""The statistical arm on a live question: the ``stat`` and ``blend`` reference rows.

A forecast run that carries a :class:`LiveStatArm` (every shadow tick, M12)
calls :meth:`LiveStatArm.record` right after Augur's aggregate is written and
before anything is delivered, so both rows exist before the question closes:

* ``stat`` — the arm's own forecast (:mod:`augur.statarm`), written only
  where the arm has one;
* ``blend`` — the registered primary candidate: one-third Augur and
  two-thirds arm where the arm has a price or a series, and Augur's forecast
  unchanged everywhere else (its ``detail`` says which, and why).

``augur floor --candidate blend`` scores the ``blend`` rows; ``augur score``
reports both beside every other reference.

"Augur" here is the forecast before M5's exact-price blend. Where research
found the question's own community prediction, the shipped aggregate is
already one-third Augur and two-thirds that price, the same combination the
candidate makes, so blending the shipped aggregate again would count the
price twice.

A series is fetched only for numeric and discrete questions the series rule
reaches (:func:`augur.research.series.match_series`); its closes are written
to the ledger as one dated source and admitted through the leakage gate
before the arm reads them. Anything that goes wrong on the way (a lookup
that fails, a source the gate refuses) makes the arm abstain with the
reason; it never stops the question.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

import httpx

from augur.agents.researcher import ResearchResult
from augur.errors import AugurError
from augur.leakage import RunMode, admit_source
from augur.ledger import Ledger
from augur.livecal import LIVE_SOURCE
from augur.log import get_logger
from augur.predictions import Prediction, values_json
from augur.questions import CONTINUOUS_TYPES, Question
from augur.replay import exact_price
from augur.research.series import SeriesFetcher, match_series, to_source
from augur.statarm import Abstain, Arm, ArmForecast, BaseRates, binary_arm, primary, series_arm

log = get_logger("augur.statlive")

STAT = "stat"
BLEND = "blend"
SOURCE = "augur statarm (M12)"
TIMEOUT_S = 60.0
"""The most the arm may take on one question (two HTTP requests, normally a second)."""


@dataclass(frozen=True)
class StatOutcome:
    """What the arm said on one question, for the run's printout."""

    arm: Arm
    blend_method: str

    def line(self) -> str:
        said = self.arm.detail if isinstance(self.arm, ArmForecast) else self.arm.reason
        kind = self.arm.kind if isinstance(self.arm, ArmForecast) else "abstains"
        return f"stat arm ({kind}): {said}; blend: {self.blend_method}"


class LiveStatArm:
    """Writes the ``stat`` and ``blend`` rows for each question a run forecasts."""

    def __init__(
        self, *, rates: BaseRates, fetcher: SeriesFetcher | None, timeout_s: float = TIMEOUT_S
    ) -> None:
        self._rates = rates
        self._fetcher = fetcher
        self._timeout_s = timeout_s

    async def arm(
        self,
        *,
        ledger: Ledger,
        run_id: str,
        q: Question,
        cutoff: datetime,
        mode: RunMode,
        research: ResearchResult,
    ) -> Arm:
        """The arm's forecast for ``q`` as of ``cutoff``, or why it abstains."""
        if q.type == "binary":
            quote = exact_price(research)
            if quote is not None:
                detail = (
                    f"the question's own {quote.platform} price {quote.probability:.3f} "
                    f"as of {quote.price_time:%Y-%m-%d %H:%M} UTC"
                )
                return binary_arm(quote.probability, LIVE_SOURCE, self._rates, detail)
            return binary_arm(None, LIVE_SOURCE, self._rates)
        if q.type not in CONTINUOUS_TYPES:
            return Abstain(f"the arm does not answer {q.type} questions")
        match = match_series(q.title)
        if match is None:
            return Abstain("the series rule does not reach this title")
        if q.scaling is None:
            return Abstain("the question has no usable scaling")
        if self._fetcher is None:
            return Abstain(f"{match.query}: no series source in this run (a replay)")
        try:
            symbol = await self._fetcher.resolve(match)
            if symbol is None:
                return Abstain(f"{match.query}: no symbol found for it")
            series = await self._fetcher.fetch(symbol, cutoff)
            if not series.observations:
                return Abstain(f"{symbol}: no close dated at or before the cutoff")
            source = to_source(series, match.query, keep=len(series.observations))
        except (AugurError, httpx.HTTPError) as exc:
            log.warning("series lookup failed", query=match.query, error=str(exc))
            return Abstain(f"{match.query}: the series lookup failed ({exc})")
        ledger.record_source(source)
        admission = admit_source(source, cutoff, mode)
        ledger.record_admission(run_id, q.question_id, admission)
        if not admission.admitted:
            return Abstain(f"{symbol}: the leakage gate refused the series ({admission.reason})")
        closes = [(o.day, o.close) for o in series.observations]
        return series_arm(closes, match.target, q.scaling, f"{series.name} ({symbol})")

    async def record(
        self,
        *,
        ledger: Ledger,
        run_id: str,
        q: Question,
        cutoff: datetime,
        mode: RunMode,
        research: ResearchResult,
        augur: Prediction,
    ) -> StatOutcome:
        """Write the ``stat`` row (where the arm answers) and the ``blend`` row.

        The arm gets :data:`TIMEOUT_S`; past it, or on any error, it abstains,
        and the ``blend`` row is still written (as Augur's forecast unchanged).
        """
        arm: Arm
        try:
            async with asyncio.timeout(self._timeout_s):
                arm = await self.arm(
                    ledger=ledger, run_id=run_id, q=q, cutoff=cutoff, mode=mode, research=research
                )
        except TimeoutError:
            arm = Abstain(f"the arm took longer than {self._timeout_s:g} s")
        except Exception as exc:  # the arm abstains; it never costs the question its blend row
            log.warning("statistical arm failed", question_id=q.question_id, error=str(exc))
            arm = Abstain(f"the arm failed ({type(exc).__name__}: {exc})")
        if isinstance(arm, ArmForecast):
            ledger.record_reference(
                question_id=q.question_id,
                post_id=q.post_id,
                question_type=q.type,
                forecaster=STAT,
                detail=f"{arm.kind}: {arm.detail}",
                values=values_json(arm.prediction()),
                as_of=cutoff,
                source=SOURCE,
            )
        candidate, method = primary(augur, arm, q.scaling)
        ledger.record_reference(
            question_id=q.question_id,
            post_id=q.post_id,
            question_type=q.type,
            forecaster=BLEND,
            detail=method,
            values=values_json(candidate),
            as_of=cutoff,
            source=SOURCE,
        )
        return StatOutcome(arm, method)
