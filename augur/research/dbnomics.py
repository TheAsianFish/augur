"""Daily observations of a DBnomics series as of an instant (M18).

DBnomics' v22 API is keyless: ``/v22/series/<provider>/<dataset>/<series>?observations=1``
returns one document with the series' ``period`` and ``value`` arrays. The
dating rule (PLAN.md, ``m18-registration``, item 2): the observation for day
D is dated 00:00 UTC of D + 1, and :meth:`DBnomicsFetcher.fetch` keeps it only
if that instant is at or before ``as_of``. Every point the server returns is
filtered here, so a server that answers with later points than were asked
for cannot leak them. ``NA`` and other non-numbers are dropped.

The API serves the current revision of every value: a later correction to an
old station reading is not visible here (a stated risk), so what was read is
pinned by writing it to a ledger (:func:`to_source`) before anything uses it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from augur.clock import Clock, ensure_utc, to_iso
from augur.errors import SearchError
from augur.log import get_logger
from augur.ratelimit import RateLimiter
from augur.research.series import observation_dated_at
from augur.research.source import DateEvidence, Source
from augur.research.wikipedia import USER_AGENT, retry_after_seconds

log = get_logger("augur.research.dbnomics")

DEFAULT_BASE_URL = "https://api.db.nomics.world"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class Point:
    day: date
    value: float
    dated_at: datetime
    """00:00 UTC of ``day`` + 1."""


@dataclass(frozen=True)
class DSeries:
    code: str
    """``provider/dataset/series``, e.g. ``meteofrance/TEMPERATURE/celsius.07005.D``."""
    name: str
    url: str
    points: tuple[Point, ...]
    """Admitted points only, oldest first."""
    refused: int
    """Points dated after ``as_of``, dropped."""
    as_of: datetime
    retrieved_at: datetime

    def until(self, as_of: datetime) -> DSeries:
        """The same series as of an earlier instant: later points move to ``refused``."""
        as_of = ensure_utc(as_of)
        if as_of > self.as_of:
            raise SearchError(f"{self.code} was fetched as of {to_iso(self.as_of)}, not later")
        kept = tuple(p for p in self.points if p.dated_at <= as_of)
        return DSeries(
            code=self.code,
            name=self.name,
            url=self.url,
            points=kept,
            refused=self.refused + len(self.points) - len(kept),
            as_of=as_of,
            retrieved_at=self.retrieved_at,
        )


def code_from_id(question_id: str) -> str:
    """ForecastBench's ``meteofrance_TEMPERATURE_celsius.07005.D`` as a DBnomics code."""
    parts = question_id.split("_", 2)
    if len(parts) != 3 or not all(parts):
        raise SearchError(f"{question_id!r} is not a DBnomics provider_dataset_series id")
    return "/".join(parts)


def _value(raw: Any) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    number = float(raw)
    return number if math.isfinite(number) else None


class DBnomicsFetcher:
    """One series' observations from DBnomics' keyless API, as of an instant."""

    name = "dbnomics"

    def __init__(
        self,
        http: httpx.AsyncClient,
        clock: Clock,
        *,
        limiter: RateLimiter | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 2,
    ) -> None:
        self._http = http
        self._clock = clock
        self._limiter = limiter or RateLimiter(rate_per_second=1.0, burst=1, max_concurrent=1)
        self._base = base_url.rstrip("/")
        self._max_retries = max_retries

    def url(self, code: str) -> str:
        return f"{self._base}/v22/series/{code}?observations=1"

    async def fetch(self, code: str, as_of: datetime) -> DSeries:
        """``code``'s observations dated at or before ``as_of``."""
        as_of = ensure_utc(as_of)
        body = await self._get(code)
        series = body.get("series") if isinstance(body, dict) else None
        docs = series.get("docs") if isinstance(series, dict) else None
        doc = docs[0] if isinstance(docs, list) and len(docs) == 1 else None
        if not isinstance(doc, dict):
            raise SearchError(
                f"DBnomics has no single series {code}; check the code or retry later"
            )
        periods = doc.get("period")
        values = doc.get("value")
        if not isinstance(periods, list) or not isinstance(values, list):
            raise SearchError(f"DBnomics series {code} has no period and value arrays")
        if len(periods) != len(values):
            raise SearchError(
                f"DBnomics series {code} has {len(periods)} periods for "
                f"{len(values)} values; retry later"
            )
        by_day: dict[date, Point] = {}
        refused = 0
        for period, raw in zip(periods, values, strict=True):
            try:
                day = date.fromisoformat(str(period))
            except ValueError:
                continue  # not a daily period: never read
            dated_at = observation_dated_at(day)
            if dated_at > as_of:
                refused += 1  # never read past as_of, whatever the server sent
                continue
            value = _value(raw)
            if value is not None:
                by_day[day] = Point(day, value, dated_at)
        return DSeries(
            code=code,
            name=str(doc.get("series_name") or code).strip(),
            url=self.url(code),
            points=tuple(by_day[d] for d in sorted(by_day)),
            refused=refused,
            as_of=as_of,
            retrieved_at=self._clock.now(),
        )

    async def _get(self, code: str) -> Any:
        last_problem = "no attempt made"
        url = f"{self._base}/v22/series/{code}"
        for attempt in range(self._max_retries + 1):
            async with self._limiter.acquire():
                try:
                    response = await self._http.get(
                        url, params={"observations": "1"}, headers={"User-Agent": USER_AGENT}
                    )
                except httpx.HTTPError as exc:
                    last_problem = f"{type(exc).__name__}: {exc}"
                    log.warning("dbnomics_transport_error", attempt=attempt)
                    continue
            if response.status_code in _RETRY_STATUSES:
                last_problem = f"HTTP {response.status_code}"
                self._limiter.pause(retry_after_seconds(response))
                continue
            if response.status_code != 200:
                raise SearchError(
                    f"DBnomics answered HTTP {response.status_code} for {code}; "
                    "check the series code or the base URL"
                )
            try:
                return response.json()
            except ValueError:
                raise SearchError(
                    "DBnomics returned a body that is not JSON; retry later"
                ) from None
        raise SearchError(
            f"DBnomics failed after {self._max_retries + 1} attempts ({last_problem}); retry later"
        )


def _number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def to_source(series: DSeries, query: str, keep: Iterable[date]) -> Source:
    """The admitted points on the days ``keep`` as one source, dated by the last of them.

    Uses the ``series`` searcher's excerpt form (one ``day value`` line each),
    so the ledger scan checks that the source's date is backed by its last line.
    """
    wanted = set(keep)
    shown = [p for p in series.points if p.day in wanted]
    if not shown:
        raise SearchError(
            f"{series.code} has no observation on the wanted days dated at or before "
            f"{to_iso(series.as_of)}"
        )
    last = shown[-1]
    lines = [
        f"{series.name} ({series.code}), DBnomics: {len(shown)} daily observations on the "
        f"days the arms read, each dated 00:00 UTC of the day after its period, all at or "
        f"before {series.as_of:%Y-%m-%d %H:%M} UTC:",
        *(f"{p.day.isoformat()} {_number(p.value)}" for p in shown),
    ]
    return Source.build(
        searcher="series",
        query=query,
        url=series.url,
        title=f"{series.name} ({series.code}) daily observations",
        excerpt="\n".join(lines),
        published_at=last.dated_at,
        retrieved_at=series.retrieved_at,
        as_of=series.as_of,
        date_evidence=DateEvidence.PRICE_HISTORY,
        evidence_detail=(
            f"dbnomics {series.code} last observation {last.day.isoformat()}; "
            "the observation for day D is dated 00:00 UTC of D + 1"
        ),
    )
