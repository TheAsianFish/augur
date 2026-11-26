"""Daily closes of a quoted series as of an instant: the ``series`` source.

M12's series rule (PLAN.md, "The series rule"): a question is about a quoted
series only if its title names a ticker in parentheses or one of the fixed
:data:`ALIASES`, asks for a close, closing price or level, and names the day
(:func:`match_series`). The value is that series' daily close on that day.

The closes come from one free keyless source, Yahoo Finance's chart endpoint
(:class:`SeriesFetcher`). The dating rule: the close of trading day D (the
day in the exchange's own time zone) is dated 00:00 UTC of D + 1, and it is
admitted only if that instant is at or before ``as_of``. Every point the
server returns is filtered here, so a server that answers with later points
than were asked for cannot leak them; the intraday point of a day still
trading is dated after now and is refused the same way. :func:`to_source`
turns the admitted closes into one dated :class:`Source`, which is written to
the ledger before anything reads it, so the leakage gate covers it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from typing import Any, Literal
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from augur.clock import Clock, ensure_utc, to_iso
from augur.errors import SearchError
from augur.log import get_logger
from augur.ratelimit import RateLimiter
from augur.research.source import DateEvidence, Source
from augur.research.wikipedia import USER_AGENT, retry_after_seconds

log = get_logger("augur.research.series")

DEFAULT_BASE_URL = "https://query1.finance.yahoo.com"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_QUOTE_TYPES = frozenset(
    {"EQUITY", "ETF", "INDEX", "CRYPTOCURRENCY", "CURRENCY", "FUTURE", "MUTUALFUND"}
)

ALIASES: dict[str, tuple[str, str]] = {
    "s&p 500": ("^GSPC", "S&P 500"),
    "nasdaq composite": ("^IXIC", "Nasdaq Composite"),
    "nasdaq-100": ("^NDX", "Nasdaq-100"),
    "nasdaq 100": ("^NDX", "Nasdaq-100"),
    "dow jones industrial average": ("^DJI", "Dow Jones Industrial Average"),
    "dow jones": ("^DJI", "Dow Jones Industrial Average"),
    "russell 2000": ("^RUT", "Russell 2000"),
    "vix": ("^VIX", "CBOE Volatility Index"),
    "ftse 100": ("^FTSE", "FTSE 100"),
    "dax": ("^GDAXI", "DAX"),
    "cac 40": ("^FCHI", "CAC 40"),
    "euro stoxx 50": ("^STOXX50E", "Euro Stoxx 50"),
    "nikkei 225": ("^N225", "Nikkei 225"),
    "nikkei": ("^N225", "Nikkei 225"),
    "hang seng": ("^HSI", "Hang Seng Index"),
    "kospi": ("^KS11", "KOSPI"),
    "ibovespa": ("^BVSP", "Ibovespa"),
    "sensex": ("^BSESN", "BSE Sensex"),
    "nifty 50": ("^NSEI", "Nifty 50"),
    "s&p/tsx": ("^GSPTSE", "S&P/TSX Composite"),
    "asx 200": ("^AXJO", "S&P/ASX 200"),
    "bitcoin": ("BTC-USD", "Bitcoin (USD)"),
    "ether": ("ETH-USD", "Ether (USD)"),
    "ethereum": ("ETH-USD", "Ether (USD)"),
    "gold": ("GC=F", "Gold futures"),
    "wti": ("CL=F", "WTI crude oil futures"),
    "brent": ("BZ=F", "Brent crude oil futures"),
    "eur/usd": ("EURUSD=X", "EUR/USD"),
    "usd/jpy": ("JPY=X", "USD/JPY"),
    "gbp/usd": ("GBPUSD=X", "GBP/USD"),
    "usd/cny": ("CNY=X", "USD/CNY"),
    "usd/brl": ("BRL=X", "USD/BRL"),
    "eur/pln": ("EURPLN=X", "EUR/PLN"),
    "usd/mxn": ("MXN=X", "USD/MXN"),
}
"""Lowercase phrase -> (Yahoo symbol, display name): the fixed list of series
a title may name without a ticker."""

_TICKER = re.compile(r"\((\^?[A-Z][A-Z0-9.\-=^]{0,11})\)")
_NOT_TICKERS = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CNY",
        "BRL",
        "PLN",
        "MXN",
        "CAD",
        "AUD",
        "CHF",
        "INR",
        "KRW",
        "HKD",
        "UTC",
        "GMT",
        "ET",
        "EST",
        "EDT",
    }
)
"""Parenthesised words that are units or time zones, not tickers."""

_ALIAS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (phrase, re.compile(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", re.IGNORECASE))
    for phrase in sorted(ALIASES, key=lambda p: (-len(p), p))
)
_ASKS_CLOSE = re.compile(r"\b(close|closes|closing|level)\b", re.IGNORECASE)
_REJECT = re.compile(
    r"\b(outperform\w*|underperform\w*|revenue|earnings|volume|market cap\w*"
    r"|percentage change|returns?|between|reference rate|bonds?|percentage of par"
    r"|highest|lowest|(?<!industrial )average|intraday)\b|%\s*change",
    re.IGNORECASE,
)
"""Phrases that make a title about something other than one day's level of
the series: a relative or difference question, a fundamental, a fixing, a
bond, or an extreme or average over several days."""

_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH = r"(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DATE_PATTERNS = (
    re.compile(rf"\b{_MONTH}\s+(?P<day>\d{{1,2}}),?\s+(?P<year>\d{{4}})\b", re.IGNORECASE),
    re.compile(rf"\b(?P<day>\d{{1,2}})\s+{_MONTH}\s+(?P<year>\d{{4}})\b", re.IGNORECASE),
    re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b"),
)


@dataclass(frozen=True)
class SeriesMatch:
    """A title the series rule reaches."""

    query: str
    """The ticker as written (``MGLU3``, ``^IXIC``) or the alias phrase."""
    symbol: str | None
    """The Yahoo symbol when known without a lookup; ``None`` means
    :meth:`SeriesFetcher.resolve` is needed."""
    target: date
    """The day whose close the title asks for."""
    how: Literal["ticker", "alias"]


@dataclass(frozen=True)
class Observation:
    day: date
    """The trading day, in the exchange's own time zone."""
    close: float
    dated_at: datetime
    """00:00 UTC of ``day`` + 1: the instant the close is taken as public."""


@dataclass(frozen=True)
class Series:
    symbol: str
    name: str
    currency: str
    url: str
    """The chart URL fetched (it carries no secret)."""
    observations: tuple[Observation, ...]
    """Admitted closes only: dated at or before ``as_of``, oldest first."""
    refused: int
    """Points the server returned that were dated after ``as_of``, dropped."""
    as_of: datetime
    retrieved_at: datetime


def _exchange_zone(meta: dict[str, Any]) -> tzinfo:
    """The exchange's time zone by name; its current fixed offset only if the name is unknown."""
    name = meta.get("exchangeTimezoneName")
    if isinstance(name, str) and name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    offset = meta.get("gmtoffset")
    seconds = int(offset) if isinstance(offset, int | float) else 0
    return timezone(timedelta(seconds=seconds))


def observation_dated_at(day: date) -> datetime:
    """When the close of trading day ``day`` is taken as public: 00:00 UTC of the next day."""
    return datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=UTC)


def _dates_in(title: str) -> set[date]:
    found: set[date] = set()
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(title):
            raw_month = m.group("month")
            month = int(raw_month) if raw_month.isdigit() else _MONTHS[raw_month.lower()]
            try:
                found.add(date(int(m.group("year")), month, int(m.group("day"))))
            except ValueError:
                continue
    return found


def _ticker_in(title: str) -> str | None:
    for m in _TICKER.finditer(title):
        ticker = m.group(1)
        if ticker not in _NOT_TICKERS:
            return ticker
    return None


def _alias_in(title: str) -> str | None:
    for phrase, pattern in _ALIAS_PATTERNS:
        if pattern.search(title):
            return phrase
    return None


_COIN_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
"""Coin tickers as titles write them; a symbol search would find exchange-traded funds."""


def _known_symbol(ticker: str) -> str | None:
    if ticker in _COIN_TICKERS:
        return _COIN_TICKERS[ticker]
    if ticker.startswith("^") or "=" in ticker or "-USD" in ticker:
        return ticker
    return None


def match_series(title: str, *, fallback: date | None = None) -> SeriesMatch | None:
    """The series and day a question title asks the close of, or ``None``.

    A match needs all of: a ticker in parentheses or an alias (the ticker
    wins), a word asking for a close, closing price or level (registration item 2),
    and one day (from the title, else ``fallback``). A title naming two
    different days, or using any phrase in the reject list (relative changes,
    fundamentals, fixings, bonds, extremes), is not a match. Pure and offline.
    """
    if _REJECT.search(title) or not _ASKS_CLOSE.search(title):
        return None
    days = _dates_in(title)
    if len(days) > 1:
        return None
    target = next(iter(days)) if days else fallback
    if target is None:
        return None
    ticker = _ticker_in(title)
    if ticker is not None:
        return SeriesMatch(ticker, _known_symbol(ticker), target, "ticker")
    phrase = _alias_in(title)
    if phrase is not None:
        return SeriesMatch(phrase, ALIASES[phrase][0], target, "alias")
    return None


def _close(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    close = float(value)
    if not math.isfinite(close) or close <= 0:
        return None
    return close


class SeriesFetcher:
    """Daily closes from Yahoo Finance's chart endpoint, as of an instant."""

    name = "series"

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

    async def resolve(self, m: SeriesMatch) -> str | None:
        """The Yahoo symbol for ``m``: its own if known, else the search's.

        The search's quotes are read in its order: the first whose symbol is
        the ticker itself, else the first that is the ticker plus an exchange
        suffix (``MGLU3`` -> ``MGLU3.SA``; ``MGLU3F.SA`` is another line and
        never taken). ``None`` if nothing fits.
        """
        if m.symbol is not None:
            return m.symbol
        ticker = m.query.upper()
        found = await self._get(
            f"{self._base}/v1/finance/search",
            {"q": ticker, "quotesCount": "10", "newsCount": "0"},
        )
        quotes = found.get("quotes") if isinstance(found, dict) else None
        candidates: list[str] = []
        for item in quotes if isinstance(quotes, list) else []:
            if not isinstance(item, dict) or item.get("quoteType") not in _QUOTE_TYPES:
                continue
            symbol = item.get("symbol")
            if isinstance(symbol, str) and symbol:
                candidates.append(symbol)
        for symbol in candidates:
            if symbol.upper() == ticker:
                return symbol
        for symbol in candidates:
            if symbol.upper().startswith(ticker + "."):
                return symbol
        return None

    def chart_url(self, symbol: str, params: dict[str, str] | None = None) -> str:
        """The chart URL :meth:`fetch` reads for ``symbol`` (with ``params``, as asked)."""
        query = params or {"range": "2y", "interval": "1d"}
        joined = "&".join(f"{k}={v}" for k, v in query.items())
        return f"{self._base}/v8/finance/chart/{quote(symbol, safe='')}?{joined}"

    @staticmethod
    def period(start: datetime, as_of: datetime) -> dict[str, str]:
        """Chart parameters for the closes from ``start`` to a day past ``as_of``.

        A backtest asks for an explicit period, since ``range=2y`` counts back
        from today. The request runs a day past ``as_of`` on purpose: what keeps
        a later close out is the filter in :meth:`fetch`, never the request.
        """
        first = int(ensure_utc(start).timestamp())
        last = int((ensure_utc(as_of) + timedelta(days=1)).timestamp())
        return {"period1": str(first), "period2": str(last), "interval": "1d"}

    async def fetch(self, symbol: str, as_of: datetime, *, start: datetime | None = None) -> Series:
        """``symbol``'s daily closes, keeping those dated at or before ``as_of``.

        Two years back from today by default (the live arm); from ``start``
        to a day past ``as_of`` when ``start`` is given (a backtest).
        """
        as_of = ensure_utc(as_of)
        url = f"{self._base}/v8/finance/chart/{quote(symbol, safe='')}"
        params = {"range": "2y", "interval": "1d"} if start is None else self.period(start, as_of)
        body = await self._get(url, params)
        chart = body.get("chart") if isinstance(body, dict) else None
        if not isinstance(chart, dict):
            raise SearchError(
                f"Yahoo chart for {symbol} has no 'chart' object; check the symbol or retry later"
            )
        error = chart.get("error")
        if error:
            detail = error.get("description") if isinstance(error, dict) else error
            raise SearchError(
                f"Yahoo chart refused {symbol}: {detail}; check the symbol "
                "(resolve the ticker first) or leave the question to the other rules"
            )
        results = chart.get("result")
        result = results[0] if isinstance(results, list) and results else None
        if not isinstance(result, dict):
            raise SearchError(f"Yahoo chart for {symbol} has no result; retry later")
        raw_meta = result.get("meta")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        zone = _exchange_zone(meta)
        stamps = result.get("timestamp")
        indicators = result.get("indicators")
        quotes = indicators.get("quote") if isinstance(indicators, dict) else None
        first = quotes[0] if isinstance(quotes, list) and quotes else None
        closes = first.get("close") if isinstance(first, dict) else None
        stamps = stamps if isinstance(stamps, list) else []
        closes = closes if isinstance(closes, list) else []
        by_day: dict[date, Observation] = {}
        refused = 0
        for stamp, raw in zip(stamps, closes, strict=False):
            if isinstance(stamp, bool) or not isinstance(stamp, int | float):
                continue
            if not math.isfinite(stamp):
                continue
            # The trading day in the exchange's own zone at that bar's date (the
            # offset in ``meta`` is today's, wrong across a daylight-saving change),
            # and never dated before the day after the bar's own UTC date.
            day = datetime.fromtimestamp(stamp, tz=zone).date()
            utc_day = datetime.fromtimestamp(stamp, tz=UTC).date()
            dated_at = max(observation_dated_at(day), observation_dated_at(utc_day))
            if dated_at > as_of:
                refused += 1  # never read past as_of, whatever the server sent
                continue
            close = _close(raw)
            if close is None:
                continue
            by_day[day] = Observation(day=day, close=close, dated_at=dated_at)
        name = meta.get("longName") or meta.get("shortName") or symbol
        return Series(
            symbol=str(meta.get("symbol") or symbol),
            name=str(name).strip(),
            currency=str(meta.get("currency") or ""),
            url=self.chart_url(symbol, params),
            observations=tuple(by_day[d] for d in sorted(by_day)),
            refused=refused,
            as_of=as_of,
            retrieved_at=self._clock.now(),
        )

    async def _get(self, url: str, params: dict[str, str]) -> Any:
        last_problem = "no attempt made"
        for attempt in range(self._max_retries + 1):
            async with self._limiter.acquire():
                try:
                    response = await self._http.get(
                        url, params=params, headers={"User-Agent": USER_AGENT}
                    )
                except httpx.HTTPError as exc:
                    last_problem = f"{type(exc).__name__}: {exc}"
                    log.warning("series_transport_error", attempt=attempt)
                    continue
            if response.status_code in _RETRY_STATUSES:
                last_problem = f"HTTP {response.status_code}"
                self._limiter.pause(retry_after_seconds(response))
                continue
            if response.status_code not in (200, 404):
                raise SearchError(
                    f"Yahoo Finance request failed with HTTP {response.status_code}; "
                    "check the base URL or leave the question to the other rules"
                )
            try:
                return response.json()
            except ValueError:
                if response.status_code == 404:
                    raise SearchError(
                        "Yahoo Finance answered HTTP 404; check the symbol or the base URL"
                    ) from None
                raise SearchError(
                    "Yahoo Finance returned a body that is not JSON; retry later"
                ) from None
        raise SearchError(
            f"Yahoo Finance failed after {self._max_retries + 1} attempts ({last_problem}); "
            "retry later or leave the question to the other rules"
        )


def _number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def to_source(series: Series, query: str, *, keep: int = 30) -> Source:
    """The admitted closes as one source, dated by the last of them.

    The excerpt is a plain-text table of the last ``keep`` admitted closes,
    after a line giving how many were admitted in total and the dating rule.
    """
    if not series.observations:
        raise SearchError(
            f"{series.symbol} has no close dated at or before {to_iso(series.as_of)}; "
            "the series rule abstains on this question"
        )
    shown = series.observations[-keep:] if keep > 0 else ()
    currency = f" in {series.currency}" if series.currency else ""
    lines = [
        f"{series.name} ({series.symbol}) daily closes{currency}: "
        f"{len(series.observations)} closes admitted at or before "
        f"{series.as_of:%Y-%m-%d %H:%M} UTC; each close of trading day D is dated "
        f"00:00 UTC of D + 1. The last {len(shown)}:",
        *(f"{o.day.isoformat()} {_number(o.close)}" for o in shown),
    ]
    last = series.observations[-1]
    return Source.build(
        searcher="series",
        query=query,
        url=series.url,
        title=f"{series.name} ({series.symbol}) daily closes",
        excerpt="\n".join(lines),
        published_at=last.dated_at,
        retrieved_at=series.retrieved_at,
        as_of=series.as_of,
        date_evidence=DateEvidence.PRICE_HISTORY,
        evidence_detail=(
            f"yahoo chart {series.symbol} last trading day {last.day.isoformat()}; "
            "the close of day D is dated 00:00 UTC of D + 1"
        ),
    )
