"""Market prices as of an instant: the ``markets`` searcher.

Two jobs, one rule. The rule: a price is the last trade, history point or
aggregate entry at or before ``as_of``, and its own timestamp is the
source's date (:class:`~augur.research.quotes.MarketQuote` refuses to exist
otherwise). Every point a server returns is filtered here as well, so a
server that answers with later points than it was asked for cannot leak them.

1. **Exact quotes** (:meth:`MarketsSearcher.exact_quote`): the price of a
   market that *is* the question. A ForecastBench market question names its
   Manifold contract or Polymarket condition id; a Metaculus question's own
   community prediction comes from the aggregate history the API shows the
   token (none, for a bot token on a scored tournament).
2. **Similar markets** (:meth:`MarketsSearcher.search`): Manifold's and
   Polymarket's search, and Kalshi's open events matched by shared words
   (Kalshi has no text search), each hit priced as of ``as_of``.

Per platform, the date evidence:

* Manifold: the last bet at or before ``as_of`` (``probAfter``),
  :attr:`DateEvidence.MARKET_TRADE`;
* Polymarket: the last point of the CLOB price history at or before
  ``as_of`` for the Yes token, :attr:`DateEvidence.PRICE_HISTORY`;
* Kalshi: the last hourly candle that ended at or before ``as_of``,
  :attr:`DateEvidence.PRICE_HISTORY`;
* Metaculus: the recency-weighted aggregate entry in force at ``as_of``,
  :attr:`DateEvidence.AGGREGATE_HISTORY`.

Known limits, stated plainly: search ranks against today's index, so
*which* markets are found can reflect later events (the *price* still
cannot); Kalshi's list holds only events open now; a Polymarket or Kalshi
history with no point in the :data:`STALE_AFTER` before ``as_of`` is no price.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from augur.clock import Clock, ensure_utc, parse_iso, to_iso
from augur.errors import AugurError, SearchError
from augur.log import get_logger
from augur.questions import Question
from augur.ratelimit import RateLimiter
from augur.research.quotes import MarketQuote, Match, Platform
from augur.research.source import DateEvidence, Source
from augur.research.wikipedia import USER_AGENT, retry_after_seconds

log = get_logger("augur.research.markets")

STALE_AFTER = timedelta(days=7)
"""How far back a Polymarket or Kalshi price history is read: a market with no
point in the week before ``as_of`` has no price. Manifold has no such limit:
its market maker's price stands until the next bet."""
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "before",
        "between",
        "by",
        "did",
        "do",
        "does",
        "during",
        "for",
        "from",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "least",
        "more",
        "most",
        "no",
        "not",
        "of",
        "on",
        "or",
        "than",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "under",
        "until",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "within",
        "would",
        "after",
        "over",
        "per",
        "yes",
    ]
)
_MONTHS = frozenset(
    [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    ]
)
_WORD = re.compile(r"[a-z0-9][a-z0-9.\-']*")


@dataclass(frozen=True)
class MarketUrls:
    manifold: str = "https://api.manifold.markets/v0"
    polymarket_gamma: str = "https://gamma-api.polymarket.com"
    polymarket_clob: str = "https://clob.polymarket.com"
    kalshi: str = "https://api.elections.kalshi.com/trade-api/v2"


def keywords(title: str, limit: int = 8) -> list[str]:
    """The content words of a question title, in order, without repeats.

    Dates are not content: month names and bare numbers (days, years) are
    dropped, or every question about "26 November 2026" would match every
    market that closes that day.
    """
    seen: list[str] = []
    for word in _WORD.findall(title.lower()):
        word = word.strip(".-'")
        if len(word) < 2 or word in _STOPWORDS or word in _MONTHS or word in seen:
            continue
        if not any(ch.isalpha() for ch in word):
            continue
        seen.append(word)
    return seen[:limit]


def overlap(query: str, title: str) -> float:
    """Share of the query's content words that appear in ``title``."""
    wanted = set(keywords(query, limit=50))
    if not wanted:
        return 0.0
    return len(wanted & set(keywords(title, limit=50))) / len(wanted)


def _ms(instant: datetime) -> int:
    return int(ensure_utc(instant).timestamp() * 1000)


def _from_epoch(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=UTC)


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _instant(value: Any) -> datetime | None:
    if isinstance(value, int | float):
        return _from_epoch(value / 1000 if value > 1e11 else value)
    if isinstance(value, str) and value:
        try:
            return ensure_utc(parse_iso(value))
        except ValueError:
            return None
    return None


def metaculus_aggregate_at(post: dict[str, Any], as_of: datetime) -> tuple[float, datetime] | None:
    """The community prediction in force at ``as_of`` from a post's aggregate history.

    Binary questions only. ``None`` when the history is hidden (a bot token on
    a scored tournament sees none) or starts after ``as_of``.
    """
    question = post.get("question")
    if not isinstance(question, dict) or question.get("type") != "binary":
        return None
    aggregations = question.get("aggregations")
    if not isinstance(aggregations, dict):
        return None
    method = aggregations.get("recency_weighted")
    history = method.get("history") if isinstance(method, dict) else None
    best: tuple[float, datetime] | None = None
    for entry in history if isinstance(history, list) else []:
        if not isinstance(entry, dict):
            continue
        start = _instant(entry.get("start_time"))
        centers = entry.get("centers")
        if start is None or start > as_of or not isinstance(centers, list) or not centers:
            continue
        end = _instant(entry.get("end_time"))
        if end is not None and end <= as_of:
            continue
        try:
            p = float(centers[0])
        except (TypeError, ValueError):
            continue
        if 0.0 <= p <= 1.0 and (best is None or start > best[1]):
            best = (p, start)
    return best


class MarketsSearcher:
    """Prediction-market prices as of an instant; see the module docstring."""

    name = "markets"

    def __init__(
        self,
        http: httpx.AsyncClient,
        clock: Clock,
        *,
        metaculus: Any | None = None,
        limiter: RateLimiter | None = None,
        urls: MarketUrls | None = None,
        max_retries: int = 2,
        kalshi_pages: int = 5,
        min_overlap: float = 0.5,
    ) -> None:
        self._http = http
        self._clock = clock
        self._metaculus = metaculus
        self._limiter = limiter or RateLimiter(rate_per_second=4.0, burst=4, max_concurrent=2)
        self._urls = urls or MarketUrls()
        self._max_retries = max_retries
        self._kalshi_pages = kalshi_pages
        self._min_overlap = min_overlap
        self._kalshi_events: list[dict[str, Any]] | None = None

    # Exact quotes.

    async def exact_quote(
        self, platform: str, market_id: str, as_of: datetime, *, title: str = "", url: str = ""
    ) -> MarketQuote | None:
        """The price of the market ``market_id`` as of ``as_of``, or ``None`` if it had none."""
        as_of = ensure_utc(as_of)
        if platform == "manifold":
            return await self._manifold_quote(market_id, as_of, "exact", title, url)
        if platform == "polymarket":
            return await self._polymarket_quote(market_id, as_of, "exact", title, url)
        if platform == "kalshi":
            return await self._kalshi_quote(market_id, as_of, "exact", title, url)
        return None

    async def metaculus_quote(self, post_id: int, as_of: datetime) -> MarketQuote | None:
        """A Metaculus question's own community prediction as of ``as_of``, where shown."""
        if self._metaculus is None:
            return None
        post: dict[str, Any] = await self._metaculus.get_post(post_id)
        found = metaculus_aggregate_at(post, ensure_utc(as_of))
        if found is None:
            return None
        p, start = found
        question = post.get("question") or {}
        return MarketQuote(
            platform="metaculus",
            market_id=str(question.get("id", post_id)),
            title=str(post.get("title", "")),
            url=f"https://www.metaculus.com/questions/{post_id}/",
            outcome="Yes",
            probability=p,
            price_time=start,
            as_of=as_of,
            match="exact",
        )

    # Similar markets.

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        """Markets similar to ``query`` on Manifold, Polymarket and Kalshi, priced at ``as_of``."""
        as_of = ensure_utc(as_of)
        if limit < 1:
            return []
        quotes: list[MarketQuote] = []
        for platform, finder in (
            ("manifold", self._manifold_search),
            ("polymarket", self._polymarket_search),
            ("kalshi", self._kalshi_search),
        ):
            try:
                quotes.extend(await finder(query, as_of, limit))
            except SearchError as exc:
                log.warning("markets_search_failed", platform=platform, error=str(exc))
        quotes.sort(key=lambda q: -overlap(query, q.title))
        return [self.to_source(q, query) for q in quotes[:limit]]

    async def search_question(self, q: Question, as_of: datetime, limit: int) -> list[Source]:
        """The question's own community prediction (if shown), then similar markets."""
        as_of = ensure_utc(as_of)
        sources: list[Source] = []
        if q.type == "binary":
            try:
                own = await self.metaculus_quote(q.post_id, as_of)
            except AugurError as exc:  # one platform never sinks the search
                log.warning("markets_metaculus_failed", post_id=q.post_id, error=str(exc))
                own = None
            if own is not None:
                sources.append(self.to_source(own, q.title))
        sources.extend(await self.search(" ".join(keywords(q.title)), as_of, limit))
        return sources

    def to_source(self, quote: MarketQuote, query: str) -> Source:
        """A quote as a source the brief can cite, dated by its price."""
        kind = (
            "the market for this question"
            if quote.match == "exact"
            else "a similar market found by search; it may ask something different"
        )
        evidence = {
            "manifold": DateEvidence.MARKET_TRADE,
            "polymarket": DateEvidence.PRICE_HISTORY,
            "kalshi": DateEvidence.PRICE_HISTORY,
            "metaculus": DateEvidence.AGGREGATE_HISTORY,
        }[quote.platform]
        volume = f" Volume {quote.volume:,.0f}." if quote.volume is not None else ""
        excerpt = (
            f'{quote.platform.capitalize()} ({kind}): "{quote.title}". '
            f"Price {quote.probability:.1%} for {quote.outcome}, set at "
            f"{quote.price_time:%Y-%m-%d %H:%M} UTC, the last price at or before "
            f"{quote.as_of:%Y-%m-%d %H:%M} UTC.{volume}"
        )
        return Source.build(
            searcher=self.name,
            query=query,
            url=quote.url,
            title=quote.title,
            excerpt=excerpt,
            published_at=quote.price_time,
            retrieved_at=self._clock.now(),
            as_of=quote.as_of,
            date_evidence=evidence,
            evidence_detail=f"{quote.platform} {quote.market_id} price_time="
            f"{to_iso(quote.price_time)}",
            quote=quote,
        )

    # Manifold.

    async def _manifold_quote(
        self, contract_id: str, as_of: datetime, match: Match, title: str, url: str
    ) -> MarketQuote | None:
        bets = await self._get(
            "manifold",
            f"{self._urls.manifold}/bets",
            {"contractId": contract_id, "beforeTime": str(_ms(as_of) + 1), "limit": "20"},
        )
        last: tuple[int, float] | None = None
        for bet in bets if isinstance(bets, list) else []:
            if not isinstance(bet, dict) or bet.get("isCancelled"):
                continue
            created, after = bet.get("createdTime"), bet.get("probAfter")
            if not isinstance(created, int | float) or not isinstance(after, int | float):
                continue
            if not (math.isfinite(created) and math.isfinite(after)):
                continue  # a NaN would pass the clamp below and fail MarketQuote mid-run
            if created > _ms(as_of):
                continue  # a server that ignored beforeTime: never read past as_of
            if last is None or created > last[0]:
                last = (int(created), float(after))
        if last is None:
            return None
        # An automated market maker's price stands until the next bet, so the
        # last bet sets the price at as_of however long ago it was placed.
        price_time = _from_epoch(last[0] / 1000)
        return MarketQuote(
            platform="manifold",
            market_id=contract_id,
            title=title or contract_id,
            url=url or f"https://manifold.markets/market/{contract_id}",
            outcome="Yes",
            probability=min(max(last[1], 0.0), 1.0),
            price_time=price_time,
            as_of=as_of,
            match=match,
        )

    async def _manifold_search(self, query: str, as_of: datetime, limit: int) -> list[MarketQuote]:
        hits = await self._get(
            "manifold",
            f"{self._urls.manifold}/search-markets",
            {"term": query, "limit": str(limit * 2), "filter": "all", "contractType": "BINARY"},
        )
        quotes: list[MarketQuote] = []
        for hit in hits if isinstance(hits, list) else []:
            if not isinstance(hit, dict) or "id" not in hit:
                continue
            created = _instant(hit.get("createdTime"))
            if created is None or created > as_of:
                continue  # the market did not exist yet
            closed = _instant(hit.get("closeTime"))
            if closed is not None and closed <= as_of:
                continue  # closed by as_of: its last price says nothing about now
            quote = await self._manifold_quote(
                str(hit["id"]),
                as_of,
                "search",
                str(hit.get("question", "")),
                str(hit.get("url", "")),
            )
            if quote is not None and overlap(query, quote.title) >= self._min_overlap:
                quotes.append(quote)
            if len(quotes) >= limit:
                break
        return quotes

    # Polymarket.

    async def _polymarket_market(self, condition_id: str) -> dict[str, Any] | None:
        for closed in ("false", "true"):
            found = await self._get(
                "polymarket",
                f"{self._urls.polymarket_gamma}/markets",
                {"condition_ids": condition_id, "closed": closed},
            )
            for market in found if isinstance(found, list) else []:
                if isinstance(market, dict):
                    return market
        return None

    async def _polymarket_quote(
        self, condition_id: str, as_of: datetime, match: Match, title: str, url: str
    ) -> MarketQuote | None:
        market = await self._polymarket_market(condition_id)
        if market is None:
            return None
        return await self._polymarket_price(market, as_of, match, title, url)

    async def _polymarket_price(
        self, market: dict[str, Any], as_of: datetime, match: Match, title: str, url: str
    ) -> MarketQuote | None:
        tokens = _json_list(market.get("clobTokenIds"))
        outcomes = [str(o) for o in _json_list(market.get("outcomes"))]
        if not tokens or (outcomes and outcomes[0].lower() != "yes"):
            return None  # only Yes/No markets, priced on the Yes token
        end = int(as_of.timestamp())
        history = await self._get(
            "polymarket",
            f"{self._urls.polymarket_clob}/prices-history",
            {
                "market": str(tokens[0]),
                "startTs": str(end - int(STALE_AFTER.total_seconds())),
                "endTs": str(end),
                "fidelity": "60",
            },
        )
        points = history.get("history") if isinstance(history, dict) else None
        last: tuple[float, float] | None = None
        for point in points if isinstance(points, list) else []:
            if not isinstance(point, dict):
                continue
            t, p = point.get("t"), point.get("p")
            if not isinstance(t, int | float) or not isinstance(p, int | float):
                continue
            if not (math.isfinite(t) and math.isfinite(p)):
                continue
            if t > end:
                continue  # never read past as_of, whatever the server sent
            if last is None or t > last[0]:
                last = (float(t), float(p))
        if last is None:
            return None
        slug = market.get("slug")
        volume = market.get("volumeNum")
        return MarketQuote(
            platform="polymarket",
            market_id=str(market.get("conditionId", "")),
            title=title or str(market.get("question", "")),
            url=url or (f"https://polymarket.com/market/{slug}" if slug else ""),
            outcome="Yes",
            probability=min(max(last[1], 0.0), 1.0),
            price_time=_from_epoch(last[0]),
            as_of=as_of,
            match=match,
            volume=float(volume) if isinstance(volume, int | float) else None,
            closes_at=_instant(market.get("endDate")),
        )

    async def _polymarket_search(
        self, query: str, as_of: datetime, limit: int
    ) -> list[MarketQuote]:
        found = await self._get(
            "polymarket",
            f"{self._urls.polymarket_gamma}/public-search",
            {"q": query, "limit_per_type": str(limit * 2), "events_status": "all"},
        )
        events = found.get("events") if isinstance(found, dict) else None
        quotes: list[MarketQuote] = []
        for event in events if isinstance(events, list) else []:
            markets = event.get("markets") if isinstance(event, dict) else None
            for market in markets if isinstance(markets, list) else []:
                if not isinstance(market, dict):
                    continue
                created = _instant(market.get("startDate") or market.get("createdAt"))
                question = str(market.get("question", ""))
                if created is None or created > as_of:
                    continue
                if overlap(query, question) < self._min_overlap:
                    continue
                quote = await self._polymarket_price(market, as_of, "search", question, "")
                if quote is not None:
                    quotes.append(quote)
                if len(quotes) >= limit:
                    return quotes
        return quotes

    # Kalshi.

    async def _kalshi_quote(
        self, ticker: str, as_of: datetime, match: Match, title: str, url: str
    ) -> MarketQuote | None:
        found = await self._get("kalshi", f"{self._urls.kalshi}/markets/{ticker}", {})
        market = found.get("market") if isinstance(found, dict) else None
        if not isinstance(market, dict):
            return None
        event = await self._get(
            "kalshi", f"{self._urls.kalshi}/events/{market.get('event_ticker', '')}", {}
        )
        series = (
            (event.get("event") or {}).get("series_ticker") if isinstance(event, dict) else None
        )
        if not series:
            return None
        return await self._kalshi_price(str(series), market, as_of, match, title, url)

    async def _kalshi_price(
        self,
        series: str,
        market: dict[str, Any],
        as_of: datetime,
        match: Match,
        title: str,
        url: str,
    ) -> MarketQuote | None:
        ticker = str(market.get("ticker", ""))
        end = int(as_of.timestamp())
        found = await self._get(
            "kalshi",
            f"{self._urls.kalshi}/series/{series}/markets/{ticker}/candlesticks",
            {
                "start_ts": str(end - int(STALE_AFTER.total_seconds())),
                "end_ts": str(end),
                "period_interval": "60",
            },
        )
        candles = found.get("candlesticks") if isinstance(found, dict) else None
        last: tuple[int, float] | None = None
        for candle in candles if isinstance(candles, list) else []:
            if not isinstance(candle, dict):
                continue
            t = candle.get("end_period_ts")
            price = candle.get("price") if isinstance(candle.get("price"), dict) else {}
            raw = price.get("close_dollars") if isinstance(price, dict) else None
            if not isinstance(t, int) or t > end or raw is None:
                continue
            try:
                p = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(p):
                continue
            if last is None or t > last[0]:
                last = (t, p)
        if last is None:
            return None
        name = title or str(market.get("title", ticker))
        return MarketQuote(
            platform="kalshi",
            market_id=ticker,
            title=name,
            url=url or f"https://kalshi.com/markets/{series.lower()}",
            outcome="Yes",
            probability=min(max(last[1], 0.0), 1.0),
            price_time=_from_epoch(last[0]),
            as_of=as_of,
            match=match,
            closes_at=_instant(market.get("close_time")),
        )

    async def _kalshi_open_events(self) -> list[dict[str, Any]]:
        if self._kalshi_events is not None:
            return self._kalshi_events
        events: list[dict[str, Any]] = []
        cursor = ""
        for _ in range(self._kalshi_pages):
            params = {"status": "open", "limit": "200", "with_nested_markets": "true"}
            if cursor:
                params["cursor"] = cursor
            found = await self._get("kalshi", f"{self._urls.kalshi}/events", params)
            page = found.get("events") if isinstance(found, dict) else None
            events.extend(
                e for e in (page if isinstance(page, list) else []) if isinstance(e, dict)
            )
            cursor = str(found.get("cursor") or "") if isinstance(found, dict) else ""
            if not cursor:
                break
        self._kalshi_events = events
        return events

    async def _kalshi_search(self, query: str, as_of: datetime, limit: int) -> list[MarketQuote]:
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for event in await self._kalshi_open_events():
            for market in event.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                opened = _instant(market.get("open_time"))
                if opened is None or opened > as_of:
                    continue
                name = f"{event.get('title', '')} {market.get('yes_sub_title', '')}".strip()
                score = overlap(query, name)
                if score >= self._min_overlap:
                    scored.append(
                        (score, str(event.get("series_ticker", "")), market | {"_n": name})
                    )
        scored.sort(key=lambda s: -s[0])
        quotes: list[MarketQuote] = []
        for _score, series, market in scored[:limit]:
            quote = await self._kalshi_price(series, market, as_of, "search", market["_n"], "")
            if quote is not None:
                quotes.append(quote)
        return quotes

    # HTTP.

    async def _get(self, platform: Platform, url: str, params: dict[str, str]) -> Any:
        last_problem = "no attempt made"
        for attempt in range(self._max_retries + 1):
            async with self._limiter.acquire():
                try:
                    response = await self._http.get(
                        url, params=params, headers={"User-Agent": USER_AGENT}
                    )
                except httpx.HTTPError as exc:
                    last_problem = f"{type(exc).__name__}: {exc}"
                    log.warning("markets_transport_error", platform=platform, attempt=attempt)
                    continue
            if response.status_code in _RETRY_STATUSES:
                wait = retry_after_seconds(response)
                last_problem = f"HTTP {response.status_code}"
                self._limiter.pause(wait)
                continue
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                raise SearchError(
                    f"{platform} request failed with HTTP {response.status_code}; "
                    "check the API URL or run without the markets searcher"
                )
            try:
                return response.json()
            except ValueError:
                raise SearchError(
                    f"{platform} returned a body that is not JSON; retry later"
                ) from None
        raise SearchError(
            f"{platform} failed after {self._max_retries + 1} attempts ({last_problem}); "
            "retry later or run without the markets searcher"
        )


def priced(sources: Sequence[Source]) -> list[MarketQuote]:
    """The quotes behind ``sources``, in order."""
    return [s.quote for s in sources if s.quote is not None]
