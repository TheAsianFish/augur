"""Wikipedia, read as each page stood at an instant.

For a query and an ``as_of`` instant the searcher:

1. asks the MediaWiki search API for matching titles;
2. for each title, asks for the newest revision saved at or before ``as_of``
   (``rvdir=older`` starting at ``rvstart=as_of``);
3. renders exactly that revision (``action=parse&oldid=...``) and keeps its
   prose as the excerpt.

The source's date is the revision's server-side save timestamp
(:attr:`DateEvidence.REVISION_TIMESTAMP`), and its URL is the permanent link
to that revision, so anyone can check what the model read.

Known limitations, stated plainly:

* **Ranking uses today's index.** ``list=search`` ranks against the current
  text of every page, so *which* titles are chosen can reflect events after
  ``as_of`` (a page may rank first because of what was added to it later).
  The *content* returned is still the revision current at ``as_of``.
* **Pages that did not exist at ``as_of`` are dropped**, as are pages whose
  only revisions are later than ``as_of`` and pages that are missing now.
* **Templates are expanded at their current version.** MediaWiki renders an
  old revision's wikitext with today's templates. Infoboxes, navboxes and
  other tables (where most template output lands) are removed from the
  excerpt, but inline templates in prose can still show later wording.
* A revision timestamp later than ``as_of`` in a response is treated as an API
  bug: the page is dropped and a warning is logged.
"""

from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from augur import __version__
from augur.clock import Clock, ensure_utc, parse_iso
from augur.errors import SearchError
from augur.log import get_logger
from augur.ratelimit import RateLimiter
from augur.research.source import DateEvidence, Source
from augur.research.text import html_to_text, truncate_words

USER_AGENT = f"augur-forecast/{__version__} (+https://github.com/TheAsianFish/augur)"
DEFAULT_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_RETRY_AFTER = 5.0
MAX_RETRY_AFTER = 120.0
# The shared client allows long reads for model calls; a wiki request that
# has not answered in this long is retried.
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

log = get_logger("augur.research.wikipedia")


def retry_after_seconds(response: httpx.Response, default: float = DEFAULT_RETRY_AFTER) -> float:
    """Seconds to wait from a ``Retry-After`` header (seconds or HTTP date), capped."""
    raw = response.headers.get("retry-after", "").strip()
    if not raw:
        return default
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
            date_header = response.headers.get("date")
            sent = parsedate_to_datetime(date_header) if date_header else None
        except (TypeError, ValueError):
            return default
        if sent is None or when.tzinfo is None or sent.tzinfo is None:
            return default
        seconds = (when - sent).total_seconds()
    return min(max(seconds, 0.0), MAX_RETRY_AFTER)


def mediawiki_timestamp(instant: datetime) -> str:
    """``instant`` as MediaWiki wants it: ``YYYY-MM-DDTHH:MM:SSZ`` in UTC."""
    return ensure_utc(instant).strftime("%Y-%m-%dT%H:%M:%SZ")


class WikipediaSearcher:
    """Wikipedia pages as they stood at ``as_of``; see the module docstring for limits."""

    name = "wikipedia"

    def __init__(
        self,
        http: httpx.AsyncClient,
        clock: Clock,
        limiter: RateLimiter | None = None,
        api_url: str = DEFAULT_API_URL,
        excerpt_chars: int = 3000,
        max_retries: int = 2,
    ) -> None:
        if excerpt_chars < 1:
            raise ValueError("excerpt_chars must be at least 1")
        self._http = http
        self._clock = clock
        self._limiter = limiter or RateLimiter(rate_per_second=1.0, burst=1, max_concurrent=1)
        self._api_url = api_url
        self._excerpt_chars = excerpt_chars
        self._max_retries = max_retries

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        """Pages matching ``query``, each as the revision current at ``as_of``."""
        as_of = ensure_utc(as_of)
        if limit < 1:
            return []
        titles = await self._search_titles(query, limit)
        sources: list[Source] = []
        for title in titles:
            found = await self._source_for(query, title, as_of)
            if found is not None:
                sources.append(found)
        return sources

    async def _search_titles(self, query: str, limit: int) -> list[str]:
        data = await self._get(
            query,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
                "srnamespace": "0",
            },
        )
        hits = data.get("query", {}).get("search", [])
        if not isinstance(hits, list):
            raise SearchError(
                f"Wikipedia search for {query!r} returned an unexpected shape; "
                "check the API URL or retry later"
            )
        return [str(hit["title"]) for hit in hits if isinstance(hit, dict) and "title" in hit]

    async def _source_for(self, query: str, title: str, as_of: datetime) -> Source | None:
        revision = await self._revision_at(query, title, as_of)
        if revision is None:
            return None
        revid, timestamp_text, page_title = revision
        published_at = parse_iso(timestamp_text)
        if published_at > as_of:
            log.warning(
                "wikipedia_revision_after_as_of",
                title=page_title,
                revid=revid,
                timestamp=timestamp_text,
                as_of=as_of.isoformat(),
            )
            return None
        html = await self._revision_html(query, revid)
        if html is None:
            return None
        if "mw:PageProp/redirect" in html:
            # At as_of the page was a redirect: there is no prose of its own to read.
            log.info("wikipedia_redirect_at_as_of", title=page_title, revid=revid)
            return None
        text = html_to_text(html)
        if not text:
            log.info("wikipedia_empty_revision", title=page_title, revid=revid)
            return None
        return Source.build(
            searcher=self.name,
            query=query,
            url=f"https://en.wikipedia.org/w/index.php?oldid={revid}",
            title=page_title,
            excerpt=truncate_words(text, self._excerpt_chars),
            published_at=published_at,
            retrieved_at=self._clock.now(),
            as_of=as_of,
            date_evidence=DateEvidence.REVISION_TIMESTAMP,
            evidence_detail=f"revid={revid} timestamp={timestamp_text}",
        )

    async def _revision_at(
        self, query: str, title: str, as_of: datetime
    ) -> tuple[int, str, str] | None:
        data = await self._get(
            query,
            {
                "action": "query",
                "prop": "revisions",
                "titles": title,
                "rvlimit": "1",
                "rvdir": "older",
                "rvstart": mediawiki_timestamp(as_of),
                "rvprop": "ids|timestamp",
            },
        )
        pages = data.get("query", {}).get("pages", [])
        for page in pages if isinstance(pages, list) else []:
            if not isinstance(page, dict) or page.get("missing") or page.get("invalid"):
                continue
            revisions = page.get("revisions") or []
            if not revisions:
                continue
            rev = revisions[0]
            try:
                return int(rev["revid"]), str(rev["timestamp"]), str(page.get("title", title))
            except (KeyError, TypeError, ValueError):
                continue
        log.info("wikipedia_no_revision_at_as_of", title=title, as_of=as_of.isoformat())
        return None

    async def _revision_html(self, query: str, revid: int) -> str | None:
        data = await self._get(
            query,
            {
                "action": "parse",
                "oldid": str(revid),
                "prop": "text",
                "disablelimitreport": "1",
                "disableeditsection": "1",
            },
            tolerate_api_error=True,
        )
        if "error" in data:
            code = data["error"].get("code") if isinstance(data["error"], dict) else None
            log.warning("wikipedia_parse_failed", revid=revid, code=code)
            return None
        text = data.get("parse", {}).get("text")
        return text if isinstance(text, str) else None

    async def _get(
        self, query: str, params: dict[str, str], *, tolerate_api_error: bool = False
    ) -> dict[str, Any]:
        full = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
        action = params.get("list") or params.get("prop") or params["action"]
        last_problem = "no attempt made"
        for attempt in range(self._max_retries + 1):
            async with self._limiter.acquire():
                try:
                    response = await self._http.get(
                        self._api_url,
                        params=full,
                        headers={"User-Agent": USER_AGENT},
                        timeout=REQUEST_TIMEOUT,
                    )
                except httpx.HTTPError as exc:
                    last_problem = f"{type(exc).__name__}: {exc}"
                    log.warning("wikipedia_transport_error", attempt=attempt, error=last_problem)
                    continue
            if response.status_code in _RETRY_STATUSES:
                wait = retry_after_seconds(response)
                last_problem = f"HTTP {response.status_code}"
                log.warning(
                    "wikipedia_retry", status=response.status_code, wait=wait, attempt=attempt
                )
                self._limiter.pause(wait)
                continue
            if response.status_code != 200:
                raise SearchError(
                    f"Wikipedia {action} for {query!r} failed with HTTP {response.status_code}; "
                    "check the API URL and User-Agent, or drop the Wikipedia searcher"
                )
            try:
                data = response.json()
            except ValueError:
                raise SearchError(
                    f"Wikipedia {action} for {query!r} returned a body that is not JSON; "
                    "check the API URL or retry later"
                ) from None
            if not isinstance(data, dict):
                raise SearchError(
                    f"Wikipedia {action} for {query!r} returned unexpected JSON; retry later"
                )
            error = data.get("error")
            if isinstance(error, dict) and error.get("code") == "maxlag":
                wait = retry_after_seconds(response)
                last_problem = "replication lag (maxlag)"
                log.warning("wikipedia_maxlag", wait=wait, attempt=attempt)
                self._limiter.pause(wait)
                continue
            if error is not None and not tolerate_api_error:
                code = error.get("code") if isinstance(error, dict) else error
                raise SearchError(
                    f"Wikipedia {action} for {query!r} returned API error {code!r}; "
                    "check the query parameters"
                )
            return data
        raise SearchError(
            f"Wikipedia {action} for {query!r} failed after {self._max_retries + 1} attempts "
            f"({last_problem}); retry later or run without the Wikipedia searcher"
        )
