#!/usr/bin/env python3
"""Is there an open question this account has not forecast? Standard library only.

The first job of the scheduled workflow: it runs on a bare runner in seconds,
so a run with nothing to do costs a few seconds rather than an install.

Environment: ``METACULUS_TOKEN`` (never printed), ``AUGUR_SCHEDULE`` (``off``,
``minibench`` or ``all``; anything else is ``off``), ``AUGUR_POLL_TOURNAMENTS``
(a comma list overriding the schedule's tournaments), ``GITHUB_OUTPUT``
(receives ``has_work=true|false`` and ``tournaments=<comma list>``).
An API error counts as work, so the forecast job runs and reports it.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

API = "https://www.metaculus.com/api"
USER_AGENT = "augur-forecast-poll (+https://github.com/TheAsianFish/augur)"
FORECAST_TYPES = "binary,multiple_choice,numeric,discrete"
SCHEDULES = {"minibench": ["minibench"], "all": ["minibench", "33121"]}
PAUSE_SECONDS = 1.0

Fetch = Callable[[str, Mapping[str, str]], Any]


class PollError(Exception):
    """A request failed; the message names the path and status, never a header."""


def http_get_json(url: str, headers: Mapping[str, str]) -> Any:
    request = urllib.request.Request(url, headers=dict(headers))  # noqa: S310 (https constant)
    path = urllib.parse.urlsplit(url).path
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise PollError(f"GET {path} returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise PollError(f"GET {path} failed: {type(reason).__name__}") from None
    except ValueError:
        raise PollError(f"GET {path} returned something other than JSON") from None


def _time(value: Any) -> datetime | None:
    """ISO 8601 text or Unix seconds, as an aware UTC datetime."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def needs_forecast(question: Mapping[str, Any], now: datetime) -> bool:
    """True unless the latest forecast has values and has not ended."""
    latest = (question.get("my_forecasts") or {}).get("latest")
    if not isinstance(latest, dict) or not latest.get("forecast_values"):
        return True
    end = _time(latest.get("end_time"))
    return end is not None and end <= now


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def poll_tournament(
    tournament: str,
    fetch: Fetch,
    headers: Mapping[str, str],
    sleep: Callable[[float], None],
    now: datetime,
) -> tuple[int, int, datetime | None]:
    """``(open, not yet forecast, soonest close among those)``; raises PollError."""
    query = urllib.parse.urlencode(
        {
            "tournaments": tournament,
            "statuses": "open",
            "forecast_type": FORECAST_TYPES,
            "limit": 100,
        }
    )
    listing = fetch(f"{API}/posts/?{query}", headers)
    results = listing.get("results") if isinstance(listing, dict) else None
    if not isinstance(results, list):
        raise PollError("GET /api/posts/ returned no results list; the API may have changed")
    opened, todo, soonest = 0, 0, None
    for post in results:
        if not isinstance(post, dict) or not isinstance(post.get("id"), int):
            continue
        if not isinstance(post.get("question"), dict):
            continue  # group or conditional post: the bot does not forecast these
        opened += 1
        sleep(PAUSE_SECONDS)
        detail = fetch(f"{API}/posts/{post['id']}/", headers)
        question = detail.get("question") if isinstance(detail, dict) else None
        if not isinstance(question, dict):
            raise PollError(f"GET /api/posts/{post['id']}/ returned no question")
        if needs_forecast(question, now):
            todo += 1
            close = _time(question.get("scheduled_close_time"))
            if close is not None and (soonest is None or close < soonest):
                soonest = close
    return opened, todo, soonest


def poll(
    env: Mapping[str, str],
    fetch: Fetch = http_get_json,
    sleep: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
) -> tuple[bool, list[str], list[str]]:
    """``(has_work, tournaments with work, lines to print)``."""
    schedule = env.get("AUGUR_SCHEDULE", "off").strip().lower() or "off"
    lines: list[str] = []
    if schedule != "off" and schedule not in SCHEDULES:
        lines.append(f"warning: AUGUR_SCHEDULE={schedule!r} is not off, minibench or all; off")
        schedule = "off"
    if schedule == "off":
        lines.append("schedule is off: nothing polled, no forecast job")
        return False, [], lines
    override = [t.strip() for t in env.get("AUGUR_POLL_TOURNAMENTS", "").split(",") if t.strip()]
    tournaments = override or SCHEDULES[schedule]
    token = env.get("METACULUS_TOKEN", "").strip()
    if not token:
        lines.append("no METACULUS_TOKEN: cannot see this account's forecasts; set the secret")
        return False, [], lines
    headers = {
        "Authorization": f"Token {token}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    moment = now or datetime.now(UTC)
    with_work: list[str] = []
    for index, tournament in enumerate(tournaments):
        if index:
            sleep(PAUSE_SECONDS)
        try:
            opened, todo, soonest = poll_tournament(tournament, fetch, headers, sleep, moment)
        except (PollError, AttributeError, TypeError) as exc:
            message = str(exc) if isinstance(exc, PollError) else "unexpected response shape"
            lines.append(f"warning: {tournament}: {message}; the forecast job will run and report")
            with_work.append(tournament)
            continue
        line = f"{tournament}: {opened} open, {todo} not yet forecast"
        if soonest is not None:
            line += f", closes soonest {_stamp(soonest)}"
        lines.append(line)
        if todo:
            with_work.append(tournament)
    return bool(with_work), with_work, lines


def main(env: Mapping[str, str] | None = None, fetch: Fetch = http_get_json) -> int:
    environ = os.environ if env is None else env
    has_work, tournaments, lines = poll(environ, fetch)
    for line in lines:
        print(line)
    print(f"has_work={'true' if has_work else 'false'} tournaments={','.join(tournaments)}")
    output = environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"has_work={'true' if has_work else 'false'}\n")
            fh.write(f"tournaments={','.join(tournaments)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
