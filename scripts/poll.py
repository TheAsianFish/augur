#!/usr/bin/env python3
"""Is there an open question this account has not forecast? Standard library only.

The first job of the scheduled workflow: it runs on a bare runner in seconds,
so a run with nothing to do costs a few seconds rather than an install.

Environment: ``METACULUS_TOKEN`` (never printed), ``AUGUR_SCHEDULE`` (``off``,
``minibench`` or ``all``; anything else is ``off``), ``AUGUR_FUTUREEVAL`` (the
FutureEval project ``all`` polls: an id or slug, default ``33121``, Fall 2026;
set the Spring 2027 id when it opens), ``AUGUR_POLL_TOURNAMENTS``
(a comma list overriding the schedule's tournaments), ``GITHUB_OUTPUT``
(receives ``has_work=true|false`` and ``tournaments=<comma list>``).
An API error counts as work, so the forecast job runs and reports it.

Shadow mode (M11, ``AUGUR_SHADOW=true``): nothing is ever posted, so
``my_forecasts`` stays empty and cannot say what was done. A question counts
as done when the shadow exports in ``AUGUR_SHADOW_LEDGER`` (the ``shadow/``
tree of the ``ledger`` branch: JSONL, one file per tick) hold an Augur
aggregate forecast for it; the listing alone is read, no post detail. The
``all`` schedule polls MiniBench first and the FutureEval project only when
MiniBench has nothing to do and the day's free calls allow a full run:
``AUGUR_SHADOW_DAILY_CALLS`` (default 1000, OpenRouter's daily allowance for
free models) against the calls the exports show today, the template's
counted at ``AUGUR_SHADOW_TEMPLATE_CALLS`` a question.
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
SCHEDULES = ("minibench", "all")
FUTUREEVAL_DEFAULT = "33121"  # Fall 2026 FutureEval
PAUSE_SECONDS = 1.0
DAILY_CALLS_DEFAULT = 1000
# Calls a question costs before the exports show any, as measured on the
# bot-testing-area on 2026-11-24 (RESULTS.md M11): Augur 7 (planner, brief, five
# draws) plus a probe a run, the template 15 (forecasting-tools 0.3.7, five
# predictions with their parsing). The poll uses the exports' own Augur average
# once they have one; the template's calls are never in the ledger.
AUGUR_CALLS_DEFAULT = 8
TEMPLATE_CALLS_DEFAULT = 15
MAX_QUESTIONS_DEFAULT = 6

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


def _listing(tournament: str, fetch: Fetch, headers: Mapping[str, str]) -> list[Any]:
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
    return results


def poll_tournament(
    tournament: str,
    fetch: Fetch,
    headers: Mapping[str, str],
    sleep: Callable[[float], None],
    now: datetime,
) -> tuple[int, int, datetime | None]:
    """``(open, not yet forecast, soonest close among those)``; raises PollError."""
    results = _listing(tournament, fetch, headers)
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


# -- shadow mode ----------------------------------------------------------------


class ShadowLedger:
    """What the shadow exports hold: question ids Augur forecast, and today's calls."""

    def __init__(self) -> None:
        self.held: set[int] = set()
        self.templated: set[int] = set()
        self.calls_today = 0
        self.templates_today = 0
        self.calls_by_question: dict[int, int] = {}
        self.files = 0

    def augur_calls_per_question(self) -> float:
        done = [n for q, n in self.calls_by_question.items() if q in self.held]
        return sum(done) / len(done) if done else float(AUGUR_CALLS_DEFAULT)


def read_shadow(directory: str, today: str) -> ShadowLedger:
    """Read every ``*.jsonl`` export under ``directory`` (missing: nothing held).

    Each line is ``{"table": name, "row": {...}}`` (``augur.export``). A file is
    read whole; one that is not JSON lines raises ``PollError``, since guessing
    would forecast a question twice or never."""
    found = ShadowLedger()
    if not directory or not os.path.isdir(directory):
        return found
    seen_calls: set[str] = set()
    seen_templates: set[str] = set()
    paths: list[str] = []
    for root, _dirs, names in os.walk(directory):
        paths.extend(os.path.join(root, n) for n in names if n.endswith(".jsonl"))
    for path in sorted(paths):
        found.files += 1
        with open(path, encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    table, row = item["table"], item["row"]
                except (ValueError, KeyError, TypeError):
                    raise PollError(f"{path}:{number} is not an export line") from None
                if table == "forecasts" and row.get("kind") == "aggregate":
                    found.held.add(int(row["question_id"]))
                elif table == "reference_forecasts" and row.get("forecaster") == "template":
                    found.templated.add(int(row["question_id"]))
                    key = str(row.get("reference_id"))
                    if key not in seen_templates:
                        seen_templates.add(key)
                        if str(row.get("recorded_at", "")).startswith(today):
                            found.templates_today += 1
                elif table == "llm_calls":
                    key = str(row.get("call_id"))
                    if key in seen_calls:
                        continue  # a tick's second push repeats its first
                    seen_calls.add(key)
                    qid = int(row.get("question_id") or 0)
                    found.calls_by_question[qid] = found.calls_by_question.get(qid, 0) + 1
                    if str(row.get("started_at", "")).startswith(today):
                        found.calls_today += 1
    return found


def shadow_listing(
    tournament: str, fetch: Fetch, headers: Mapping[str, str], held: set[int]
) -> tuple[int, int, datetime | None]:
    """``(open, not yet in the shadow ledger, soonest close among those)``, from the
    listing alone (a shadow tick needs no ``my_forecasts``)."""
    results = _listing(tournament, fetch, headers)
    opened, todo, soonest = 0, 0, None
    for post in results:
        question = post.get("question") if isinstance(post, dict) else None
        if not isinstance(question, dict) or not isinstance(question.get("id"), int):
            continue
        opened += 1
        if question["id"] in held:
            continue
        todo += 1
        close = _time(question.get("scheduled_close_time"))
        if close is not None and (soonest is None or close < soonest):
            soonest = close
    return opened, todo, soonest


def _int_env(env: Mapping[str, str], name: str, default: int, lines: list[str]) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    lines.append(f"warning: {name}={raw!r} is not a positive whole number; using {default}")
    return default


def _is_name(value: str) -> bool:
    """A tournament id or slug: lowercase letters, digits and dashes only."""
    return bool(value) and all(c.isdigit() or c == "-" or "a" <= c <= "z" for c in value)


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
    futureeval = env.get("AUGUR_FUTUREEVAL", "").strip().lower() or FUTUREEVAL_DEFAULT
    if not _is_name(futureeval):
        lines.append(
            f"warning: AUGUR_FUTUREEVAL={futureeval!r} is not an id or slug; using the default"
        )
        futureeval = FUTUREEVAL_DEFAULT
    tournaments = override or (["minibench", futureeval] if schedule == "all" else ["minibench"])
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
    if env.get("AUGUR_SHADOW", "").strip().lower() == "true":
        return _poll_shadow(env, fetch, sleep, moment, tournaments, override, headers, lines)
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


def _poll_shadow(
    env: Mapping[str, str],
    fetch: Fetch,
    sleep: Callable[[float], None],
    moment: datetime,
    tournaments: list[str],
    override: list[str],
    headers: Mapping[str, str],
    lines: list[str],
) -> tuple[bool, list[str], list[str]]:
    """Shadow mode: work is an open question the shadow exports do not hold."""
    directory = env.get("AUGUR_SHADOW_LEDGER", "").strip()
    try:
        shadow = read_shadow(directory, moment.strftime("%Y-%m-%d"))
    except (PollError, OSError) as exc:
        # Unreadable exports would make every question look new: stop, loudly.
        lines.append(f"error: shadow ledger unreadable ({exc}); nothing forecast this tick")
        return False, [], lines
    allowance = _int_env(env, "AUGUR_SHADOW_DAILY_CALLS", DAILY_CALLS_DEFAULT, lines)
    per_template = _int_env(env, "AUGUR_SHADOW_TEMPLATE_CALLS", TEMPLATE_CALLS_DEFAULT, lines)
    per_run = _int_env(env, "AUGUR_MAX_QUESTIONS_PER_RUN", MAX_QUESTIONS_DEFAULT, lines)
    per_question = shadow.augur_calls_per_question() + per_template
    used = shadow.calls_today + shadow.templates_today * per_template
    lines.append(
        f"shadow ledger: {shadow.files} export(s), {len(shadow.held)} question(s) held, "
        f"{len(shadow.templated)} with a template forecast; today about {used} of "
        f"{allowance} free calls used ({per_question:.1f} a question)"
    )
    with_work: list[str] = []
    for index, tournament in enumerate(tournaments):
        if index and not override:
            # The schedule's later tournaments (FutureEval) get only what is left.
            if with_work:
                lines.append(f"{tournament}: not polled, minibench comes first this tick")
                continue
            if used + per_run * per_question > allowance:
                lines.append(
                    f"{tournament}: not polled, a run of {per_run} would pass the day's "
                    f"{allowance} free calls ({used} used)"
                )
                continue
        if index:
            sleep(PAUSE_SECONDS)
        try:
            opened, todo, soonest = shadow_listing(tournament, fetch, headers, shadow.held)
        except (PollError, AttributeError, TypeError) as exc:
            message = str(exc) if isinstance(exc, PollError) else "unexpected response shape"
            lines.append(f"warning: {tournament}: {message}; the shadow job will run and report")
            with_work.append(tournament)
            continue
        line = f"{tournament}: {opened} open, {todo} not in the shadow ledger"
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
