"""Measure how long MiniBench questions stay open, and when they open.

Standard library only. Lists every post of the current MiniBench round
(tournament slug ``minibench``) and of earlier rounds, found by walking back
from the current round's start in 14-day steps (earlier rounds have the slug
``minibench-YYYY-MM-DD``), and prints per-question windows and summary
statistics.

    METACULUS_TOKEN=... python3 scripts/question_windows.py \
        --json data/runs/m3-question-windows.json \
        --txt data/runs/m3-question-windows.txt

The token is read from the environment, sent only in the Authorization
header, and never printed or written.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

API = "https://www.metaculus.com/api"
USER_AGENT = "augur-forecast-measure (+https://github.com/TheAsianFish/augur)"
PAGE = 100
ROUND_STEP = timedelta(days=14)


class FetchError(Exception):
    """A request failed; the message names the URL and status, never the token."""


class NotFound(FetchError):
    pass


class Client:
    def __init__(self, token: str, *, interval: float = 1.0, max_retries: int = 5) -> None:
        self._token = token
        self._interval = interval
        self._max_retries = max_retries
        self._last = 0.0
        self.requests = 0

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        url = f"{API}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        for attempt in range(self._max_retries + 1):
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            req = urllib.request.Request(  # noqa: S310 (fixed https host)
                url,
                headers={"Authorization": f"Token {self._token}", "User-Agent": USER_AGENT},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https host)
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise NotFound(f"GET {url}: 404") from None
                if exc.code == 429 and attempt < self._max_retries:
                    retry_after = exc.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.isdigit() else 10.0 * (attempt + 1)
                    print(f"429 on {url}; waiting {delay:.0f}s", file=sys.stderr)
                    time.sleep(min(delay, 120.0))
                    continue
                raise FetchError(f"GET {url}: HTTP {exc.code}") from None
            except urllib.error.URLError as exc:
                raise FetchError(f"GET {url}: {exc.reason}") from None
        raise FetchError(f"GET {url}: still rate limited after {self._max_retries} retries")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty list")
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def questions_from_posts(posts: list[dict[str, Any]], round_slug: str) -> list[dict[str, Any]]:
    """One row per question; a group post contributes each of its sub-questions."""
    rows: list[dict[str, Any]] = []
    for post in posts:
        subs: list[dict[str, Any]] = []
        if post.get("question"):
            subs = [post["question"]]
        elif post.get("group_of_questions"):
            subs = list(post["group_of_questions"].get("questions", []))
        for q in subs:
            opened = q.get("open_time") or post.get("open_time")
            closes = q.get("scheduled_close_time") or post.get("scheduled_close_time")
            if not opened or not closes:
                continue
            rows.append(
                {
                    "round": round_slug,
                    "post_id": post["id"],
                    "question_id": q["id"],
                    "type": q.get("type", "unknown"),
                    "status": q.get("status") or post.get("status"),
                    "open_time": iso(parse_time(opened)),
                    "scheduled_close_time": iso(parse_time(closes)),
                }
            )
    rows.sort(key=lambda r: (r["open_time"], r["question_id"]))
    return rows


def _dist(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "min": min(values),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": max(values),
    }


def window_stats(
    questions: list[dict[str, Any]],
    round_starts: dict[str, datetime],
    days: tuple[int, ...] = (1, 2, 3, 4, 7, 14),
) -> dict[str, Any]:
    """Summary statistics over question rows (times in hours unless named otherwise)."""
    by_type = Counter(q["type"] for q in questions)
    windows = [
        (parse_time(q["scheduled_close_time"]) - parse_time(q["open_time"])).total_seconds() / 3600
        for q in questions
    ]
    gaps: list[float] = []
    simultaneous = 0
    per_round: dict[str, Any] = {}
    rounds = sorted(
        {q["round"] for q in questions},
        key=lambda slug: round_starts.get(slug, datetime.min.replace(tzinfo=UTC)),
    )
    for slug in rounds:
        rq = [q for q in questions if q["round"] == slug]
        opens = sorted(parse_time(q["open_time"]) for q in rq)
        distinct = sorted(set(opens))
        simultaneous += len(opens) - len(distinct)
        rgaps = [(b - a).total_seconds() / 3600 for a, b in itertools.pairwise(distinct)]
        gaps.extend(rgaps)
        # Anchor at the round's start_date, or at its first open if that came earlier.
        start = min(round_starts.get(slug, opens[0]), opens[0])
        rwin = [
            (parse_time(q["scheduled_close_time"]) - parse_time(q["open_time"])).total_seconds()
            / 3600
            for q in rq
        ]
        share = {
            str(n): sum(1 for o in opens if o < start + timedelta(days=n)) / len(opens)
            for n in days
        }
        span = (opens[-1] - opens[0]).total_seconds() / 3600
        per_round[slug] = {
            "anchor": iso(start),
            "n_questions": len(rq),
            "by_type": dict(sorted(Counter(q["type"] for q in rq).items())),
            "first_open": iso(opens[0]),
            "last_open": iso(opens[-1]),
            "last_close": iso(max(parse_time(q["scheduled_close_time"]) for q in rq)),
            "open_span_hours": span,
            "window_hours": _dist(rwin),
            "gap_hours": _dist(rgaps) if rgaps else None,
            "share_opened_in_first_n_days": share,
        }
    by_hour = Counter(parse_time(q["open_time"]).hour for q in questions)
    return {
        "n_questions": len(questions),
        "n_rounds": len(rounds),
        "by_type": dict(sorted(by_type.items())),
        "window_hours": _dist(windows),
        "window_hours_counts": {
            f"{k:.2f}": v for k, v in sorted(Counter(round(w, 2) for w in windows).items())
        },
        "gap_between_consecutive_opens_hours": _dist(gaps) if gaps else None,
        "simultaneous_opens": simultaneous,
        "opens_by_utc_hour": {str(h): by_hour.get(h, 0) for h in range(24)},
        "rounds": per_round,
    }


def fetch_round(client: Client, slug: str) -> list[dict[str, Any]]:
    posts: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = client.get(
            "/posts/",
            {
                "tournaments": slug,
                "limit": str(PAGE),
                "offset": str(offset),
                "order_by": "-open_time",
            },
        )
        results = page.get("results", [])
        posts.extend(results)
        if len(results) < PAGE or not page.get("next"):
            return posts
        offset += PAGE


def discover_rounds(
    client: Client, max_rounds: int, max_misses: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """The current round plus earlier ones at 14-day steps; returns (rounds, slugs not found)."""
    current = client.get("/projects/tournaments/minibench/")
    rounds = [current]
    missing: list[str] = []
    start = parse_time(current["start_date"])
    misses = 0
    k = 1
    while len(rounds) < max_rounds and misses < max_misses:
        slug = f"minibench-{(start - k * ROUND_STEP):%Y-%m-%d}"
        k += 1
        try:
            rounds.append(client.get(f"/projects/tournaments/{slug}/"))
            misses = 0
        except NotFound:
            missing.append(slug)
            misses += 1
    return rounds, missing


def _fmt_dist(d: dict[str, float] | None, unit: str = "h") -> str:
    if d is None:
        return "n/a"
    return (
        f"n={d['n']:.0f} min {d['min']:.2f}{unit} median {d['median']:.2f}{unit} "
        f"p90 {d['p90']:.2f}{unit} max {d['max']:.2f}{unit}"
    )


def render(result: dict[str, Any]) -> str:
    s = result["summary"]
    lines = [
        "MiniBench question windows",
        f"command: {result['command']}",
        f"fetched: {result['fetched_at']} ({result['requests']} requests)",
        f"rounds: {', '.join(r['slug'] for r in result['rounds_found'])}",
        f"round slugs tried and not found: {', '.join(result['rounds_not_found']) or 'none'}",
        "",
        f"questions: {s['n_questions']} in {s['n_rounds']} rounds",
        "by type: " + ", ".join(f"{k} {v}" for k, v in s["by_type"].items()),
        f"window (scheduled close - open): {_fmt_dist(s['window_hours'])}",
        "window values: " + ", ".join(f"{k}h x{v}" for k, v in s["window_hours_counts"].items()),
        "gap between consecutive distinct opens, within a round: "
        + _fmt_dist(s["gap_between_consecutive_opens_hours"]),
        f"questions sharing an open instant with another: {s['simultaneous_opens']}",
        "opens by UTC hour: " + " ".join(f"{h:0>2}:{n}" for h, n in s["opens_by_utc_hour"].items()),
        "",
        "per round:",
    ]
    for slug, r in s["rounds"].items():
        shares = " ".join(
            f"{n}d {v * 100:.0f}%" for n, v in r["share_opened_in_first_n_days"].items()
        )
        lines += [
            f"  {slug}: anchor {r['anchor']}, {r['n_questions']} questions "
            f"({', '.join(f'{k} {v}' for k, v in r['by_type'].items())})",
            f"    opens {r['first_open']} .. {r['last_open']} "
            f"(span {r['open_span_hours']:.1f}h), last close {r['last_close']}",
            f"    windows: {_fmt_dist(r['window_hours'])}",
            f"    gaps: {_fmt_dist(r['gap_hours'])}",
            f"    share opened within the first N days of the round: {shares}",
        ]
    lines += [""] + [f"note: {n}" for n in result["notes"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", default="data/runs/m3-question-windows.json")
    parser.add_argument("--txt", default="data/runs/m3-question-windows.txt")
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--max-misses", type=int, default=2)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args(argv)
    token = os.environ.get("METACULUS_TOKEN", "")
    if not token:
        print("METACULUS_TOKEN is not set; export it and rerun.", file=sys.stderr)
        return 2
    client = Client(token, interval=args.interval)
    fetched_at = datetime.now(UTC)
    try:
        rounds, missing = discover_rounds(client, args.max_rounds, args.max_misses)
        questions: list[dict[str, Any]] = []
        listed: dict[str, int] = {}
        for rnd in rounds:
            posts = fetch_round(client, rnd["slug"])
            listed[rnd["slug"]] = len(posts)
            questions.extend(questions_from_posts(posts, rnd["slug"]))
    except FetchError as exc:
        print(f"fetch failed: {exc}. Check the token and network, then rerun.", file=sys.stderr)
        return 1
    starts = {r["slug"]: parse_time(r["start_date"]) for r in rounds}
    notes = [
        "Rounds before the current one are found only by slug: minibench (current), then "
        "minibench-YYYY-MM-DD at 14-day steps back from its start_date, stopping after "
        f"{args.max_misses} consecutive 404s or {args.max_rounds} rounds. A round off that "
        "grid would be missed.",
        "MiniBench projects are unlisted, so /api/projects/tournaments/ does not return them.",
        "Window = scheduled_close_time - open_time from the API. Gaps are between distinct "
        "open instants within one round (not across rounds).",
        "First N days are counted from the round's start_date, or from its first open when a "
        "question opened before start_date.",
    ]
    for rnd in rounds:
        if rnd.get("questions_count") is not None and rnd["questions_count"] != listed[rnd["slug"]]:
            notes.append(
                f"{rnd['slug']}: project says questions_count={rnd['questions_count']}, the "
                f"post listing returned {listed[rnd['slug']]} posts."
            )
    cmd_args = sys.argv[1:] if argv is None else argv
    result: dict[str, Any] = {
        "command": shlex.join(["python3", "scripts/question_windows.py", *cmd_args]),
        "fetched_at": iso(fetched_at),
        "requests": client.requests,
        "rounds_found": [
            {
                "id": r["id"],
                "slug": r["slug"],
                "start_date": r.get("start_date"),
                "forecasting_end_date": r.get("forecasting_end_date"),
                "questions_count": r.get("questions_count"),
                "posts_listed": listed[r["slug"]],
            }
            for r in rounds
        ],
        "rounds_not_found": missing,
        "notes": notes,
        "summary": window_stats(questions, starts),
        "questions": questions,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    text = render(result)
    with open(args.txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
