"""Measure how late GitHub Actions fires scheduled workflows, on public bot forks.

Standard library only, unauthenticated. Lists the forks of
Metaculus/metac-bot-template, keeps the ones pushed recently (GitHub disables a
schedule after 60 days without repository activity), fetches each one's latest
``event=schedule`` runs, reads the cron expression(s) of the workflow file at
each run's commit from raw.githubusercontent.com, and compares each run's
``created_at`` with its nominal tick.

    python3 scripts/cron_delay.py \
        --json data/runs/m3-cron-delay.json --txt data/runs/m3-cron-delay.txt

Every response is cached under --cache (default /tmp/augur-cron-cache), so a
rerun costs no API calls. The unauthenticated core limit is 60 requests an
hour; the script stops spending when the remaining budget reaches --reserve.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shlex
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

API = "https://api." + "github.com"
RAW = "https://raw.githubusercontent.com"
USER_AGENT = "augur-forecast-measure (+https://github.com/TheAsianFish/augur)"
TEMPLATE = "Metaculus/metac-bot-template"

# --- cron -------------------------------------------------------------------


class UnsupportedCron(ValueError):
    """The expression uses a feature this parser does not model."""


def _field(spec: str, lo: int, hi: int) -> frozenset[int]:
    values: set[int] = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            if not step_s.isdigit() or int(step_s) < 1:
                raise UnsupportedCron(f"bad step in {spec!r}")
            step = int(step_s)
        if part == "*":
            a, b = lo, hi
        elif "-" in part:
            a_s, b_s = part.split("-", 1)
            if not (a_s.isdigit() and b_s.isdigit()):
                raise UnsupportedCron(f"bad range in {spec!r}")
            a, b = int(a_s), int(b_s)
        elif part.isdigit():
            a = int(part)
            b = hi if step > 1 else a
        else:
            raise UnsupportedCron(f"bad field {spec!r}")
        if not (lo <= a <= b <= hi):
            raise UnsupportedCron(f"out of range {spec!r}")
        values.update(range(a, b + 1, step))
    return frozenset(values)


@dataclass(frozen=True)
class Cron:
    expr: str
    minutes: frozenset[int]
    hours: frozenset[int]

    @classmethod
    def parse(cls, expr: str) -> Cron:
        fields = expr.split()
        if len(fields) != 5:
            raise UnsupportedCron(f"{expr!r} does not have 5 fields")
        minute, hour, dom, month, dow = fields
        if (dom, month, dow) != ("*", "*", "*"):
            raise UnsupportedCron(f"{expr!r}: day, month and weekday must be '*'")
        return cls(expr, _field(minute, 0, 59), _field(hour, 0, 23))

    def ticks_of_day(self) -> list[int]:
        """Minutes after midnight UTC at which the expression fires."""
        return sorted(h * 60 + m for h in self.hours for m in self.minutes)


class Schedule:
    """The union of a workflow's cron lines (GitHub fires each one)."""

    def __init__(self, crons: list[Cron]) -> None:
        if not crons:
            raise ValueError("a schedule needs at least one cron")
        self.crons = crons
        self.day = sorted({t for c in crons for t in c.ticks_of_day()})

    @property
    def exprs(self) -> list[str]:
        return [c.expr for c in self.crons]

    def intervals(self) -> list[int]:
        """Minutes between consecutive ticks, wrapping across midnight."""
        d = self.day
        return [b - a for a, b in itertools.pairwise(d)] + [d[0] + 1440 - d[-1]]

    def min_interval(self) -> int:
        return min(self.intervals())

    def tick_at_or_before(self, t: datetime) -> datetime:
        midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
        minute = (t - midnight).total_seconds() / 60
        earlier = [m for m in self.day if m <= minute]
        if earlier:
            return midnight + timedelta(minutes=earlier[-1])
        return midnight - timedelta(days=1) + timedelta(minutes=self.day[-1])

    def ticks_between(self, start: datetime, end: datetime) -> list[datetime]:
        """All ticks with start <= tick <= end."""
        out: list[datetime] = []
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= end:
            for m in self.day:
                tick = day + timedelta(minutes=m)
                if start <= tick <= end:
                    out.append(tick)
            day += timedelta(days=1)
        return out


_CRON_LINE = re.compile(r"""^\s*-?\s*cron:\s*(?:'([^']*)'|"([^"]*)"|([^\s#'"][^#]*))""")


def parse_workflow_crons(text: str) -> list[str]:
    """The cron expressions in a workflow file (commented-out lines ignored)."""
    out: list[str] = []
    for line in text.splitlines():
        m = _CRON_LINE.match(line)
        if m:
            expr = next(g for g in m.groups() if g is not None)
            out.append(" ".join(expr.split()))
    return out


# --- statistics ---------------------------------------------------------------


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty list")
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def dist(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def analyse_runs(
    schedule: Schedule, runs: list[tuple[datetime, int]], long_gap: int = 6
) -> dict[str, Any]:
    """Delays, coverage and missed ticks for one workflow's schedule runs.

    ``runs`` holds (created_at, run_number). Each run is assigned the latest
    tick at or before its created_at, so a delay longer than the interval is
    attributed to a later tick (and the earlier tick counts as missed).
    Expected ticks run from the first assigned tick to the last; a streak of
    ``long_gap`` or more consecutive missed ticks is counted separately.
    Run numbers count every run of the workflow whatever its trigger, so a
    jump between consecutive schedule runs counts runs from other events.
    """
    runs = sorted(runs)
    created = [c for c, _ in runs]
    assigned = [schedule.tick_at_or_before(c) for c in created]
    delays = [(c - t).total_seconds() / 60 for c, t in zip(created, assigned, strict=True)]
    hit = set(assigned)
    expected = schedule.ticks_between(assigned[0], assigned[-1]) if assigned else []
    missed_flags = [t not in hit for t in expected]
    missed = sum(missed_flags)
    in_long = 0
    longest = 0
    for is_missed, group in itertools.groupby(missed_flags):
        n = len(list(group))
        if is_missed:
            longest = max(longest, n)
            if n >= long_gap:
                in_long += n
    gaps = [(b - a).total_seconds() / 60 for a, b in itertools.pairwise(created)]
    numbers = sorted(n for _, n in runs)
    other = sum(max(0, b - a - 1) for a, b in itertools.pairwise(numbers))
    return {
        "n_runs": len(created),
        "first_run": iso(created[0]) if created else None,
        "last_run": iso(created[-1]) if created else None,
        "span_hours": (created[-1] - created[0]).total_seconds() / 3600 if created else 0.0,
        "expected_ticks": len(expected),
        "ticks_with_a_run": len(hit),
        "coverage": len(hit) / len(expected) if expected else None,
        "missed_ticks": missed,
        "missed_in_long_gaps": in_long,
        "missed_isolated": missed - in_long,
        "longest_missed_streak": longest,
        "runs_sharing_a_tick": len(assigned) - len(hit),
        "other_event_runs_between": other,
        "delays_min": delays,
        "gaps_min": gaps,
        "tick_hours": [t.hour for t in assigned],
    }


# --- fetching -------------------------------------------------------------------


class BudgetExhausted(Exception):
    pass


class Fetcher:
    def __init__(self, cache: str, reserve: int) -> None:
        self.cache = cache
        self.reserve = reserve
        self.api_calls = 0
        self.cache_hits = 0
        self.raw_calls = 0
        os.makedirs(cache, exist_ok=True)
        self.remaining: int | None = None

    def _path(self, url: str) -> str:
        return os.path.join(self.cache, hashlib.sha256(url.encode()).hexdigest()[:32] + ".json")

    def _open(self, url: str) -> tuple[int, bytes, dict[str, str]]:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 (fixed https hosts)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https hosts)
                return resp.status, resp.read(), dict(resp.headers.items())
        except urllib.error.HTTPError as exc:
            return exc.code, b"", dict(exc.headers.items()) if exc.headers else {}

    def rate_limit(self) -> dict[str, Any]:
        status, body, _ = self._open(f"{API}/rate_limit")
        if status != 200:
            raise RuntimeError(f"/rate_limit returned {status}")
        core: dict[str, Any] = json.loads(body)["resources"]["core"]
        self.remaining = int(core["remaining"])
        return core

    def api(self, path: str) -> Any:
        url = f"{API}{path}"
        cached = self._path(url)
        if os.path.exists(cached):
            self.cache_hits += 1
            with open(cached, encoding="utf-8") as f:
                return json.load(f)["body"]
        if self.remaining is None:
            self.rate_limit()
        assert self.remaining is not None
        if self.remaining <= self.reserve:
            raise BudgetExhausted(f"{self.remaining} API calls left (reserve {self.reserve})")
        status, body, headers = self._open(url)
        self.api_calls += 1
        if "X-RateLimit-Remaining" in headers:
            self.remaining = int(headers["X-RateLimit-Remaining"])
        if status in (403, 429):
            raise BudgetExhausted(f"GET {path}: HTTP {status}")
        data = json.loads(body) if status == 200 else None
        if status in (200, 404, 451):
            with open(cached, "w", encoding="utf-8") as f:
                json.dump(
                    {"url": url, "status": status, "fetched": iso(datetime.now(UTC)), "body": data},
                    f,
                )
        return data

    def raw(self, owner_repo: str, ref: str, path: str) -> str | None:
        url = f"{RAW}/{owner_repo}/{ref}/{path}"
        cached = self._path(url)
        if os.path.exists(cached):
            with open(cached, encoding="utf-8") as f:
                body: str | None = json.load(f)["body"]
                return body
        status, data, _ = self._open(url)
        self.raw_calls += 1
        text = data.decode("utf-8", "replace") if status == 200 else None
        if status in (200, 404):
            with open(cached, "w", encoding="utf-8") as f:
                json.dump({"url": url, "status": status, "body": text}, f)
        return text


# --- report ---------------------------------------------------------------------


def _fmt(d: dict[str, float] | None, unit: str = "min") -> str:
    if d is None:
        return "n/a"
    return (
        f"n={d['n']:.0f} p50 {d['p50']:.1f} p90 {d['p90']:.1f} p99 {d['p99']:.1f} "
        f"max {d['max']:.1f} {unit}"
    )


def interval_class(minutes: int) -> str:
    if minutes <= 20:
        return "<=20 min"
    if minutes <= 60:
        return "30-60 min"
    if minutes < 720:
        return "1-12 h"
    return ">=12 h"


def pool(
    workflows: list[dict[str, Any]],
    *,
    unambiguous_min: int,
    complete_coverage: float,
    complete_min_runs: int,
) -> dict[str, Any]:
    """Pooled statistics over analysed workflows (rows carry the private _ lists)."""
    unamb = [w for w in workflows if w["min_interval_min"] >= unambiguous_min]
    complete = [
        w
        for w in workflows
        if w["min_interval_min"] < unambiguous_min
        and (w["coverage"] or 0) >= complete_coverage
        and w["n_runs"] >= complete_min_runs
        and w["runs_sharing_a_tick"] == 0
    ]
    trusted = unamb + complete
    by_hour: dict[int, list[float]] = defaultdict(list)
    for w in trusted:
        for h, d in zip(w["_hours"], w["_delays"], strict=True):
            by_hour[h].append(d)
    classes: dict[str, Any] = {}
    for name in ("<=20 min", "30-60 min", "1-12 h", ">=12 h"):
        ws = [w for w in workflows if interval_class(w["min_interval_min"]) == name]
        exp = sum(w["expected_ticks"] for w in ws)
        hit = sum(w["ticks_with_a_run"] for w in ws)
        classes[name] = {
            "workflows": len(ws),
            "repos": len({w["repo"] for w in ws}),
            "runs": sum(w["n_runs"] for w in ws),
            "expected_ticks": exp,
            "ticks_with_a_run": hit,
            "coverage": hit / exp if exp else None,
            "workflows_complete": sum(1 for w in ws if (w["coverage"] or 0) >= complete_coverage),
            "gap_between_runs_min": dist([g for w in ws for g in w["_gaps"]]),
        }
    short = [
        w
        for w in workflows
        if w["min_interval_min"] <= 60 and (w["coverage"] or 0) < complete_coverage
    ]
    c_exp = sum(w["expected_ticks"] for w in complete)
    c_hit = sum(w["ticks_with_a_run"] for w in complete)
    return {
        "unambiguous": {
            "rule": f"shortest interval >= {unambiguous_min} min",
            "workflows": len(unamb),
            "repos": len({w["repo"] for w in unamb}),
            "delay_min": dist([d for w in unamb for d in w["_delays"]]),
            "delays_min": sorted(d for w in unamb for d in w["_delays"]),
        },
        "complete": {
            "rule": (
                f"shortest interval < {unambiguous_min} min, coverage >= {complete_coverage}, "
                f">= {complete_min_runs} runs, no two runs on one tick"
            ),
            "workflows": [f"{w['repo']}:{w['workflow']}" for w in complete],
            "repos": len({w["repo"] for w in complete}),
            "runs": sum(w["n_runs"] for w in complete),
            "delay_min": dist([d for w in complete for d in w["_delays"]]),
            "delays_min": sorted(d for w in complete for d in w["_delays"]),
            "expected_ticks": c_exp,
            "missed_ticks": c_exp - c_hit,
            "missed_share": (c_exp - c_hit) / c_exp if c_exp else None,
        },
        "throttled_short_interval": {
            "rule": f"shortest interval <= 60 min and coverage < {complete_coverage}",
            "workflows": len(short),
            "repos": len({w["repo"] for w in short}),
            "runs": sum(w["n_runs"] for w in short),
            "gap_between_runs_min": dist([g for w in short for g in w["_gaps"]]),
            "gaps_min": sorted(g for w in short for g in w["_gaps"]),
            "other_event_runs_between": sum(w["other_event_runs_between"] for w in short),
        },
        "by_interval": classes,
        "trusted_delay_by_tick_hour": {str(h): dist(v) for h, v in sorted(by_hour.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", default="data/runs/m3-cron-delay.json")
    parser.add_argument("--txt", default="data/runs/m3-cron-delay.txt")
    parser.add_argument("--cache", default="/tmp/augur-cron-cache")  # noqa: S108 (a cache, not a secret)
    parser.add_argument("--reserve", type=int, default=3)
    parser.add_argument("--fork-pages", type=int, default=5)
    parser.add_argument("--pushed-within-days", type=int, default=60)
    parser.add_argument(
        "--recent-days", type=int, default=14, help="keep repos with a schedule run this recent"
    )
    parser.add_argument("--max-repos", type=int, default=200)
    parser.add_argument(
        "--unambiguous-min",
        type=int,
        default=720,
        help="crons at least this many minutes apart give unambiguous delays",
    )
    parser.add_argument("--complete-coverage", type=float, default=0.95)
    parser.add_argument("--complete-min-runs", type=int, default=20)
    parser.add_argument(
        "--now", default=None, help="ISO time to treat as now (default: fetch time)"
    )
    args = parser.parse_args(argv)

    fx = Fetcher(args.cache, args.reserve)
    fetched_at = datetime.now(UTC)
    now = parse_time(args.now) if args.now else fetched_at
    core_before = fx.rate_limit()
    notes: list[str] = []

    forks: list[dict[str, Any]] = []
    try:
        for page in range(1, args.fork_pages + 1):
            batch = fx.api(f"/repos/{TEMPLATE}/forks?sort=newest&per_page=100&page={page}")
            if not batch:
                break
            forks.extend(batch)
            if len(batch) < 100:
                break
    except BudgetExhausted as exc:
        notes.append(f"fork listing stopped early: {exc}")
    cutoff = now - timedelta(days=args.pushed_within_days)
    candidates = sorted(
        (f for f in forks if f.get("pushed_at") and parse_time(f["pushed_at"]) >= cutoff),
        key=lambda f: f["pushed_at"],
        reverse=True,
    )[: args.max_repos]

    workflows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for fork in candidates:
        name = fork["full_name"]
        try:
            runs_page = fx.api(f"/repos/{name}/actions/runs?event=schedule&per_page=100")
        except BudgetExhausted as exc:
            skipped[name] = f"not fetched: {exc}"
            continue
        runs = (runs_page or {}).get("workflow_runs", [])
        if not runs:
            skipped[name] = "no schedule runs"
            continue
        last = max(parse_time(r["created_at"]) for r in runs)
        if last < now - timedelta(days=args.recent_days):
            skipped[name] = f"latest schedule run is older than {args.recent_days} days"
            continue
        by_wf: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in runs:
            by_wf[r.get("path") or str(r.get("workflow_id"))].append(r)
        for path, wf_runs in sorted(by_wf.items()):
            # Group by the cron lines in force at each run's commit.
            groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
            for r in wf_runs:
                text = fx.raw(name, r["head_sha"], path)
                exprs = tuple(parse_workflow_crons(text)) if text is not None else ()
                if not exprs:
                    skipped[f"{name}:{path}@{r['head_sha'][:7]}"] = (
                        "workflow file unreadable or without a cron at the run's commit"
                    )
                    continue
                groups[exprs].append(r)
            for exprs, g_runs in groups.items():
                try:
                    schedule = Schedule([Cron.parse(e) for e in exprs])
                except UnsupportedCron as exc:
                    skipped[f"{name}:{path}:{' | '.join(exprs)}"] = f"unsupported cron: {exc}"
                    continue
                stats = analyse_runs(
                    schedule,
                    [(parse_time(r["created_at"]), int(r["run_number"])) for r in g_runs],
                )
                queue = [
                    (parse_time(r["run_started_at"]) - parse_time(r["created_at"])).total_seconds()
                    / 60
                    for r in g_runs
                    if r.get("run_started_at") and r.get("run_attempt", 1) == 1
                ]
                interval = schedule.min_interval()
                last_g = parse_time(stats["last_run"])
                quiet = now - last_g
                if quiet > max(timedelta(hours=24), timedelta(minutes=2 * interval + 720)):
                    status = "stopped"
                elif (stats["coverage"] or 0) >= args.complete_coverage:
                    status = "complete"
                else:
                    status = "skips"
                workflows.append(
                    {
                        "repo": name,
                        "workflow": path,
                        "cron": list(exprs),
                        "min_interval_min": interval,
                        "ticks_per_day": len(schedule.day),
                        "status": status,
                        "schedule_runs_in_repo": (runs_page or {}).get("total_count"),
                        **{
                            k: v
                            for k, v in stats.items()
                            if k not in ("delays_min", "gaps_min", "tick_hours")
                        },
                        "delay_min": dist(stats["delays_min"]),
                        "gap_between_runs_min": dist(stats["gaps_min"]),
                        "queue_min": dist(queue),
                        "_delays": stats["delays_min"],
                        "_gaps": stats["gaps_min"],
                        "_hours": stats["tick_hours"],
                    }
                )
    pooled = pool(
        workflows,
        unambiguous_min=args.unambiguous_min,
        complete_coverage=args.complete_coverage,
        complete_min_runs=args.complete_min_runs,
    )
    core_after = fx.rate_limit()
    cmd_args = sys.argv[1:] if argv is None else argv
    notes += [
        "delay = run created_at - nominal tick; the tick is the latest one at or before "
        "created_at under the cron lines in the workflow file at the run's head_sha. A delay "
        "longer than the interval would be attributed to a later tick, so delays are only "
        "reported as trustworthy for crons at least --unambiguous-min apart, and for "
        "workflows where (nearly) every tick has exactly one run.",
        "coverage = ticks with a run / ticks between the first and last observed run. Missed "
        "ticks are counted inside that span only.",
        "Only the latest 100 schedule runs per repository are fetched (one page, shared by "
        "all its scheduled workflows), so a workflow's span is whatever those runs cover.",
        "other_event_runs_between: run numbers count every run of a workflow, so jumps "
        "between consecutive schedule runs are runs from other triggers (workflow_dispatch, "
        "push, ...), e.g. an external pinger that dispatches the workflow.",
        "queue = run_started_at - created_at for first attempts.",
        "status: stopped = no schedule run for max(24 h, 2 intervals + 12 h) before now; "
        f"complete = coverage >= {args.complete_coverage}; skips = otherwise.",
        "Sample: public forks of the template pushed in the last "
        f"{args.pushed_within_days} days (GitHub disables schedules after 60 days without "
        "activity), newest push first, as far as the unauthenticated budget reached. Private "
        "repositories and accounts on paid plans are not represented.",
    ]
    result: dict[str, Any] = {
        "command": shlex.join(["python3", "scripts/cron_delay.py", *cmd_args]),
        "fetched_at": iso(fetched_at),
        "now": iso(now),
        "api": {
            "calls_made": fx.api_calls,
            "cache_hits": fx.cache_hits,
            "raw_fetches": fx.raw_calls,
            "core_remaining_before": core_before["remaining"],
            "core_remaining_after": core_after["remaining"],
        },
        "forks_listed": len(forks),
        "candidates": len(candidates),
        "repos_fetched": len(
            {k.split(":")[0] for k in skipped if "not fetched" not in skipped[k]}
            | {w["repo"] for w in workflows}
        ),
        "repos_analysed": len({w["repo"] for w in workflows}),
        "runs_analysed": sum(w["n_runs"] for w in workflows),
        "pooled": pooled,
        "workflows": [{k: v for k, v in w.items() if not k.startswith("_")} for w in workflows],
        "skipped": skipped,
        "notes": notes,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    text = render(result)
    with open(args.txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text, end="")
    return 0


def render(result: dict[str, Any]) -> str:
    p = result["pooled"]
    a = result["api"]
    u, c, t = p["unambiguous"], p["complete"], p["throttled_short_interval"]
    lines = [
        "GitHub Actions scheduled-trigger delay on public forks of " + TEMPLATE,
        f"command: {result['command']}",
        f"fetched: {result['fetched_at']} (API calls {a['calls_made']}, cache hits "
        f"{a['cache_hits']}, raw fetches {a['raw_fetches']}; core remaining "
        f"{a['core_remaining_before']} -> {a['core_remaining_after']})",
        f"forks listed {result['forks_listed']}; recently pushed candidates "
        f"{result['candidates']}; repos fetched {result['repos_fetched']}; repos with recent "
        f"schedule runs analysed {result['repos_analysed']} "
        f"({len(result['workflows'])} workflow/cron groups, {result['runs_analysed']} runs)",
        "",
        f"delay, unambiguous ({u['rule']}; {u['repos']} repos, {u['workflows']} workflows): "
        + _fmt(u["delay_min"]),
        f"delay, complete-coverage workflows ({c['repos']} repos, {c['runs']} runs): "
        + _fmt(c["delay_min"]),
        f"  missed ticks there: {c['missed_ticks']} of {c['expected_ticks']}"
        + (f" ({c['missed_share'] * 100:.2f}%)" if c["missed_share"] is not None else ""),
        f"  workflows: {', '.join(c['workflows']) or 'none'}",
        f"throttled short-interval workflows ({t['rule']}; {t['repos']} repos, "
        f"{t['workflows']} workflows, {t['runs']} runs):",
        f"  time between consecutive schedule runs: {_fmt(t['gap_between_runs_min'])}",
        f"  runs from other triggers between their schedule runs: {t['other_event_runs_between']}",
        "",
        "coverage by cron interval (ticks with a run / expected):",
    ]
    for name, k in p["by_interval"].items():
        cov = f"{k['coverage'] * 100:.1f}%" if k["coverage"] is not None else "n/a"
        lines.append(
            f"  {name:>9}: {k['repos']} repos, {k['workflows']} workflows, {k['runs']} runs, "
            f"{k['ticks_with_a_run']}/{k['expected_ticks']} = {cov}; complete "
            f"{k['workflows_complete']}/{k['workflows']}; gap between runs "
            f"{_fmt(k['gap_between_runs_min'])}"
        )
    lines += ["", "trusted delay (unambiguous + complete) by UTC hour of the tick:"]
    for h, d in p["trusted_delay_by_tick_hour"].items():
        if d:
            lines.append(
                f"  {int(h):02d}: n={d['n']:.0f} p50 {d['p50']:.1f} p90 {d['p90']:.1f} "
                f"max {d['max']:.1f}"
            )
    lines += ["", "per workflow:"]
    for w in result["workflows"]:
        q = w["queue_min"]
        cov = f"{w['coverage'] * 100:.0f}%" if w["coverage"] is not None else "n/a"
        lines += [
            f"  {w['repo']} {w['workflow']} cron {' | '.join(w['cron'])} "
            f"(every {w['min_interval_min']} min) [{w['status']}]",
            f"    runs {w['n_runs']} over {w['span_hours']:.1f}h ({w['first_run']} .. "
            f"{w['last_run']}); coverage {w['ticks_with_a_run']}/{w['expected_ticks']} = {cov}; "
            f"longest missed streak {w['longest_missed_streak']}; other-trigger runs between "
            f"{w['other_event_runs_between']}",
            f"    delay {_fmt(w['delay_min'])}; gap between runs "
            f"{_fmt(w['gap_between_runs_min'])}; queue p50 "
            f"{q['p50'] if q else float('nan'):.1f} min",
        ]
    lines += ["", "skipped:"]
    reasons = Counter(result["skipped"].values())
    lines += [f"  {n} x {reason}" for reason, n in reasons.most_common()]
    lines += [""] + [f"note: {n}" for n in result["notes"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
