"""Choose a cron cadence from the measured question windows and cron behaviour.

Standard library only, offline. Reads the outputs of question_windows.py and
cron_delay.py and simulates candidate cadences on the measured open times:

* model "on-time": every tick fires unless dropped (probability = the missed
  share of the complete-coverage workflows), created after a delay resampled
  from their measured delays;
* model "throttled": schedule runs arrive as a renewal process whose gaps are
  resampled from the measured gaps of the throttled short-interval workflows,
  whatever the cron says (that is what the throttled forks showed).

A question counts as caught when the first run created at or after its open
time, plus the forecast-job duration, ends at least --margin minutes before
its scheduled close.

    python3 scripts/cadence.py --txt data/runs/m3-cadence.txt
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import math
import random
import shlex
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

DAY = 1440


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def minutes_since(t: datetime, epoch: datetime) -> float:
    return (t - epoch).total_seconds() / 60


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty list")
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class Cadence:
    label: str
    day_ticks: tuple[int, ...]  # minutes after midnight UTC
    interval: int = 0
    offset: int = 0
    fallback: bool = False

    @classmethod
    def every(cls, interval: int, offset: int, *, fallback: bool = False) -> Cadence:
        if DAY % interval:
            raise ValueError(f"{interval} does not divide a day")
        ticks = {(offset + k * interval) % DAY for k in range(DAY // interval)}
        label = f"every {interval} min at +{offset}"
        if fallback:
            half = interval // 2
            ticks |= {(offset + half + k * interval) % DAY for k in range(DAY // interval)}
            label += f" + fallback {half} min later"
        return cls(label, tuple(sorted(ticks)), interval, offset, fallback)

    def cron_lines(self) -> list[str]:
        """Cron expressions (minute and hour fields only) that fire at these ticks."""
        by_minute: dict[int, list[int]] = {}
        for t in self.day_ticks:
            by_minute.setdefault(t % 60, []).append(t // 60)
        by_hours: dict[tuple[int, ...], list[int]] = {}
        for minute, hrs in sorted(by_minute.items()):
            by_hours.setdefault(tuple(sorted(hrs)), []).append(minute)
        lines = []
        for hours, minutes in by_hours.items():
            h = "*" if len(hours) == 24 else ",".join(map(str, hours))
            lines.append(f"{','.join(map(str, minutes))} {h} * * *")
        return lines

    def ticks(self, start: float, end: float, epoch_minute_of_day: float) -> list[float]:
        """Tick instants (minutes since epoch) in [start, end]."""
        out: list[float] = []
        day0 = math.floor((start + epoch_minute_of_day) / DAY)
        day = day0
        while day * DAY - epoch_minute_of_day <= end:
            base = day * DAY - epoch_minute_of_day
            out.extend(base + t for t in self.day_ticks if start <= base + t <= end)
            day += 1
        return out


@dataclass
class Question:
    open: float  # minutes since epoch
    close: float
    window_h: float


@dataclass
class SimResult:
    caught: float
    caught_by_window: dict[float, float]
    worst_left: float
    p01_left: float
    runs_with_work_per_rep: float
    no_run: float


def _first_at_or_after(runs: list[float], t: float) -> float | None:
    i = bisect.bisect_left(runs, t)
    return runs[i] if i < len(runs) else None


def evaluate(
    questions: list[Question], runs: list[float], job: float, margin: float
) -> tuple[list[float], int]:
    """Time left at submission for each question, and the number of runs that find work.

    A run finds work when some question is open at its creation and was not
    already handled by an earlier run.
    """
    runs = sorted(runs)
    left: list[float] = []
    working: set[float] = set()
    for q in questions:
        r = _first_at_or_after(runs, q.open)
        if r is None or r >= q.close:
            left.append(-math.inf)  # no run starts inside the window
            continue
        left.append(q.close - r - job)
        working.add(r)
    return left, len(working)


def simulate_on_time(
    questions: list[Question],
    cadence: Cadence,
    delays: list[float],
    p_miss: float,
    *,
    epoch_minute_of_day: float,
    job: float,
    margin: float,
    reps: int,
    seed: int,
) -> SimResult:
    rng = random.Random(seed)
    lookback = max(delays) + max(cadence_gaps(cadence)) + 1
    spans = merge_spans([(q.open - lookback, q.close) for q in questions])
    base_ticks = [t for a, b in spans for t in cadence.ticks(a, b, epoch_minute_of_day)]
    lefts: list[list[float]] = []
    work = 0
    for _ in range(reps):
        runs = [t + rng.choice(delays) for t in base_ticks if rng.random() >= p_miss]
        left, n_work = evaluate(questions, runs, job, margin)
        lefts.append(left)
        work += n_work
    return summarise(questions, lefts, margin, work / reps)


def simulate_throttled(
    questions: list[Question],
    gaps: list[float],
    *,
    job: float,
    margin: float,
    reps: int,
    seed: int,
) -> SimResult:
    rng = random.Random(seed)
    start = min(q.open for q in questions) - max(gaps)
    end = max(q.close for q in questions)
    lefts: list[list[float]] = []
    work = 0
    for _ in range(reps):
        t = start + rng.random() * rng.choice(gaps)  # a random phase
        runs: list[float] = []
        while t <= end:
            runs.append(t)
            t += rng.choice(gaps)
        left, n_work = evaluate(questions, runs, job, margin)
        lefts.append(left)
        work += n_work
    return summarise(questions, lefts, margin, work / reps)


def summarise(
    questions: list[Question], lefts: list[list[float]], margin: float, work: float
) -> SimResult:
    flat = [x for rep in lefts for x in rep]
    caught = sum(1 for x in flat if x >= margin) / len(flat)
    by_window: dict[float, float] = {}
    for w in sorted({q.window_h for q in questions}):
        idx = [i for i, q in enumerate(questions) if q.window_h == w]
        vals = [rep[i] for rep in lefts for i in idx]
        by_window[w] = sum(1 for x in vals if x >= margin) / len(vals)
    finite = [x for x in flat if x != -math.inf]
    worst = min(finite) if finite else -math.inf
    p01 = sorted(flat)[int(0.01 * len(flat))]  # nearest rank; -inf when >= 1% get no run
    no_run = (len(flat) - len(finite)) / len(flat)
    return SimResult(caught, by_window, worst, p01, work, no_run)


def cadence_gaps(c: Cadence) -> list[int]:
    d = c.day_ticks
    return [b - a for a, b in itertools.pairwise(d)] + [d[0] + DAY - d[-1]]


def merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def robust_share(
    questions: list[Question],
    cadence: Cadence,
    *,
    epoch_minute_of_day: float,
    job: float,
    margin: float,
    delay: float,
) -> float:
    """Share of questions with at least two ticks that would each catch them in time."""
    ok = 0
    for q in questions:
        ticks = cadence.ticks(q.open, q.close - margin - job - delay, epoch_minute_of_day)
        ok += len(ticks) >= 2
    return ok / len(questions)


def billed_per_month(
    cadence: Cadence, p_miss: float, runs_with_work: float, span_days: float, job: float
) -> tuple[float, float]:
    """(poll minutes, forecast minutes) billed per 30 days on a private repository."""
    polls = len(cadence.day_ticks) * 30 * (1 - p_miss)
    forecast = runs_with_work / span_days * 30 * math.ceil(job)
    return polls, forecast


def load_questions(path: str) -> tuple[list[Question], str, float]:
    """Questions in minutes since the UTC midnight before the first open."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["questions"]
    epoch = min(parse_time(r["open_time"]) for r in rows).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    qs = []
    for r in rows:
        o, c = parse_time(r["open_time"]), parse_time(r["scheduled_close_time"])
        qs.append(
            Question(
                minutes_since(o, epoch),
                minutes_since(c, epoch),
                round((c - o).total_seconds() / 3600, 2),
            )
        )
    span_days = (max(q.close for q in qs) - min(q.open for q in qs)) / DAY
    return qs, str(data["fetched_at"]), span_days


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--windows", default="data/runs/m3-question-windows.json")
    parser.add_argument("--delays", default="data/runs/m3-cron-delay.json")
    parser.add_argument("--txt", default="data/runs/m3-cadence.txt")
    parser.add_argument(
        "--job-minutes",
        type=float,
        default=8.0,
        help="forecast job duration: about 1 min install plus 2.5 min a question (M2)",
    )
    parser.add_argument("--margin", type=float, default=15.0, help="minutes left before close")
    parser.add_argument("--target", type=float, default=0.99)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args(argv)

    questions, windows_fetched, span_days = load_questions(args.windows)
    with open(args.delays, encoding="utf-8") as f:
        cron = json.load(f)
    complete = cron["pooled"]["complete"]
    delays: list[float] = complete["delays_min"]
    p_miss: float = complete["missed_share"] or 0.0
    gaps: list[float] = cron["pooled"]["throttled_short_interval"]["gaps_min"]
    if not delays:
        print("no complete-coverage delays in the cron measurement; rerun it.", file=sys.stderr)
        return 1
    emd = 0.0  # the epoch is a UTC midnight
    d99 = percentile(delays, 99)

    plain: list[Cadence] = [Cadence.every(20, 7)]
    for interval, offsets in (
        (30, (0, 7, 13, 23)),
        (60, (0, 7, 17, 37, 52)),
        (90, (0, 7, 17, 37)),
        (120, (0, 7, 17, 37)),
    ):
        plain += [Cadence.every(interval, o) for o in offsets]
    with_fallback = [
        Cadence.every(c.interval, c.offset, fallback=True) for c in plain if c.interval >= 60
    ]

    rows: list[dict[str, Any]] = []
    for c in plain + with_fallback:
        sim = simulate_on_time(
            questions,
            c,
            delays,
            p_miss,
            epoch_minute_of_day=emd,
            job=args.job_minutes,
            margin=args.margin,
            reps=args.reps,
            seed=args.seed,
        )
        polls, fc = billed_per_month(
            c, p_miss, sim.runs_with_work_per_rep, span_days, args.job_minutes
        )
        rows.append(
            {
                "cadence": c,
                "sim": sim,
                "polls": polls,
                "forecast": fc,
                "total": polls + fc,
                "robust": robust_share(
                    questions,
                    c,
                    epoch_minute_of_day=emd,
                    job=args.job_minutes,
                    margin=args.margin,
                    delay=d99,
                ),
                "robust_by_window": {
                    w: robust_share(
                        [q for q in questions if q.window_h == w],
                        c,
                        epoch_minute_of_day=emd,
                        job=args.job_minutes,
                        margin=args.margin,
                        delay=d99,
                    )
                    for w in sorted({q.window_h for q in questions})
                },
            }
        )
    throttled = (
        simulate_throttled(
            questions,
            gaps,
            job=args.job_minutes,
            margin=args.margin,
            reps=args.reps,
            seed=args.seed,
        )
        if gaps
        else None
    )

    windows = sorted({q.window_h for q in questions})
    head = (
        f"{'cadence':<44} {'ticks/d':>7} {'caught':>7} "
        + " ".join(f"{f'{w:g}h win':>8}" for w in windows)
        + f" {'2 ticks':>7} {'no run':>7} {'p1 left':>8} {'worst':>7}"
        + f" {'poll':>6} {'fcst':>6} {'total':>6}"
    )

    def line(r: dict[str, Any]) -> str:
        s: SimResult = r["sim"]
        return (
            f"{r['cadence'].label:<44} {len(r['cadence'].day_ticks):>7} "
            f"{s.caught * 100:>6.2f}% "
            + " ".join(f"{s.caught_by_window[w] * 100:>7.2f}%" for w in windows)
            + f" {r['robust'] * 100:>6.1f}% {s.no_run * 100:>6.2f}% "
            f"{s.p01_left:>8.1f} {s.worst_left:>7.1f} "
            f"{r['polls']:>6.0f} {r['forecast']:>6.0f} {r['total']:>6.0f}"
        )

    cmd_args = sys.argv[1:] if argv is None else argv
    out = [
        "Cadence simulation against measured MiniBench windows and GitHub cron behaviour",
        f"command: {shlex.join(['python3', 'scripts/cadence.py', *cmd_args])}",
        f"inputs: {args.windows} (fetched {windows_fetched}, "
        f"{len(questions)} questions over {span_days:.1f} days), {args.delays} (fetched "
        f"{cron['fetched_at']})",
        f"parameters: job {args.job_minutes:g} min (billed {math.ceil(args.job_minutes)}), "
        f"margin {args.margin:g} min, target {args.target * 100:g}%, {args.reps} reps, "
        f"seed {args.seed}",
        f"on-time model: delays from {len(delays)} runs of {complete['repos']} "
        f"complete-coverage repo(s) (p50 {percentile(delays, 50):.1f}, p99 {d99:.1f}, max "
        f"{max(delays):.1f} min), dropped-tick probability {p_miss * 100:.2f}%",
        "",
        "on-time model (caught = share of question-replications with >= margin left; "
        "'2 ticks' = share of questions with two ticks that would each catch them at the "
        "p99 delay, i.e. surviving one dropped tick; left in minutes; billed minutes per "
        "30 days on a private repo):",
        head,
    ]
    out += [line(r) for r in rows]
    if throttled:
        tg = cron["pooled"]["throttled_short_interval"]
        out += [
            "",
            f"throttled model (gaps between schedule runs resampled from {len(gaps)} gaps of "
            f"{tg['repos']} repos; p50 {percentile(gaps, 50):.0f}, p90 "
            f"{percentile(gaps, 90):.0f} min; independent of the cron): caught "
            f"{throttled.caught * 100:.2f}% ("
            + ", ".join(
                f"{w:g}h windows {v * 100:.2f}%" for w, v in throttled.caught_by_window.items()
            )
            + f"), p1 left {throttled.p01_left:.1f} min",
        ]

    def recommend(window: float | None) -> list[str]:
        def share(r: dict[str, Any]) -> float:
            sim: SimResult = r["sim"]
            return sim.caught if window is None else sim.caught_by_window[window]

        def rob(r: dict[str, Any]) -> float:
            return float(r["robust"] if window is None else r["robust_by_window"][window])

        scope = "all windows" if window is None else f"{window:g}h windows only"
        ok = [r for r in rows if not r["cadence"].fallback and share(r) >= args.target]
        if not ok:
            return [f"{scope}: no plain cadence reaches {args.target * 100:g}%"]
        best = min(ok, key=lambda r: (r["total"], -share(r)))
        c: Cadence = best["cadence"]
        lines = [
            f"{scope}: cheapest plain cadence with caught >= {args.target * 100:g}% is "
            f"{c.label}: cron {' | '.join(repr(x) for x in c.cron_lines())}; caught "
            f"{share(best) * 100:.2f}%, about {best['total']:.0f} billed minutes per 30 days "
            f"({best['polls']:.0f} poll + {best['forecast']:.0f} forecast)"
        ]
        if rob(best) >= args.target:
            lines.append("  every question has two ticks in time; one dropped tick loses none")
            return lines
        fb = next(
            (
                r
                for r in rows
                if r["cadence"].fallback
                and (r["cadence"].interval, r["cadence"].offset) == (c.interval, c.offset)
            ),
            None,
        )
        lines.append(
            f"  only {rob(best) * 100:.1f}% of questions have a second tick in time, so a "
            "dropped tick loses the others"
            + (
                f"; a second line {c.interval // 2} min later ({fb['cadence'].label}: cron "
                f"{' | '.join(repr(x) for x in fb['cadence'].cron_lines())}) gives caught "
                f"{share(fb) * 100:.2f}%, two ticks {rob(fb) * 100:.1f}%, about "
                f"{fb['total']:.0f} billed minutes"
                if fb
                else ""
            )
        )
        return lines

    out += ["", "recommendation (on-time model):"]
    out += recommend(None)
    out += recommend(max(windows))
    if throttled:
        out.append(
            f"if the schedule is throttled as on most public forks, no cron cadence helps: "
            f"{throttled.caught * 100:.1f}% caught; the fallback then has to be a trigger "
            "outside GitHub's scheduler (a workflow_dispatch from an external cron)."
        )
    text = "\n".join(out) + "\n"
    with open(args.txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
