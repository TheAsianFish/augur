"""``augur floor --pooled``: M14's pooled Q1 analysis over several rounds.

The analysis is PLAN.md's ``m14-registration`` block, registered before the
2026-11-30 round opened; this module computes it as written:

- each round's questions are the posts in its standings snapshot, and each
  round's coverage (forecast by each side, scored, annulled, uncovered,
  unresolved) is counted exactly as :func:`augur.floor.decide` counts it;
- the primary quantity is the mean head-to-head of Augur's aggregate against
  the shadow template over every pooled question; the registered blend
  against the template is secondary;
- the interval is a 90% percentile bootstrap, 10,000 resamples, seed
  20261119, resampling *opening days* (the UTC calendar day of each
  question's ``open_time``); below eight clusters the question-level
  interval is printed beside it, labelled the optimistic one;
- the n a lead of +2 or +5 points needs is printed at the assumed spread
  of 32 and at the measured one;
- only a reading on :data:`ANALYSIS_DATE` is the registered result; one before
  it is a progress reading and one after it a late reading, each labelled so.

It reads the floor rule's counts but never its verdict: nothing it returns
or prints can say whether a route may submit. Only ``augur floor --round R``
applies the floor rule, one round at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date
from pathlib import Path
from typing import Any

from augur import floor, scorecard
from augur.clock import Clock, SystemClock, to_iso
from augur.errors import AugurError
from augur.score import Estimate, clustered_bootstrap
from augur.standings import RoundStandings, load_snapshot

LEVEL = floor.LEVEL
RESAMPLES = floor.RESAMPLES
SEED = floor.SEED
MIN_CLUSTERS = 8
"""Below this many opening days the question-level interval is printed as well."""
ANALYSIS_DATE = date(2027, 1, 11)
"""Item 5: the one day (UTC) the registered result is read."""
ASSUMED_SPREAD = 32.0
LEADS: tuple[float, ...] = (2.0, 5.0)
POWER = 0.80
ONE_SIDED = 0.05
_Z = statistics.NormalDist()

LABEL = (
    "an unsubmitted shadow comparison on a below-floor model: nothing was posted, the "
    "route is below the submit floor, and this report decides nothing about submission "
    "(only augur floor --round R reads the floor rule, one round at a time)"
)
PROGRESS = "progress"
REGISTERED = "registered"
LATE = "late"
REGISTERED_ROUNDS = ("fall-futureeval-2026",)
"""Item 1, besides MiniBench rounds (``minibench`` and its dated ``minibench-*`` slugs)."""
ABOVE = "above 0"
BELOW = "below 0"
UNRESOLVED = "not resolvable this season"


def needed(spread: float, lead: float) -> int:
    """Item 7: questions for a true ``lead`` at 80% power, read by a one-sided 5% lower end.

    n = ceil(((z_0.95 + z_0.80) * spread / lead) ** 2), questions independent.
    """
    if spread <= 0 or lead <= 0:
        raise ValueError("spread and lead must be positive")
    z = _Z.inv_cdf(1.0 - ONE_SIDED) + _Z.inv_cdf(POWER)
    return math.ceil((z * spread / lead) ** 2)


def opening_day(question: scorecard.ScoredQuestion) -> str:
    """Item 4's cluster: the UTC calendar day the question opened."""
    return question.opened.astimezone(UTC).date().isoformat()


@dataclass(frozen=True)
class RoundCoverage:
    round: str
    fetched_at: str
    finalized: bool
    posts: int
    augur: int
    template: int
    both: int
    scored: int
    annulled: int
    uncovered: int
    unresolved: int
    blend_scored: int

    def line(self) -> str:
        return (
            f"{self.round} (snapshot {self.fetched_at}, "
            f"{'finalized' if self.finalized else 'not finalized'}): {self.posts} posts; "
            f"forecast by Augur {self.augur}, by the template {self.template}, by both "
            f"{self.both}; scored {self.scored}; annulled or ambiguous {self.annulled}; "
            f"uncovered {self.uncovered}; unresolved {self.unresolved}; "
            f"blend scored {self.blend_scored}"
        )


@dataclass(frozen=True)
class Interval:
    """A pooled mean with its bootstrap interval and how it was clustered."""

    n: int
    clusters: int
    mean: float
    low: float | None
    high: float | None

    @classmethod
    def of(cls, values: Sequence[float], labels: Sequence[Hashable]) -> Interval:
        est: Estimate = clustered_bootstrap(
            values, labels, level=LEVEL, resamples=RESAMPLES, seed=SEED
        )
        if est.clusters < 2:
            return cls(est.n, est.clusters, est.mean, None, None)
        return cls(est.n, est.clusters, est.mean, est.low, est.high)

    def text(self) -> str:
        if self.low is None or self.high is None:
            return f"{self.mean:.2f} (no interval: {self.clusters} cluster)"
        return f"{self.mean:.2f} [{self.low:.2f}, {self.high:.2f}]"


@dataclass(frozen=True)
class Arm:
    """One contestant against the template, pooled."""

    name: str
    by_day: Interval | None
    by_question: Interval | None
    """Printed only below :data:`MIN_CLUSTERS` opening days."""
    spread: float | None


def when(day: date) -> str:
    """Item 5: only a reading on the analysis date itself is the registered result."""
    if day < ANALYSIS_DATE:
        return PROGRESS
    return REGISTERED if day == ANALYSIS_DATE else LATE


def registered_round(slug: str) -> bool:
    """Item 1's rounds: MiniBench and the Fall 2026 FutureEval project."""
    return slug == "minibench" or slug.startswith("minibench-") or slug in REGISTERED_ROUNDS


@dataclass(frozen=True)
class PooledReport:
    analysed_at: str
    reading_time: str
    """``progress`` before the analysis date, ``registered`` on it, ``late`` after it."""
    rounds: tuple[RoundCoverage, ...]
    duplicates: int
    primary: Arm
    secondary: Arm
    reading: str
    needed_assumed: tuple[int, ...]
    needed_measured: tuple[int, ...] | None

    def lines(self) -> list[str]:
        day, registered = self.analysed_at[:10], ANALYSIS_DATE.isoformat()
        heading = {
            PROGRESS: f"progress reading on {day}, before the registered analysis date "
            f"{registered}: not the registered result",
            REGISTERED: f"registered analysis, read {day} (analysis date {registered})",
            LATE: f"late reading on {day}, after the registered analysis date {registered}: not "
            "the registered result, which is the reading made on that date",
        }[self.reading_time]
        out = [f"pooled Q1 analysis (PLAN.md, m14-registration): {heading}", f"label: {LABEL}"]
        out.append("rounds:")
        out.extend(f"  {r.line()}" for r in self.rounds)
        out.extend(
            f"  note: {r.round} is not one of the registered rounds (registration item 1)"
            for r in self.rounds
            if not registered_round(r.round)
        )
        if self.duplicates:
            out.append(f"  {self.duplicates} posts appear in more than one snapshot; counted once")
        for arm, role in ((self.primary, "primary"), (self.secondary, "secondary")):
            out.extend(_arm_lines(arm, role))
        out.append(f"reading (primary, day-clustered interval): {self.reading}")
        assumed = ", ".join(
            f"+{lead:g} needs {n:,}" for lead, n in zip(LEADS, self.needed_assumed, strict=True)
        )
        out.append(
            f"questions needed at 80% power, one-sided 5%, questions independent: at the "
            f"assumed spread {ASSUMED_SPREAD:g}: {assumed}"
        )
        if self.needed_measured is not None and self.primary.spread is not None:
            measured = ", ".join(
                f"+{lead:g} needs {n:,}"
                for lead, n in zip(LEADS, self.needed_measured, strict=True)
            )
            out.append(f"  at the measured spread {self.primary.spread:.2f}: {measured}")
        else:
            out.append("  at the measured spread: n/a (fewer than two scored questions)")
        return out

    def json_text(self) -> str:
        """Byte-stable JSON: sorted keys, indent 1, one trailing newline."""
        body: dict[str, Any] = asdict(self)
        body["label"] = LABEL
        body["registration"] = {
            "analysis_date": ANALYSIS_DATE.isoformat(),
            "level": LEVEL,
            "resamples": RESAMPLES,
            "seed": SEED,
            "min_clusters": MIN_CLUSTERS,
            "assumed_spread": ASSUMED_SPREAD,
            "leads": list(LEADS),
        }
        return json.dumps(body, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def _arm_lines(arm: Arm, role: str) -> list[str]:
    head = f"{role}: {arm.name} against the shadow template"
    if arm.by_day is None:
        return [f"{head}: n = 0, nothing scored yet"]
    out = [
        f"{head}: mean head-to-head {arm.by_day.text()} ({LEVEL:.0%} bootstrap over opening "
        f"days, {RESAMPLES:,} resamples, seed {SEED}); n = {arm.by_day.n}, "
        f"clusters (opening days) = {arm.by_day.clusters}, measured spread (SD of h) "
        + ("n/a" if arm.spread is None else f"{arm.spread:.2f}")
    ]
    if arm.by_question is not None:
        out.append(
            f"  fewer than {MIN_CLUSTERS} clusters, so also the question-level interval (the "
            f"optimistic one): {arm.by_question.text()}"
        )
    return out


def _arm(name: str, items: Sequence[tuple[str, int, float]]) -> Arm:
    """``items`` are (opening day, question id, h); sorted first, so the seeded resamples, and
    so the interval, do not depend on the order the rounds were given in."""
    if not items:
        return Arm(name, None, None, None)
    ordered = sorted(items)
    values = [h for _, _, h in ordered]
    days = [d for d, _, _ in ordered]
    by_day = Interval.of(values, days)
    by_question = (
        Interval.of(values, list(range(len(values)))) if by_day.clusters < MIN_CLUSTERS else None
    )
    spread = statistics.stdev(values) if len(values) >= 2 else None
    return Arm(name, by_day, by_question, spread)


def reading(arm: Arm) -> str:
    """Item 6, on the primary arm's day-clustered interval."""
    if arm.by_day is None or arm.by_day.low is None or arm.by_day.high is None:
        return UNRESOLVED
    if arm.by_day.low > 0:
        return ABOVE
    if arm.by_day.high < 0:
        return BELOW
    return UNRESOLVED


def pool(
    snapshots: Sequence[RoundStandings],
    ledger_paths: Sequence[Path],
    *,
    clock: Clock,
) -> PooledReport:
    """Pool every round's scored questions, cluster them by opening day, report."""
    if not snapshots:
        raise AugurError("give at least one round's standings snapshot")
    opened = {s.question_id: opening_day(s) for s in scorecard.load(ledger_paths).scored}
    seen: dict[str, set[int]] = {floor.AUGUR: set(), floor.BLEND: set()}
    duplicates = 0
    primary: list[tuple[str, int, float]] = []
    secondary: list[tuple[str, int, float]] = []
    coverage: list[RoundCoverage] = []
    for snap in snapshots:
        own = floor.decide(snap, ledger_paths, floor.AUGUR)
        blend = floor.decide(snap, ledger_paths, floor.BLEND)
        for got, into in ((own, primary), (blend, secondary)):
            held = seen[got.candidate]
            for s in got.scored:
                if s.question_id in held:
                    # A post in two snapshots is pooled once, under the first round given.
                    if got is own:
                        duplicates += 1
                    continue
                held.add(s.question_id)
                into.append((opened[s.question_id], s.question_id, s.h))
        coverage.append(
            RoundCoverage(
                round=own.round,
                fetched_at=own.fetched_at,
                finalized=own.finalized,
                posts=own.round_posts,
                augur=own.augur,
                template=own.template,
                both=own.both,
                scored=own.n,
                annulled=own.annulled,
                uncovered=own.uncovered,
                unresolved=own.unresolved,
                blend_scored=blend.n,
            )
        )
    first = _arm("Augur's aggregate", primary)
    second = _arm("the registered blend (M12's primary candidate)", secondary)
    now = clock.now()
    return PooledReport(
        analysed_at=to_iso(now),
        reading_time=when(now.astimezone(UTC).date()),
        rounds=tuple(coverage),
        duplicates=duplicates,
        primary=first,
        secondary=second,
        reading=reading(first),
        needed_assumed=tuple(needed(ASSUMED_SPREAD, lead) for lead in LEADS),
        needed_measured=(
            None
            if first.spread is None or first.spread <= 0
            else tuple(needed(first.spread, lead) for lead in LEADS)
        ),
    )


def main(args: argparse.Namespace, clock: Clock | None = None) -> int:
    """``augur floor --pooled --round R ... --ledger L``: print the pooled report."""
    rounds: list[str] = args.round or []
    standings: list[Path] = args.standings or []
    if standings and len(standings) != len(rounds):
        raise AugurError(
            "with --pooled, give one --standings per --round, in the same order, or none "
            "(each round's snapshot then defaults to data/standings/SLUG.json)"
        )
    paths = standings or [Path("data/standings") / f"{slug}.json" for slug in rounds]
    snapshots: list[RoundStandings] = []
    for slug, path in zip(rounds, paths, strict=True):
        snap = load_snapshot(path)
        if snap.round.slug != slug:
            raise AugurError(
                f"{path} is the snapshot of {snap.round.slug}, not {slug}; give each round's "
                "own snapshot with --standings"
            )
        snapshots.append(snap)
    report = pool(snapshots, args.ledger, clock=clock or SystemClock())
    for line in report.lines():
        print(line)
    if args.json is not None:
        args.json.write_text(report.json_text(), encoding="utf-8")
    return 0
