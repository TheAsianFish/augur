"""``augur floor``: PLAN.md's floor rule for the free model, applied to one round.

The rule (registered 2026-11-24, before the 2026-11-30 MiniBench round) is
PLAN.md's block between the ``floor-rule`` markers; this module applies it as
written and decides nothing by hand:

- the round's questions are the posts in its standings snapshot;
- a question is scored when the ledgers hold Augur's aggregate and a template
  forecast for it and its latest resolution is ``resolved`` (annulled and
  ambiguous questions are counted, and so are questions only one side
  forecast); the forecasts and resolutions are read as ``augur score`` reads
  them (:func:`augur.scorecard.load`, every run mode but ``unrestricted``, so
  the shadow mode's dry runs count);
- per question h = 100 ln(p_A / p_T), halved on numeric and discrete
  questions; a 90% percentile bootstrap of mean(h), 10,000 resamples, seed
  20261119;
- the implied peer score is S + mean(h), S the average spot peer score of
  ``metac-nemotron-3-ultra+asknews`` on the snapshot's leaderboard;
- the verdict is *not yet decidable*, *not admitted* or *admitted*, by item 6.

A verdict is a result, not an error: every verdict exits 0.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, date
from pathlib import Path
from typing import Any

from augur import scorecard
from augur.clock import to_iso
from augur.errors import AugurError
from augur.score import DEFAULT_SEED, clustered_bootstrap, head_to_head
from augur.standings import SAME_MODEL_TEMPLATE, RoundStandings, load_snapshot

LEVEL = 0.90
RESAMPLES = 10_000
SEED = DEFAULT_SEED
MIN_N = 20
DECISION_DAY = date(2026, 12, 13)
"""Item 7: the last snapshot fetched on or before this day (UTC) decides the next round."""
DECIDED_ROUND = "2026-12-14"
PEER = SAME_MODEL_TEMPLATE
TEMPLATE = "template"
AUGUR = "augur"
BLEND = "blend"
CANDIDATES: tuple[str, ...] = (AUGUR, BLEND)
"""``augur``: Augur's aggregate (M11's item 1). ``blend``: the ``blend`` reference rows,
M12's registered primary candidate (PLAN.md, ``m12-registration`` item 5)."""

ADMITTED = "admitted"
NOT_ADMITTED = "not admitted"
UNDECIDABLE = "not yet decidable"

CANDIDATE_NAMES = {
    AUGUR: "Augur's aggregate",
    BLEND: "the blend (M12's primary candidate: one-third Augur, two-thirds the statistical "
    "arm where it has a price or a series, Augur elsewhere)",
}

BIASES = (
    "proxy biases: the shadow template is another run of the same code, not the public "
    "bot's forecasts; S enters as a constant, so the implied peer interval is too narrow; "
    "the two can cover different questions"
)


@dataclass(frozen=True)
class Scored:
    post_id: int
    question_id: int
    question_type: str
    h: float


@dataclass(frozen=True)
class FloorResult:
    round: str
    candidate: str
    fetched_at: str
    finalized: bool
    round_posts: int
    max_coverage: float
    augur: int
    template: int
    both: int
    n: int
    annulled: int
    uncovered: int
    unresolved: int
    unscorable: int
    mean_h: float | None
    h_low: float | None
    h_high: float | None
    sd_h: float | None
    peer: str
    s: float | None
    implied: float | None
    implied_low: float | None
    implied_high: float | None
    verdict: str
    reason: str
    after_decision_day: bool
    scored: tuple[Scored, ...]
    vs_augur_n: int = 0
    """With the blend as candidate: questions scored for P39 (blend against Augur alone)."""
    vs_augur_mean: float | None = None
    vs_augur_low: float | None = None
    vs_augur_high: float | None = None

    def lines(self) -> list[str]:
        def num(x: float | None, digits: int = 2) -> str:
            return "n/a" if x is None else f"{x:.{digits}f}"

        def interval(mean: float | None, low: float | None, high: float | None) -> str:
            if mean is None:
                return "n/a"
            return f"{num(mean)} [{num(low)}, {num(high)}]"

        out = [
            f"round: {self.round}",
            f"candidate: {CANDIDATE_NAMES[self.candidate]}",
            f"snapshot: fetched {self.fetched_at}, "
            f"{'finalized' if self.finalized else 'not finalized'}, "
            f"max_coverage {self.max_coverage:g} of {self.round_posts} posts",
            f"questions: {self.round_posts} in the round; forecast by Augur {self.augur}, "
            f"by the template {self.template}, by both {self.both}",
            f"scored: n = {self.n}; annulled or ambiguous {self.annulled}; uncovered (one side "
            f"only) {self.uncovered}; unresolved {self.unresolved}"
            + (f"; resolved but unscorable {self.unscorable}" if self.unscorable else ""),
            f"head-to-head mean(h): {interval(self.mean_h, self.h_low, self.h_high)} "
            f"({LEVEL:.0%} bootstrap over questions, {RESAMPLES:,} resamples, seed {SEED})",
            f"measured paired spread (SD of h): {num(self.sd_h)}",
            f"S ({self.peer}, average spot peer score): "
            + ("no entry" if self.s is None else num(self.s)),
            f"implied peer score S + mean(h): "
            f"{interval(self.implied, self.implied_low, self.implied_high)}",
            BIASES,
            f"verdict: {self.verdict} ({self.reason})",
        ]
        if self.candidate == BLEND:
            out.append(
                f"reported (P39), the blend against Augur's aggregate alone: n = "
                f"{self.vs_augur_n}, mean head-to-head "
                f"{interval(self.vs_augur_mean, self.vs_augur_low, self.vs_augur_high)} "
                f"(same bootstrap); not part of the verdict"
            )
        if self.after_decision_day:
            out.append(
                f"note: this snapshot was fetched after {DECISION_DAY.isoformat()} (UTC), so the "
                f"verdict decides nothing for the {DECIDED_ROUND} round; it is recorded for the "
                "season only"
            )
        return out

    def json_text(self) -> str:
        """Byte-stable JSON: sorted keys, indent 1, one trailing newline."""
        body: dict[str, Any] = asdict(self)
        body["rule"] = {
            "level": LEVEL,
            "resamples": RESAMPLES,
            "seed": SEED,
            "min_n": MIN_N,
            "decision_day": DECISION_DAY.isoformat(),
        }
        return json.dumps(body, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


@dataclass
class _Pooled:
    """What the ledgers hold for each question id, pooled as ``augur score`` pools them."""

    post_of: dict[int, int]
    augur: set[int]
    blend: set[int]
    template: set[int]
    resolution: dict[int, tuple[str, str]]  # question id -> (recorded_at, status)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _pool(paths: Sequence[Path]) -> _Pooled:
    pooled = _Pooled(post_of={}, augur=set(), blend=set(), template=set(), resolution={})
    for path in paths:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            if _has_table(conn, "questions"):
                for qid, post_id in conn.execute(
                    "SELECT DISTINCT question_id, post_id FROM questions"
                ):
                    pooled.post_of.setdefault(int(qid), int(post_id))
            if _has_table(conn, "forecasts"):
                for (qid,) in conn.execute(
                    "SELECT DISTINCT f.question_id FROM forecasts f "
                    "JOIN runs r ON r.run_id = f.run_id "
                    "WHERE f.kind = 'aggregate' AND r.mode != 'unrestricted'"
                ):
                    pooled.augur.add(int(qid))
            if _has_table(conn, "reference_forecasts"):
                for qid, post_id, who in conn.execute(
                    "SELECT DISTINCT question_id, post_id, forecaster FROM reference_forecasts "
                    "WHERE forecaster IN (?, ?)",
                    (TEMPLATE, BLEND),
                ):
                    (pooled.template if who == TEMPLATE else pooled.blend).add(int(qid))
                    pooled.post_of.setdefault(int(qid), int(post_id))
            if _has_table(conn, "resolutions"):
                for qid, at, status in conn.execute(
                    "SELECT question_id, recorded_at, status FROM resolutions "
                    "ORDER BY recorded_at, rowid"
                ):
                    held = pooled.resolution.get(int(qid))
                    if held is None or str(at) >= held[0]:
                        pooled.resolution[int(qid)] = (str(at), str(status))
    return pooled


def _peer_average(snapshot: RoundStandings) -> float | None:
    for entry in snapshot.board().entries:
        if entry.user is not None and entry.user.username == PEER:
            return entry.average
    return None


def decide(
    snapshot: RoundStandings, ledger_paths: Sequence[Path], candidate: str = AUGUR
) -> FloorResult:
    """Apply the floor rule to one round's snapshot and the ledgers holding its forecasts.

    ``candidate`` names whose forecast is scored against the template: Augur's
    aggregate, or the ``blend`` rows M12 registered as the primary candidate.
    A blend row exists only beside an aggregate, so the two cover the same
    questions when the statistical arm ran on every one.
    """
    if candidate not in CANDIDATES:
        raise AugurError(f"unknown candidate {candidate!r}; use one of {', '.join(CANDIDATES)}")
    for path in ledger_paths:
        if not path.exists():
            raise AugurError(f"no ledger at {path}; give the shadow ledger with --ledger")
    posts = {q.post_id for q in snapshot.questions}
    pooled = _pool(ledger_paths)

    def in_round(qids: set[int]) -> set[int]:
        return {q for q in qids if pooled.post_of.get(q) in posts}

    augur = in_round(pooled.augur if candidate == AUGUR else pooled.augur & pooled.blend)
    template = in_round(pooled.template)
    both = augur & template
    either = augur | template
    unresolved = {q for q in either if q not in pooled.resolution}
    annulled = {q for q in both if q in pooled.resolution and pooled.resolution[q][1] != "resolved"}
    resolved_both = {
        q for q in both if q in pooled.resolution and pooled.resolution[q][1] == "resolved"
    }

    scored: list[Scored] = []
    vs_augur: list[tuple[int, float]] = []
    for s in scorecard.load(ledger_paths).scored:
        if (
            candidate == BLEND
            and s.question_id in in_round(pooled.blend)
            and BLEND in s.references
            and s.resolution
        ):
            p_blend, p_augur = s.p(s.references[BLEND]), s.p(s.augur)
            vs_augur.append((s.post_id, head_to_head(s.question_type, p_blend, p_augur)))
        if s.question_id not in resolved_both or TEMPLATE not in s.references:
            continue
        mine = s.augur if candidate == AUGUR else s.references.get(BLEND)
        if mine is None:
            continue
        h = head_to_head(s.question_type, s.p(mine), s.p(s.references[TEMPLATE]))
        scored.append(Scored(s.post_id, s.question_id, s.question_type, h))
    scored.sort(key=lambda x: x.post_id)
    hs = [x.h for x in scored]
    n = len(hs)

    mean_h: float | None = None
    h_low: float | None = None
    h_high: float | None = None
    sd_h: float | None = None
    if hs:
        est = clustered_bootstrap(hs, level=LEVEL, resamples=RESAMPLES, seed=SEED)
        mean_h, h_low, h_high = est.mean, est.low, est.high
    if n >= 2:
        sd_h = statistics.stdev(hs)

    s_avg = _peer_average(snapshot)
    implied: float | None = None
    implied_low: float | None = None
    implied_high: float | None = None
    if s_avg is not None and mean_h is not None and h_low is not None and h_high is not None:
        implied, implied_low, implied_high = s_avg + mean_h, s_avg + h_low, s_avg + h_high

    board = snapshot.board()
    waiting: list[str] = []
    if unresolved:
        waiting.append(
            f"{len(unresolved)} of the round's questions with an Augur or template forecast "
            "have no resolution; run augur resolve, or augur resolve --manual-file"
        )
    if board.max_coverage < 0.5 * len(snapshot.questions):
        waiting.append(
            f"the leaderboard has scored {board.max_coverage:g} of the round's "
            f"{len(snapshot.questions)} questions, fewer than half"
        )
    if s_avg is None:
        waiting.append(f"the snapshot has no {PEER} entry with coverage")

    if waiting:
        verdict, reason = UNDECIDABLE, "; ".join(waiting)
    elif n < MIN_N or h_low is None or implied_low is None:
        verdict, reason = NOT_ADMITTED, f"n = {n} scored questions, below {MIN_N}"
    else:
        ends = (
            f"lower ends: head-to-head {h_low:.2f}, implied peer score {implied_low:.2f}; "
            "both must be above 0"
        )
        if h_low > 0 and implied_low > 0:
            verdict, reason = ADMITTED, ends
        else:
            verdict, reason = NOT_ADMITTED, ends

    vs = [h for _, h in sorted(vs_augur)]
    vs_est = clustered_bootstrap(vs, level=LEVEL, resamples=RESAMPLES, seed=SEED) if vs else None

    return FloorResult(
        round=snapshot.round.slug,
        candidate=candidate,
        vs_augur_n=len(vs),
        vs_augur_mean=None if vs_est is None else vs_est.mean,
        vs_augur_low=None if vs_est is None else vs_est.low,
        vs_augur_high=None if vs_est is None else vs_est.high,
        fetched_at=to_iso(snapshot.fetched_at),
        finalized=board.finalized,
        round_posts=len(posts),
        max_coverage=board.max_coverage,
        augur=len(augur),
        template=len(template),
        both=len(both),
        n=n,
        annulled=len(annulled),
        uncovered=len(either - both),
        unresolved=len(unresolved),
        unscorable=len(resolved_both) - n,
        mean_h=mean_h,
        h_low=h_low,
        h_high=h_high,
        sd_h=sd_h,
        peer=PEER,
        s=s_avg,
        implied=implied,
        implied_low=implied_low,
        implied_high=implied_high,
        verdict=verdict,
        reason=reason,
        after_decision_day=snapshot.fetched_at.astimezone(UTC).date() > DECISION_DAY,
        scored=tuple(scored),
    )


def add_parser(subparsers: Any) -> None:
    """Register ``augur floor``."""
    p = subparsers.add_parser(
        "floor",
        help="apply PLAN.md's floor rule for the free model to one MiniBench round",
        description=(
            "Score Augur's shadow aggregate against the shadow template on the round's "
            "questions (the posts in its standings snapshot), add the head-to-head to the "
            f"average spot peer score of {PEER}, and print the verdict: admitted, not "
            "admitted or not yet decidable. Offline; every verdict exits 0."
        ),
    )
    p.add_argument("--round", required=True, metavar="SLUG", help="the round's project slug")
    p.add_argument(
        "--ledger",
        type=Path,
        action="append",
        required=True,
        metavar="L",
        help="the shadow ledger; repeatable (resolutions and references pool by question)",
    )
    p.add_argument(
        "--standings",
        type=Path,
        metavar="S",
        help="the round's standings snapshot (default data/standings/SLUG.json)",
    )
    p.add_argument(
        "--candidate",
        choices=CANDIDATES,
        default=AUGUR,
        help="whose forecast faces the template: Augur's aggregate (M11's item 1, the "
        "default) or blend, M12's registered primary candidate (the ledger's blend rows)",
    )
    p.add_argument("--json", type=Path, metavar="OUT", help="also write the numbers as JSON")


def main(args: argparse.Namespace) -> int:
    path: Path = args.standings or Path("data/standings") / f"{args.round}.json"
    snapshot = load_snapshot(path)
    if snapshot.round.slug != args.round:
        raise AugurError(
            f"{path} is the snapshot of {snapshot.round.slug}, not {args.round}; give the "
            "round's own snapshot with --standings"
        )
    result = decide(snapshot, args.ledger, args.candidate)
    for line in result.lines():
        print(line)
    if args.json is not None:
        args.json.write_text(result.json_text(), encoding="utf-8")
    return 0
