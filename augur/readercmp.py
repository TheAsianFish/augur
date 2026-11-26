"""M16: is the newer free model a better reader on the same questions?

Item numbers below are PLAN.md's ``m16-registration`` block, which RESULTS.md
renders byte for byte.

* *Backtest* (items 2 and 3): one draw a question from each route, no
  retrieval, on the market questions of ForecastBench's 2026-11-08 and
  2026-11-22 sets, one ledger a route. A question is paired when both
  ledgers hold an aggregate for it and its latest resolution (written by
  ``augur resolve --forecastbench``) is yes or no.
* *Live* (item 4), optional: a ledger rebuilt from the shadow exports, where
  each question the arm reached holds a ``reader`` row (three draws on the
  newer route) and a ``reader-base`` row (Augur's own members 0 to 2 from the
  same tick, combined the same way).
* *Statistic* (item 5): per question, Brier(newer) − Brier(current), with a
  95% paired bootstrap interval resampling questions (10,000 resamples, seed
  20261119) and the detectable effect; head-to-head points on the outcome for
  live questions.
* *Rule* (item 6) and *date* (item 7): :func:`verdict`. The reading date is
  the newest fetch of a committed resolution file, never the clock, so
  ``augur readers --check`` is reproducible.

No model is called here. ``augur readers`` writes :data:`SNAPSHOT`;
``--check`` re-runs and compares, as ``make eval`` does.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import statistics
from collections.abc import Hashable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from augur.score import (
    Z_95,
    Z_POWER_80,
    Estimate,
    brier,
    clustered_bootstrap,
    head_to_head,
    outcome_probability,
)

CURRENT = "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
NEWER = "openrouter:nvidia/nemotron-3.5-super-vl:free"
LEDGERS = {
    "current": Path("data/ledgers/m16-reader-current.sqlite"),
    "newer": Path("data/ledgers/m16-reader-newer.sqlite"),
}
SETS = ("2026-11-08", "2026-11-22")
FIRST_SET = "2026-11-08"
"""P58 and P59 read this set's questions resolved in its first resolution file."""
PROSPECTIVE = {"2026-11-08": False, "2026-11-22": True}
"""Item 3: the 2026-11-08 resolution file existed (unread) before any forecast."""
RESOLUTIONS = Path("data/forecastbench")
SNAPSHOT = Path("data/reader/m16-readers.json")
ANALYSIS_DATE = date(2027, 1, 11)
SWITCH_MARGIN = 0.01
TARGETS = (0.01, 0.02)
READER = "reader"
READER_BASE = "reader-base"


@dataclass(frozen=True)
class Paired:
    question_id: int
    question_set: str
    p_newer: float
    p_current: float
    outcome: int
    market: float | None

    @property
    def diff(self) -> float:
        return brier(self.p_newer, self.outcome) - brier(self.p_current, self.outcome)


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _set_of(tournament: str) -> str:
    """``forecastbench:2026-11-08-llm.json`` -> ``2026-11-08``."""
    return tournament.removeprefix("forecastbench:").removesuffix("-llm.json")


Key = tuple[str, int]
"""``(set, question_id)``: a question posed in both sets (47 Metaculus questions are,
frozen on different days) has one id in both runs of a ledger, so everything is
keyed by the set as well."""

_EVIDENCE = re.compile(r"^(?P<set>\d{4}-\d{2}-\d{2})-llm\.json resolution set: ")
_PRICE = re.compile(r"^ForecastBench (?P<set>\d{4}-\d{2}-\d{2})-llm\.json freeze_datetime_value$")


def aggregates(path: Path) -> dict[Key, float]:
    """``(set, question_id) -> probability`` for every aggregate in a backtest ledger."""
    out: dict[Key, float] = {}
    if not path.exists():
        return out
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT f.question_id, r.tournament, f.probability FROM forecasts f "
            "JOIN runs r ON r.run_id = f.run_id WHERE f.kind = 'aggregate' "
            "AND r.mode = 'backtest' ORDER BY f.recorded_at"
        ).fetchall()
    for qid, tournament, p in rows:
        out.setdefault((_set_of(str(tournament)), int(qid)), float(p))
    return out


def resolutions(path: Path) -> dict[Key, int]:
    """``(set, question_id) -> 1 or 0``, each set read only from its own resolution file.

    ``augur resolve --forecastbench`` names the file in each row's evidence; the
    latest row for a (set, question) counts.
    """
    out: dict[Key, int] = {}
    if not path.exists():
        return out
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT question_id, status, value, evidence FROM resolutions "
            "ORDER BY recorded_at, rowid"
        ).fetchall()
    latest: dict[Key, tuple[str, str | None]] = {}
    for qid, status, value, evidence in rows:
        m = _EVIDENCE.match(str(evidence))
        if m is not None:
            latest[(m.group("set"), int(qid))] = (str(status), value)
    for key, (status, value) in latest.items():
        if status == "resolved" and value in ("yes", "no"):
            out[key] = 1 if value == "yes" else 0
    return out


def market_prices(path: Path) -> dict[Key, float]:
    """The freeze price ``augur resolve`` records as a ``market`` reference (never shown)."""
    out: dict[Key, float] = {}
    if not path.exists():
        return out
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT question_id, probability, source FROM reference_forecasts "
            "WHERE forecaster = 'market' AND probability IS NOT NULL ORDER BY recorded_at"
        ).fetchall()
    for qid, p, source in rows:
        m = _PRICE.match(str(source))
        if m is not None:
            out.setdefault((m.group("set"), int(qid)), float(p))
    return out


@dataclass(frozen=True)
class RouteUsage:
    """Item 5's latency, and what each route's calls did."""

    forecast: dict[str, int]
    failed: dict[str, int]
    calls: int
    errors: int
    median_latency_ms: float | None
    mean_latency_ms: float | None
    output_tokens_median: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "forecast": self.forecast,
            "failed": self.failed,
            "calls": self.calls,
            "errors": self.errors,
            "median_latency_ms": self.median_latency_ms,
            "mean_latency_ms": None
            if self.mean_latency_ms is None
            else round(self.mean_latency_ms, 1),
            "output_tokens_median": self.output_tokens_median,
        }


def route_usage(path: Path) -> RouteUsage:
    forecast = {s: 0 for s in SETS}
    failed = {s: 0 for s in SETS}
    if not path.exists():
        return RouteUsage(forecast, failed, 0, 0, None, None, None)
    with closing(_connect(path)) as conn:
        for qset, n in conn.execute(
            "SELECT r.tournament, COUNT(DISTINCT f.question_id) FROM forecasts f "
            "JOIN runs r ON r.run_id = f.run_id WHERE f.kind = 'aggregate' GROUP BY r.tournament"
        ):
            forecast[_set_of(str(qset))] = int(n)
        # Not forecast: recorded in a run of the set with no aggregate in any run of that set
        # (a resume retries a failed question, so a later success clears it).
        for qset, n in conn.execute(
            "SELECT r.tournament, COUNT(DISTINCT q.question_id) FROM questions q "
            "JOIN runs r ON r.run_id = q.run_id WHERE r.mode = 'backtest' AND NOT EXISTS "
            "(SELECT 1 FROM forecasts f JOIN runs r2 ON r2.run_id = f.run_id "
            "WHERE f.question_id = q.question_id AND f.kind = 'aggregate' "
            "AND r2.tournament = r.tournament) GROUP BY r.tournament"
        ):
            failed[_set_of(str(qset))] = int(n)
        rows = conn.execute(
            "SELECT latency_ms, error, output_tokens FROM llm_calls WHERE role = 'forecaster'"
        ).fetchall()
    ok = [int(r[0]) for r in rows if r[1] is None]
    tokens = [int(r[2]) for r in rows if r[1] is None and r[2] is not None]
    return RouteUsage(
        forecast=forecast,
        failed=failed,
        calls=len(rows),
        errors=sum(1 for r in rows if r[1] is not None),
        median_latency_ms=float(statistics.median(ok)) if ok else None,
        mean_latency_ms=statistics.fmean(ok) if ok else None,
        output_tokens_median=float(statistics.median(tokens)) if tokens else None,
    )


def pairs(root: Path) -> tuple[list[Paired], dict[str, int]]:
    """The questions both routes forecast that have resolved yes or no, and what was left out."""
    newer = aggregates(root / LEDGERS["newer"])
    current = aggregates(root / LEDGERS["current"])
    outcomes = resolutions(root / LEDGERS["current"])
    outcomes_newer = resolutions(root / LEDGERS["newer"])
    prices = market_prices(root / LEDGERS["current"])
    left = {"one route only": 0, "not resolved to 0 or 1": 0, "resolutions disagree": 0}
    out: list[Paired] = []
    for key in sorted(set(newer) | set(current)):
        if key not in newer or key not in current:
            left["one route only"] += 1
            continue
        y = outcomes.get(key)
        if y is None or outcomes_newer.get(key) != y:
            left["not resolved to 0 or 1" if y is None else "resolutions disagree"] += 1
            continue
        out.append(Paired(key[1], key[0], newer[key], current[key], y, prices.get(key)))
    return out, left


def _est(values: Sequence[float], clusters: Sequence[Hashable] | None = None) -> Estimate | None:
    """The mean with item 5's interval; ``clusters`` names the question each value is on,
    so a question posed in both sets is resampled once, with both its rows."""
    return clustered_bootstrap(values, clusters) if values else None


def _diffs(ps: Sequence[Paired]) -> tuple[list[float], list[Hashable]]:
    return [p.diff for p in ps], [("fb", p.question_id) for p in ps]


def _est_json(e: Estimate | None) -> dict[str, Any] | None:
    if e is None:
        return None
    sd = e.se * math.sqrt(e.n) if e.n > 1 and not math.isnan(e.se) else None
    return {
        "mean": round(e.mean, 5),
        "low": round(e.low, 5),
        "high": round(e.high, 5),
        "n": e.n,
        "mde": None if math.isnan(e.mde) else round(e.mde, 5),
        "sd": None if sd is None else round(sd, 5),
        "unrounded": [e.mean, e.low, e.high],
    }


def n_needed(sd: float, target: float) -> int:
    """Questions a paired difference of ``target`` needs at 80% power, two-sided 5%."""
    return math.ceil(((Z_95 + Z_POWER_80) * sd / target) ** 2)


def verdict(pooled: Estimate | None, latency: dict[str, float | None], reading: str) -> str:
    """Item 6 on the pooled estimate; before item 7's date it is a progress reading."""
    if pooled is None:
        return "not decidable: no question has resolved"
    low, high = pooled.low, pooled.high
    faster = (
        latency["newer"] is not None
        and latency["current"] is not None
        and latency["newer"] <= latency["current"]
    )
    if high < 0.0 or (high < SWITCH_MARGIN and faster):
        what = "switched"
    elif low > 0.0:
        what = "kept"
    else:
        what = "not decidable at this n"
    if reading == "progress":
        return f"{what} (progress reading: the rule binds only on {ANALYSIS_DATE})"
    return what


def _held(ok: bool) -> str:
    return "held" if ok else "failed"


def hypotheses(
    first: Estimate | None, latency: dict[str, float | None], live: dict[str, Any] | None
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if first is None:
        out.append({"id": "P58", "result": "not yet readable: no question of the set resolved"})
        out.append({"id": "P59", "result": "not yet readable: no question of the set resolved"})
    else:
        half = (first.high - first.low) / 2.0
        p58 = half > 0.05 and first.low <= 0.0 <= first.high
        out.append(
            {
                "id": "P58",
                "result": f"{_held(p58)}: half width {half:.4f}, interval "
                f"[{first.low:+.4f}, {first.high:+.4f}] on {first.n} questions",
            }
        )
        p59 = -0.03 <= first.mean <= 0.0
        out.append({"id": "P59", "result": f"{_held(p59)}: {first.mean:+.4f}"})
    if latency["newer"] is None or latency["current"] is None:
        out.append({"id": "P60", "result": "not yet readable"})
    else:
        p60 = latency["newer"] < latency["current"]
        out.append(
            {
                "id": "P60",
                "result": f"{_held(p60)}: median {latency['newer'] / 1000:.1f} s against "
                f"{latency['current'] / 1000:.1f} s",
            }
        )
    out.append(
        {
            "id": "P61",
            "result": "not yet readable: the live arm starts with the 2026-12-14 round"
            if live is None
            else str(live.get("p61", "not yet readable")),
        }
    )
    return out


def resolution_files(root: Path) -> list[dict[str, Any]]:
    """The committed resolution cuts' provenance (``*.open.resolutions.json``), by set."""
    out = []
    for s in SETS:
        meta = root / RESOLUTIONS / f"{s}.open.resolutions.json"
        if meta.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            out.append({"set": s} | {k: data[k] for k in ("fetched", "sha256", "upstream")})
    return out


def reading_kind(day: date | None) -> str:
    if day is None or day < ANALYSIS_DATE:
        return "progress"
    return "registered" if day == ANALYSIS_DATE else "late"


# -- the live arm (item 4), from a ledger rebuilt from the shadow exports -----------------


def live_rows(path: Path) -> list[tuple[int, str, float, float, float, float]]:
    """``(question_id, type, h2h, brier_diff or nan, p_reader, p_base)`` per resolved question.

    Head-to-head and Brier are on the outcome, for binary questions; other
    types report head-to-head on the probability each row put on the outcome.
    """
    from augur.questions import question_from_post

    out: list[tuple[int, str, float, float, float, float]] = []
    with closing(_connect(path)) as conn:
        refs: dict[tuple[int, str], dict[str, Any]] = {}
        for qid, who, values in conn.execute(
            "SELECT question_id, forecaster, values_json FROM reference_forecasts "
            "WHERE forecaster IN (?, ?) ORDER BY recorded_at",
            (READER, READER_BASE),
        ):
            refs.setdefault((int(qid), str(who)), json.loads(values))
        latest: dict[int, tuple[str, str | None]] = {}
        for qid, status, value in conn.execute(
            "SELECT question_id, status, value FROM resolutions ORDER BY recorded_at, rowid"
        ):
            latest[int(qid)] = (str(status), value)
        payloads: dict[int, str] = {}
        for qid, payload in conn.execute(
            "SELECT question_id, payload_json FROM questions ORDER BY rowid"
        ):
            payloads[int(qid)] = str(payload)
    for qid in sorted({q for q, _ in refs}):
        mine, base = refs.get((qid, READER)), refs.get((qid, READER_BASE))
        resolved = latest.get(qid)
        if mine is None or base is None or resolved is None or resolved[0] != "resolved":
            continue
        q = question_from_post(json.loads(payloads[qid]))
        value = str(resolved[1])
        how: dict[str, Any] = {"options": q.options, "scaling": q.scaling}
        p_mine = outcome_probability(q.type, mine, value, **how)
        p_base = outcome_probability(q.type, base, value, **how)
        h2h = head_to_head(q.type, p_mine, p_base)
        diff = math.nan
        if q.type == "binary":
            y = 1 if value == "yes" else 0
            diff = brier(float(mine["probability"]), y) - brier(float(base["probability"]), y)
        out.append((qid, q.type, h2h, diff, p_mine, p_base))
    return out


def snapshot(root: Path, live: Path | None = None) -> dict[str, Any]:
    """Every M16 number, from the committed ledgers (and a live ledger, if given)."""
    usage = {k: route_usage(root / p) for k, p in LEDGERS.items()}
    latency = {k: u.median_latency_ms for k, u in usage.items()}
    ps, left = pairs(root)
    by_set = {s: [p for p in ps if p.question_set == s] for s in SETS}
    backtest = _est(*_diffs(ps))
    live_part: dict[str, Any] | None = None
    live_binary: list[float] = []
    live_ids: list[Hashable] = []
    if live is not None:
        rows = live_rows(live)
        live_binary = [r[3] for r in rows if not math.isnan(r[3])]
        live_ids = [("live", r[0]) for r in rows if not math.isnan(r[3])]
        live_part = {
            "questions": len(rows),
            "head_to_head": _est_json(_est([r[2] for r in rows])),
            "brier_diff": _est_json(_est(live_binary)),
        }
    values, ids = _diffs(ps)
    pooled = _est(values + live_binary, ids + live_ids)
    files = resolution_files(root)
    fetched = max((date.fromisoformat(f["fetched"][:10]) for f in files), default=None)
    reading = reading_kind(fetched)
    sd = (
        None
        if pooled is None or pooled.n < 2 or math.isnan(pooled.se)
        else pooled.se * math.sqrt(pooled.n)
    )
    first = _est(*_diffs(by_set[FIRST_SET]))
    sets: dict[str, Any] = {}
    for s in SETS:
        sub = by_set[s]
        sets[s] = {
            "prospective": PROSPECTIVE[s],
            "paired": len(sub),
            "brier_newer": round(statistics.fmean(brier(p.p_newer, p.outcome) for p in sub), 5)
            if sub
            else None,
            "brier_current": round(statistics.fmean(brier(p.p_current, p.outcome) for p in sub), 5)
            if sub
            else None,
            "brier_market": _market_brier(sub),
            "yes_share": round(statistics.fmean(p.outcome for p in sub), 4) if sub else None,
            "diff": _est_json(_est(*_diffs(sub))),
        }
    return {
        "routes": {"current": CURRENT, "newer": NEWER},
        "method": {
            "statistic": "Brier(newer) - Brier(current) per question and set, 95% paired "
            "bootstrap resampling questions (a question posed in both sets is one), 10,000 "
            "resamples, seed 20261119",
            "analysis_date": str(ANALYSIS_DATE),
        },
        "usage": {k: u.to_json() for k, u in usage.items()},
        "left_out": left,
        "sets": sets,
        "backtest": _est_json(backtest),
        "backtest_brier": {
            "newer": round(statistics.fmean(brier(p.p_newer, p.outcome) for p in ps), 5)
            if ps
            else None,
            "current": round(statistics.fmean(brier(p.p_current, p.outcome) for p in ps), 5)
            if ps
            else None,
            "market": _market_brier(ps),
        },
        "live": live_part,
        "pooled": _est_json(pooled),
        "n_needed": None if sd is None else {str(t): n_needed(sd, t) for t in TARGETS},
        "resolution_files": files,
        "reading": {"date": None if fetched is None else str(fetched), "kind": reading},
        "verdict": verdict(pooled, latency, reading),
        "hypotheses": hypotheses(first, latency, live_part),
    }


def _market_brier(ps: Sequence[Paired]) -> dict[str, Any] | None:
    priced = [p for p in ps if p.market is not None]
    if not priced:
        return None
    return {
        "n": len(priced),
        "market": round(statistics.fmean(brier(p.market or 0.0, p.outcome) for p in priced), 5),
        "newer": round(statistics.fmean(brier(p.p_newer, p.outcome) for p in priced), 5),
        "current": round(statistics.fmean(brier(p.p_current, p.outcome) for p in priced), 5),
    }


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def add_parser(subparsers: Any) -> None:
    rd = subparsers.add_parser(
        "readers",
        help="M16: the newer free reader against the current route, same questions",
        description=(
            "Pair the two routes' one-draw forecasts on ForecastBench's 2026-11-08 and "
            "2026-11-22 market questions (and, with --live, the shadow arm's reader rows), "
            "score the resolved ones and apply the registered route rule. Writes the "
            "snapshot RESULTS.md renders from. Offline, free, no model call."
        ),
    )
    rd.add_argument("--out", type=Path, default=SNAPSHOT)
    rd.add_argument(
        "--live",
        type=Path,
        metavar="LEDGER",
        help="a ledger rebuilt from the shadow exports (augur ledger import), for the live arm",
    )
    rd.add_argument(
        "--check", action="store_true", help="re-run and compare with the committed snapshot"
    )


def main(args: Any) -> int:
    root = Path.cwd()
    text = dumps(snapshot(root, args.live))
    path = root / args.out
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            print(f"{args.out} differs from a fresh run; run augur readers")
            return 1
        print(f"{args.out} matches a fresh run")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    data = json.loads(text)
    print(f"wrote {args.out}: {data['verdict']}")
    for h in data["hypotheses"]:
        print(f"  {h['id']}: {h['result']}")
    return 0
