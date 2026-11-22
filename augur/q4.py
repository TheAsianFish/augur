"""Q4: can calibration be learned from the bot's own resolved history?

M8's experiment (PLAN.md, "M8's predictions"), scored from the ledgers.

The corpus, C4, is nine ForecastBench sets (2026-06-21 to 2026-10-11) of
market questions that resolved to 0 or 1, forecast by one free model with
no retrieval and one draw. A *question instance* is one question in one
set: a market posed in three sets is three instances, forecast three times,
and the bootstrap resamples markets so those instances move together.

The rule that makes it a backtest, and the one thing this module must never
get wrong: **a test instance is calibrated only with instances that resolved
before the day it opened** (resolution date < freeze date).
:func:`history` selects that history and :func:`check_time_order` refuses
any fit that would see a later resolution; every fit is recorded with the
latest resolution it saw, so the committed snapshot can be re-checked.

Two designs, both fixed before the run:

* **W, walk-forward:** every instance with at least :data:`W_MIN`
  resolutions before it opened, each calibrated on all of them;
* **L, learning curve:** the instances with at least :data:`L_MIN`
  resolutions before they opened, each calibrated on its ``k`` most recent
  resolutions, for each ``k`` in :data:`KS`, on the same instances at every
  ``k``.

Everything here is deterministic (the bootstrap is seeded), so ``augur
calibrate`` writes a snapshot that ``augur calibrate --check`` reproduces
byte for byte, and ``augur report`` renders RESULTS.md from it.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from collections.abc import Callable, Hashable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from augur.calibrate import CALIBRATORS, CalibrationMap, Calibrator, by_name
from augur.errors import AugurError, LeakageError
from augur.forecastbench import load_set
from augur.q3 import members
from augur.replay import clip
from augur.score import Estimate, brier, clustered_bootstrap, log_score, paired

SETS: tuple[str, ...] = (
    "2026-06-21",
    "2026-07-05",
    "2026-07-19",
    "2026-08-02",
    "2026-08-16",
    "2026-08-30",
    "2026-09-13",
    "2026-09-27",
    "2026-10-11",
)
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
RELEASED = date(2026, 6, 4)
"""The model's OpenRouter listing date, the release gate's bound (as in M5 to M7)."""
FORECASTBENCH = Path("data/forecastbench")
LEDGER = Path("data/ledgers/m8-c4.sqlite")
FIRST_SET_LEDGER = Path("data/ledgers/m5-replay-c2.sqlite")
"""2026-06-21 reuses member 0 of M5's C2 arm (b), the first draw C2 already reports."""
SNAPSHOT = Path("data/calibration/m8-calibration.json")

W_MIN = 50
L_MIN = 400
KS: tuple[int, ...] = (25, 50, 100, 200, 400)
BINS = 10
READABLE = frozenset({"polymarket", "manifold"})
"""Markets whose price the bot can read live (M5); Metaculus and INFER prices are hidden."""


class CalibrationDataError(AugurError):
    """The corpus is incomplete or inconsistent."""


@dataclass(frozen=True)
class Instance:
    """One question in one set: the raw forecast, its outcome and its dates."""

    key: str
    market: str
    source: str
    question_set: str
    opened: date
    resolved: date
    outcome: int
    p: float
    price: float | None


def tournament(question_set: str) -> str:
    return f"forecastbench:{question_set}-llm.json"


def load_corpus(root: Path) -> list[Instance]:
    """C4's instances in a fixed order (set, then the set's question order).

    Raises if a set's forecasts are missing from its ledger, so a partial
    corpus can never be scored as if it were whole.
    """
    out: list[Instance] = []
    for qs in SETS:
        report = load_set(
            root / FORECASTBENCH / f"{qs}-llm.subset.json",
            root / FORECASTBENCH / f"{qs}_resolution_set.subset.json",
            released=RELEASED,
        )
        if qs == SETS[0]:
            draws = members(root / FIRST_SET_LEDGER, tournament(qs))
            raw = {q: ms[0] for q, ms in draws.items() if 0 in ms}
        else:
            raw = _aggregates(root / LEDGER, tournament(qs))
        missing = [q.fb_id for q in report.kept if q.question_id not in raw]
        if len(missing) > len(report.kept) // 10:
            raise CalibrationDataError(
                f"{qs}: {len(missing)} of {len(report.kept)} questions have no forecast in "
                "the ledger; finish the run (make backtest-q4) before scoring"
            )
        for q in report.kept:
            if q.question_id not in raw:
                continue
            out.append(
                Instance(
                    key=f"{qs}:{q.source}:{q.fb_id}",
                    market=f"{q.source}:{q.fb_id}",
                    source=q.source,
                    question_set=qs,
                    opened=q.freeze_datetime.date(),
                    resolved=q.resolution_date,
                    outcome=q.outcome,
                    p=raw[q.question_id],
                    price=q.freeze_value,
                )
            )
    return out


def _aggregates(path: Path, name: str) -> dict[int, float]:
    if not path.exists():
        raise CalibrationDataError(f"no {path}; run make backtest-q4 first")
    latest: dict[int, tuple[str, float]] = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        for qid, made, p in conn.execute(
            "SELECT f.question_id, f.recorded_at, f.probability FROM forecasts f "
            "JOIN runs r ON r.run_id = f.run_id "
            "WHERE r.tournament = ? AND r.mode = 'backtest' AND f.kind = 'aggregate' "
            "AND r.model = ?",
            (name, MODEL),
        ):
            if int(qid) not in latest or made > latest[int(qid)][0]:
                latest[int(qid)] = (str(made), float(p))
    return {q: v[1] for q, v in latest.items()}


# -- the time order ---------------------------------------------------------------


def history(instances: Sequence[Instance], opened: date) -> list[Instance]:
    """Every instance resolved before ``opened``, oldest resolution first."""
    return sorted(
        (i for i in instances if i.resolved < opened),
        key=lambda i: (i.resolved, i.opened, i.key),
    )


def check_time_order(train: Sequence[Instance], opened: date) -> None:
    """Refuse a fit that would see an outcome not known when the question opened."""
    late = [i.key for i in train if i.resolved >= opened]
    if late:
        raise LeakageError(
            f"{len(late)} training outcome(s) resolved on or after {opened} "
            f"(first: {late[0]}); calibrate only on history resolved before a question opened"
        )


@dataclass(frozen=True)
class Fit:
    """One fit: the day its test instances opened and the history it saw."""

    opened: date
    n: int
    latest_resolution: date
    tests: int
    params: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Calibrated:
    """Test instances and each calibrator's forecast for them, in the same order."""

    tests: list[Instance]
    forecasts: dict[str, list[float]]
    fits: list[Fit]


def walk_forward(
    instances: Sequence[Instance],
    calibrators: Sequence[Calibrator] = CALIBRATORS,
    *,
    min_history: int,
    window: int | None = None,
    select: Callable[[Sequence[Instance], date], list[Instance]] = history,
) -> Calibrated:
    """Calibrate each instance with at least ``min_history`` earlier resolutions.

    Instances that opened on the same day share one fit. With ``window``,
    each fit uses only the ``window`` most recent resolutions. ``select``
    is the history rule (a parameter only so a test can prove a wrong one is
    refused).
    """
    days = sorted({i.opened for i in instances})
    tests: list[Instance] = []
    forecasts: dict[str, list[float]] = {c.name: [] for c in calibrators}
    fits: list[Fit] = []
    for day in days:
        train = select(instances, day)
        if len(train) < min_history:
            continue
        if window is not None:
            train = train[-window:]
        check_time_order(train, day)
        group = [i for i in instances if i.opened == day]
        ps = [i.p for i in train]
        ys = [i.outcome for i in train]
        maps: dict[str, CalibrationMap] = {c.name: c.fit(ps, ys) for c in calibrators}
        for name, m in maps.items():
            forecasts[name].extend(m(i.p) for i in group)
        tests.extend(group)
        fits.append(
            Fit(
                opened=day,
                n=len(train),
                latest_resolution=max(i.resolved for i in train),
                tests=len(group),
                params={name: m.params() for name, m in maps.items()},
            )
        )
    return Calibrated(tests, forecasts, fits)


# -- the measures -----------------------------------------------------------------


def briers(ps: Sequence[float], ys: Sequence[int]) -> list[float]:
    return [brier(clip(p), y) for p, y in zip(ps, ys, strict=True)]


def logs(ps: Sequence[float], ys: Sequence[int]) -> list[float]:
    return [log_score(clip(p) if y else 1.0 - clip(p)) for p, y in zip(ps, ys, strict=True)]


@dataclass(frozen=True)
class Bin:
    n: int
    mean_p: float
    freq: float


def equal_mass_bins(ps: Sequence[float], ys: Sequence[int], bins: int = BINS) -> list[Bin]:
    """``bins`` groups of forecasts sorted by value, sizes differing by at most one.

    Ties are kept in input order (a stable sort), so the split is deterministic.
    """
    if not ps:
        return []
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    k = min(bins, len(ps))
    out: list[Bin] = []
    for b in range(k):
        idx = order[b * len(ps) // k : (b + 1) * len(ps) // k]
        out.append(
            Bin(
                len(idx),
                math.fsum(ps[i] for i in idx) / len(idx),
                sum(ys[i] for i in idx) / len(idx),
            )
        )
    return out


def ece(ps: Sequence[float], ys: Sequence[int], bins: int = BINS) -> float:
    """Expected calibration error on equal-mass bins: Σ (n_b / N) |mean p_b − freq_b|."""
    groups = equal_mass_bins(ps, ys, bins)
    return math.fsum(g.n * abs(g.mean_p - g.freq) for g in groups) / len(ps)


def sharpness(ps: Sequence[float]) -> float:
    """The standard deviation of the forecasts (0 for a constant forecaster)."""
    return statistics.pstdev(ps)


def _est(e: Estimate, digits: int = 4) -> dict[str, float | int]:
    return {
        "mean": round(e.mean, digits),
        "low": round(e.low, digits),
        "high": round(e.high, digits),
        "n": e.n,
        "clusters": e.clusters,
        "mde": round(e.mde, digits) if math.isfinite(e.mde) else -1.0,
    }


def _clusters(tests: Sequence[Instance]) -> list[Hashable]:
    return [i.market for i in tests]


def _row(
    name: str, ps: Sequence[float], raw: Sequence[float], tests: Sequence[Instance]
) -> dict[str, Any]:
    ys = [i.outcome for i in tests]
    cl = _clusters(tests)
    clipped = [clip(p) for p in ps]
    return {
        "calibrator": name,
        "brier": _est(clustered_bootstrap(briers(ps, ys), cl)),
        "log": round(statistics.fmean(logs(ps, ys)), 4),
        "vs_raw_brier": _est(paired(briers(ps, ys), briers(raw, ys), cl)),
        "vs_raw_log": _est(paired(logs(ps, ys), logs(raw, ys), cl)),
        "ece": round(ece(clipped, ys), 4),
        "sharpness": round(sharpness(clipped), 4),
        "mean": round(statistics.fmean(clipped), 4),
        "bins": [
            {"n": b.n, "mean_p": round(b.mean_p, 4), "freq": round(b.freq, 4)}
            for b in equal_mass_bins(clipped, ys)
        ],
    }


def _edge(name: str, ps: Sequence[float], tests: Sequence[Instance]) -> list[dict[str, Any]]:
    """Brier(calibrated) − Brier(price at the freeze), apart by who can read the price."""
    out: list[dict[str, Any]] = []
    for group, readable in (("readable", True), ("hidden", False)):
        idx = [
            k
            for k, i in enumerate(tests)
            if (i.source in READABLE) == readable and i.price is not None
        ]
        if not idx:
            continue
        sub = [tests[k] for k in idx]
        ys = [i.outcome for i in sub]
        prices = [i.price for i in sub if i.price is not None]
        mine = [ps[k] for k in idx]
        out.append(
            {
                "calibrator": name,
                "group": group,
                "n": len(sub),
                "price_brier": round(statistics.fmean(briers(prices, ys)), 4),
                "edge": _est(paired(briers(mine, ys), briers(prices, ys), _clusters(sub))),
            }
        )
    return out


def verdict(e: dict[str, Any]) -> str:
    """M7's replacement rule on Brier(calibrator) − Brier(raw); no calibrator is cheaper."""
    if float(e["high"]) < 0.0:
        return "replaces raw (interval below 0)"
    return "raw stays"


# -- live -------------------------------------------------------------------------


def live_resolved(root: Path) -> dict[str, int]:
    """Resolved binary questions with a posted (live) Augur forecast, over every ledger."""
    forecasts = resolved = 0
    for path in sorted((root / "data/ledgers").glob("*.sqlite")):
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
            if "forecasts" not in tables:
                continue
            live = {
                int(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT f.question_id FROM forecasts f "
                    "JOIN runs r ON r.run_id = f.run_id "
                    "WHERE r.mode = 'live' AND f.kind = 'aggregate' "
                    "AND f.probability IS NOT NULL"
                )
            }
            forecasts += len(live)
            if "resolutions" in tables and live:
                done = {
                    int(r[0])
                    for r in conn.execute(
                        "SELECT DISTINCT question_id FROM resolutions WHERE status = 'resolved'"
                    )
                }
                resolved += len(live & done)
    return {"binary_forecasts": forecasts, "resolved": resolved}


# -- the snapshot -----------------------------------------------------------------


def _fits(c: Calibrated) -> list[dict[str, Any]]:
    return [
        {
            "opened": f.opened.isoformat(),
            "n": f.n,
            "latest_resolution": f.latest_resolution.isoformat(),
            "tests": f.tests,
        }
        for f in c.fits
    ]


def _design(c: Calibrated) -> dict[str, Any]:
    ys = [i.outcome for i in c.tests]
    raw = c.forecasts["raw"]
    rows = [_row(name, ps, raw, c.tests) for name, ps in c.forecasts.items()]
    for r in rows:
        r["verdict"] = "" if r["calibrator"] == "raw" else verdict(r["vs_raw_brier"])
    return {
        "instances": len(c.tests),
        "markets": len({i.market for i in c.tests}),
        "sets": sorted({i.question_set for i in c.tests}),
        "base_rate": round(sum(ys) / len(ys), 4),
        "fits": _fits(c),
        "rows": rows,
        "edge": [e for name, ps in c.forecasts.items() for e in _edge(name, ps, c.tests)],
    }


def snapshot(root: Path) -> dict[str, Any]:
    """Every number M8 reports, from the committed ledgers and ForecastBench subsets."""
    corpus = load_corpus(root)
    w = walk_forward(corpus, min_history=W_MIN)
    curve: list[dict[str, Any]] = []
    last: Calibrated | None = None
    for k in KS:
        last = walk_forward(corpus, min_history=L_MIN, window=k)
        ys = [i.outcome for i in last.tests]
        raw = last.forecasts["raw"]
        for name, ps in last.forecasts.items():
            if name == "raw":
                continue
            curve.append(
                {
                    "k": k,
                    "calibrator": name,
                    "vs_raw_brier": _est(
                        paired(briers(ps, ys), briers(raw, ys), _clusters(last.tests))
                    ),
                    "ece": round(ece([clip(p) for p in ps], ys), 4),
                }
            )
    assert last is not None
    ys_l = [i.outcome for i in last.tests]
    walk = _design(w)
    ys_w = [i.outcome for i in w.tests]
    walk["platt_vs_base_rate"] = _est(
        paired(
            briers(w.forecasts["platt"], ys_w),
            briers(w.forecasts["base-rate"], ys_w),
            _clusters(w.tests),
        )
    )
    return {
        "corpus": {
            "sets": list(SETS),
            "model": MODEL,
            "released": RELEASED.isoformat(),
            "instances": len(corpus),
            "markets": len({i.market for i in corpus}),
            "base_rate": round(sum(i.outcome for i in corpus) / len(corpus), 4),
            "by_set": {qs: sum(1 for i in corpus if i.question_set == qs) for qs in SETS},
        },
        "method": {
            "bootstrap": "paired, resampling markets, 10,000 resamples, seeded",
            "ece_bins": BINS,
            "w_min": W_MIN,
            "l_min": L_MIN,
            "ks": list(KS),
            "clip": [0.01, 0.99],
        },
        "walk_forward": walk,
        "learning_curve": {
            "instances": len(last.tests),
            "markets": len({i.market for i in last.tests}),
            "raw_brier": round(statistics.fmean(briers(last.forecasts["raw"], ys_l)), 4),
            "fits": _fits(last),
            "rows": curve,
        },
        "live": live_resolved(root),
    }


# -- the live choice --------------------------------------------------------------


def live_choice(snap: dict[str, Any]) -> str:
    """The calibrator M7's replacement rule picks on W, or ``raw`` if none passes."""
    winners = [
        r
        for r in snap["walk_forward"]["rows"]
        if r["calibrator"] != "raw" and float(r["vs_raw_brier"]["high"]) < 0.0
    ]
    if not winners:
        return "raw"
    return str(min(winners, key=lambda r: float(r["vs_raw_brier"]["mean"]))["calibrator"])


def live_calibrator(root: Path, snap: dict[str, Any]) -> dict[str, Any]:
    """The chosen calibrator fitted on every resolution in C4, for ``augur.livecal``."""
    corpus = load_corpus(root)
    name = live_choice(snap)
    fitted = by_name(name).fit([i.p for i in corpus], [i.outcome for i in corpus])
    latest = max(i.resolved for i in corpus)
    return {
        "model": MODEL,
        "fitted_on": f"M8 corpus C4: {len(corpus)} resolutions through {latest.isoformat()}, "
        "no retrieval, one draw",
        "params": fitted.params(),
    }
