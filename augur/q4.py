"""Q4: can calibration be learned from the bot's own resolved history?

M8's experiment (PLAN.md, "M8's predictions"), scored from the ledgers.

The corpus, C4, is nine ForecastBench sets (2026-06-21 to 2026-10-11) of
market questions that resolved to 0 or 1, forecast by one free model with
no retrieval and one draw. A *question instance* is one question in one
set: a market posed in three sets is three instances, forecast three times,
and the bootstrap resamples markets so those instances move together.

The rule that makes it a backtest, and the one thing this module must never
get wrong: **a test instance is calibrated only with instances whose outcome
was public before the day it opened.** The day an outcome became public is
the later of ForecastBench's resolution date (often only the market's close)
and the resolution time the market's own API reports
(``scripts/resolution_times.py``, committed in :data:`RESOLUTION_TIMES`); an
instance with no such evidence (INFER, which has no public API) is tested
but never trained on.
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

import json
import math
import sqlite3
import statistics
from collections.abc import Callable, Hashable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from augur.calibrate import CALIBRATORS, CalibrationMap, Calibrator, by_name
from augur.errors import AugurError, LeakageError
from augur.forecastbench import load_set
from augur.q3 import members
from augur.replay import clip
from augur.score import Estimate, brier, clustered_bootstrap, log_score, paired
from augur.sourcecal import SOURCED, SourcedCalibrator, SourcedMap

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
RESOLUTION_TIMES = Path("data/calibration/m8-resolution-times.json")

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
    """ForecastBench's resolution date (often the market's close)."""
    known: date | None
    """The day the outcome was public, on the market's own evidence; ``None``: no evidence."""
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
    times = _known_times(root / RESOLUTION_TIMES)
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
                    known=_known(q.resolution_date, times.get(f"{q.source}:{q.fb_id}")),
                    outcome=q.outcome,
                    p=raw[q.question_id],
                    price=q.freeze_value,
                )
            )
    return out


def _known_times(path: Path) -> dict[str, date | None]:
    if not path.exists():
        raise CalibrationDataError(
            f"no {path}; run python scripts/resolution_times.py {path} to record when each "
            "market's outcome became public"
        )
    markets = json.loads(path.read_text(encoding="utf-8"))["markets"]
    return {
        key: None if m["known_at"] is None else datetime.fromisoformat(m["known_at"]).date()
        for key, m in markets.items()
    }


def _known(resolution_date: date, evidence: date | None) -> date | None:
    return None if evidence is None else max(resolution_date, evidence)


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
    """Every instance whose outcome was public before ``opened``, oldest first."""
    known = [i for i in instances if i.known is not None and i.known < opened]
    return sorted(known, key=lambda i: (i.known, i.opened, i.key))


def check_time_order(train: Sequence[Instance], opened: date) -> None:
    """Refuse a fit that would see an outcome not known when the question opened."""
    late = [i.key for i in train if i.known is None or i.known >= opened]
    if late:
        raise LeakageError(
            f"{len(late)} training outcome(s) not shown to be public before {opened} "
            f"(first: {late[0]}); calibrate only on outcomes known before a question opened"
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
                latest_resolution=max(i.known for i in train if i.known is not None),
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


def _edge(
    name: str, ps: Sequence[float], raw: Sequence[float], tests: Sequence[Instance]
) -> list[dict[str, Any]]:
    """Brier(calibrated) − Brier(price at the freeze), apart by who can read the price.

    Each group also carries Brier(calibrated) − Brier(raw) on the same instances
    and the group's base rate: a map fitted on the pooled history can help one
    group and hurt the other.
    """
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
        own_raw = [raw[k] for k in idx]
        out.append(
            {
                "calibrator": name,
                "group": group,
                "n": len(sub),
                "base_rate": round(sum(ys) / len(ys), 4),
                "vs_raw_brier": _est(paired(briers(mine, ys), briers(own_raw, ys), _clusters(sub))),
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


# -- source-aware maps (the 2026-11-22 follow-up) ----------------------------------

SOURCES: tuple[str, ...] = ("polymarket", "manifold", "metaculus", "infer")
NON_INFERIORITY = 0.01
"""M7's non-inferiority margin, the source-aware decision rule's bound (PLAN.md)."""
WITHHELD = "metaculus"
"""The source a map must pass on before it reaches live forecasts (every live question)."""


def walk_forward_sourced(
    instances: Sequence[Instance],
    calibrators: Sequence[SourcedCalibrator] = SOURCED,
    *,
    min_history: int,
) -> Calibrated:
    """M8's walk-forward with maps that also see each instance's source.

    The same days, the same histories and the same time-order check as
    :func:`walk_forward`, so its tests are W's instances in W's order.
    """
    tests: list[Instance] = []
    forecasts: dict[str, list[float]] = {"raw": [], **{c.name: [] for c in calibrators}}
    fits: list[Fit] = []
    for day in sorted({i.opened for i in instances}):
        train = history(instances, day)
        if len(train) < min_history:
            continue
        check_time_order(train, day)
        group = [i for i in instances if i.opened == day]
        ps = [i.p for i in train]
        ys = [i.outcome for i in train]
        ss = [i.source for i in train]
        maps: dict[str, SourcedMap] = {c.name: c.fit(ps, ys, ss) for c in calibrators}
        forecasts["raw"].extend(clip(i.p) for i in group)
        for name, m in maps.items():
            forecasts[name].extend(m(i.p, i.source) for i in group)
        tests.extend(group)
        fits.append(
            Fit(
                opened=day,
                n=len(train),
                latest_resolution=max(i.known for i in train if i.known is not None),
                tests=len(group),
                params={name: m.params() for name, m in maps.items()},
            )
        )
    return Calibrated(tests, forecasts, fits)


def _by_source(c: Calibrated) -> list[dict[str, Any]]:
    """Each map against raw and against pooled Platt, on all of W and on each source."""
    out: list[dict[str, Any]] = []
    for group in ("all", *SOURCES):
        idx = [k for k, i in enumerate(c.tests) if group in ("all", i.source)]
        if not idx:
            continue
        sub = [c.tests[k] for k in idx]
        ys = [i.outcome for i in sub]
        cl = _clusters(sub)
        own = {name: [ps[k] for k in idx] for name, ps in c.forecasts.items()}
        raw_b, platt_b = briers(own["raw"], ys), briers(own["platt"], ys)
        rows = []
        for name, ps in own.items():
            b = briers(ps, ys)
            rows.append(
                {
                    "calibrator": name,
                    "brier": round(statistics.fmean(b), 4),
                    "mean": round(statistics.fmean(clip(p) for p in ps), 4),
                    "vs_raw_brier": _est(paired(b, raw_b, cl)),
                    "vs_platt_brier": _est(paired(b, platt_b, cl)),
                }
            )
        out.append(
            {
                "group": group,
                "n": len(sub),
                "markets": len({i.market for i in sub}),
                "base_rate": round(sum(ys) / len(ys), 4),
                "rows": rows,
            }
        )
    return out


def source_choice(snap: dict[str, Any]) -> str | None:
    """The source-aware map the decision rule admits, or ``None``.

    (i) On W's Metaculus instances, Brier(map) − Brier(raw) has its upper
    bound below +0.01; (ii) on all of W, Brier(map) − Brier(pooled Platt)
    has its upper bound below +0.01. The lower overall point estimate wins.
    """
    groups = {g["group"]: {r["calibrator"]: r for r in g["rows"]} for g in snap["by_source"]}
    passing = [
        name
        for name in groups["all"]
        if name not in ("raw", "platt")
        and float(groups[WITHHELD][name]["vs_raw_brier"]["high"]) < NON_INFERIORITY
        and float(groups["all"][name]["vs_platt_brier"]["high"]) < NON_INFERIORITY
    ]
    if not passing:
        return None
    return str(min(passing, key=lambda n: float(groups["all"][n]["vs_raw_brier"]["mean"])))


def source_aware(root: Path, corpus: Sequence[Instance] | None = None) -> dict[str, Any]:
    """The source-aware comparison on W, by source, and the map the rule admits."""
    c = walk_forward_sourced(load_corpus(root) if corpus is None else corpus, min_history=W_MIN)
    partial = [f.params["platt-partial"] for f in c.fits]
    snap: dict[str, Any] = {
        "instances": len(c.tests),
        "markets": len({i.market for i in c.tests}),
        "fits": len(c.fits),
        "partial_last_fit": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in partial[-1].items()
            if k != "offsets"
        }
        | {"offsets": {s: round(v, 4) for s, v in partial[-1]["offsets"].items()}},
        "by_source": _by_source(c),
    }
    snap["choice"] = source_choice(snap)
    return snap


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
        "edge": [e for name, ps in c.forecasts.items() for e in _edge(name, ps, raw, c.tests)],
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
            "resolution_evidence": {
                "markets": len({i.market for i in corpus if i.known is not None}),
                "later_than_forecastbench": len(
                    {i.market for i in corpus if i.known is not None and i.known > i.resolved}
                ),
                "none": len({i.market for i in corpus if i.known is None}),
                "no_evidence_instances": sum(1 for i in corpus if i.known is None),
            },
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
        "source_aware": source_aware(root, corpus),
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
    """The chosen calibrator fitted on every resolution in C4, for ``augur.livecal``.

    A source-aware map that passes the 2026-11-22 rule replaces the pooled
    choice and applies everywhere; otherwise the pooled choice is kept and
    withheld from Metaculus questions, on which it has not been shown to
    be no worse than raw.
    """
    corpus = load_corpus(root)
    latest = max(i.known or i.resolved for i in corpus)
    ps, ys = [i.p for i in corpus], [i.outcome for i in corpus]
    sourced = snap.get("source_aware", {}).get("choice")
    params: dict[str, Any]
    if sourced:
        cal = next(c for c in SOURCED if c.name == sourced)
        params = cal.fit(ps, ys, [i.source for i in corpus]).params()
        withheld: list[str] = []
    else:
        params = by_name(live_choice(snap)).fit(ps, ys).params()
        withheld = [] if params["name"] == "raw" else [WITHHELD]
    return {
        "model": MODEL,
        "fitted_on": f"M8 corpus C4: {len(corpus)} resolutions through {latest.isoformat()}, "
        "no retrieval, one draw",
        "params": params,
        "withheld_from": withheld,
    }
