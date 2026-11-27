"""M18's rows: the day-of-year arms beside M15's arms and ForecastBench's entrants.

Reads only committed files, calls no model and no network:

* M15's dataset extract and comparator cut (through
  :func:`augur.fbdataset_eval.build_rows`), for the corpus replication;
* the observations ledger (:data:`augur.fbseason.LEDGER`), through
  :func:`augur.fbseason.ledger_observations` (admitted sources only);
* the confirmatory manifest and P65's instance list (``data/forecastbench/m18``);
* the hold-out sets' extract and their baselines' forecasts
  (``m18/holdout``, ``m18/forecast-sets``);
* the later re-cuts that carry the confirmatory outcomes as they resolve
  (``m18/later``, ``m18/later-forecasts``), when they exist.

Every difference is Brier(first) − Brier(second), paired by instance, with a
95% interval from 10,000 bootstrap resamples (seed 20261119) of clusters
(PLAN.md, ``m18-registration``, item 9). ``augur fbcompare`` writes
:data:`SNAPSHOT`; ``--check`` re-runs and compares byte for byte, as ``make
eval`` does; ``augur report`` renders it.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from augur import fbdataset, fbdataset_eval, fbseason
from augur.clock import ensure_utc, parse_iso
from augur.fbdataset import DQuestion, History, arm_b, arm_c, brier
from augur.fbseason import SeasonError
from augur.score import DEFAULT_RESAMPLES, DEFAULT_SEED, paired_many

SNAPSHOT = Path("data/forecastbench/m18-seasonal.json")
HOLDOUT_DIR = fbseason.M18_DIR / "holdout"
HOLDOUT_FORECASTS = fbseason.M18_DIR / "forecast-sets"
LATER_DIR = fbseason.M18_DIR / "later"
LATER_FORECASTS = fbseason.M18_DIR / "later-forecasts"
GATE = fbseason.M18_DIR / "gate.json"
HOLDOUT_SETS = ("2026-10-11", "2026-10-25")
HORIZONS = (7, 30, 90, 180, 365)
FEW_CLUSTERS = 8
LABEL = (
    "a backtest row in raw Brier on matched instances, not a Brier Index; "
    "BLF's 59.1 against 50.0 on DBnomics (arXiv 2604.18576v5, appendix G.1) is a "
    "different sample and metric, quoted and not compared"
)
REPLICATION_LABEL = (
    "a replication of a seen result: the 799 corpus instances had been looked at "
    "(0.1740 in the plan's scratch run); no prediction is read on them"
)
ENTRANTS = ("Naive", "stat-v1", "hybrid ensemble 1", "model-4")


def horizon_of(due: date, r: date) -> int:
    days = (r - due).days
    for h in HORIZONS:
        if abs(days - h) <= 5:
            return h
    raise SeasonError(f"{due} to {r} is {days} days, not a known horizon")


# -- the arms on any (question, date) ------------------------------------------------------


@dataclass(frozen=True)
class SeasonArms:
    s: float
    s_prime: float
    k: float
    b: float
    shifts: int
    exceed_n: int
    pooled_change: float | None


def season_arms(
    targets: Sequence[tuple[DQuestion, date]],
    obs: Mapping[int, Mapping[date, float]],
    history: History,
    stations: Mapping[str, Sequence[DQuestion]],
) -> dict[tuple[str, str, date], SeasonArms]:
    """S, S′, K and B on each Meteo-France (question, r); ``stations`` by set (item 6)."""
    k_cache: dict[tuple[str, date], tuple[float, float | None]] = {}

    def k_for(qs: str, r: date) -> tuple[float, float | None]:
        if (qs, r) not in k_cache:
            changes = []
            for q in stations.get(qs, ()):
                got = obs.get(q.question_id)
                if got is not None:
                    changes.append(
                        fbseason.station_change(fbseason.usable_shifts(got, q.due, r, q.freeze))
                    )
            k_cache[(qs, r)] = fbseason.arm_k(changes)
        return k_cache[(qs, r)]

    out: dict[tuple[str, str, date], SeasonArms] = {}
    for q, r in targets:
        if not fbseason.is_meteo(q):
            raise SeasonError(f"{q.series} is not a Meteo-France question")
        b = arm_b(history.at(q.freeze), q.source)
        got = obs.get(q.question_id, {})
        shifts = fbseason.usable_shifts(got, q.due, r, q.freeze)
        ex = fbseason.exceedance(got, r, q.freeze, q.freeze_value)
        k, pooled = k_for(q.question_set, r)
        out[(q.question_set, q.id, r)] = SeasonArms(
            s=fbseason.arm_s(shifts, b),
            s_prime=fbseason.arm_s_prime(ex, b),
            k=k,
            b=b,
            shifts=len(shifts),
            exceed_n=0 if ex is None else ex.n,
            pooled_change=pooled,
        )
    return out


def meteo_stations(questions: Sequence[DQuestion]) -> dict[str, list[DQuestion]]:
    out: dict[str, list[DQuestion]] = {}
    for q in questions:
        if fbseason.is_meteo(q):
            out.setdefault(q.question_set, []).append(q)
    return out


# -- statistics ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One scored instance: its forecasters' values, outcome and clusters."""

    key: tuple[str, str, str, str]
    y: int
    values: dict[str, float]
    series: str
    cluster: str
    """(set, horizon) on the confirmatory set; the series elsewhere."""
    horizon: int
    meteo: bool


def _r(x: float, digits: int = 4) -> float:
    return round(x, digits)


def _with(items: Sequence[Item], *names: str) -> list[Item]:
    return [i for i in items if all(n in i.values for n in names)]


def _briers(items: Sequence[Item], names: Sequence[str]) -> dict[str, float]:
    return {n: _r(statistics.fmean(brier(i.values[n], i.y) for i in items)) for n in names}


def _diffs(
    wants: Sequence[tuple[str, str, str, Sequence[Item]]], *, by: str
) -> list[dict[str, Any]]:
    """Each wanted difference; ``by`` is ``series`` or ``cluster`` (with the question level)."""
    pairs: list[tuple[list[float], list[float], list[Hashable] | None]] = []
    for _g, first, second, items in wants:
        a = [brier(i.values[first], i.y) for i in items]
        b = [brier(i.values[second], i.y) for i in items]
        pairs.append((a, b, [getattr(i, by) for i in items]))
        if by == "cluster":
            pairs.append((a, b, None))
    ests = paired_many(pairs)
    step = 2 if by == "cluster" else 1
    out = []
    for n, (group, first, second, items) in enumerate(wants):
        if not items:
            out.append({"group": group, "first": first, "second": second, "n": 0})
            continue
        e = ests[step * n]
        row: dict[str, Any] = {
            "group": group,
            "first": first,
            "second": second,
            "n": e.n,
            "mean": e.mean,
            "low": e.low,
            "high": e.high,
            "clusters": e.clusters,
        }
        if by == "cluster":
            q = ests[step * n + 1]
            row["question_level"] = {"low": q.low, "high": q.high, "n": q.n}
        out.append(row)
    return out


def _nonempty(
    wants: Sequence[tuple[str, str, str, Sequence[Item]]],
) -> list[tuple[str, str, str, Sequence[Item]]]:
    return [w for w in wants if w[3]]


def _find(rows: Sequence[dict[str, Any]], group: str, first: str, second: str) -> dict[str, Any]:
    for c in rows:
        if (c["group"], c["first"], c["second"]) == (group, first, second):
            return c
    raise SeasonError(f"internal: no comparison {group} {first} − {second}")


# -- the corpus replication -----------------------------------------------------------------


def replication(root: Path, obs: Mapping[int, Mapping[date, float]]) -> dict[str, Any]:
    """M15's corpus with S, S′, K and B-with-S added (item 13): seen, so no verdict."""
    rows, meta = fbdataset_eval.build_rows(root)
    loaded: fbdataset.Loaded = meta["loaded"]
    history = History(loaded.questions)
    stations = meteo_stations(loaded.questions)
    targets = [
        (r.inst.question, r.inst.resolution_date)
        for r in rows
        if fbseason.is_meteo(r.inst.question)
    ]
    arms = season_arms(targets, obs, history, stations)
    items: list[Item] = []
    for r in rows:
        q = r.inst.question
        values = dict(r.values)
        meteo = fbseason.is_meteo(q)
        if meteo:
            a = arms[(q.question_set, q.id, r.inst.resolution_date)]
            if abs(a.b - values["B"]) > 1e-12:
                raise SeasonError(f"{r.inst.key}: B differs from M15's")
            values.update({"S": a.s, "S′": a.s_prime, "K": a.k, "B+S": a.s})
        else:
            values["B+S"] = values["B"]
        items.append(
            Item(
                key=r.inst.key,
                y=r.y,
                values=values,
                series=q.series,
                cluster=q.series,
                horizon=horizon_of(q.due, r.inst.resolution_date),
                meteo=meteo,
            )
        )
    meteo_items = [i for i in items if i.meteo]
    arm_names = ["A", "B", "S", "S′", "K"]
    by_horizon = []
    for h in (None, 7, 30, 90):
        sub = [i for i in meteo_items if h is None or i.horizon == h]
        by_horizon.append(
            {
                "horizon": "all" if h is None else h,
                "n": len(sub),
                "yes_share": _r(statistics.fmean(i.y for i in sub)),
                "brier": _briers(sub, arm_names),
            }
        )
    wants: list[tuple[str, str, str, Sequence[Item]]] = []
    for h in (None, 7, 30, 90):
        sub = [i for i in meteo_items if h is None or i.horizon == h]
        g = "meteo" if h is None else f"meteo {h}d"
        for first, second in (("S", "B"), ("S", "A"), ("S", "K"), ("S′", "S"), ("S′", "A")):
            wants.append((g, first, second, sub))
    entrants = []
    for name in ENTRANTS:
        sub = _with(meteo_items, name)
        if not sub:
            continue
        entrants.append(
            {
                "entrant": name,
                "n": len(sub),
                "sets": sorted({i.key[0] for i in sub}),
                "brier": _briers(sub, [name, *arm_names]),
            }
        )
        wants.append(("meteo", "S", name, sub))
        wants.append(("meteo", "S′", name, sub))
    pooled_rows = []
    for group, sub in (
        ("pooled", items),
        ("dbnomics", [i for i in items if i.key[1] == "dbnomics"]),
    ):
        pooled_rows.append(
            {
                "group": group,
                "n": len(sub),
                "brier": _briers(sub, ["A", "B", "B+S", "Naive"]),
            }
        )
        wants.append((group, "B+S", "B", sub))
        wants.append((group, "B+S", "A", sub))
        wants.append((group, "B+S", "Naive", _with(sub, "Naive")))
        for name in ENTRANTS[1:]:
            got = _with(sub, name)
            if got:
                wants.append((group, "B+S", name, got))
    gate_seen = _with(items, "stat-v1")
    gate_temp = [i for i in gate_seen if i.meteo]
    wants.append(("gate seen", "B+S", "stat-v1", gate_seen))
    wants.append(("gate seen, temperatures", "B+S", "stat-v1", gate_temp))
    for src in fbdataset.SOURCES:
        sub = [i for i in gate_seen if i.key[1] == src and not i.meteo]
        label = "ECB series" if src == "dbnomics" else src
        wants.append((f"gate seen, {label}", "B+S", "stat-v1", sub))
    diffs = _diffs(_nonempty(wants), by="series")
    not_reached = {
        "S": sum(1 for a in arms.values() if a.shifts == 0),
        "S′": sum(1 for a in arms.values() if a.exceed_n == 0),
    }
    shifts = [a.shifts for a in arms.values()]
    return {
        "label": REPLICATION_LABEL,
        "instances": len(meteo_items),
        "stations": len({i.series for i in meteo_items}),
        "not_reached": not_reached,
        "shifts": {"min": min(shifts), "median": statistics.median(shifts), "max": max(shifts)},
        "by_horizon": by_horizon,
        "entrants": entrants,
        "pooled": pooled_rows,
        "comparisons": diffs,
    }


# -- the hold-out sets ---------------------------------------------------------------------


def load_extract(directory: Path, role: str) -> list[DQuestion]:
    """Every question of the extract files of ``role`` in ``directory``."""
    out: list[DQuestion] = []
    for path in sorted(directory.glob("2026-*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("role") != role or doc.get("forecast_due_date") != path.stem:
            raise SeasonError(f"{path} is not a {role} set of the extract")
        for raw in doc["questions"]:
            out.append(
                DQuestion(
                    question_set=path.stem,
                    source=str(raw["source"]),
                    id=str(raw["id"]),
                    freeze=ensure_utc(parse_iso(str(raw["freeze_datetime"]))),
                    freeze_value=raw.get("freeze_value"),
                    url=str(raw.get("url") or ""),
                    resolutions=tuple(
                        (date.fromisoformat(d), int(y))
                        for d, y in sorted(raw["resolutions"].items())
                    ),
                )
            )
    return out


def _removed(directory: Path) -> dict[str, dict[str, int]]:
    return {
        p.stem: dict(json.loads(p.read_text(encoding="utf-8"))["removed"])
        for p in sorted(directory.glob("2026-*.json"))
    }


def _baseline_forecasts(directory: Path) -> dict[tuple[str, str, str, str], dict[str, float]]:
    """ForecastBench's baselines (and named entries) from a forecast-set cut, by instance."""
    out: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for path in sorted(directory.glob("2026-*.json")):
        fs = fbdataset_eval._Set.model_validate_json(path.read_text(encoding="utf-8"))
        names = [fbdataset_eval._short(e) for e in fs.entrants]
        for raw in fs.instances:
            key = (fs.question_set, raw.source, raw.id, raw.resolution_date)
            got = out.setdefault(key, {})
            for j, name in enumerate(names):
                p = raw.forecasts[j]
                if name is None or p is None:
                    continue
                if j in raw.imputed and name not in fbdataset_eval.BASELINES.values():
                    continue
                got[name] = p
    return out


def holdout(root: Path, loaded: fbdataset.Loaded) -> dict[str, Any]:
    """P66: M15's arms on the 2026-10-11 and 2026-10-25 sets, beside Naive Forecaster."""
    directory = root / HOLDOUT_DIR
    questions = load_extract(directory, "holdout")
    if sorted({q.question_set for q in questions}) != list(HOLDOUT_SETS):
        raise SeasonError(f"{directory} must hold the sets {', '.join(HOLDOUT_SETS)}")
    history = History([*loaded.questions, *questions])
    forecasts = _baseline_forecasts(root / HOLDOUT_FORECASTS)
    items: list[Item] = []
    for q in questions:
        known = history.at(q.freeze)
        if known.latest is not None and known.latest > History.known_on(q.freeze.date()):
            raise SeasonError(f"{q.series}: an outcome known after its freeze was counted")
        b = arm_b(known, q.source)
        c = arm_c(known, q.series, b)
        for r, y in q.resolutions:
            key = (q.question_set, q.source, q.id, r.isoformat())
            values = {"A": 0.5, "B": b, "C": c, **forecasts.get(key, {})}
            items.append(Item(key, y, values, q.series, q.series, horizon_of(q.due, r), False))
    groups: list[tuple[str, list[Item]]] = [("pooled", items)]
    groups += [(s, [i for i in items if i.key[1] == s]) for s in ("acled", "dbnomics", "fred")]
    groups += [(s, [i for i in items if i.key[0] == s]) for s in HOLDOUT_SETS]
    table = []
    wants: list[tuple[str, str, str, Sequence[Item]]] = []
    for g, sub in groups:
        names = ["A", "B", "C"] + [
            n for n in ("Naive", "Imputed") if all(n in i.values for i in sub)
        ]
        table.append(
            {
                "group": g,
                "n": len(sub),
                "yes_share": _r(statistics.fmean(i.y for i in sub)) if sub else None,
                "brier": _briers(sub, names) if sub else {},
            }
        )
        for first, second in (("B", "A"), ("B", "Naive"), ("C", "B"), ("Naive", "A")):
            wants.append((g, first, second, _with(sub, first, second)))
    diffs = _diffs(_nonempty(wants), by="series")
    bn = _find(diffs, "pooled", "B", "Naive")
    held = bn["mean"] <= -0.02 and bn["high"] < 0
    return {
        "sets": list(HOLDOUT_SETS),
        "instances": len(items),
        "series": len({i.series for i in items}),
        "left_out": _removed(directory),
        "temperature_questions": sum(1 for q in questions if fbseason.is_meteo(q)),
        "table": table,
        "comparisons": diffs,
        "p66": {
            "verdict": "held" if held else "failed",
            "reading": f"B − Naive {bn['mean']:+.5f} [{bn['low']:+.5f}, {bn['high']:+.5f}] on "
            f"{bn['n']} instances, {bn['clusters']} series",
        },
    }


# -- the confirmatory set ----------------------------------------------------------------------


def later_outcomes(root: Path) -> tuple[dict[tuple[str, str, str, str], int], str | None]:
    """Outcomes in the later re-cuts, by instance, and the latest fetch day among them."""
    directory = root / LATER_DIR
    if not directory.exists():
        return {}, None
    out: dict[tuple[str, str, str, str], int] = {}
    fetched: list[str] = []
    for path in sorted(directory.glob("2026-*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("role") != "later":
            raise SeasonError(f"{path} is not a later re-cut")
        fetched.append(str(doc["provenance"]["fetched"]))
        for q in doc["questions"]:
            for d, y in q["resolutions"].items():
                out[(path.stem, str(q["source"]), str(q["id"]), d)] = int(y)
    return out, (max(fetched) if fetched else None)


def _readable(as_of: str | None) -> bool:
    return as_of is not None and date.fromisoformat(as_of) >= fbseason.READING_DAY


def confirmatory(
    root: Path, loaded: fbdataset.Loaded, obs: Mapping[int, Mapping[date, float]]
) -> dict[str, Any]:
    """The 313 instances: every arm's forecast, pinned now; scored as outcomes appear."""
    manifest = fbseason.load_manifest(root)
    by_q = {(q.question_set, q.id): q for q in loaded.questions}
    history = History(loaded.questions)
    stations = meteo_stations(loaded.questions)
    targets = [(by_q[(p.question_set, p.id)], p.resolution_date) for p in manifest.instances]
    arms = season_arms(targets, obs, history, stations)
    outcomes, as_of = later_outcomes(root)
    forecasts = []
    items: list[Item] = []
    for p in manifest.instances:
        a = arms[(p.question_set, p.id, p.resolution_date)]
        if horizon_of(date.fromisoformat(p.question_set), p.resolution_date) != p.horizon:
            raise SeasonError(f"{p.key}: the manifest's horizon is wrong")
        forecasts.append(
            {
                "question_set": p.question_set,
                "id": p.id,
                "resolution_date": p.resolution_date.isoformat(),
                "horizon": p.horizon,
                "S": _r(a.s, 6),
                "S′": _r(a.s_prime, 6),
                "K": a.k,
                "B": _r(a.b, 6),
                "shifts": a.shifts,
            }
        )
        y = outcomes.get(p.key)
        if y is not None:
            values = {"A": 0.5, "B": a.b, "S": a.s, "S′": a.s_prime, "K": a.k}
            items.append(
                Item(
                    p.key,
                    y,
                    values,
                    f"dbnomics:{p.id}",
                    f"{p.question_set}/{p.horizon}",
                    p.horizon,
                    True,
                )
            )
    strata = []
    wants: list[tuple[str, str, str, Sequence[Item]]] = []
    for h in (None, 90, 180, 365):
        planned = [p for p in manifest.instances if h is None or p.horizon == h]
        sub = [i for i in items if h is None or i.horizon == h]
        g = "all" if h is None else f"{h}d"
        strata.append(
            {
                "stratum": g,
                "planned": len(planned),
                "published": len(sub),
                "clusters": len({f"{p.question_set}/{p.horizon}" for p in planned}),
                "brier": _briers(sub, ["A", "B", "S", "S′", "K"]) if sub else {},
            }
        )
        for first, second in (("S", "K"), ("S", "B"), ("S′", "S"), ("S", "A")):
            wants.append((g, first, second, sub))
    diffs = _diffs(_nonempty(wants), by="cluster")
    gate = gate_reading(root, loaded, obs, outcomes)
    return {
        "planned": len(manifest.instances),
        "left_out": manifest.left_out,
        "window": [d.isoformat() for d in manifest.window],
        "reading_day": fbseason.READING_DAY.isoformat(),
        "outcomes_as_of": as_of,
        "kind": "registered reading" if _readable(as_of) else "progress reading",
        "k_up": sum(1 for f in forecasts if f["K"] == fbseason.K_UP),
        "s_not_reached": sum(1 for f in forecasts if f["shifts"] == 0),
        "strata": strata,
        "comparisons": diffs,
        "gate": gate,
        "hypotheses": hypotheses(strata, diffs, gate, as_of),
        "forecasts": forecasts,
    }


def gate_reading(
    root: Path,
    loaded: fbdataset.Loaded,
    obs: Mapping[int, Mapping[date, float]],
    outcomes: Mapping[tuple[str, str, str, str], int],
) -> dict[str, Any]:
    """P65 on the 2026-09-27 set's instances resolving in the window."""
    doc = json.loads((root / GATE).read_text(encoding="utf-8"))
    qs = str(doc["question_set"])
    by_q = {(q.source, q.id): q for q in loaded.questions if q.question_set == qs}
    history = History(loaded.questions)
    stations = meteo_stations(loaded.questions)
    stat = _baseline_forecasts(root / LATER_FORECASTS) if (root / LATER_FORECASTS).exists() else {}
    items: list[Item] = []
    for row in doc["instances"]:
        q = by_q[(str(row["source"]), str(row["id"]))]
        r = date.fromisoformat(str(row["resolution_date"]))
        key = (qs, q.source, q.id, r.isoformat())
        y = outcomes.get(key)
        p = stat.get(key, {}).get("stat-v1")
        if y is None or p is None:
            continue
        if fbseason.is_meteo(q):
            bs = season_arms([(q, r)], obs, history, stations)[(qs, q.id, r)].s
        else:
            bs = arm_b(history.at(q.freeze), q.source)
        items.append(
            Item(
                key,
                y,
                {"B+S": bs, "stat-v1": p},
                q.series,
                q.series,
                horizon_of(q.due, r),
                fbseason.is_meteo(q),
            )
        )
    wants = _nonempty(
        [
            ("gate", "B+S", "stat-v1", items),
            ("gate, temperatures", "B+S", "stat-v1", [i for i in items if i.meteo]),
        ]
    )
    return {
        "question_set": qs,
        "planned": len(doc["instances"]),
        "left_out": len(doc["left_out"]),
        "scored": len(items),
        "comparisons": _diffs(wants, by="series"),
    }


def _ci(c: dict[str, Any]) -> tuple[float, float]:
    """The interval a verdict reads: clustered, or question level with one cluster (item 9)."""
    if c.get("clusters", 0) < 2 and "question_level" in c:
        return float(c["question_level"]["low"]), float(c["question_level"]["high"])
    return float(c["low"]), float(c["high"])


def hypotheses(
    strata: Sequence[dict[str, Any]],
    diffs: Sequence[dict[str, Any]],
    gate: dict[str, Any],
    as_of: str | None,
) -> list[dict[str, Any]]:
    """P62 to P65 (item 12): not yet readable before the reading day or with no outcome."""
    by = {s["stratum"]: s for s in strata}
    readable = _readable(as_of)

    def mark(ok: bool, n: int) -> str:
        if not readable or n == 0:
            return "not yet readable"
        return "held" if ok else "failed"

    out = []
    s_all = by["all"]
    if s_all["published"]:
        sb = _find(diffs, "all", "S", "B")
        ok = s_all["brier"]["S"] <= 0.12 and sb["mean"] <= -0.15 and _ci(sb)[1] < 0
        reading = f"Brier(S) {s_all['brier']['S']:.4f}; S − B {sb['mean']:+.4f} on {sb['n']}"
    else:
        ok, reading = False, "no confirmatory outcome published"
    out.append({"id": "P62", "verdict": mark(ok, s_all["published"]), "reading": reading})
    parts, oks, n = [], [], 0
    for g in ("90d", "180d"):
        if by[g]["published"]:
            c = _find(diffs, g, "S", "K")
            oks.append(-0.02 <= c["mean"] <= 0.02)
            parts.append(f"{g} S − K {c['mean']:+.4f} on {c['n']}")
            n += 1
        else:
            parts.append(f"{g}: none published")
    out.append(
        {
            "id": "P63",
            "verdict": mark(all(oks) and n == 2, n if n == 2 else 0),
            "reading": "; ".join(parts),
        }
    )
    one = by["365d"]
    if one["published"]:
        ok = abs(one["brier"]["S"] - 0.25) <= 0.03 and one["brier"]["S′"] < one["brier"]["S"]
        reading = f"Brier(S) {one['brier']['S']:.4f}, Brier(S′) {one['brier']['S′']:.4f}"
    else:
        ok, reading = False, "none published"
    out.append({"id": "P64", "verdict": mark(ok, one["published"]), "reading": reading})
    if gate["scored"]:
        gc = _find(gate["comparisons"], "gate", "B+S", "stat-v1")
        ok = gc["high"] < 0.01
        reading = (
            f"B+S − stat-v1 {gc['mean']:+.4f} [{gc['low']:+.4f}, {gc['high']:+.4f}] on {gc['n']}"
        )
    else:
        ok, reading = False, f"0 of {gate['planned']} instances scored (no outcome or forecast yet)"
    out.append({"id": "P65", "verdict": mark(ok, gate["scored"]), "reading": reading})
    return out


# -- the snapshot --------------------------------------------------------------------------------


def snapshot(root: Path) -> dict[str, Any]:
    """Every number M18's rows report."""
    obs = fbseason.ledger_observations(root / fbseason.LEDGER)
    loaded = fbdataset.load(root)
    rep = replication(root, obs)
    return {
        "label": LABEL,
        "method": {
            "statistic": "Brier(first) − Brier(second), paired by instance; negative: "
            "first is better",
            "bootstrap": f"95%, {DEFAULT_RESAMPLES:,} resamples, seed {DEFAULT_SEED}; "
            "clusters: series (replication, hold-out, gate), (set, horizon) on the "
            "confirmatory set with the question-level interval beside it",
            "arms": {
                "S": "(wins + 1) / (n + 2) over earlier years: mean within 3 days of r against "
                "mean within 3 days of the due date, both shifted back k ≥ 1 years",
                "S′": "(#{x > freeze value} + 1) / (N + 2) over observations within 10 days of "
                "r in earlier years",
                "K": "0.9 if the station-pooled mean change between the two days is above 0, "
                "0.1 if below",
                "B+S": "S on Meteo-France instances, M15's B elsewhere",
            },
        },
        "provenance": {
            "observations_ledger": str(fbseason.LEDGER),
            "stations": len({q.id for q in loaded.questions if fbseason.is_meteo(q)}),
        },
        "replication": rep,
        "holdout": holdout(root, loaded),
        "confirmatory": confirmatory(root, loaded, obs),
    }


def dumps(snap: dict[str, Any]) -> str:
    return json.dumps(snap, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
