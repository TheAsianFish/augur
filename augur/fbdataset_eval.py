"""M15's same-instance row: the no-model arms beside ForecastBench's own entrants.

Reads only committed files, calls no model and no network:

* the dataset extract (:data:`augur.fbdataset.DATASET_DIR`), for the corpus
  and the walk-forward history;
* the closes ledger (:data:`augur.fbdataset.LEDGER`), for arm D, through
  :func:`augur.fbdataset.ledger_closes` (admitted sources only);
* the comparators' forecasts on the same instances (:data:`SETS_DIR`, cut
  from ForecastBench's processed forecast sets by
  ``scripts/extract_forecast_sets.py --dataset``).

Every difference is Brier(first) − Brier(second), paired by instance, with a
95% interval from 10,000 bootstrap resamples (seed 20261119) of clusters:
once by series and once by resolution date (PLAN.md, ``m15-registration``,
item 9). ``augur fbcompare`` writes :data:`SNAPSHOT`; ``--check`` re-runs and
compares byte for byte, as ``make eval`` does; ``augur report`` renders it.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from augur import fbdataset
from augur.fbdataset import Forecasts, Instance, brier
from augur.score import DEFAULT_RESAMPLES, DEFAULT_SEED, Estimate, paired_many

SETS_DIR = Path("data/forecastbench/forecast-sets-dataset")
SNAPSHOT = Path("data/forecastbench/m15-dataset.json")
LABEL = (
    "a backtest row in raw Brier on matched instances, not a Brier Index: "
    "ForecastBench's index rescales against its baselines and imputes missing forecasts"
)
BASELINES: dict[str, str] = {
    "naive-forecaster": "Naive",
    "imputed-forecaster": "Imputed",
    "always-0.5": "Always 0.5",
}
"""ForecastBench's baselines by file stem, and their short names here."""
ENTRANTS: dict[tuple[str, str], tuple[str, str]] = {
    ("Anonymous 24", "model-4"): ("model-4", "leader"),
    ("NeuroFusion", "hybrid ensemble 1"): ("hybrid ensemble 1", "leader"),
    ("Anonymous 39", "stat-v1 (calibrated market anchor + dataset base rates)"): (
        "stat-v1",
        "no model",
    ),
    ("Anonymous 23", "undertow-mix (statbase x gpt-5.6-sol)"): ("undertow-mix", "mixed"),
}
"""The named external entries (item 8): short name and kind."""
LEADERS = ("model-4", "hybrid ensemble 1")
POOLED_PAIRS: tuple[tuple[str, str], ...] = (
    ("B", "A"),
    ("C", "A"),
    ("C", "B"),
    ("B", "Naive"),
    ("C", "Naive"),
    ("Naive", "A"),
)
YFINANCE_PAIRS: tuple[tuple[str, str], ...] = (("D", "B"), ("D", "A"), ("D", "Naive"))
ENTRANT_SECONDS: tuple[str, ...] = ("A", "B", "C", "Naive")


class M15Error(fbdataset.DatasetError):
    """The committed comparator extract does not match the dataset extract."""


class _Entrant(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    file: str
    organization: str
    model: str


class _Instance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    source: str
    id: str
    resolution_date: str
    resolved_to: float | None = None
    forecasts: list[float | None]
    imputed: list[int]


class _Set(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    question_set: str
    entrants: list[_Entrant]
    instances: list[_Instance]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Row:
    """One instance with every forecaster that has a forecast for it."""

    f: Forecasts
    values: dict[str, float]

    @property
    def inst(self) -> Instance:
        return self.f.instance

    @property
    def y(self) -> int:
        return self.inst.outcome


def _short(e: _Entrant) -> str | None:
    if e.file.startswith("ForecastBench."):
        return BASELINES.get(e.file[len("ForecastBench.") : -5])
    named = ENTRANTS.get((e.organization, e.model))
    return None if named is None else named[0]


def build_rows(root: Path) -> tuple[list[Row], dict[str, Any]]:
    """Every corpus instance with the arms and the comparators' published forecasts."""
    loaded = fbdataset.load(root)
    closes = fbdataset.ledger_closes(root / fbdataset.LEDGER)
    forecasts = fbdataset.forecast_all(loaded, closes)
    by_key = {f.instance.key: f for f in forecasts}
    values: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for f in forecasts:
        v = {"A": f.a, "B": f.b, "C": f.c}
        if f.d is not None:
            v["D"] = f.d
        values[f.instance.key] = v
    mismatched = 0
    provenance: set[str] = set()
    sets_seen: set[str] = set()
    for path in sorted((root / SETS_DIR).glob("2026-*.json")):
        fs = _Set.model_validate_json(path.read_text(encoding="utf-8"))
        sets_seen.add(fs.question_set)
        provenance.add(json.dumps(fs.provenance, sort_keys=True))
        names = [_short(e) for e in fs.entrants]
        for raw in fs.instances:
            key = (fs.question_set, raw.source, raw.id, raw.resolution_date)
            got = by_key.get(key)
            if got is None:
                raise M15Error(f"{path.name}: {key} is not an instance of the dataset extract")
            if raw.resolved_to is not None and int(raw.resolved_to) != got.instance.outcome:
                mismatched += 1
            for j, name in enumerate(names):
                p = raw.forecasts[j]
                if name is None or p is None:
                    continue
                if j in raw.imputed and name not in BASELINES.values():
                    continue  # an external entry's imputed forecast is left out (item 8)
                if not 0.0 <= p <= 1.0:
                    raise M15Error(f"{path.name}: {name} forecast {p} on {key}")
                values[key][name] = p
    missing = [s for s in fbdataset.CORPUS_SETS if s not in sets_seen]
    if missing:
        raise M15Error(f"{SETS_DIR} lacks the corpus sets {', '.join(missing)}; re-extract")
    if len(provenance) != 1:
        raise M15Error(f"{SETS_DIR} mixes extracts of different archives; re-extract")
    rows = [Row(f, values[f.instance.key]) for f in forecasts]
    meta = {
        "archive": json.loads(next(iter(provenance))),
        "outcome_mismatches": mismatched,
        "loaded": loaded,
        "closes": closes,
    }
    return rows, meta


# -- statistics -------------------------------------------------------------------------


def _r(x: float, digits: int = 4) -> float:
    return round(x, digits)


def _est(e: Estimate) -> dict[str, Any]:
    """Full precision: the verdicts read these, so nothing is rounded before them."""
    return {"low": e.low, "high": e.high, "clusters": e.clusters}


@dataclass(frozen=True)
class Want:
    """One paired comparison to compute."""

    group: str
    first: str
    second: str
    rows: Sequence[Row]


def comparisons(wants: Sequence[Want]) -> list[dict[str, Any]]:
    """Each wanted difference with its series- and date-clustered intervals."""
    pairs: list[tuple[list[float], list[float], list[Hashable]]] = []
    for w in wants:
        a = [brier(r.values[w.first], r.y) for r in w.rows]
        b = [brier(r.values[w.second], r.y) for r in w.rows]
        pairs.append((a, b, [r.inst.question.series for r in w.rows]))
        pairs.append((a, b, [r.inst.resolution_date.isoformat() for r in w.rows]))
    ests = paired_many(pairs)
    out: list[dict[str, Any]] = []
    for k, w in enumerate(wants):
        s, d = ests[2 * k], ests[2 * k + 1]
        out.append(
            {
                "group": w.group,
                "first": w.first,
                "second": w.second,
                "n": s.n,
                "mean": s.mean,
                "by_series": _est(s),
                "by_date": _est(d),
            }
        )
    return out


def _with(rows: Sequence[Row], *names: str) -> list[Row]:
    return [r for r in rows if all(n in r.values for n in names)]


def _brier_rows(rows: Sequence[Row], names: Sequence[str]) -> dict[str, float]:
    return {n: _r(statistics.fmean(brier(r.values[n], r.y) for r in rows)) for n in names}


def brier_table(rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Brier pooled and by source for every arm and baseline on the whole corpus."""
    out: list[dict[str, Any]] = []
    groups: list[tuple[str, list[Row]]] = [("pooled", list(rows))]
    groups += [(s, [r for r in rows if r.inst.source == s]) for s in fbdataset.SOURCES]
    for name, rs in groups:
        names = ["A", "B", "C"] + (["D"] if name == "yfinance" else [])
        names += [b for b in BASELINES.values() if all(b in r.values for r in rs)]
        out.append(
            {
                "group": name,
                "n": len(rs),
                "yes_share": _r(statistics.fmean(r.y for r in rs)),
                "brier": _brier_rows(rs, names),
            }
        )
    return out


def entrant_table(rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Each named entry on its own instances, pooled and by source, beside the arms."""
    out: list[dict[str, Any]] = []
    for (_org, _model), (name, kind) in ENTRANTS.items():
        rs = _with(rows, name)
        sets = sorted({r.inst.question.question_set for r in rs})
        by_source = []
        for s in fbdataset.SOURCES:
            sub = [r for r in rs if r.inst.source == s]
            if sub:
                by_source.append(
                    {"source": s, "n": len(sub), "brier": _brier_rows(sub, [name, "A", "B", "C"])}
                )
        out.append(
            {
                "entrant": name,
                "organization": _org,
                "model": _model,
                "kind": kind,
                "sets": sets,
                "n": len(rs),
                "brier": _brier_rows(rs, [name, *ENTRANT_SECONDS]),
                "by_source": by_source,
            }
        )
    return out


def coverage(rows: Sequence[Row], meta: dict[str, Any]) -> dict[str, Any]:
    loaded: fbdataset.Loaded = meta["loaded"]
    closes = meta["closes"]
    yf = fbdataset.yfinance_questions(loaded)
    by_source = []
    for s in fbdataset.SOURCES:
        rs = [r for r in rows if r.inst.source == s]
        by_source.append(
            {
                "source": s,
                "instances": len(rs),
                "questions": len({(r.inst.question.question_set, r.inst.question.id) for r in rs}),
                "series": len({r.inst.question.series for r in rs}),
                "b_from_history": sum(1 for r in rs if r.f.source_n >= fbdataset.MIN_SOURCE),
                "c_with_own_history": sum(1 for r in rs if r.f.series_n > 0),
                "d_reached": sum(1 for r in rs if r.f.share is not None),
                "series_arm": "the ticker's closes (Yahoo chart, keyless)"
                if s == "yfinance"
                else "none: base rates only (registration item 7)",
            }
        )
    return {
        "instances": len(rows),
        "series": len({r.inst.question.series for r in rows}),
        "dataset_questions": loaded.dataset_questions,
        "left_out": loaded.removed,
        "yfinance_questions": len(yf),
        "yfinance_with_closes": sum(1 for q in yf if q.question_id in closes),
        "outcome_mismatches": meta["outcome_mismatches"],
        "by_source": by_source,
    }


def _find(cs: Sequence[dict[str, Any]], group: str, first: str, second: str) -> dict[str, Any]:
    for c in cs:
        if (c["group"], c["first"], c["second"]) == (group, first, second):
            return c
    raise M15Error(f"internal: no comparison {group} {first} − {second}")


def _width(e: dict[str, Any]) -> float:
    return float(e["high"]) - float(e["low"])


def hypotheses(cs: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """P52 to P57 as registered (item 10), and the follow-up rule (item 11)."""

    def held(ok: bool) -> str:
        return "held" if ok else "failed"

    ba = _find(cs, "pooled", "B", "A")
    bn = _find(cs, "pooled", "B", "Naive")
    cb = _find(cs, "pooled", "C", "B")
    db = _find(cs, "yfinance", "D", "B")
    leaders = [_find(cs, name, name, "B") for name in LEADERS]
    stat = _find(cs, "stat-v1", "B", "stat-v1")
    ws, wd = _width(ba["by_series"]), _width(ba["by_date"])
    out = [
        {
            "id": "P52",
            "verdict": held(ba["mean"] <= -0.03 and ba["by_series"]["high"] < 0),
            "reading": f"B − A {ba['mean']:+.5f}, by series [{ba['by_series']['low']:+.5f}, "
            f"{ba['by_series']['high']:+.5f}]",
        },
        {
            "id": "P53",
            "verdict": held(bn["mean"] <= -0.02 and bn["by_series"]["high"] < 0),
            "reading": f"B − Naive {bn['mean']:+.5f}, by series [{bn['by_series']['low']:+.5f}, "
            f"{bn['by_series']['high']:+.5f}]",
        },
        {
            "id": "P54",
            "verdict": held(wd >= 2 * ws),
            "reading": f"B − A interval width by date {wd:.4f} against by series {ws:.4f} "
            + (f"({wd / ws:.2f}x)" if ws > 0 else "(zero by series)"),
        },
        {
            "id": "P55",
            "verdict": held(-db["mean"] < 0.005),
            "reading": f"on yfinance B − D {-db['mean']:+.5f} (D − B {db['mean']:+.5f}, by series "
            f"[{db['by_series']['low']:+.5f}, {db['by_series']['high']:+.5f}])",
        },
        {
            "id": "P56",
            "verdict": held(all(c["mean"] <= -0.02 for c in leaders)),
            "reading": "; ".join(f"{c['first']} − B {c['mean']:+.5f} on {c['n']}" for c in leaders),
        },
        {
            "id": "P57",
            "verdict": held(cb["mean"] <= -0.01 and cb["by_series"]["high"] < 0),
            "reading": f"C − B {cb['mean']:+.5f}, by series [{cb['by_series']['low']:+.5f}, "
            f"{cb['by_series']['high']:+.5f}]",
        },
    ]
    p52, p53 = out[0]["verdict"] == "held", out[1]["verdict"] == "held"
    no_worse = stat["by_series"]["high"] < 0.01
    follow = {
        "opens": (p52 and p53) or no_worse,
        "p52_and_p53": p52 and p53,
        "b_no_worse_than_stat_v1": no_worse,
        "reading": f"B − stat-v1 {stat['mean']:+.5f}, by series [{stat['by_series']['low']:+.5f}, "
        f"{stat['by_series']['high']:+.5f}] on {stat['n']}",
    }
    return out, follow


def snapshot(root: Path) -> dict[str, Any]:
    """Every number M15's row reports."""
    rows, meta = build_rows(root)
    wants: list[Want] = []
    groups: list[tuple[str, list[Row]]] = [("pooled", list(rows))]
    groups += [(s, [r for r in rows if r.inst.source == s]) for s in fbdataset.SOURCES]
    for name, rs in groups:
        for first, second in POOLED_PAIRS:
            wants.append(Want(name, first, second, _with(rs, first, second)))
    yf = [r for r in rows if r.inst.source == "yfinance"]
    for first, second in YFINANCE_PAIRS:
        wants.append(Want("yfinance", first, second, _with(yf, first, second)))
    for _key, (name, _kind) in ENTRANTS.items():
        rs = _with(rows, name)
        for second in ENTRANT_SECONDS:
            wants.append(Want(name, name, second, _with(rs, second)))
    wants.append(Want("stat-v1", "B", "stat-v1", _with(rows, "stat-v1")))
    cs = comparisons(wants)
    hyp, follow = hypotheses(cs)
    loaded: fbdataset.Loaded = meta["loaded"]
    return {
        "label": LABEL,
        "method": {
            "statistic": "Brier(first) − Brier(second), paired by instance; negative: "
            "first is better",
            "bootstrap": f"95%, {DEFAULT_RESAMPLES:,} resamples, seed {DEFAULT_SEED}, "
            "resampling series (source, id) and, separately, resolution dates",
            "corpus": f"sets {fbdataset.CORPUS_SETS[0]} to {fbdataset.CORPUS_SETS[-1]}; "
            f"history from {fbdataset.HISTORY_FIRST}; an outcome dated D known at freeze F "
            f"when D + {fbdataset.LAG_DAYS} days is on or before F's day",
            "arms": {
                "A": "0.5",
                "B": f"(yes + 1) / (n + 2) over the source's known outcomes; 0.5 while n < "
                f"{fbdataset.MIN_SOURCE}",
                "C": f"(yes_s + {fbdataset.SERIES_PSEUDO:g}·B) / (n_s + "
                f"{fbdataset.SERIES_PSEUDO:g}) over the series' known outcomes",
                "D": f"{fbdataset.SERIES_WEIGHT:g}·(share of past h-observation changes above "
                f"0 in the last {fbdataset.LOOKBACK_DAYS} days of admitted closes) + "
                f"{1 - fbdataset.SERIES_WEIGHT:g}·B; B with fewer than "
                f"{fbdataset.MIN_CHANGES} changes",
            },
            "comparators": "forecasts as published, unclipped; an external entry's imputed "
            "forecasts left out; the baselines' flag ignored",
        },
        "provenance": {
            "sets": loaded.provenance,
            "archive": meta["archive"],
            "closes_ledger": str(fbdataset.LEDGER),
        },
        "coverage": coverage(rows, meta),
        "briers": brier_table(rows),
        "entrants": entrant_table(rows),
        "comparisons": cs,
        "hypotheses": hyp,
        "follow_up": follow,
    }


def dumps(snap: dict[str, Any]) -> str:
    return json.dumps(snap, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
