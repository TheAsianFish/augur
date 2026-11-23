"""M10's same-question row: Augur's backtests beside every ForecastBench entrant.

ForecastBench publishes every entrant's forecasts for each question set (its
processed forecast sets; CC BY-SA 4.0, Forecasting Research Institute). The
rows for the question instances Augur's backtests forecast are committed in
:data:`SETS_DIR` (``scripts/extract_forecast_sets.py`` cut them from the
published archive). On exactly those instances, with the outcome Augur was
scored against, each entrant's Brier score is set beside Augur's, paired per
question (PLAN.md, "M10's same-question row").

* **Arms** (:data:`CORPORA`): C1 (M4), C2 (M5's arm (b)), O (M6's one pass as
  of open, 64 of C2's questions) and C4 (M8's nine sets). Each arm's forecast
  is the one RESULTS.md scores, clipped to [0.01, 0.99] as there.
* **Entrants**: every file of the set, grouped by ForecastBench's file name
  (:data:`GROUPS`); an entrant is one (organization, model) pair, pooled over
  the sets it entered (on C4) and compared only on its forecasts that are not
  imputed. ForecastBench's baselines are the exception: the naive and imputed
  forecasters are imputed by construction (the market's value the day before
  and on the due date), so their flag is ignored. Entrants' forecasts are
  scored as published, unclipped. The market's value on the due date and the
  question set's freeze value are rows of their own.
* **Statistic**: Brier(Augur) − Brier(entrant) on the entrant's questions,
  bootstrap over markets (10,000 resamples, seeded); a positive difference
  means the entrant beat Augur. Augur's rank in a group is one plus the
  number of entrants whose point estimate is above 0, among the group and
  Augur; *bottom quarter* means that rank is above three quarters of the
  number ranked.

Deterministic, so ``augur fbcompare`` writes a snapshot that ``augur
fbcompare --check`` reproduces byte for byte and ``augur report`` renders.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from augur import q2, q4
from augur.errors import AugurError
from augur.forecastbench import FBQuestion, load_set
from augur.replay import clip
from augur.score import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Estimate,
    brier,
    clustered_bootstrap,
    paired_many,
)

SETS_DIR = Path("data/forecastbench/forecast-sets")
SNAPSHOT = Path("data/forecastbench/m10-same-question.json")
FORECASTBENCH = Path("data/forecastbench")
LEDGERS = Path("data/ledgers")

Group = Literal["zero-shot", "zero-shot-freeze", "external", "baseline"]
GROUPS: tuple[Group, ...] = ("zero-shot", "zero-shot-freeze", "external", "baseline")
GROUP_NAMES: dict[str, str] = {
    "zero-shot": "ForecastBench zero-shot LLMs, no freeze value",
    "zero-shot-freeze": "ForecastBench zero-shot LLMs, shown the freeze value",
    "external": "external entries",
    "baseline": "ForecastBench baselines",
    "market": "market",
}
MARKET_DUE = "market value on the due date"
MARKET_FREEZE = "freeze value (question set)"
BOTTOM_QUARTER = 0.75
P31_SHARE = 0.9
SHOWN_SHARE = 0.5


class FBCompareError(AugurError):
    """The extract or a ledger does not cover the instances Augur scored."""


# -- the extract (untrusted data crossing into the code) ---------------------------


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    bytes: int
    sha256: str
    last_modified: str
    fetched: str
    licence: str
    attribution: str
    extracted_by: str


class SetEntrant(BaseModel):
    model_config = ConfigDict(frozen=True)

    file: str
    organization: str
    model: str
    model_organization: str
    leaderboard_eligible: bool
    group: Group
    web_search: bool


class SetQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    id: str
    resolution_date: str
    resolved: bool
    resolved_to: float | None
    market_value_on_due_date: float | None
    forecasts: list[float | None]
    imputed: list[int]


class ForecastSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_set: str
    provenance: Provenance
    entrants: list[SetEntrant]
    questions: list[SetQuestion]


def load_sets(directory: Path) -> dict[str, ForecastSet]:
    """Every committed forecast set extract, by question set (``2026-06-21``)."""
    out: dict[str, ForecastSet] = {}
    for path in sorted(directory.glob("*.json")):
        fs = ForecastSet.model_validate_json(path.read_text(encoding="utf-8"))
        for q in fs.questions:
            if len(q.forecasts) != len(fs.entrants):
                raise FBCompareError(f"{path}: {q.source}:{q.id} has a forecast count mismatch")
        out[fs.question_set] = fs
    if not out:
        raise FBCompareError(
            f"no forecast sets in {directory}; run scripts/extract_forecast_sets.py"
        )
    return out


def entrant_names(fs: ForecastSet) -> list[str]:
    """``organization / model``; a pair entered twice in one set is told apart by file."""
    base = [f"{e.organization} / {e.model}" for e in fs.entrants]
    return [
        f"{name} [{e.file}]" if base.count(name) > 1 else name
        for name, e in zip(base, fs.entrants, strict=True)
    ]


# -- Augur's arms ------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One question instance Augur scored: its forecast, outcome and freeze value."""

    question_set: str
    source: str
    fb_id: str
    outcome: int
    resolution_date: date
    p: float
    freeze_value: float | None

    @property
    def market(self) -> str:
        return f"{self.source}:{self.fb_id}"


@dataclass(frozen=True)
class Corpus:
    name: str
    description: str
    model: str
    retrieval: bool
    items: list[Item] = field(default_factory=list)


def _kept(root: Path, question_set: str) -> list[FBQuestion]:
    return load_set(
        root / FORECASTBENCH / f"{question_set}-llm.subset.json",
        root / FORECASTBENCH / f"{question_set}_resolution_set.subset.json",
        released=date.min,
    ).kept


def _need(path: Path) -> Path:
    if not path.exists():
        raise FBCompareError(f"no {path}; M10 reads M4 to M8's committed ledgers")
    return path


def _items(question_set: str, qs: Sequence[FBQuestion], p: dict[int, float]) -> list[Item]:
    return [
        Item(
            question_set,
            q.source,
            q.fb_id,
            q.outcome,
            q.resolution_date,
            p[q.question_id],
            q.freeze_value,
        )
        for q in qs
        if q.question_id in p
    ]


def _tournament(question_set: str) -> str:
    return f"forecastbench:{question_set}-llm.json"


C1_SET = "2026-05-24"
C2_SET = "2026-06-21"
O_SAMPLE = 64
NEMOTRON = "nvidia/nemotron-3-ultra-550b-a55b:free"


def corpus_c1(root: Path) -> Corpus:
    path = _need(root / LEDGERS / "m4-forecastbench-qwen.sqlite")
    p = q2.forecasts(path, _tournament(C1_SET))
    return Corpus(
        "C1",
        f"ForecastBench {C1_SET}, no retrieval, one draw (M4)",
        "qwen2.5:7b",
        False,
        _items(C1_SET, _kept(root, C1_SET), p),
    )


def corpus_c2(root: Path) -> Corpus:
    path = _need(root / LEDGERS / "m5-replay-c2.sqlite")
    p = q2.forecasts(path, _tournament(C2_SET))
    return Corpus(
        "C2",
        f"ForecastBench {C2_SET}, no retrieval, median of three draws (M5's arm (b))",
        NEMOTRON,
        False,
        _items(C2_SET, _kept(root, C2_SET), p),
    )


def corpus_o(root: Path) -> Corpus:
    path = _need(root / LEDGERS / "m6-c2-open.sqlite")
    p = q2.forecasts(path, _tournament(C2_SET) + "+onepass-asof-open")
    sample = sorted(_kept(root, C2_SET), key=lambda q: q.question_id)[:O_SAMPLE]
    return Corpus(
        "O",
        f"ForecastBench {C2_SET}, the first {O_SAMPLE} questions by id, one research pass "
        "as of open, median of three draws (M6)",
        NEMOTRON,
        True,
        _items(C2_SET, sample, p),
    )


def corpus_c4(root: Path) -> Corpus:
    items = [
        Item(
            i.question_set,
            i.source,
            i.market.split(":", 1)[1],
            i.outcome,
            i.resolved,
            i.p,
            i.price,
        )
        for i in q4.load_corpus(root)
    ]
    return Corpus(
        "C4",
        f"ForecastBench {q4.SETS[0]} to {q4.SETS[-1]} (nine sets), no retrieval, one draw (M8)",
        q4.MODEL,
        False,
        items,
    )


CORPORA = (corpus_c1, corpus_c2, corpus_o, corpus_c4)


# -- matching and scoring ----------------------------------------------------------


@dataclass
class Series:
    """One entrant's (or market row's) forecasts on a corpus, by item index."""

    name: str
    group: str
    organization: str = ""
    model: str = ""
    web_search: bool = False
    leaderboard_eligible: bool = True
    sets: set[str] = field(default_factory=set)
    forecasts: dict[int, float] = field(default_factory=dict)
    imputed: int = 0
    missing: int = 0


@dataclass(frozen=True)
class Matched:
    series: list[Series]
    outcome_disagreements: list[str]
    date_disagreements: int
    unresolved: int


def match(corpus: Corpus, sets: dict[str, ForecastSet]) -> Matched:
    """Every entrant's forecast on each of the corpus's instances.

    Raises if an instance Augur scored is not in the extract. The outcome is
    always Augur's; instances where ForecastBench's processed file resolved
    differently are listed (and counted), never re-scored.
    """
    rows: dict[tuple[str, str, str], tuple[ForecastSet, SetQuestion]] = {}
    for qs, fs in sets.items():
        for q in fs.questions:
            rows[(qs, q.source, q.id)] = (fs, q)
    series: dict[str, Series] = {
        MARKET_DUE: Series(MARKET_DUE, "market"),
        MARKET_FREEZE: Series(MARKET_FREEZE, "market"),
    }
    names = {qs: entrant_names(fs) for qs, fs in sets.items()}
    disagree: list[str] = []
    dates = unresolved = 0
    for k, item in enumerate(corpus.items):
        got = rows.get((item.question_set, item.source, item.fb_id))
        if got is None:
            raise FBCompareError(
                f"{corpus.name}: {item.question_set} {item.market} is not in {SETS_DIR}; "
                "re-run scripts/extract_forecast_sets.py"
            )
        fs, q = got
        if not q.resolved:
            unresolved += 1
        if q.resolved_to is None or float(q.resolved_to) != float(item.outcome):
            disagree.append(f"{item.question_set}:{item.market}")
        if q.resolution_date != item.resolution_date.isoformat():
            dates += 1
        if q.market_value_on_due_date is not None:
            series[MARKET_DUE].forecasts[k] = float(q.market_value_on_due_date)
        if item.freeze_value is not None:
            series[MARKET_FREEZE].forecasts[k] = item.freeze_value
        imputed = set(q.imputed)
        for j, (e, name) in enumerate(zip(fs.entrants, names[item.question_set], strict=True)):
            s = series.get(name)
            if s is None:
                s = series[name] = Series(
                    name,
                    e.group,
                    e.organization,
                    e.model,
                    e.web_search,
                    e.leaderboard_eligible,
                )
            s.sets.add(item.question_set)
            p = q.forecasts[j]
            if p is None:
                s.missing += 1
            elif j in imputed and e.group != "baseline":
                s.imputed += 1
            else:
                s.forecasts[k] = p
    return Matched(list(series.values()), disagree, dates, unresolved)


def _r(x: float, digits: int = 4) -> float:
    return round(x, digits)


def _est(e: Estimate) -> dict[str, Any]:
    """The estimate; one market gives no interval (``low`` and ``high`` are null)."""
    interval = e.clusters >= 2
    return {
        "mean": _r(e.mean),
        "low": _r(e.low) if interval else None,
        "high": _r(e.high) if interval else None,
        "n": e.n,
        "clusters": e.clusters,
    }


def above(d: dict[str, Any]) -> bool:
    """The interval of Augur − entrant is wholly above 0 (the entrant beat Augur)."""
    return d["low"] is not None and float(d["low"]) > 0


def below(d: dict[str, Any]) -> bool:
    """The interval of Augur − entrant is wholly below 0 (Augur beat the entrant)."""
    return d["high"] is not None and float(d["high"]) < 0


def compare(corpus: Corpus, sets: dict[str, ForecastSet]) -> dict[str, Any]:
    """One corpus: Augur's arm, each entrant against it, and each group's summary."""
    if not corpus.items:
        raise FBCompareError(f"{corpus.name}: no forecasts in its ledger")
    m = match(corpus, sets)
    ys = [i.outcome for i in corpus.items]
    markets: list[Hashable] = [i.market for i in corpus.items]
    augur = [brier(clip(i.p), y) for i, y in zip(corpus.items, ys, strict=True)]
    scored = [s for s in m.series if s.forecasts]
    pairs: list[tuple[Sequence[float], Sequence[float], Sequence[Hashable] | None]] = []
    for s in scored:
        idx = sorted(s.forecasts)
        pairs.append(
            (
                [augur[k] for k in idx],
                [brier(s.forecasts[k], ys[k]) for k in idx],
                [markets[k] for k in idx],
            )
        )
    diffs = paired_many(pairs)
    entrants: list[dict[str, Any]] = []
    for s, (a, b, _), d in zip(scored, pairs, diffs, strict=True):
        entrants.append(
            {
                "name": s.name,
                "group": s.group,
                "organization": s.organization,
                "model": s.model,
                "web_search": s.web_search,
                "leaderboard_eligible": s.leaderboard_eligible,
                "sets": sorted(s.sets),
                "n": len(b),
                "markets": d.clusters,
                "imputed_excluded": s.imputed,
                "missing": s.missing,
                "brier": _r(statistics.fmean(b)),
                "augur_brier": _r(statistics.fmean(a)),
                "diff": _est(d),
            }
        )
    entrants.sort(key=lambda e: (e["group"], -e["diff"]["mean"], e["name"]))
    unscored = sorted(
        (
            {"name": s.name, "group": s.group, "imputed_excluded": s.imputed, "missing": s.missing}
            for s in m.series
            if not s.forecasts
        ),
        key=lambda u: str(u["name"]),
    )
    groups = {g: group_summary([e for e in entrants if e["group"] == g]) for g in GROUPS}
    return {
        "name": corpus.name,
        "description": corpus.description,
        "model": corpus.model,
        "retrieval": corpus.retrieval,
        "sets": sorted({i.question_set for i in corpus.items}),
        "n": len(corpus.items),
        "markets": len(set(markets)),
        "base_rate": _r(sum(ys) / len(ys)),
        "augur_brier": _est(clustered_bootstrap(augur, markets)),
        "outcome_disagreements": len(m.outcome_disagreements),
        "outcome_disagreement_instances": m.outcome_disagreements,
        "resolution_date_disagreements": m.date_disagreements,
        "unresolved_in_forecast_set": m.unresolved,
        "entrants": entrants,
        "unscored": unscored,
        "groups": groups,
    }


def group_summary(entrants: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count, Augur's rank, and the medians of a group (entrants with n ≥ 1).

    Best and worst (by Augur − entrant) are named among the entrants compared
    on at least :data:`SHOWN_SHARE` of the group's median n, so an entrant
    left with a handful of questions does not stand for the group; every
    count, rank and median uses the whole group.
    """
    count = len(entrants)
    if not count:
        return {"count": 0}
    means = [float(e["diff"]["mean"]) for e in entrants]
    better = sum(1 for d in means if d > 0)
    ranked = count + 1
    rank = better + 1
    median_n = statistics.median(int(e["n"]) for e in entrants)
    shown = math.ceil(SHOWN_SHARE * median_n)
    wide = sorted(
        (e for e in entrants if int(e["n"]) >= shown),
        key=lambda e: (-float(e["diff"]["mean"]), str(e["name"])),
    )
    return {
        "count": count,
        "beat_augur": better,
        "beat_augur_interval": sum(1 for e in entrants if above(e["diff"])),
        "augur_beats_interval": sum(1 for e in entrants if below(e["diff"])),
        "no_interval": sum(1 for e in entrants if e["diff"]["low"] is None),
        "augur_rank": rank,
        "ranked": ranked,
        "bottom_quarter": rank > BOTTOM_QUARTER * ranked,
        "median_diff": _r(statistics.median(means)),
        "median_brier": _r(statistics.median(float(e["brier"]) for e in entrants)),
        "median_n": median_n,
        "shown_min_n": shown,
        "best": wide[0]["name"],
        "worst": wide[-1]["name"],
    }


# -- the predictions ---------------------------------------------------------------


PREDICTIONS: dict[str, str] = {
    "P30": "On C2 and on C4, Augur's no-retrieval arm is worse than the median zero-shot LLM "
    "entrant shown no freeze value (point estimate), and ranks in that group's bottom quarter.",
    "P31": "On C2, at least 90% of the zero-shot entrants shown the freeze value beat Augur's "
    "no-retrieval arm with an interval of Augur − entrant wholly above 0.",
    "P32": "Researched as of open (O, 64 questions), Augur is still worse than the median "
    "external entry on the same questions (point estimate).",
    "P33": "No Augur arm beats any external entry with an interval wholly below 0.",
}
"""As committed in PLAN.md before the comparison (2c3b87d); never edited."""
READINGS: dict[str, str] = {
    "P30": "the median entrant is the median of Augur − entrant over the group's entrants, "
    "each on its own non-imputed questions; the group is every ForecastBench zero-shot "
    "file without freeze values, including runs that name web search and entrants left "
    "with few questions after imputed ones are excluded; bottom quarter: Augur's rank above "
    f"{BOTTOM_QUARTER} × the number ranked (the group and Augur)",
    "P31": "share = entrants with the interval wholly above 0 / entrants in the group",
    "P32": "the median of Augur − entrant over the external entries, on O's 64 questions",
    "P33": "every external entry on C1, C2, O and C4; an entry compared on one market has "
    "no interval and cannot count",
}


def _signed(x: float) -> str:
    return f"{x:+.4f}"


def verdicts(corpora: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """P30 to P33 read as PLAN.md states them, from the snapshot's numbers."""
    out: list[dict[str, Any]] = []

    parts = []
    held30 = True
    for name in ("C2", "C4"):
        g = corpora[name]["groups"]["zero-shot"]
        worse = float(g["median_diff"]) > 0
        held30 = held30 and worse and bool(g["bottom_quarter"])
        parts.append(
            f"{name}: median Augur − entrant {_signed(g['median_diff'])} over {g['count']} "
            f"entrants ({'worse than' if worse else 'not worse than'} the median); Augur "
            f"ranks {g['augur_rank']} of {g['ranked']} ({g['beat_augur']} beat it), "
            f"{'in' if g['bottom_quarter'] else 'not in'} the bottom quarter "
            f"(rank above {BOTTOM_QUARTER * g['ranked']:g})"
        )
    out.append({"id": "P30", "held": held30, "numbers": "; ".join(parts)})

    g = corpora["C2"]["groups"]["zero-shot-freeze"]
    share = g["beat_augur_interval"] / g["count"]
    out.append(
        {
            "id": "P31",
            "held": share >= P31_SHARE,
            "numbers": f"C2: {g['beat_augur_interval']} of {g['count']} entrants shown the "
            f"freeze value ({100 * share:.0f}%) beat Augur with an interval of Augur − entrant "
            f"wholly above 0 (needed: at least {100 * P31_SHARE:.0f}%)",
        }
    )

    g = corpora["O"]["groups"]["external"]
    out.append(
        {
            "id": "P32",
            "held": float(g["median_diff"]) > 0,
            "numbers": f"O: median Augur − entrant over {g['count']} external entries "
            f"{_signed(g['median_diff'])} on the same questions (median n {g['median_n']:g}); "
            f"{g['beat_augur']} of {g['count']} beat Augur (point estimate)",
        }
    )

    beaten = [
        f"{name}: {e['name']} {_signed(e['diff']['mean'])} "
        f"[{_signed(e['diff']['low'])}, {_signed(e['diff']['high'])}]"
        for name, c in corpora.items()
        for e in c["entrants"]
        if e["group"] == "external" and below(e["diff"])
    ]
    counts = ", ".join(f"{name} {c['groups']['external']['count']}" for name, c in corpora.items())
    out.append(
        {
            "id": "P33",
            "held": not beaten,
            "numbers": f"external entries with an interval of Augur − entrant wholly below 0: "
            f"{len(beaten)} ({counts} entries compared)"
            + (f": {'; '.join(beaten)}" if beaten else ""),
        }
    )
    for v in out:
        v["prediction"] = PREDICTIONS[v["id"]]
        v["reading"] = READINGS[v["id"]]
    return out


# -- the snapshot ------------------------------------------------------------------


def snapshot(root: Path) -> dict[str, Any]:
    """Every number M10's same-question row reports."""
    sets = load_sets(root / SETS_DIR)
    provenance = {fs.provenance.model_dump_json() for fs in sets.values()}
    if len(provenance) != 1:
        raise FBCompareError(f"{SETS_DIR} mixes extracts of different archives; re-extract")
    corpora = {c["name"]: c for c in (compare(make(root), sets) for make in CORPORA)}
    return {
        "provenance": next(iter(sets.values())).provenance.model_dump(),
        "method": {
            "statistic": "Brier(Augur) − Brier(entrant) on the entrant's non-imputed "
            "forecasts of the instances Augur scored; positive: the entrant beat Augur",
            "bootstrap": f"paired, resampling markets, {DEFAULT_RESAMPLES:,} resamples, "
            f"seed {DEFAULT_SEED}",
            "clip": "Augur's forecasts clipped to [0.01, 0.99] as RESULTS.md scores them; "
            "entrants' forecasts and market values as published",
            "imputed": "an entrant's imputed forecasts are excluded, except ForecastBench's "
            "baselines (the naive and imputed forecasters are imputed by construction)",
            "interval": "none for an entrant compared on a single market (null low and high); "
            "it counts for no interval-based statement",
            "rank": "1 + entrants with a point estimate above 0, among the group and Augur; "
            f"bottom quarter: rank above {BOTTOM_QUARTER} × the number ranked",
            "entrant": "one (organization, model) pair as published; on C4 pooled over the "
            "sets it entered",
        },
        "corpora": list(corpora.values()),
        "verdicts": verdicts(corpora),
    }


def dumps(snap: dict[str, Any]) -> str:
    return json.dumps(snap, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
