"""M12 on C4: the statistical arm and both blends in M8's walk-forward harness.

No model call: every forecast here is either Augur's committed raw
no-retrieval forecast on corpus C4 (:func:`augur.q4.load_corpus`), a
ForecastBench freeze price, or a base rate fitted on outcomes public before
the instance opened. The design is item 6 of PLAN.md's ``m12-registration``
block:

* the instances are M8's walk-forward set W (every instance with at least
  :data:`augur.q4.W_MIN` resolutions public before it opened), in W's order;
* each day's base rates are fitted on :func:`augur.q4.history` for that
  day, and :func:`augur.q4.check_time_order` refuses any later outcome;
* *priced* means a Polymarket or Manifold instance with a freeze price, the
  prices the bot can read live; Metaculus and INFER prices are hidden, so
  there the arm is a base rate;
* every difference is Brier(arm or blend) − Brier(other), paired by
  instance, with a 95% bootstrap interval resampling markets (10,000
  resamples, seeded), as M8's table has it.

``augur statarm`` writes the snapshot (and the base rates the live arm
uses, fitted on every C4 outcome); ``augur statarm --check`` re-runs and
compares, as ``make eval`` does; ``augur report`` renders RESULTS.md from it.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from augur.clock import to_iso
from augur.metaculus import MetaculusClient
from augur.predictions import BinaryPrediction
from augur.q4 import (
    MODEL,
    READABLE,
    SOURCES,
    W_MIN,
    Instance,
    _clusters,
    _est,
    briers,
    check_time_order,
    history,
    load_corpus,
)
from augur.questions import CONTINUOUS_TYPES
from augur.replay import clip
from augur.research.series import SeriesFetcher, match_series
from augur.score import paired
from augur.standings import RoundStandings
from augur.statarm import (
    BASE_RATES_PATH,
    Abstain,
    ArmForecast,
    BaseRates,
    base_rate_blend,
    binary_arm,
    primary,
    series_arm,
)

SNAPSHOT = Path("data/statarm/m12-c4.json")
COVERAGE = {
    "minibench-2026-11-16": (
        Path("data/standings/minibench.json"),
        Path("data/statarm/m12-coverage-minibench-2026-11-16.json"),
    ),
}
"""Rounds whose question list the series rule was run on: snapshot and coverage file."""
ARMS: tuple[str, ...] = ("augur", "arm", "candidate", "base-rate-blend", "price")
"""``price`` exists only on priced instances; the others on every instance of W."""


@dataclass(frozen=True)
class Row:
    """One W instance and every arm's forecast for it."""

    instance: Instance
    priced: bool
    arm_kind: str
    forecasts: dict[str, float]


def priced(i: Instance) -> bool:
    """A price the bot could read live: Polymarket or Manifold, with a freeze price."""
    return i.source in READABLE and i.price is not None


def rows(instances: Sequence[Instance], *, min_history: int = W_MIN) -> list[Row]:
    """Every W instance with Augur, the arm, both blends and (where priced) the price."""
    out: list[Row] = []
    for day in sorted({i.opened for i in instances}):
        train = history(instances, day)
        if len(train) < min_history:
            continue
        check_time_order(train, day)
        rates = BaseRates.fit((i.source, i.outcome) for i in train)
        for i in (i for i in instances if i.opened == day):
            is_priced = priced(i)
            arm = binary_arm(i.price if is_priced else None, i.source, rates)
            if not isinstance(arm, ArmForecast) or arm.probability is None:
                raise AssertionError(f"internal: the binary arm abstained on {i.key}")
            augur = BinaryPrediction(clip(i.p))
            cand, _ = primary(augur, arm)
            brb, _ = base_rate_blend(augur, arm)
            if not isinstance(cand, BinaryPrediction) or not isinstance(brb, BinaryPrediction):
                raise AssertionError(f"internal: a binary blend gave another type on {i.key}")
            forecasts = {
                "augur": augur.probability,
                "arm": arm.probability,
                "candidate": cand.probability,
                "base-rate-blend": brb.probability,
            }
            if is_priced and i.price is not None:
                forecasts["price"] = i.price
            out.append(Row(i, is_priced, arm.kind, forecasts))
    return out


GROUPS: tuple[str, ...] = ("all", "priced", "unpriced", *SOURCES)


def _members(rs: Sequence[Row], group: str) -> list[Row]:
    if group == "all":
        return list(rs)
    if group == "priced":
        return [r for r in rs if r.priced]
    if group == "unpriced":
        return [r for r in rs if not r.priced]
    return [r for r in rs if r.instance.source == group]


def _diff(rs: Sequence[Row], a: str, b: str) -> dict[str, float | int]:
    ys = [r.instance.outcome for r in rs]
    return _est(
        paired(
            briers([r.forecasts[a] for r in rs], ys),
            briers([r.forecasts[b] for r in rs], ys),
            _clusters([r.instance for r in rs]),
        )
    )


def group_table(rs: Sequence[Row]) -> list[dict[str, Any]]:
    """Each group's Brier per arm, and each arm against Augur (and the price, where priced)."""
    out: list[dict[str, Any]] = []
    for group in GROUPS:
        sub = _members(rs, group)
        if not sub:
            continue
        ys = [r.instance.outcome for r in sub]
        arms = [a for a in ARMS if all(a in r.forecasts for r in sub)]
        out.append(
            {
                "group": group,
                "n": len(sub),
                "markets": len({r.instance.market for r in sub}),
                "base_rate": round(sum(ys) / len(ys), 4),
                "arm_kinds": {
                    k: sum(1 for r in sub if r.arm_kind == k)
                    for k in sorted({r.arm_kind for r in sub})
                },
                "brier": {
                    a: round(statistics.fmean(briers([r.forecasts[a] for r in sub], ys)), 4)
                    for a in arms
                },
                "vs_augur": {a: _diff(sub, a, "augur") for a in arms if a != "augur"},
                "vs_price": (
                    {a: _diff(sub, a, "price") for a in arms if a != "price"}
                    if "price" in arms
                    else {}
                ),
            }
        )
    return out


def _held(ok: bool) -> str:
    return "held" if ok else "failed"


def hypotheses(table: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """P34 to P37 as registered, each read off the table and marked held or failed."""
    g = {row["group"]: row for row in table}
    p34 = g["priced"]["vs_augur"]["candidate"]
    p35 = g["priced"]["vs_price"]["candidate"]
    p36 = g["metaculus"]["vs_augur"]["base-rate-blend"]
    p37 = g["all"]["vs_augur"]["arm"]
    return [
        {
            "id": "P34",
            "statistic": "priced: Brier(candidate) - Brier(Augur)",
            "estimate": p34,
            "rule": "interval below 0",
            "verdict": _held(float(p34["high"]) < 0.0),
        },
        {
            "id": "P35",
            "statistic": "priced: Brier(candidate) - Brier(price)",
            "estimate": p35,
            "rule": "point estimate above 0",
            "verdict": _held(float(p35["mean"]) > 0.0),
        },
        {
            "id": "P36",
            "statistic": "Metaculus: Brier(base-rate blend) - Brier(Augur)",
            "estimate": p36,
            "rule": "point estimate between +0.01 and +0.04",
            "verdict": _held(0.01 <= float(p36["mean"]) <= 0.04),
        },
        {
            "id": "P37",
            "statistic": "all of W: Brier(arm) - Brier(Augur)",
            "estimate": p37,
            "rule": "interval below 0",
            "verdict": _held(float(p37["high"]) < 0.0),
        },
    ]


def live_base_rates(corpus: Sequence[Instance]) -> BaseRates:
    """The live arm's base rates: every C4 outcome, by source."""
    latest = max(i.known or i.resolved for i in corpus)
    return BaseRates.fit(
        ((i.source, i.outcome) for i in corpus),
        fitted_on=f"M8 corpus C4: {len(corpus)} resolutions through {latest.isoformat()}",
    )


def snapshot(root: Path, corpus: Sequence[Instance] | None = None) -> dict[str, Any]:
    """Every C4 number M12 reports, from the committed ledgers and ForecastBench subsets."""
    corpus = load_corpus(root) if corpus is None else corpus
    rs = rows(corpus)
    table = group_table(rs)
    return {
        "corpus": {
            "model": MODEL,
            "instances": len(corpus),
            "w_instances": len(rs),
            "w_markets": len({r.instance.market for r in rs}),
        },
        "method": {
            "design": "M8's walk-forward W; base rates fitted on outcomes public before each "
            "instance opened",
            "bootstrap": "paired, resampling markets, 10,000 resamples, seeded, 95%",
            "clip": [0.01, 0.99],
            "weight": "one-third Augur, two-thirds arm, in log-odds",
        },
        "groups": table,
        "hypotheses": hypotheses(table),
    }


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def files(root: Path, out: Path = SNAPSHOT) -> dict[Path, str]:
    """What ``augur statarm`` writes: the snapshot and the live base rates."""
    corpus = load_corpus(root)
    return {
        root / out: dumps(snapshot(root, corpus)),
        BASE_RATES_PATH: dumps(live_base_rates(corpus).to_json()),
    }


# -- the series rule's coverage on a real round's question list --------------------------


def title_matches(titles: Mapping[int, tuple[str, str]]) -> dict[int, str]:
    """``post_id -> query`` for the numeric and discrete titles the series rule reaches.

    Offline and deterministic: ``augur statarm --check`` re-derives it from the
    standings snapshot and compares it with the committed coverage file.
    """
    out: dict[int, str] = {}
    for post_id, (qtype, title) in sorted(titles.items()):
        if qtype not in CONTINUOUS_TYPES:
            continue
        m = match_series(title)
        if m is not None:
            out[post_id] = m.query
    return out


async def round_coverage(
    snapshot: RoundStandings, client: MetaculusClient, fetcher: SeriesFetcher
) -> dict[str, Any]:
    """What the series rule does on every question of a round, as of each one's open.

    Each post is read from the API for its scaling and open time; a question
    the rule reaches has its series resolved and fetched as of its open
    (the earliest instant a shadow tick could forecast it), and the arm is
    built or abstains with its reason. Nothing is forecast or written.
    """
    rows: list[dict[str, Any]] = []
    for rq in sorted(snapshot.questions, key=lambda r: r.post_id):
        row: dict[str, Any] = {"post_id": rq.post_id, "type": rq.type, "title": rq.title}
        rows.append(row)
        if rq.type not in CONTINUOUS_TYPES:
            row["outcome"] = "not numeric or discrete"
            continue
        q, _ = await client.get_question(rq.post_id)
        fallback = q.scheduled_resolve_time.date() if q.scheduled_resolve_time else None
        m = match_series(q.title, fallback=fallback)
        if m is None:
            row["outcome"] = "no series in the title"
            continue
        row.update(query=m.query, how=m.how, target=m.target.isoformat())
        row["as_of"] = to_iso(q.open_time)
        symbol = await fetcher.resolve(m)
        if symbol is None:
            row["outcome"] = "abstains: no symbol found"
            continue
        series = await fetcher.fetch(symbol, q.open_time)
        row.update(symbol=symbol, closes=len(series.observations), refused=series.refused)
        if q.scaling is None:
            row["outcome"] = "abstains: no usable scaling"
            continue
        row["range"] = [q.scaling.range_min, q.scaling.range_max]
        closes = [(o.day, o.close) for o in series.observations]
        arm = series_arm(closes, m.target, q.scaling, f"{series.name} ({symbol})")
        if isinstance(arm, Abstain):
            row["outcome"] = f"abstains: {arm.reason}"
            continue
        row["outcome"] = "forecast"
        row["last_close"] = closes[-1][1]
        row["last_day"] = closes[-1][0].isoformat()
        row["percentiles"] = [[lv, round(v, 4)] for lv, v in arm.percentiles]
    reached = [r for r in rows if "query" in r]
    return {
        "round": snapshot.round.slug,
        "project": snapshot.round.id,
        "questions": len(rows),
        "numeric_or_discrete": sum(1 for r in rows if r["type"] in CONTINUOUS_TYPES),
        "reached": len(reached),
        "forecast": sum(1 for r in rows if r.get("outcome") == "forecast"),
        "rows": rows,
    }


def check_coverage(path: Path, snapshot: RoundStandings) -> list[str]:
    """Differences between a committed coverage file and the rule re-run on the titles."""
    data = json.loads(path.read_text(encoding="utf-8"))
    committed = {int(r["post_id"]): str(r["query"]) for r in data["rows"] if "query" in r}
    fresh = title_matches({q.post_id: (q.type or "", q.title) for q in snapshot.questions})
    if committed == fresh:
        return []
    return [f"{path}: the series rule now reaches {fresh}, the file says {committed}"]
