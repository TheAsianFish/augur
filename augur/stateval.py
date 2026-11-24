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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from augur.replay import clip
from augur.score import paired
from augur.statarm import (
    BASE_RATES_PATH,
    ArmForecast,
    BaseRates,
    base_rate_blend,
    binary_arm,
    primary,
)

SNAPSHOT = Path("data/statarm/m12-c4.json")
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
