"""Q6: does the fleet know when it does not know? M9's analysis, from the ledgers.

Everything here runs on draws already recorded (PLAN.md, "M9's design"):

* **S138**: C2's 138 questions, no retrieval, three draws of one model; the
  final forecast is their median.
* **S40**: M7's 40 questions, one research pass as of open, draws 0 to 4
  (the live default's five), plus three draws of a second model on the same
  briefs and the loop and two-pass briefs.

Three disagreement measures (:func:`draw_spread`, :func:`model_gap`,
:func:`pass_spread`) are scored by their AUROC for a large error (a Brier
score strictly above the sample's median). Then policies that use them are
compared, on the same questions, against two controls that do not: constant
shrinkage (Platt, control A) and the source-aware calibrator (partially
pooled Platt, control B). This model's log-odds are worth about a third of
their face value (M8), so any shrink looks good against raw; only a policy
that beats both controls shows that disagreement itself carries
information.

Every fitted map is fitted leave-one-out: the question being forecast never
enters its own fit. Deterministic (seeded bootstraps), so ``augur q6``
writes a snapshot that ``augur q6 --check`` reproduces and ``augur report``
renders.

The live half (:func:`live_units`, :func:`live_analysis`) recomputes the
pre-registered policy from the member rows of every live or shadow-round
binary forecast in the committed ledgers; nothing extra is recorded live.
"""

from __future__ import annotations

import math
import random
import sqlite3
import statistics
from collections.abc import Callable, Hashable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

from augur.calibrate import platt_targets
from augur.errors import AugurError
from augur.forecastbench import FBQuestion, load_set
from augur.q2 import forecasts
from augur.q3 import combined, members
from augur.replay import clip, logit, sigmoid
from augur.score import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Estimate,
    ScoringError,
    binary_outcome,
    brier,
    paired,
)
from augur.sourcecal import PRIOR_SD, PartialPlatt, fit_logistic

LEDGERS = Path("data/ledgers")
C2_SET = Path("data/forecastbench/2026-06-21")
C2 = "forecastbench:2026-06-21-llm.json"
ONEPASS = C2 + "+onepass-asof-open"
ARMS: dict[str, tuple[Path, str]] = {
    "none": (LEDGERS / "m5-replay-c2.sqlite", C2),
    "onepass": (LEDGERS / "m6-c2-open.sqlite", ONEPASS),
    "draws": (LEDGERS / "m7-c2-draws.sqlite", ONEPASS + "+draws"),
    "nano": (LEDGERS / "m7-c2-nano.sqlite", ONEPASS + "+model-nano"),
    "loop": (LEDGERS / "m6-c2-loop.sqlite", C2 + "+loop3-asof-open"),
    "twopass": (LEDGERS / "m7-c2-twopass.sqlite", C2 + "+twopass-asof-open"),
}
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SECOND_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
SNAPSHOT = Path("data/q6/m9-q6.json")

S40_SIZE = 40
S40_DRAWS = 5
"""The live default's head count: S40's final forecast is the median of draws 0 to 4."""
SKIP_SHARE = 4
"""Skip the 1/SKIP_SHARE of questions with the largest disagreement (rounded down)."""
BRIER_TARGET = 0.01
"""The Brier difference the 'count needed' is computed for (PLAN.md)."""
AUROC_TARGET = 0.6
POWER = 0.8
LIVE_TARGET = 60
"""The odd half's resolved binary questions at which the live test is analysed."""
LIVE_DEADLINE = date(2027, 3, 31)
LIVE_MIN_DRAWS = 3
LIVE_MODES = ("live", "dry_run")
LIVE_TOURNAMENTS = ("minibench", "futureeval", "fall-aib", "spring-aib")
"""Tournament names a live unit may come from (backtests and the bot-testing-area never)."""

_Z = NormalDist().inv_cdf(0.975)
_Z_POWER = NormalDist().inv_cdf(POWER)


class Q6DataError(AugurError):
    """The ledgers M9 reads are missing or incomplete."""


# -- the measures -----------------------------------------------------------------


def _lo(p: float) -> float:
    return logit(clip(p))


def draw_spread(draws: Sequence[float]) -> float:
    """The population standard deviation of the draws' log-odds (0 for one draw)."""
    return statistics.pstdev([_lo(p) for p in draws]) if len(draws) > 1 else 0.0


def model_gap(first: Sequence[float], second: Sequence[float]) -> float:
    """|log-odds of one model's median − log-odds of the other's|."""
    return abs(_lo(statistics.median(first)) - _lo(statistics.median(second)))


def pass_spread(briefs: Sequence[float]) -> float:
    """The population standard deviation of the research briefs' forecasts, in log-odds."""
    return draw_spread(briefs)


def confidence(p: float) -> float:
    """|logit p|: how far from 50% the final forecast is (a reference, not disagreement)."""
    return abs(_lo(p))


def spread_shrink(p: float, d: float) -> float:
    """The pre-registered fixed policy: ``sigmoid(logit(p) / (1 + d))``."""
    return clip(sigmoid(_lo(p) / (1.0 + d)))


def constant_shrink(p: float, k: float) -> float:
    """The matched control: ``sigmoid(k · logit(p))``."""
    return clip(sigmoid(k * _lo(p)))


# -- AUROC ------------------------------------------------------------------------


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """P(score of a positive > score of a negative), ties counted a half."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        raise ScoringError("AUROC needs at least one large and one small error")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))


@dataclass(frozen=True)
class Auroc:
    value: float
    low: float
    high: float
    n: int
    positives: int


def auroc_interval(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Auroc:
    """AUROC with a percentile bootstrap over questions.

    A resample that draws only one class has no AUROC; it is drawn again,
    so every one of the ``resamples`` values is defined.
    """
    value = auroc(scores, labels)
    rng = random.Random(seed)
    n = len(scores)
    out: list[float] = []
    while len(out) < resamples:
        idx = rng.choices(range(n), k=n)
        ys = [labels[i] for i in idx]
        if 0 < sum(ys) < n:
            out.append(auroc([scores[i] for i in idx], ys))
    out.sort()
    lo = out[math.floor(0.025 * resamples)]
    hi = out[math.ceil(0.975 * resamples) - 1]
    return Auroc(value, lo, hi, n, sum(labels))


def auroc_se(a: float, positives: int, negatives: int) -> float:
    """Hanley and McNeil's (1982) standard error of an AUROC."""
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (positives - 1) * (q1 - a * a) + (negatives - 1) * (q2 - a * a)) / (
        positives * negatives
    )
    return math.sqrt(var)


def auroc_needed(target: float = AUROC_TARGET, power: float = POWER) -> int:
    """The smallest balanced n at which a true AUROC of ``target`` clears 0.5 with ``power``."""
    z = _Z + NormalDist().inv_cdf(power)
    n = 4
    while (target - 0.5) < z * auroc_se(target, n // 2, n - n // 2):
        n += 2
    return n


def wrong_side(ps: Sequence[float], ys: Sequence[int]) -> list[int]:
    """1 where the forecast is on the wrong side of 50% (|p − outcome| > 0.5).

    Exploratory, added after the run: with a model this close to 50%, a
    Brier above the median mostly marks a hedged forecast, so this label asks
    the other question, whether disagreement flags a confident miss.
    """
    return [int(abs(clip(p) - y) > 0.5) for p, y in zip(ps, ys, strict=True)]


def large_errors(ps: Sequence[float], ys: Sequence[int]) -> list[int]:
    """1 where the forecast's Brier is strictly above the sample's median Brier."""
    bs = [brier(clip(p), y) for p, y in zip(ps, ys, strict=True)]
    cut = statistics.median(bs)
    return [int(b > cut) for b in bs]


# -- the samples ------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One question of a sample, with every forecast the measures need."""

    qid: int
    source: str
    outcome: int
    price: float | None
    final: float
    draws: tuple[float, ...]
    second: tuple[float, ...] = ()
    briefs: tuple[float, ...] = ()


@dataclass(frozen=True)
class Sample:
    name: str
    description: str
    model: str
    items: list[Item]
    measures: dict[str, list[float]]


def _questions(root: Path) -> list[FBQuestion]:
    report = load_set(
        root / f"{C2_SET}-llm.subset.json",
        root / f"{C2_SET}_resolution_set.subset.json",
        released=date.min,
    )
    return sorted(report.kept, key=lambda q: q.question_id)


def _need(root: Path, arm: str) -> tuple[Path, str]:
    path, tournament = ARMS[arm]
    if not (root / path).exists():
        raise Q6DataError(f"no {path}; M9 reads M5 to M7's committed ledgers (git checkout them)")
    return root / path, tournament


def load_s138(root: Path) -> Sample:
    path, t = _need(root, "none")
    draws = members(path, t)
    items = []
    for q in _questions(root):
        ds = [draws.get(q.question_id, {}).get(m) for m in range(3)]
        got = [d for d in ds if d is not None]
        if len(got) < 3:
            continue
        items.append(
            Item(q.question_id, q.source, q.outcome, q.freeze_value, combined(got), tuple(got))
        )
    return Sample(
        "S138",
        "C2 (ForecastBench 2026-06-21), no retrieval, three draws, median",
        MODEL,
        items,
        {
            "draws": [draw_spread(i.draws) for i in items],
            "confidence": [confidence(i.final) for i in items],
        },
    )


def load_s40(root: Path) -> Sample:
    first = members(*_need(root, "onepass"))
    more = members(*_need(root, "draws"))
    nano = members(*_need(root, "nano"))
    o_agg = forecasts(*_need(root, "onepass"))
    loop = forecasts(*_need(root, "loop"))
    two = forecasts(*_need(root, "twopass"))
    items = []
    for q in _questions(root)[:S40_SIZE]:
        qid = q.question_id
        ms = {**first.get(qid, {}), **more.get(qid, {})}
        if not all(m in ms for m in range(S40_DRAWS)):
            continue
        if len(nano.get(qid, {})) < 3 or qid not in loop or qid not in two or qid not in o_agg:
            continue
        draws = tuple(ms[m] for m in range(S40_DRAWS))
        items.append(
            Item(
                qid,
                q.source,
                q.outcome,
                q.freeze_value,
                combined(draws),
                draws,
                tuple(nano[qid][m] for m in range(3)),
                (o_agg[qid], loop[qid], two[qid]),
            )
        )
    return Sample(
        "S40",
        "M7's 40 (ForecastBench 2026-06-21), one research pass as of open, five draws, median",
        MODEL,
        items,
        {
            "draws": [draw_spread(i.draws) for i in items],
            "models": [model_gap(i.draws[:3], i.second) for i in items],
            "passes": [pass_spread(i.briefs) for i in items],
            "confidence": [confidence(i.final) for i in items],
        },
    )


# -- leave-one-out fits -----------------------------------------------------------

Fit = Callable[[Sequence[int]], Callable[[int], float]]
"""Given the training indices, a map from a held-out index to its forecast."""


def loo(n: int, fit: Fit) -> list[float]:
    """Each index forecast by a map fitted on every other index."""
    out = []
    for i in range(n):
        train = [j for j in range(n) if j != i]
        out.append(clip(fit(train)(i)))
    return out


def control_a(items: Sequence[Item]) -> list[float]:
    """Constant shrinkage: Platt on the final forecast, leave-one-out."""
    xs = [_lo(i.final) for i in items]
    ys = [i.outcome for i in items]

    def fit(train: Sequence[int]) -> Callable[[int], float]:
        (a, b), _ = fit_logistic(
            [[xs[j], 1.0] for j in train],
            platt_targets([ys[j] for j in train]),
            [0.0, 0.0],
            [1.0, 0.0],
        )
        return lambda i: sigmoid(a * xs[i] + b)

    return loo(len(items), fit)


def control_b(items: Sequence[Item]) -> list[float]:
    """The source-aware calibrator (partially pooled Platt), leave-one-out."""

    def fit(train: Sequence[int]) -> Callable[[int], float]:
        m = PartialPlatt().fit(
            [items[j].final for j in train],
            [items[j].outcome for j in train],
            [items[j].source for j in train],
        )
        return lambda i: m(items[i].final, items[i].source)

    return loo(len(items), fit)


def fitted_slope(items: Sequence[Item], measure: Sequence[float]) -> list[float]:
    """B plus a disagreement term in the slope: ``(a + c·z)·x + b + u_s``, leave-one-out.

    ``z`` is the measure standardised on the training questions.
    """
    xs = [_lo(i.final) for i in items]
    ys = [i.outcome for i in items]
    sources = sorted({i.source for i in items})
    col = {s: k for k, s in enumerate(sources)}
    precision = 1.0 / (PRIOR_SD * PRIOR_SD)

    def fit(train: Sequence[int]) -> Callable[[int], float]:
        ds = [measure[j] for j in train]
        mu = statistics.fmean(ds)
        sd = statistics.pstdev(ds) or 1.0

        def row(i: int) -> list[float]:
            z = (measure[i] - mu) / sd
            r = [xs[i], xs[i] * z, 1.0] + [0.0] * len(sources)
            r[3 + col[items[i].source]] = 1.0
            return r

        beta, _ = fit_logistic(
            [row(j) for j in train],
            platt_targets([ys[j] for j in train]),
            [0.0, 0.0, 0.0] + [precision] * len(sources),
            [1.0, 0.0, 0.0] + [0.0] * len(sources),
        )
        seen = {items[j].source for j in train}

        def predict(i: int) -> float:
            r = row(i)
            if items[i].source not in seen:  # an unseen source keeps the shared intercept
                r[3 + col[items[i].source]] = 0.0
            return sigmoid(math.fsum(a * b for a, b in zip(r, beta, strict=True)))

        return predict

    return loo(len(items), fit)


# -- comparisons ------------------------------------------------------------------


def _briers(ps: Sequence[float], ys: Sequence[int]) -> list[float]:
    return [brier(clip(p), y) for p, y in zip(ps, ys, strict=True)]


def _h2h(a: Sequence[float], b: Sequence[float], ys: Sequence[int]) -> list[float]:
    """100 × ln(a(outcome) / b(outcome)): the peer-score change of submitting a for b."""
    out = []
    for p, q, y in zip(a, b, ys, strict=True):
        pa, pb = clip(p), clip(q)
        out.append(100.0 * (math.log(pa if y else 1 - pa) - math.log(pb if y else 1 - pb)))
    return out


def _est(e: Estimate, digits: int = 4) -> dict[str, float | int]:
    return {
        "mean": round(e.mean, digits),
        "low": round(e.low, digits),
        "high": round(e.high, digits),
        "n": e.n,
        "mde": round(e.mde, digits) if math.isfinite(e.mde) else -1.0,
    }


def needed(sd: float, delta: float = BRIER_TARGET) -> int:
    """Questions for 80% power on a paired difference of ``delta`` at per-question ``sd``."""
    return math.ceil(((_Z + _Z_POWER) * sd / delta) ** 2)


def compare(
    label: str, policy: Sequence[float], control: Sequence[float], ys: Sequence[int]
) -> dict[str, Any]:
    """Brier(policy) − Brier(control) and the head-to-head score, paired over questions."""
    diffs = [a - b for a, b in zip(_briers(policy, ys), _briers(control, ys), strict=True)]
    d = paired(_briers(policy, ys), _briers(control, ys))
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return {
        "label": label,
        "brier": _est(d),
        "h2h": _est(paired(_h2h(policy, control, ys), [0.0] * len(ys)), 2),
        "sd": round(sd, 5),
        "needed": needed(sd) if sd > 0 else 0,
        "needed_observed": needed(sd, abs(d.mean)) if sd > 0 and d.mean != 0 else 0,
    }


def skip_values(
    control: Sequence[float],
    prices: Sequence[float],
    ys: Sequence[int],
    measure: Sequence[float],
    qids: Sequence[int],
) -> tuple[list[float], list[float], int]:
    """Per-question tournament-unit gains of skipping the most-disagreed quarter.

    A question's peer score against the field (the price) is
    ``100 × ln(control(outcome) / price(outcome))``; a skipped question scores
    0. Against forecasting everything the gain on question i is
    ``−skip_i · peer_i``; against skipping the same number at random it is
    ``−(skip_i − k/n) · peer_i``. Means of these lists are per question
    forecast, averaged over all n.
    """
    n = len(control)
    k = n // SKIP_SHARE
    order = sorted(range(n), key=lambda i: (-measure[i], qids[i]))
    skip = set(order[:k])
    peer = _h2h(control, prices, ys)
    against_all = [-(1.0 if i in skip else 0.0) * peer[i] for i in range(n)]
    against_random = [-((1.0 if i in skip else 0.0) - k / n) * peer[i] for i in range(n)]
    return against_all, against_random, k


def _verdict_brier(cs: Sequence[dict[str, Any]]) -> str:
    if all(c["brier"]["high"] < 0.0 for c in cs):
        return "adopted"
    if any(c["brier"]["low"] > 0.0 for c in cs):
        return "rejected"
    return "not resolvable at this n"


def _verdict_gain(cs: Sequence[dict[str, Any]]) -> str:
    if all(c["gain"]["low"] > 0.0 for c in cs):
        return "adopted"
    if any(c["gain"]["high"] < 0.0 for c in cs):
        return "rejected"
    return "not resolvable at this n"


def analyse(sample: Sample) -> dict[str, Any]:
    """Every M9 number for one sample."""
    items = sample.items
    ys = [i.outcome for i in items]
    finals = [clip(i.final) for i in items]
    labels = large_errors(finals, ys)
    aurocs = {}
    for name, values in sample.measures.items():
        a = auroc_interval(values, labels)
        aurocs[name] = {
            "auroc": round(a.value, 4),
            "low": round(a.low, 4),
            "high": round(a.high, 4),
            "n": a.n,
            "large": a.positives,
        }
    wrong = wrong_side(finals, ys)
    exploratory = {}
    for name, values in sample.measures.items():
        a = auroc_interval(values, wrong)
        exploratory[name] = {
            "auroc": round(a.value, 4),
            "low": round(a.low, 4),
            "high": round(a.high, 4),
            "n": a.n,
            "large": a.positives,
        }
    a_ps = control_a(items)
    b_ps = control_b(items)
    shrink_k = statistics.fmean(1.0 / (1.0 + d) for d in sample.measures["draws"])
    fixed = [
        spread_shrink(i.final, d) for i, d in zip(items, sample.measures["draws"], strict=True)
    ]
    matched = [constant_shrink(i.final, shrink_k) for i in items]
    controls = {
        "A − raw": compare("control A (Platt) − raw", a_ps, finals, ys),
        "B − raw": compare("control B (source-aware) − raw", b_ps, finals, ys),
        "B − A": compare("control B − control A", b_ps, a_ps, ys),
        "matched − raw": compare("matched constant shrink − raw", matched, finals, ys),
    }
    policies: list[dict[str, Any]] = []
    for name, values in sample.measures.items():
        if name == "confidence":
            continue
        ps = fitted_slope(items, values)
        cs = [
            compare("against A", ps, a_ps, ys),
            compare("against B", ps, b_ps, ys),
        ]
        policies.append(
            {
                "policy": f"fitted slope on {name}",
                "kind": "fitted",
                "measure": name,
                "comparisons": cs,
                "verdict": _verdict_brier(cs),
            }
        )
    cs = [
        compare("against A", fixed, a_ps, ys),
        compare("against B", fixed, b_ps, ys),
        compare("against the matched constant", fixed, matched, ys),
    ]
    policies.append(
        {
            "policy": "fixed spread shrink (draws)",
            "kind": "fixed",
            "measure": "draws",
            "comparisons": cs,
            "verdict": _verdict_brier(cs),
        }
    )
    priced = [k for k, i in enumerate(items) if i.price is not None]
    py = [ys[k] for k in priced]
    prices = [clip(float(items[k].price or 0.0)) for k in priced]
    qids = [items[k].qid for k in priced]
    for name, values in sample.measures.items():
        if name == "confidence":
            continue
        cs = []
        for cname, cps in (("A", a_ps), ("B", b_ps)):
            mine = [cps[k] for k in priced]
            m = [values[k] for k in priced]
            against_all, against_random, k = skip_values(mine, prices, py, m, qids)
            for label, vals in (
                (f"on top of {cname}, against forecasting every question", against_all),
                (f"on top of {cname}, against skipping {k} at random", against_random),
            ):
                cs.append(
                    {
                        "label": label,
                        "gain": _est(paired(vals, [0.0] * len(vals)), 2),
                        "skipped": k,
                        "n": len(vals),
                    }
                )
        policies.append(
            {
                "policy": f"skip the top quarter by {name}",
                "kind": "skip",
                "measure": name,
                "comparisons": cs,
                "verdict": _verdict_gain(cs),
            }
        )
    return {
        "name": sample.name,
        "description": sample.description,
        "model": sample.model,
        "n": len(items),
        "priced": len(priced),
        "large_errors": sum(labels),
        "median_brier": round(statistics.median(_briers(finals, ys)), 4),
        "raw_brier": round(statistics.fmean(_briers(finals, ys)), 4),
        "shrink_k": round(shrink_k, 4),
        "mean_measure": {k: round(statistics.fmean(v), 4) for k, v in sample.measures.items()},
        "auroc": aurocs,
        "auroc_wrong_side": exploratory,
        "controls": controls,
        "policies": policies,
    }


# -- the live half ----------------------------------------------------------------


@dataclass(frozen=True)
class LiveUnit:
    """One live or shadow-round binary question: its draws and, if resolved, its outcome."""

    ledger: str
    question_id: int
    opened: date
    model: str
    draws: tuple[float, ...]
    outcome: int | None

    @property
    def half(self) -> str:
        return "policy" if self.question_id % 2 else "control"

    @property
    def week(self) -> str:
        y, w, _ = self.opened.isocalendar()
        return f"{y}-W{w:02d}"


def _eligible_tournament(name: str) -> bool:
    low = name.lower()
    return "bot-testing" not in low and any(t in low for t in LIVE_TOURNAMENTS)


def live_units(root: Path) -> list[LiveUnit]:
    """Every pre-registered unit in the committed ledgers (PLAN.md, the live test).

    The first single-model binary forecast with at least three draws per
    question counts, across ledgers in name order and runs in start order.
    """
    seen: dict[int, LiveUnit] = {}
    outcomes: dict[int, int] = {}
    for path in sorted((root / LEDGERS).glob("*.sqlite")):
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
            if not {"runs", "forecasts", "questions", "llm_calls"} <= tables:
                continue
            if "resolutions" in tables:
                for qid, status, value, qtype in conn.execute(
                    "SELECT question_id, status, value, question_type FROM resolutions "
                    "ORDER BY recorded_at"
                ):
                    if status == "resolved" and qtype == "binary" and value is not None:
                        outcomes[int(qid)] = binary_outcome(str(value))
            marks = ", ".join("?" for _ in LIVE_MODES)
            runs = conn.execute(
                f"SELECT run_id, tournament FROM runs WHERE mode IN ({marks}) "  # noqa: S608
                "ORDER BY started_at",
                LIVE_MODES,
            ).fetchall()
            for run_id, tournament in runs:
                if not _eligible_tournament(str(tournament)):
                    continue
                rows: dict[int, list[tuple[float, str]]] = {}
                # A binary member has a probability (a ledger from before
                # migration 0004 has no question_type column, and only binary rows).
                for qid, p, model in conn.execute(
                    "SELECT f.question_id, f.probability, c.model FROM forecasts f "
                    "JOIN llm_calls c ON c.call_id = f.call_id "
                    "WHERE f.run_id = ? AND f.kind = 'member' "
                    "AND f.probability IS NOT NULL ORDER BY f.member",
                    (run_id,),
                ):
                    rows.setdefault(int(qid), []).append((float(p), str(model)))
                for qid, ds in rows.items():
                    models = {m for _, m in ds}
                    if qid in seen or len(ds) < LIVE_MIN_DRAWS or len(models) != 1:
                        continue
                    (opened,) = conn.execute(
                        "SELECT open_time FROM questions WHERE run_id = ? AND question_id = ?",
                        (run_id, qid),
                    ).fetchone()
                    seen[qid] = LiveUnit(
                        path.name,
                        qid,
                        datetime.fromisoformat(str(opened)).date(),
                        models.pop(),
                        tuple(p for p, _ in ds),
                        None,
                    )
    return [
        LiveUnit(u.ledger, u.question_id, u.opened, u.model, u.draws, outcomes.get(u.question_id))
        for u in sorted(seen.values(), key=lambda u: u.question_id)
    ]


def live_analysis(units: Sequence[LiveUnit], k: float, today: date) -> dict[str, Any]:
    """The pre-registered analysis, run once the odd half has enough resolutions.

    Until then it reports the counts only: no interval is computed on a
    half that has not reached the fixed n (nor the deadline).
    """
    out: dict[str, Any] = {
        "units": len(units),
        "by_half": {},
        "target": LIVE_TARGET,
        "deadline": LIVE_DEADLINE.isoformat(),
        "k": round(k, 4),
    }
    resolved = {
        h: [u for u in units if u.half == h and u.outcome is not None]
        for h in ("policy", "control")
    }
    for h in ("policy", "control"):
        out["by_half"][h] = {
            "units": sum(1 for u in units if u.half == h),
            "resolved": len(resolved[h]),
        }
    due = len(resolved["policy"]) >= LIVE_TARGET or today >= LIVE_DEADLINE
    out["due"] = due
    if not due:
        return out
    for h, us in resolved.items():
        if len(us) < 2:
            continue
        ys = [int(u.outcome or 0) for u in us]
        default = [combined(u.draws) for u in us]
        policy = [spread_shrink(p, draw_spread(u.draws)) for p, u in zip(default, us, strict=True)]
        matched = [constant_shrink(p, k) for p in default]
        weeks: list[Hashable] = [u.week for u in us]
        out["by_half"][h]["comparisons"] = [
            {
                "label": label,
                "brier": _est(paired(_briers(policy, ys), _briers(ctrl, ys), weeks)),
                "h2h": _est(paired(_h2h(policy, ctrl, ys), [0.0] * len(ys), weeks), 2),
            }
            for label, ctrl in (
                ("against the default", default),
                ("against the matched constant", matched),
            )
        ]
    return out


# -- the snapshot -----------------------------------------------------------------


def snapshot(root: Path, today: date) -> dict[str, Any]:
    """Every number M9 reports (the live part's counts depend on ``today`` only past
    the deadline)."""
    s138, s40 = analyse(load_s138(root)), analyse(load_s40(root))
    units = live_units(root)
    return {
        "method": {
            "bootstrap": "percentile, resampling questions, 10,000 resamples, seeded",
            "large_error": "Brier strictly above the sample's median Brier",
            "fits": "leave-one-out within the sample",
            "prior_sd": PRIOR_SD,
            "skip_share": f"1/{SKIP_SHARE}",
            "brier_target": BRIER_TARGET,
        },
        "power": {
            "auroc_target": AUROC_TARGET,
            "auroc_needed": auroc_needed(),
            "auroc_halfwidth_40": round(_Z * auroc_se(0.5, 20, 20), 3),
            "auroc_halfwidth_138": round(_Z * auroc_se(0.5, 69, 69), 3),
        },
        "samples": [s138, s40],
        "live": live_analysis(units, s40["shrink_k"], today),
    }
