"""Q3: which part carries the score. M7's ablation, scored from the ledgers.

Every arm is on the same questions: the first 40 of M6's fixed C2 sample
(PLAN.md, "M7's predictions"). One factor changes at a time against the
default, and every row names its model:

* **research depth:** no retrieval (M5's N), one pass (M6's O), two passes
  with sub-questions (M7);
* **head count:** the median of the first k of nine draws on O's recorded
  briefs (members 0 to 2 from M6, 3 to 8 from M7), k = 1, 3, 5, 9;
* **aggregation:** four rules over the same nine draws, no new calls;
* **model mix:** three draws of a second model on the same briefs, mixed
  with three of the first, against six of the first.

Deterministic (the bootstrap is seeded), so ``augur report`` renders it.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from augur.aggregate import combine_binary
from augur.forecastbench import FBQuestion
from augur.q2 import ArmSource, forecasts
from augur.replay import clip, recorded_prices
from augur.score import Estimate, brier, clustered_bootstrap, log_score, paired

DEFAULT_K = 5
HEAD_COUNTS = (1, 3, 5, 9)
RULE_NAMES = {
    "median": "median (default)",
    "trimmed_mean": "trimmed mean (20% each side)",
    "mean_logodds": "mean of log-odds",
    "extremized_logodds": "extremized log-odds, factor 1.5",
}
NON_INFERIORITY = 0.01
"""The replacement rule's margin (PLAN.md, fixed before the run)."""


def members(path: Path, tournament: str) -> dict[int, dict[int, float]]:
    """Each question's member probabilities by member number (the latest of each)."""
    out: dict[int, dict[int, tuple[str, float]]] = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        for qid, member, made, p in conn.execute(
            "SELECT f.question_id, f.member, f.recorded_at, f.probability FROM forecasts f "
            "JOIN runs r ON r.run_id = f.run_id "
            "WHERE r.tournament = ? AND f.kind = 'member' AND r.mode = 'backtest'",
            (tournament,),
        ):
            slot = out.setdefault(int(qid), {})
            if int(member) not in slot or made > slot[int(member)][0]:
                slot[int(member)] = (str(made), float(p))
    return {q: {m: v[1] for m, v in ms.items()} for q, ms in out.items()}


@dataclass(frozen=True)
class Usage:
    """Per-question model calls, output tokens and wall seconds of one arm's run(s)."""

    calls: float
    research_calls: float
    output_tokens: float
    seconds: float
    draw_seconds: float
    """Median latency of one forecaster call."""


def usage(path: Path, tournament: str, sample: set[int]) -> Usage | None:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        runs = [
            r
            for (r,) in conn.execute("SELECT run_id FROM runs WHERE tournament = ?", (tournament,))
        ]
        if not runs:
            return None
        marks = ", ".join("?" for _ in runs)
        done = {
            int(q): float(s or 0.0)
            for q, s in conn.execute(
                f"SELECT question_id, seconds FROM question_outcomes "  # noqa: S608
                f"WHERE run_id IN ({marks}) AND status = 'forecast'",
                runs,
            )
            if int(q) in sample
        }
        if not done:
            return None
        calls = research = tokens = 0
        draws: list[float] = []
        for qid, role, out, ms in conn.execute(
            f"SELECT question_id, role, output_tokens, latency_ms FROM llm_calls "  # noqa: S608
            f"WHERE run_id IN ({marks}) AND error IS NULL",
            runs,
        ):
            if int(qid) in done:
                calls += 1
                research += role != "forecaster"
                tokens += int(out or 0)
                if role == "forecaster":
                    draws.append(int(ms) / 1000.0)
    n = len(done)
    return Usage(
        calls / n,
        research / n,
        tokens / n,
        statistics.fmean(done.values()),
        statistics.median(draws) if draws else 0.0,
    )


@dataclass(frozen=True)
class Row:
    label: str
    model: str
    brier: Estimate
    log: float
    n: int
    calls: float | None = None
    tokens: float | None = None
    seconds: float | None = None


@dataclass(frozen=True)
class Diff:
    label: str
    brier: Estimate
    h2h: Estimate
    """Mean head-to-head score of the first arm against the second, 100 × ln on the outcome."""
    verdict: str = ""


def _briers(ps: Sequence[float], ys: Sequence[int]) -> list[float]:
    return [brier(clip(p), y) for p, y in zip(ps, ys, strict=True)]


def _logs(ps: Sequence[float], ys: Sequence[int]) -> list[float]:
    return [log_score(clip(p) if y else 1.0 - clip(p)) for p, y in zip(ps, ys, strict=True)]


def h2h_scores(a: Sequence[float], b: Sequence[float], ys: Sequence[int]) -> list[float]:
    """Per question, 100 × ln(p_a(outcome) / p_b(outcome)): Metaculus's head-to-head."""
    return [
        100.0 * (math.log(clip(x) if y else 1 - clip(x)) - math.log(clip(z) if y else 1 - clip(z)))
        for x, z, y in zip(a, b, ys, strict=True)
    ]


def row(
    label: str, model: str, ps: Sequence[float], ys: Sequence[int], u: Usage | None = None
) -> Row:
    return Row(
        label,
        model,
        clustered_bootstrap(_briers(ps, ys)),
        statistics.fmean(_logs(ps, ys)),
        len(ps),
        None if u is None else u.calls,
        None if u is None else u.output_tokens,
        None if u is None else u.seconds,
    )


def replacement_verdict(d: Estimate, cheaper: bool) -> str:
    """M7's replacement rule on a Brier difference (variant − default)."""
    if d.high < 0.0:
        return "replaces the default (interval below 0)"
    if cheaper and d.high < NON_INFERIORITY:
        return f"replaces the default (cheaper, no worse: upper bound < +{NON_INFERIORITY})"
    return "default stays"


def diff(
    label: str,
    a: Sequence[float],
    b: Sequence[float],
    ys: Sequence[int],
    cheaper: bool | None = None,
) -> Diff:
    d = paired(_briers(a, ys), _briers(b, ys))
    return Diff(
        label,
        d,
        clustered_bootstrap(h2h_scores(a, b, ys)),
        "" if cheaper is None else replacement_verdict(d, cheaper),
    )


def combined(ps: Sequence[float], rule: str = "median") -> float:
    """The aggregate as the bot records it (three decimals)."""
    return round(combine_binary(list(ps), rule), 3)


@dataclass(frozen=True)
class Q3:
    n: int
    short: int
    """Questions where a draw returned nothing, so 'nine draws' is eight."""
    model: str
    second_model: str
    depth: list[Row]
    depth_diffs: list[Diff]
    heads: list[Row]
    head_diffs: list[Diff]
    rules: list[Row]
    rule_diffs: list[Diff]
    mix: list[Row]
    mix_diffs: list[Diff]
    priced: int
    edge: list[Diff]
    unpriced: list[Row]
    paired_sd: dict[str, tuple[float, float]]
    """Per-question SD of the Brier and head-to-head differences, by comparison."""
    differences: dict[str, tuple[list[float], list[float]]]
    """Per question, Brier(default) − Brier(variant) and the variant's head-to-head
    score against the default (both larger = the variant better), by comparison."""


def score_q3(
    *,
    questions: Sequence[FBQuestion],
    sample: int,
    none: ArmSource,
    onepass: ArmSource,
    draws: ArmSource,
    twopass: ArmSource,
    second: ArmSource,
    prices: ArmSource,
    model: str,
    second_model: str,
) -> Q3 | None:
    """Every M7 table, on the questions every arm forecast."""
    ordered = sorted(questions, key=lambda q: q.question_id)[:sample]
    outcome = {q.question_id: q.outcome for q in ordered}
    n_agg = forecasts(none.path, none.tournament)
    o_agg = forecasts(onepass.path, onepass.tournament)
    d2_agg = forecasts(twopass.path, twopass.tournament) if twopass.path.exists() else {}
    first = members(onepass.path, onepass.tournament)
    more = members(draws.path, draws.tournament) if draws.path.exists() else {}
    other = members(second.path, second.tournament) if second.path.exists() else {}
    nine: dict[int, list[float]] = {}
    for q in outcome:
        ms = {**first.get(q, {}), **more.get(q, {})}
        # A draw that returned nothing is dropped, as a live run drops it: k
        # forecasters are the first k members that answered, in member order.
        answered = [ms[m] for m in range(9) if m in ms]
        if len(answered) >= 8 and all(m in ms for m in range(3)):
            nine[q] = answered
    common = [
        q
        for q in outcome
        if q in n_agg and q in o_agg and q in d2_agg and q in nine and len(other.get(q, {})) >= 3
    ]
    if not common:
        return None
    ys = [outcome[q] for q in common]
    sample_ids = set(common)

    def use(src: ArmSource) -> Usage | None:
        return usage(src.path, src.tournament, sample_ids)

    u_o, u_d2, u_more = use(onepass), use(twopass), use(draws)
    research_calls = u_o.research_calls if u_o else 0.0
    per_draw_tokens = (u_more.output_tokens / 6.0) if u_more else 0.0
    research_tokens = (u_o.output_tokens - 3 * per_draw_tokens) if u_o else 0.0

    col: dict[str, list[float]] = {
        "D0": [n_agg[q] for q in common],
        "D1": [o_agg[q] for q in common],
        "D2": [d2_agg[q] for q in common],
    }
    depth = [
        row("D0: no retrieval", model, col["D0"], ys, None),
        row("D1: one pass (default)", model, col["D1"], ys, u_o),
        row("D2: two passes with sub-questions", model, col["D2"], ys, u_d2),
    ]
    depth_diffs = [
        diff("D1 − D0 (one pass against none)", col["D1"], col["D0"], ys),
        diff("D2 − D1 (second pass against one pass)", col["D2"], col["D1"], ys, cheaper=False),
        diff("D2 − D0", col["D2"], col["D0"], ys),
    ]

    def first_k(k: int, rule: str = "median") -> list[float]:
        return [combined(nine[q][:k], rule) for q in common]

    heads = [
        Row(
            f"{k} forecaster{'s' if k > 1 else ''}" + (" (default)" if k == DEFAULT_K else ""),
            model,
            clustered_bootstrap(_briers(first_k(k), ys)),
            statistics.fmean(_logs(first_k(k), ys)),
            len(common),
            research_calls + k,
            research_tokens + k * per_draw_tokens,
            None,
        )
        for k in HEAD_COUNTS
    ]
    base = first_k(DEFAULT_K)
    head_diffs = [
        diff(f"{k} − {DEFAULT_K}", first_k(k), base, ys, cheaper=k < DEFAULT_K)
        for k in HEAD_COUNTS
        if k != DEFAULT_K
    ]
    rules = [row(f"{RULE_NAMES[r]}, 9 draws", model, first_k(9, r), ys) for r in RULE_NAMES]
    med9 = first_k(9)
    rule_diffs = [
        diff(f"{RULE_NAMES[r]} − median", first_k(9, r), med9, ys, cheaper=False)
        for r in RULE_NAMES
        if r != "median"
    ]

    def theirs(k: int) -> list[float]:
        return [combined([other[q][m] for m in range(k)]) for q in common]

    ultra3, ultra6 = first_k(3), first_k(6)
    nano3 = theirs(3)
    mixed = [combined(nine[q][:3] + [other[q][m] for m in range(3)]) for q in common]
    mix = [
        row("3 draws", model, ultra3, ys),
        row("3 draws", second_model, nano3, ys),
        row("6 draws", model, ultra6, ys),
        row("mixed: 3 + 3 draws", f"{model} + {second_model}", mixed, ys),
    ]
    mix_diffs = [
        diff("mixed 3 + 3 − 6 of the first model", mixed, ultra6, ys, cheaper=False),
        diff("second model × 3 − first model × 3", nano3, ultra3, ys),
    ]
    quotes = {
        q: s.quote.probability
        for q, s in recorded_prices(prices.path, prices.tournament).items()
        if s.quote is not None
    }
    priced = [i for i, q in enumerate(common) if q in quotes]
    yp = [ys[i] for i in priced]
    price = [quotes[common[i]] for i in priced]
    arms: dict[str, list[float]] = {
        "D0 no retrieval": col["D0"],
        "D1 one pass, 5 draws": base,
        "D2 two passes": col["D2"],
        "9 draws, median": med9,
        "mixed fleet": mixed,
    }
    edge = [
        diff(f"{name} − price", [ps[i] for i in priced], price, yp) for name, ps in arms.items()
    ]
    unpriced_i = [i for i, q in enumerate(common) if q not in quotes]
    yu = [ys[i] for i in unpriced_i]
    unpriced = [row(name, model, [ps[i] for i in unpriced_i], yu) for name, ps in arms.items()]

    def per_question(variant: list[float], default: list[float]) -> tuple[list[float], list[float]]:
        gain = [z - x for x, z in zip(_briers(variant, ys), _briers(default, ys), strict=True)]
        return gain, h2h_scores(variant, default, ys)

    differences = {
        "research (D2 against D1)": per_question(col["D2"], col["D1"]),
        "retrieval (D1 against D0)": per_question(col["D1"], col["D0"]),
        "model mix (mixed against 6 draws)": per_question(mixed, ultra6),
        "head count (9 against 5)": per_question(med9, base),
    }
    paired_sd = {k: (statistics.stdev(b), statistics.stdev(h)) for k, (b, h) in differences.items()}
    return Q3(
        n=len(common),
        short=sum(1 for q in common if len(nine[q]) < 9),
        model=model,
        second_model=second_model,
        depth=depth,
        depth_diffs=depth_diffs,
        heads=heads,
        head_diffs=head_diffs,
        rules=rules,
        rule_diffs=rule_diffs,
        mix=mix,
        mix_diffs=mix_diffs,
        priced=len(priced),
        edge=edge,
        unpriced=unpriced if unpriced_i else [],
        paired_sd=paired_sd,
        differences=differences,
    )


@dataclass(frozen=True)
class Throughput:
    """What one arm's model calls achieved under the endpoint's limits, from the ledger."""

    arm: str
    questions: int
    calls: int
    failed: int
    minutes: float
    per_minute: float
    median_latency_s: float
    p90_latency_s: float


def throughput(arm: str, path: Path, tournament: str) -> Throughput | None:
    """Calls per minute of wall time, and each call's latency.

    A backtest's call times are its pinned clock (each question's freeze
    instant), so wall time comes from the questions' own measured seconds
    (``question_outcomes.seconds``, a monotonic clock): the questions of a
    run go one after another, so their sum is the run's working time.
    """
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        runs = [
            r
            for (r,) in conn.execute("SELECT run_id FROM runs WHERE tournament = ?", (tournament,))
        ]
        if not runs:
            return None
        marks = ", ".join("?" for _ in runs)
        seconds = [
            float(x or 0.0)
            for q, x in conn.execute(
                f"SELECT question_id, seconds FROM question_outcomes "  # noqa: S608
                f"WHERE run_id IN ({marks})",
                runs,
            )
        ]
        rows = conn.execute(
            f"SELECT latency_ms, error FROM llm_calls WHERE run_id IN ({marks})",  # noqa: S608
            runs,
        ).fetchall()
    wall = math.fsum(seconds)
    if len(rows) < 2 or wall <= 0:
        return None
    latencies = sorted(int(r[0]) / 1000.0 for r in rows)
    return Throughput(
        arm=arm,
        questions=len(seconds),
        calls=len(rows),
        failed=sum(1 for r in rows if r[1] is not None),
        minutes=wall / 60.0,
        per_minute=len(rows) / (wall / 60.0),
        median_latency_s=statistics.median(latencies),
        p90_latency_s=latencies[min(len(latencies) - 1, int(0.9 * len(latencies)))],
    )
