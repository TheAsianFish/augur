"""Q2: what leakage is worth, and whether the research loop helps. Scored from the ledgers.

The arms (PLAN.md, "M6's predictions"), all on the same fixed sample of C2:

* **N**, no retrieval: M5's C2 arm (b);
* **O**, one research pass as of each question's open instant (a backtest);
* **T**, the same pass as of the moment the run reached the question (an
  ``unrestricted`` run: leaky on purpose, never scored as a backtest);
* **L**, the bounded research loop as of open, on the first part of the
  sample.

Everything here reads committed ledgers and is deterministic (the
bootstrap is seeded), so ``augur report`` renders it and ``make eval``
checks it.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from augur.forecastbench import FBQuestion
from augur.judge import JUDGE_PREFIX, MAJOR, Agreement, agreement
from augur.replay import clip, recorded_prices
from augur.score import Estimate, brier, clustered_bootstrap, log_score, paired

ARM_NAMES = {
    "N": "no retrieval",
    "O": "one pass, sources as of open",
    "T": "one pass, sources as of today (leaky on purpose)",
    "L": "research loop, sources as of open",
}


def forecasts(path: Path, tournament: str) -> dict[int, float]:
    """The latest aggregate per question of one arm, whatever the run's mode."""
    latest: dict[int, tuple[str, float]] = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        for qid, made, p in conn.execute(
            "SELECT f.question_id, f.recorded_at, f.probability FROM forecasts f "
            "JOIN runs r ON r.run_id = f.run_id "
            "WHERE r.tournament = ? AND f.kind = 'aggregate' AND r.mode != 'live'",
            (tournament,),
        ):
            if int(qid) not in latest or made > latest[int(qid)][0]:
                latest[int(qid)] = (str(made), float(p))
    return {q: v[1] for q, v in latest.items()}


@dataclass(frozen=True)
class ArmRow:
    arm: str
    n: int
    brier: Estimate
    log: float


@dataclass(frozen=True)
class DiffRow:
    label: str
    brier: Estimate


def _briers(ps: Sequence[float], ys: Sequence[int]) -> list[float]:
    return [brier(clip(p), y) for p, y in zip(ps, ys, strict=True)]


def arm_row(arm: str, ps: Sequence[float], ys: Sequence[int]) -> ArmRow:
    logs = [log_score(clip(p) if y else 1.0 - clip(p)) for p, y in zip(ps, ys, strict=True)]
    return ArmRow(arm, len(ps), clustered_bootstrap(_briers(ps, ys)), statistics.fmean(logs))


def diff_row(label: str, a: Sequence[float], b: Sequence[float], ys: Sequence[int]) -> DiffRow:
    """Brier(a) − Brier(b): negative means ``a`` is better."""
    return DiffRow(label, paired(_briers(a, ys), _briers(b, ys)))


@dataclass(frozen=True)
class JudgeRow:
    arm: str
    questions: int
    sources: int
    by_score: tuple[int, int, int, int, int]
    major_sources: int
    major_questions: int


def judge_row(arm: str, path: Path, tournament: str, sample: set[int]) -> JudgeRow | None:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT j.question_id, j.source_id, j.score, j.recorded_at FROM leak_judgments j "
            "JOIN runs r ON r.run_id = j.run_id WHERE r.tournament = ? ORDER BY j.recorded_at",
            (JUDGE_PREFIX + tournament,),
        ).fetchall()
    latest: dict[tuple[int, str], int] = {}
    for qid, sid, score, _ in rows:
        if int(qid) in sample:
            latest[(int(qid), str(sid))] = int(score)
    if not latest:
        return None
    counts = [0] * 5
    for v in latest.values():
        counts[v] += 1
    flagged = {q for (q, _), v in latest.items() if v >= MAJOR}
    return JudgeRow(
        arm=arm,
        questions=len({q for q, _ in latest}),
        sources=len(latest),
        by_score=(counts[0], counts[1], counts[2], counts[3], counts[4]),
        major_sources=sum(1 for v in latest.values() if v >= MAJOR),
        major_questions=len(flagged),
    )


def major_questions(path: Path, tournament: str) -> set[int]:
    """Questions with a source whose latest judgment is 3 or more (as ``judge_row`` counts)."""
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT j.question_id, j.source_id, j.score FROM leak_judgments j "
            "JOIN runs r ON r.run_id = j.run_id WHERE r.tournament = ? ORDER BY j.recorded_at",
            (JUDGE_PREFIX + tournament,),
        ).fetchall()
    latest = {(int(q), str(s)): int(v) for q, s, v in rows}
    return {q for (q, _), v in latest.items() if v >= MAJOR}


def reference_agreement(labels_path: Path, arms: dict[str, tuple[Path, str]]) -> Agreement | None:
    """The judge against the reference labels in ``labels_path`` (one list of objects).

    Each label names its ``arm``, ``question_id``, ``source_id`` and
    ``score``; it is paired with the judge's score for that source in that
    arm's ledger.
    """
    if not labels_path.exists():
        return None
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
    pairs: list[tuple[int, int]] = []
    cache: dict[str, dict[tuple[int, str], int]] = {}
    for item in labels:
        arm = str(item["arm"])
        if arm not in arms:
            continue
        if arm not in cache:
            path, tournament = arms[arm]
            with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
                cache[arm] = {
                    (int(q), str(s)): int(v)
                    for q, s, v in conn.execute(
                        "SELECT j.question_id, j.source_id, j.score FROM leak_judgments j "
                        "JOIN runs r ON r.run_id = j.run_id WHERE r.tournament = ? "
                        "ORDER BY j.recorded_at",
                        (JUDGE_PREFIX + tournament,),
                    )
                }
        judged = cache[arm].get((int(item["question_id"]), str(item["source_id"])))
        if judged is not None:
            pairs.append((int(item["score"]), judged))
    return agreement(pairs) if pairs else None


@dataclass(frozen=True)
class CostRow:
    arm: str
    questions: int
    calls: float
    research_calls: float
    output_tokens: float
    seconds: float


def cost_row(arm: str, path: Path, tournament: str, sample: set[int]) -> CostRow | None:
    """Model calls, output tokens and wall seconds per question forecast, by arm."""
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
        for qid, role, out in conn.execute(
            f"SELECT question_id, role, output_tokens FROM llm_calls "  # noqa: S608
            f"WHERE run_id IN ({marks}) AND error IS NULL",
            runs,
        ):
            if int(qid) in done:
                calls += 1
                research += role != "forecaster"
                tokens += int(out or 0)
    n = len(done)
    return CostRow(arm, n, calls / n, research / n, tokens / n, statistics.fmean(done.values()))


@dataclass(frozen=True)
class Q2:
    sample: int
    arms: list[ArmRow]
    diffs: list[DiffRow]
    loop_n: int
    loop_arms: list[ArmRow]
    loop_diffs: list[DiffRow]
    priced: int
    edge: list[DiffRow]
    unpriced: list[ArmRow]
    judges: list[JudgeRow]
    flagged_n: int
    flagged: list[ArmRow]
    agree: Agreement | None
    costs: list[CostRow]


@dataclass(frozen=True)
class ArmSource:
    path: Path
    tournament: str


def score_q2(
    *,
    questions: Sequence[FBQuestion],
    sample: int,
    loop_sample: int,
    arms: dict[str, ArmSource],
    prices: ArmSource,
    labels: Path,
) -> Q2:
    """Every M6 table, paired on the questions every compared arm forecast."""
    ordered = sorted(questions, key=lambda q: q.question_id)[:sample]
    outcome = {q.question_id: q.outcome for q in ordered}
    got = {a: forecasts(s.path, s.tournament) for a, s in arms.items()}
    main = [a for a in ("N", "O", "T") if a in got]
    common = [q.question_id for q in ordered if all(q.question_id in got[a] for a in main)]
    ys = [outcome[q] for q in common]
    rows = [arm_row(a, [got[a][q] for q in common], ys) for a in main]
    col = {a: [got[a][q] for q in common] for a in main}
    diffs = []
    if "N" in col and "T" in col:
        diffs.append(
            diff_row("T − N (what retrieval as of today is worth)", col["T"], col["N"], ys)
        )
    if "N" in col and "O" in col:
        diffs.append(diff_row("O − N (what leak-free retrieval is worth)", col["O"], col["N"], ys))
    if "O" in col and "T" in col:
        diffs.append(diff_row("T − O (the leak, same queries)", col["T"], col["O"], ys))
    # The loop, on its own (smaller) sample, against the one pass and no retrieval.
    loop_q = [
        q.question_id
        for q in ordered[:loop_sample]
        if q.question_id in got.get("L", {}) and q.question_id in got.get("O", {})
    ]
    loop_arms: list[ArmRow] = []
    loop_diffs: list[DiffRow] = []
    if loop_q:
        yl = [outcome[q] for q in loop_q]
        for a in ("N", "O", "L"):
            if all(q in got.get(a, {}) for q in loop_q):
                loop_arms.append(arm_row(a, [got[a][q] for q in loop_q], yl))
        loop_diffs.append(
            diff_row(
                "L − O (loop against one pass)",
                [got["L"][q] for q in loop_q],
                [got["O"][q] for q in loop_q],
                yl,
            )
        )
    # Edge over the price as of the freeze, priced questions only.
    quotes = {
        q: s.quote.probability
        for q, s in recorded_prices(prices.path, prices.tournament).items()
        if s.quote is not None
    }
    priced = [q for q in common if q in quotes]
    yp = [outcome[q] for q in priced]
    edge = [
        diff_row(f"{a} − price", [got[a][q] for q in priced], [quotes[q] for q in priced], yp)
        for a in main
        if priced
    ]
    loop_priced = [q for q in loop_q if q in quotes]
    if loop_priced:
        edge.append(
            diff_row(
                f"L − price (loop sample, n = {len(loop_priced)})",
                [got["L"][q] for q in loop_priced],
                [quotes[q] for q in loop_priced],
                [outcome[q] for q in loop_priced],
            )
        )
    unpriced_q = [q for q in common if q not in quotes]
    yu = [outcome[q] for q in unpriced_q]
    unpriced = [arm_row(a, [got[a][q] for q in unpriced_q], yu) for a in main if unpriced_q]
    in_sample = set(common)
    judges = [
        row
        for a in ("O", "T", "L")
        if a in arms
        for row in [judge_row(a, arms[a].path, arms[a].tournament, in_sample)]
        if row is not None
    ]
    flagged_rows: list[ArmRow] = []
    flagged_q: list[int] = []
    if "T" in arms:
        hit = major_questions(arms["T"].path, arms["T"].tournament)
        flagged_q = [q for q in common if q in hit]
        if flagged_q:
            yf = [outcome[q] for q in flagged_q]
            flagged_rows = [arm_row(a, [got[a][q] for q in flagged_q], yf) for a in main]
    agree = reference_agreement(
        labels, {a: (arms[a].path, arms[a].tournament) for a in ("O", "T", "L") if a in arms}
    )
    costs = [
        row
        for a in ("O", "T", "L")
        if a in arms
        for row in [cost_row(a, arms[a].path, arms[a].tournament, in_sample)]
        if row is not None
    ]
    return Q2(
        sample=len(common),
        arms=rows,
        diffs=diffs,
        loop_n=len(loop_q),
        loop_arms=loop_arms,
        loop_diffs=loop_diffs,
        priced=len(priced),
        edge=edge,
        unpriced=unpriced,
        judges=judges,
        flagged_n=len(flagged_q),
        flagged=flagged_rows,
        agree=agree,
        costs=costs,
    )
