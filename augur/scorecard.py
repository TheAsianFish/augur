"""Score the bot from its ledgers: every resolved question, every comparison, with intervals.

A question is scored when a ledger holds Augur's aggregate forecast for it
and a resolution (the latest row; annulled and ambiguous questions are not
scored, and are counted). The forecast scored is the one Augur posted when
it posted one, otherwise its latest aggregate (dry runs and backtests).

Comparisons are always on the questions both forecasters have:

- **references** from the ledger: the shadow template bot, the Metaculus
  community prediction, a market price;
- **constant forecasts** on binary questions: always 50%, and the share of
  the scored binary questions that resolved Yes (the base rate, known only
  after the fact, so a little flattering to the constant).

Every comparison is a paired, clustered bootstrap (:mod:`augur.score`) of the
per-question head-to-head score, and on binary questions of the Brier
difference, each with its minimum detectable effect.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Hashable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from augur.clock import parse_iso
from augur.distributions import Scaling
from augur.questions import question_from_post
from augur.score import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Estimate,
    ScoringError,
    baseline_score,
    binary_outcome,
    brier,
    bucket_index,
    clustered_bootstrap,
    head_to_head,
    log_score,
    outcome_probability,
    paired,
)

Cluster = Literal["question", "week", "source"]


@dataclass(frozen=True)
class ScoredQuestion:
    question_id: int
    post_id: int
    question_type: str
    title: str
    opened: datetime
    resolution: str
    source: str
    submitted: bool
    model: str
    augur: dict[str, Any]
    references: dict[str, dict[str, Any]]
    options: tuple[str, ...] = ()
    scaling: Scaling | None = None
    metaculus: dict[str, float] = field(default_factory=dict)

    def p(self, values: dict[str, Any]) -> float:
        return outcome_probability(
            self.question_type,
            values,
            self.resolution,
            options=self.options or None,
            scaling=self.scaling,
        )

    def baseline(self, values: dict[str, Any]) -> float:
        p = self.p(values)
        bucket = (
            bucket_index(self.resolution, self.scaling)
            if self.scaling is not None and self.question_type in ("numeric", "discrete")
            else None
        )
        return baseline_score(
            self.question_type,
            p,
            option_count=max(2, len(self.options)),
            bucket=bucket,
            scaling=self.scaling,
        )

    def cluster(self, how: Cluster) -> Hashable:
        if how == "week":
            year, week, _ = self.opened.isocalendar()
            return f"{year}-W{week:02d}"
        if how == "source":
            return self.source
        return self.question_id


@dataclass
class Loaded:
    scored: list[ScoredQuestion]
    unresolved: int = 0
    annulled: int = 0
    unscorable: list[str] = field(default_factory=list)


def _latest_by_question(
    conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()
) -> dict[Any, sqlite3.Row]:
    out: dict[Any, sqlite3.Row] = {}
    for row in conn.execute(sql, params):
        out[row[0]] = row
    return out


def load(paths: Sequence[Path], *, modes: Sequence[str] | None = None) -> Loaded:
    """Every question with an Augur aggregate in the ledgers at ``paths``."""
    result = Loaded(scored=[])
    seen: set[int] = set()
    for path in paths:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            if not _has_table(conn, "resolutions"):
                continue
            _load_one(conn, result, seen, modes)
    result.scored.sort(key=lambda s: (s.opened, s.question_id))
    return result


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _load_one(
    conn: sqlite3.Connection, result: Loaded, seen: set[int], modes: Sequence[str] | None
) -> None:
    mode_sql = ""
    params: list[Any] = []
    if modes:
        mode_sql = f" AND r.mode IN ({', '.join('?' for _ in modes)})"
        params = list(modes)
    # Posted forecasts sort after unposted ones, then by time: the last row wins.
    # Only "?" placeholders are formatted into the text; no input reaches it.
    sql = (
        "SELECT f.question_id, f.values_json, f.run_id, r.mode, r.tournament, r.model, "  # noqa: S608
        "EXISTS (SELECT 1 FROM submissions s WHERE s.run_id = f.run_id "
        "  AND s.question_id = f.question_id AND s.kind = 'forecast' "
        "  AND s.status = 'submitted') AS posted "
        "FROM forecasts f JOIN runs r ON r.run_id = f.run_id "
        f"WHERE f.kind = 'aggregate'{mode_sql} "
        "ORDER BY posted, f.recorded_at, f.rowid"
    )
    forecasts = _latest_by_question(conn, sql, params)
    resolutions = _latest_by_question(
        conn,
        "SELECT question_id, status, value FROM resolutions ORDER BY recorded_at, rowid",
    )
    references: dict[int, dict[str, dict[str, Any]]] = {}
    for row in conn.execute(
        "SELECT question_id, forecaster, values_json FROM reference_forecasts "
        "ORDER BY recorded_at, rowid"
    ):
        references.setdefault(int(row[0]), {})[str(row[1])] = json.loads(row[2])
    metaculus: dict[int, dict[str, float]] = {}
    for row in conn.execute(
        "SELECT question_id, score_type, score FROM metaculus_scores "
        "WHERE forecaster = 'augur' ORDER BY recorded_at, rowid"
    ):
        metaculus.setdefault(int(row[0]), {})[str(row[1])] = float(row[2])
    for qid, f in forecasts.items():
        qid = int(qid)
        if qid in seen:
            continue
        res = resolutions.get(qid)
        if res is None:
            result.unresolved += 1
            continue
        if res["status"] != "resolved":
            result.annulled += 1
            continue
        q = conn.execute(
            "SELECT post_id, type, title, open_time, payload_json FROM questions "
            "WHERE run_id = ? AND question_id = ?",
            (f["run_id"], qid),
        ).fetchone()
        payload = json.loads(q["payload_json"])
        options: tuple[str, ...] = ()
        scaling = None
        source = str(f["tournament"])
        if isinstance(payload.get("question"), dict):
            question = question_from_post(payload)
            options = tuple(question.options or ())
            scaling = question.scaling
        elif "fb_id" in payload:
            source = str(payload.get("source"))
        try:
            scored = ScoredQuestion(
                question_id=qid,
                post_id=int(q["post_id"]),
                question_type=str(q["type"]),
                title=str(q["title"]),
                opened=parse_iso(str(q["open_time"])),
                resolution=str(res["value"]),
                source=source,
                submitted=bool(f["posted"]),
                model=str(f["model"]),
                augur=json.loads(f["values_json"]),
                references=references.get(qid, {}),
                options=options,
                scaling=scaling,
                metaculus=metaculus.get(qid, {}),
            )
            scored.p(scored.augur)
        except (ScoringError, ValueError, KeyError) as exc:
            result.unscorable.append(f"{qid}: {exc}")
            continue
        seen.add(qid)
        result.scored.append(scored)


# -- the numbers -------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """Augur against one other forecaster on the questions both have."""

    against: str
    n: int
    head_to_head: Estimate
    brier_diff: Estimate | None
    augur_brier: float | None
    other_brier: float | None
    augur_baseline: float
    other_baseline: float


@dataclass(frozen=True)
class Summary:
    n: int
    by_type: dict[str, int]
    baseline: Estimate
    log: Estimate
    brier: Estimate | None
    binary_n: int
    base_rate: float | None
    comparisons: list[Comparison]
    cluster: Cluster
    metaculus_checks: list[tuple[int, str, float, float]]


def _constant(p_yes: float) -> dict[str, Any]:
    return {"probability": p_yes}


def summarise(
    scored: Sequence[ScoredQuestion],
    *,
    cluster: Cluster = "question",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Summary:
    if not scored:
        raise ScoringError("no resolved question has an Augur forecast")
    labels = [s.cluster(cluster) for s in scored]
    kw: dict[str, Any] = {"resamples": resamples, "seed": seed}
    by_type: dict[str, int] = {}
    for s in scored:
        by_type[s.question_type] = by_type.get(s.question_type, 0) + 1
    baseline = clustered_bootstrap([s.baseline(s.augur) for s in scored], labels, **kw)
    logs = clustered_bootstrap([log_score(s.p(s.augur)) for s in scored], labels, **kw)
    binary = [s for s in scored if s.question_type == "binary"]
    bin_labels = [s.cluster(cluster) for s in binary]
    briers = (
        clustered_bootstrap(
            [brier(s.augur["probability"], binary_outcome(s.resolution)) for s in binary],
            bin_labels,
            **kw,
        )
        if binary
        else None
    )
    base_rate = sum(binary_outcome(s.resolution) for s in binary) / len(binary) if binary else None
    comparisons: list[Comparison] = []
    names = sorted({k for s in scored for k in s.references})
    for name in names:
        both = [s for s in scored if name in s.references]
        comparisons.append(_compare(name, both, [s.references[name] for s in both], cluster, kw))
    if binary and base_rate is not None:
        comparisons.append(
            _compare("always 50%", binary, [_constant(0.5)] * len(binary), cluster, kw)
        )
        comparisons.append(
            _compare(
                f"base rate ({base_rate:.0%})",
                binary,
                [_constant(base_rate)] * len(binary),
                cluster,
                kw,
            )
        )
    checks = [
        (s.question_id, name, s.baseline(s.augur), value)
        for s in scored
        for name, value in s.metaculus.items()
        if name in ("spot_baseline_score", "baseline_score")
    ]
    return Summary(
        n=len(scored),
        by_type=dict(sorted(by_type.items())),
        baseline=baseline,
        log=logs,
        brier=briers,
        binary_n=len(binary),
        base_rate=base_rate,
        comparisons=comparisons,
        cluster=cluster,
        metaculus_checks=checks,
    )


def _compare(
    name: str,
    scored: Sequence[ScoredQuestion],
    others: Sequence[dict[str, Any]],
    cluster: Cluster,
    kw: dict[str, Any],
) -> Comparison:
    labels = [s.cluster(cluster) for s in scored]
    h2h = [
        head_to_head(s.question_type, s.p(s.augur), s.p(o))
        for s, o in zip(scored, others, strict=True)
    ]
    pairs = [(s, o) for s, o in zip(scored, others, strict=True) if s.question_type == "binary"]
    brier_diff = None
    augur_brier = other_brier = None
    if pairs:
        a = [brier(s.augur["probability"], binary_outcome(s.resolution)) for s, _ in pairs]
        b = [brier(float(o["probability"]), binary_outcome(s.resolution)) for s, o in pairs]
        brier_diff = paired(a, b, [s.cluster(cluster) for s, _ in pairs], **kw)
        augur_brier = sum(a) / len(a)
        other_brier = sum(b) / len(b)
    return Comparison(
        against=name,
        n=len(scored),
        head_to_head=clustered_bootstrap(h2h, labels, **kw),
        brier_diff=brier_diff,
        augur_brier=augur_brier,
        other_brier=other_brier,
        augur_baseline=sum(s.baseline(s.augur) for s in scored) / len(scored),
        other_baseline=sum(s.baseline(o) for s, o in zip(scored, others, strict=True))
        / len(scored),
    )


def format_summary(summary: Summary, loaded: Loaded) -> str:
    types = ", ".join(f"{t} {n}" for t, n in summary.by_type.items())
    lines = [
        f"{summary.n} scored questions ({types}); {loaded.unresolved} without a resolution, "
        f"{loaded.annulled} annulled; intervals: 95% bootstrap over {summary.cluster} "
        f"clusters ({summary.baseline.clusters}), {summary.baseline.resamples} resamples, "
        f"seed {summary.baseline.seed}",
        f"  baseline score {summary.baseline.fmt(1)}  (MDE {summary.baseline.mde:.1f})",
        f"  log score      {summary.log.fmt(3)}",
    ]
    if summary.brier is not None:
        lines.append(f"  Brier          {summary.brier.fmt(4)}  (binary, n = {summary.binary_n})")
    for c in summary.comparisons:
        lines.append(
            f"  vs {c.against} (n = {c.n}): head-to-head {c.head_to_head.fmt(1)} "
            f"MDE {c.head_to_head.mde:.1f}"
            + (
                f"; Brier {c.augur_brier:.4f} vs {c.other_brier:.4f}, "
                f"difference {c.brier_diff.fmt(4)} MDE {c.brier_diff.mde:.4f}"
                if c.brier_diff is not None
                and c.augur_brier is not None
                and c.other_brier is not None
                else ""
            )
        )
    for qid, name, ours, theirs in summary.metaculus_checks:
        lines.append(f"  check {qid}: our {name} {ours:.3f}, Metaculus {theirs:.3f}")
    return "\n".join(lines)
