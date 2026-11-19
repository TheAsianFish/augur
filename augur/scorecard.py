"""Score the bot from its ledgers: every resolved question, every comparison, with intervals.

A question is scored when a ledger holds Augur's aggregate forecast for it
and a resolution (the latest row; annulled and ambiguous questions are not
scored, and are counted). The forecast scored is the one Augur posted when
it posted one, otherwise its latest aggregate (dry runs and backtests).

Comparisons are always on the questions both forecasters have:

- **references** from any of the ledgers given (pooled by question id):
  the shadow template bot, the Metaculus community prediction, a market
  price;
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
from dataclasses import dataclass, field, replace
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
MIN_P = 0.001
MAX_P = 0.999


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
    forecast_at: str = ""
    reference_as_of: dict[str, str] = field(default_factory=dict)

    def later(self, name: str) -> bool:
        """Whether reference ``name`` is dated after Augur's forecast (it knew more)."""
        as_of = self.reference_as_of.get(name, "")
        return bool(self.forecast_at) and as_of > self.forecast_at

    def p(self, values: dict[str, Any]) -> float:
        """The probability on the outcome; binary and option probabilities are held to
        [0.001, 0.999], the range Metaculus accepts, so a market price of 0 or 1 is
        scored as the most extreme forecast a bot could post, not as certainty."""
        p = outcome_probability(
            self.question_type,
            values,
            self.resolution,
            options=self.options or None,
            scaling=self.scaling,
        )
        if self.question_type in ("binary", "multiple_choice"):
            return min(MAX_P, max(MIN_P, p))
        return p

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


@dataclass
class _Shared:
    """What every ledger given says about each question, the latest row winning.

    A resolution or a reference forecast may sit in a different ledger from
    Augur's forecast (the shadow template has its own), so they are pooled
    across ledgers by question id before any forecast is scored.
    """

    resolutions: dict[int, tuple[str, str, str | None]] = field(default_factory=dict)
    references: dict[int, dict[str, tuple[str, dict[str, Any], str]]] = field(default_factory=dict)
    metaculus: dict[int, dict[str, tuple[str, float]]] = field(default_factory=dict)

    def read(self, conn: sqlite3.Connection) -> None:
        for qid, at, status, value in conn.execute(
            "SELECT question_id, recorded_at, status, value FROM resolutions "
            "ORDER BY recorded_at, rowid"
        ):
            if int(qid) not in self.resolutions or at >= self.resolutions[int(qid)][0]:
                self.resolutions[int(qid)] = (str(at), str(status), value)
        for qid, at, who, values, as_of in conn.execute(
            "SELECT question_id, recorded_at, forecaster, values_json, as_of "
            "FROM reference_forecasts ORDER BY recorded_at, rowid"
        ):
            mine = self.references.setdefault(int(qid), {})
            if str(who) not in mine or at >= mine[str(who)][0]:
                mine[str(who)] = (str(at), json.loads(values), str(as_of))
        for qid, at, name, score in conn.execute(
            "SELECT question_id, recorded_at, score_type, score FROM metaculus_scores "
            "WHERE forecaster = 'augur' ORDER BY recorded_at, rowid"
        ):
            mine_scores = self.metaculus.setdefault(int(qid), {})
            if str(name) not in mine_scores or at >= mine_scores[str(name)][0]:
                mine_scores[str(name)] = (str(at), float(score))


@dataclass(frozen=True)
class _Candidate:
    """One aggregate forecast, with everything needed to score it."""

    posted: bool
    recorded_at: str
    order: int
    question_id: int
    post_id: int
    question_type: str
    title: str
    open_time: str
    payload: dict[str, Any]
    values: dict[str, Any]
    tournament: str
    model: str


def load(paths: Sequence[Path], *, modes: Sequence[str] | None = None) -> Loaded:
    """Every question with an Augur aggregate in the ledgers at ``paths``.

    Each question is counted once however many ledgers hold it. Its forecast
    is the posted one if any ledger posted it, otherwise the latest; the
    order of ``paths`` never decides. Ledgers from before schema 5 (no
    ``resolutions`` table) still contribute forecasts.
    """
    result = Loaded(scored=[])
    shared = _Shared()
    best: dict[int, _Candidate] = {}
    order = 0
    for path in paths:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            if _has_table(conn, "resolutions"):
                shared.read(conn)
            if not _has_table(conn, "forecasts"):
                continue
            for c in _candidates(conn, modes):
                order += 1
                c = _Candidate(**{**c.__dict__, "order": order})
                key = (c.posted, c.recorded_at, c.order)
                held = best.get(c.question_id)
                if held is None or key > (held.posted, held.recorded_at, held.order):
                    best[c.question_id] = c
    for qid, c in best.items():
        res = shared.resolutions.get(qid)
        if res is None:
            result.unresolved += 1
            continue
        if res[1] != "resolved":
            result.annulled += 1
            continue
        try:
            scored = _scored(c, str(res[2]), shared)
            scored.p(scored.augur)
        except (ScoringError, ValueError, KeyError) as exc:
            result.unscorable.append(f"{qid}: {exc}")
            continue
        result.scored.append(_without_bad_references(scored, result.unscorable))
    result.scored.sort(key=lambda s: (s.opened, s.question_id))
    return result


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _candidates(conn: sqlite3.Connection, modes: Sequence[str] | None) -> list[_Candidate]:
    mode_sql = ""
    params: list[Any] = []
    if modes:
        mode_sql = f" AND r.mode IN ({', '.join('?' for _ in modes)})"
        params = list(modes)
    # Ledgers from before schema 4 hold binary forecasts as a bare probability.
    columns = {str(r[1]) for r in conn.execute("PRAGMA table_info(forecasts)")}
    values_sql = (
        "f.values_json"
        if "values_json" in columns
        else "json_object('probability', f.probability) AS values_json"
    )
    # Only "?" placeholders and the fixed column text above are formatted
    # into the query; no input reaches it.
    sql = (
        f"SELECT f.question_id, {values_sql}, f.recorded_at, r.tournament, r.model, "  # noqa: S608
        "q.post_id, q.type, q.title, q.open_time, q.payload_json, "
        "EXISTS (SELECT 1 FROM submissions s WHERE s.run_id = f.run_id "
        "  AND s.question_id = f.question_id AND s.kind = 'forecast' "
        "  AND s.status = 'submitted') AS posted "
        "FROM forecasts f JOIN runs r ON r.run_id = f.run_id "
        "JOIN questions q ON q.run_id = f.run_id AND q.question_id = f.question_id "
        f"WHERE f.kind = 'aggregate'{mode_sql} ORDER BY f.rowid"
    )
    out: list[_Candidate] = []
    for row in conn.execute(sql, params):
        values = row["values_json"]
        if values is None:
            continue
        out.append(
            _Candidate(
                posted=bool(row["posted"]),
                recorded_at=str(row["recorded_at"]),
                order=0,
                question_id=int(row["question_id"]),
                post_id=int(row["post_id"]),
                question_type=str(row["type"]),
                title=str(row["title"]),
                open_time=str(row["open_time"]),
                payload=json.loads(row["payload_json"]),
                values=json.loads(values),
                tournament=str(row["tournament"]),
                model=str(row["model"]),
            )
        )
    return out


def _scored(c: _Candidate, resolution: str, shared: _Shared) -> ScoredQuestion:
    options: tuple[str, ...] = ()
    scaling = None
    source = c.tournament
    if isinstance(c.payload.get("question"), dict):
        question = question_from_post(c.payload)
        options = tuple(question.options or ())
        scaling = question.scaling
    elif "fb_id" in c.payload:
        source = str(c.payload.get("source"))
    return ScoredQuestion(
        question_id=c.question_id,
        post_id=c.post_id,
        question_type=c.question_type,
        title=c.title,
        opened=parse_iso(c.open_time),
        resolution=resolution,
        source=source,
        submitted=c.posted,
        model=c.model,
        augur=c.values,
        references={k: v for k, (_, v, _) in shared.references.get(c.question_id, {}).items()},
        options=options,
        scaling=scaling,
        metaculus={k: v for k, (_, v) in shared.metaculus.get(c.question_id, {}).items()},
        forecast_at=c.recorded_at,
        reference_as_of={k: a for k, (_, _, a) in shared.references.get(c.question_id, {}).items()},
    )


def _without_bad_references(scored: ScoredQuestion, unscorable: list[str]) -> ScoredQuestion:
    """Drop (and list) any reference that cannot be scored, so one bad row
    does not stop the whole scorecard."""
    kept: dict[str, dict[str, Any]] = {}
    for name, values in scored.references.items():
        try:
            scored.p(values)
            if scored.question_type == "binary":
                float(values["probability"])
        except (ScoringError, ValueError, KeyError, TypeError) as exc:
            unscorable.append(f"{scored.question_id} ({name}): {exc}")
            continue
        kept[name] = values
    if len(kept) == len(scored.references):
        return scored
    return replace(scored, references=kept)


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
    later: int = 0


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
        later=sum(s.later(name) for s in scored),
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
            + (f"; {c.later} of {c.n} dated after Augur's forecast (favours it)" if c.later else "")
        )
    for qid, name, ours, theirs in summary.metaculus_checks:
        lines.append(f"  check {qid}: our {name} {ours:.3f}, Metaculus {theirs:.3f}")
    return "\n".join(lines)
