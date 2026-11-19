"""``augur report``: the generated parts of RESULTS.md, rendered from the ledgers.

RESULTS.md mixes prose with blocks this module writes. A block sits between
two markers::

    <!-- augur:begin NAME -->
    ...generated, do not edit...
    <!-- augur:end NAME -->

``augur report`` replaces each block's body with a fresh render from the
committed ledgers and snapshots, and ``augur report --check`` (and a unit
test) fails when the committed file differs from a render, so a number in a
block can never drift from the data it came from. Every render is offline
and deterministic: bootstraps are seeded, and nothing reads the clock.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from augur import replay, scorecard, standings
from augur.forecastbench import load_set
from augur.score import Estimate, detectable_effect

# The spread of one bot's per-question spot peer score, derived in PLAN.md from
# the Spring 2026 winner's interval: 3.71 / 1.96 * sqrt(288) = 32 points.
PEER_SD = 32.0

REPO = Path(__file__).resolve().parent.parent
LEDGERS = Path("data/ledgers")
FORECASTBENCH_LEDGER = LEDGERS / "m4-forecastbench-qwen.sqlite"
FORECASTBENCH_SOURCE = Path("data/forecastbench/2026-05-24.source.json")

_BLOCK = re.compile(
    r"(<!-- augur:begin (?P<name>[a-z0-9-]+) -->\n)(?P<body>.*?)(<!-- augur:end (?P=name) -->)",
    re.S,
)


class ReportError(ValueError):
    pass


@dataclass(frozen=True)
class Block:
    name: str
    render: Callable[[Path], str]


def _ledgers(root: Path) -> list[Path]:
    return sorted((root / LEDGERS).glob("*.sqlite"))


def _est(e: Estimate, digits: int) -> str:
    return f"{e.mean:.{digits}f} [{e.low:.{digits}f}, {e.high:.{digits}f}]"


def _signed(e: Estimate, digits: int) -> str:
    return f"{e.mean:+.{digits}f} [{e.low:+.{digits}f}, {e.high:+.{digits}f}]"


# -- blocks -------------------------------------------------------------------


def render_forecastbench(root: Path) -> str:
    path = root / FORECASTBENCH_LEDGER
    source = json.loads((root / FORECASTBENCH_SOURCE).read_text(encoding="utf-8"))
    loaded = scorecard.load([path], modes=["backtest"])
    if not loaded.scored:
        return (
            f"No resolved question in `{FORECASTBENCH_LEDGER}` yet: run "
            "`augur resolve --forecastbench` on it.\n"
        )
    summary = scorecard.summarise(loaded.scored)
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        runs = conn.execute(
            "SELECT run_id, started_at, model, code_version, settings_json FROM runs "
            "WHERE mode = 'backtest' ORDER BY started_at"
        ).fetchall()
        calls, failed, out_tokens, cost, ms = conn.execute(
            "SELECT count(*), sum(error IS NOT NULL), sum(output_tokens), sum(cost_usd), "
            "sum(latency_ms) FROM llm_calls"
        ).fetchone()
        outcomes = dict(
            conn.execute("SELECT status, count(*) FROM question_outcomes GROUP BY status")
        )
    draws = sorted({json.loads(r[4]).get("forecasters") for r in runs} - {None})
    models = sorted({str(r[2]) for r in runs})
    b = summary.brier
    if b is None:
        raise ReportError("the ForecastBench set should be binary")
    rows = [
        "| Forecaster | Brier [95% CI] | Baseline score | Augur minus it: Brier [95% CI] "
        "| MDE (Brier) | Augur over it: head-to-head [95% CI] | MDE (points) |",
        "|---|---|---|---|---|---|---|",
        f"| **Augur**, {', '.join(models)}, no retrieval | {_est(b, 4)} "
        f"| {_est(summary.baseline, 1)} | | | | |",
    ]
    for c in summary.comparisons:
        if c.brier_diff is None or c.other_brier is None:
            continue
        label = {"market": "Market price at the freeze date (not shown to Augur)"}.get(
            c.against, c.against.capitalize()
        )
        rows.append(
            f"| {label} | {c.other_brier:.4f} | {c.other_baseline:.1f} "
            f"| {_signed(c.brier_diff, 4)} | {c.brier_diff.mde:.4f} "
            f"| {_signed(c.head_to_head, 1)} | {c.head_to_head.mde:.1f} |"
        )
    removed = "; ".join(f"{n} {why}" for why, n in source["removed"].items())
    freezes = sorted({s.opened.strftime("%Y-%m-%d %H:%M") for s in loaded.scored})
    lines = [
        f"Question set `{Path(source['urls']['questions']).name}` (ForecastBench, "
        f"CC BY-SA 4.0): {source['total']} questions; removed {removed}; "
        f"**{summary.n} scored** ({summary.binary_n} binary, "
        f"{summary.base_rate:.1%} resolved Yes).",
        "",
        *rows,
        "",
        f"Method: {len(runs)} backtest run(s) in `{FORECASTBENCH_LEDGER}` "
        f"(code {', '.join(sorted({str(r[3]) for r in runs}))}); "
        f"{', '.join(str(d) for d in draws)} draw(s) per question, median; clock pinned to "
        f"each question's freeze date ({', '.join(freezes)} UTC); no sources. "
        f"{calls} model calls ({failed or 0} failed), {out_tokens or 0:,} output tokens, "
        f"{(ms or 0) / 3_600_000:.1f} h of model time on a 4-core CPU, cost ${cost or 0:.2f}. "
        f"Outcomes: {', '.join(f'{k} {v}' for k, v in sorted(outcomes.items()))}. "
        f"Intervals: paired bootstrap over questions, {summary.baseline.resamples:,} "
        f"resamples, seed {summary.baseline.seed}; MDE is the true difference this n would "
        "detect with 80% power at the 5% level. The base rate is the scored set's own, known "
        "only after the fact.",
    ]
    return "\n".join(lines) + "\n"


def render_live(root: Path) -> str:
    """Augur on resolved live questions, against the template and the community."""
    paths = _ledgers(root)
    live = [p for p in paths if _count(p, "SELECT count(*) FROM runs WHERE mode = 'live'")]
    questions: dict[int, tuple[str, bool]] = {}
    references: dict[str, set[int]] = {}
    for p in live:
        with closing(sqlite3.connect(f"file:{p}?mode=ro", uri=True)) as conn:
            for qid, tournament, posted in conn.execute(
                "SELECT q.question_id, r.tournament, EXISTS (SELECT 1 FROM submissions s "
                "WHERE s.question_id = q.question_id AND s.run_id = q.run_id "
                "AND s.kind = 'forecast' AND s.status = 'submitted') "
                "FROM questions q JOIN runs r ON r.run_id = q.run_id WHERE r.mode = 'live'"
            ):
                before = questions.get(int(qid), ("", False))
                questions[int(qid)] = (str(tournament), before[1] or bool(posted))
    for p in paths:
        with closing(sqlite3.connect(f"file:{p}?mode=ro", uri=True)) as conn:
            if _has(conn, "reference_forecasts"):
                for qid, who in conn.execute(
                    "SELECT question_id, forecaster FROM reference_forecasts"
                ):
                    references.setdefault(str(who), set()).add(int(qid))
    posted = {q: t for q, (t, p) in questions.items() if p}
    by_tournament: dict[str, int] = {}
    for t in posted.values():
        by_tournament[t] = by_tournament.get(t, 0) + 1
    loaded = scorecard.load(paths, modes=["live"])
    tour = ", ".join(f"{t} {n}" for t, n in sorted(by_tournament.items())) or "none"
    refs = (
        ", ".join(f"{who} on {len(qs & set(questions))}" for who, qs in sorted(references.items()))
        or "none"
    )
    lines = [
        f"Live questions in the committed ledgers: {len(questions)}; "
        f"posted by Augur: {len(posted)} ({tour}). Reference forecasts recorded: {refs}. "
        f"With a resolution in the ledger: {len(loaded.scored) + loaded.annulled}; "
        f"annulled {loaded.annulled}; scored **{len(loaded.scored)}**.",
        "",
    ]
    if not loaded.scored:
        lines += [
            "Nothing is scored yet, so no interval and no detectable effect can be computed "
            "from data. Until there is a measured spread, the detectable effect at the sizes "
            f"this season can reach, for a per-question spread of {PEER_SD:.0f} points "
            "(PLAN.md, from the Spring 2026 winner's interval; a paired difference against "
            "the same-model template is likely tighter, and M4's first resolved round "
            "replaces this with the measured one):",
            "",
            "| Resolved questions | Detectable mean difference (points per question, "
            "80% power, 5% two-sided) |",
            "|---|---|",
        ]
        for n in (30, 60, 120, 300):
            lines.append(f"| {n} | {detectable_effect(PEER_SD, n):.1f} |")
        return "\n".join(lines) + "\n"
    summary = scorecard.summarise(loaded.scored, cluster="week")
    lines += [
        "| Against | Questions | Augur over it: head-to-head [95% CI] | MDE (points) |",
        "|---|---|---|---|",
    ]
    for c in summary.comparisons:
        lines.append(
            f"| {c.against} | {c.n} | {_signed(c.head_to_head, 1)} | {c.head_to_head.mde:.1f} |"
        )
    lines.append("")
    lines.append(f"Augur's baseline score: {_est(summary.baseline, 1)} over {summary.n}.")
    return "\n".join(lines) + "\n"


def render_ledgers(root: Path) -> str:
    rows = [
        "| Ledger | Runs (modes) | Questions | Model calls | Forecasts (aggregate) "
        "| Posted | Resolutions | References |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in _ledgers(root):
        with closing(sqlite3.connect(f"file:{p}?mode=ro", uri=True)) as conn:
            modes = ", ".join(
                f"{m} {n}"
                for m, n in conn.execute(
                    "SELECT mode, count(*) FROM runs GROUP BY mode ORDER BY mode"
                )
            )
            runs = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
            qs = conn.execute("SELECT count(DISTINCT question_id) FROM questions").fetchone()[0]
            calls = conn.execute("SELECT count(*) FROM llm_calls").fetchone()[0]
            aggs = conn.execute(
                "SELECT count(*) FROM forecasts WHERE kind = 'aggregate'"
            ).fetchone()[0]
            posted = conn.execute(
                "SELECT count(*) FROM submissions WHERE kind = 'forecast' AND status = 'submitted'"
            ).fetchone()[0]
            res = refs = 0
            if _has(conn, "resolutions"):
                res = conn.execute("SELECT count(*) FROM resolutions").fetchone()[0]
                refs = conn.execute("SELECT count(*) FROM reference_forecasts").fetchone()[0]
        rows.append(
            f"| `{p.name}` | {runs} ({modes or '-'}) | {qs} | {calls} | {aggs} | {posted} "
            f"| {res} | {refs} |"
        )
    return "\n".join(rows) + "\n"


STANDINGS = Path("data/standings")


def render_standings(root: Path) -> str:
    """Each committed round snapshot: the table, and why Augur has no row in it."""
    ledgers = _ledgers(root)
    outcomes = standings.read_outcomes(ledgers)
    parts: list[str] = []
    for path in sorted((root / STANDINGS).glob("*.json")):
        round_ = standings.load_snapshot(path)
        text = standings.render(round_, None)
        kept: list[str] = []
        for line in text.splitlines():
            if line.startswith("### Questions Augur did not forecast"):
                break
            kept.append("#### " + line[3:] if line.startswith("## ") else line)
        while kept and not kept[-1].strip():
            kept.pop()
        board = round_.board()
        bots = sorted(
            (e for e in board.entries if e.user is not None and e.user.is_bot),
            key=lambda e: e.score,
        )
        if bots:
            mid = bots[len(bots) // 2]
            average = mid.average
            kept += [
                "",
                f"Median bot by summed score (the line Q5 predicts Augur clears): "
                f"{mid.score:.1f} over {mid.coverage:g} questions"
                + (f", {average:.2f} a question" if average is not None else "")
                + f" (rank {mid.rank}; the middle of {len(bots)} bot entries, many of which "
                "covered only part of the round, which a summed score punishes).",
            ]
        reasons: dict[str, int] = {}
        forecast = 0
        for q in round_.questions:
            done, why = standings.explain(q, outcomes)
            if done:
                forecast += 1
            else:
                reasons[why] = reasons.get(why, 0) + 1
        why = "; ".join(f"{n} {r}" for r, n in sorted(reasons.items(), key=lambda x: -x[1]))
        kept += [
            "",
            f"Questions in the round: {len(round_.questions)}; forecast and submitted by Augur: "
            f"{forecast}; not forecast: {why or 'none'} (from the committed ledgers; "
            f"`augur standings --from {path.relative_to(root)} --ledger ...` lists each one).",
        ]
        parts.append("\n".join(kept))
    return "\n\n".join(parts) + "\n" if parts else "No standings snapshot committed.\n"


# -- the replay evaluation (M5) ---------------------------------------------------


@dataclass(frozen=True)
class ReplayCorpus:
    label: str
    stem: str
    """``data/forecastbench/<date>``: the committed question and resolution subsets."""
    forecasts: Path
    prices: Path
    with_prices: Path | None


REPLAY_CORPORA = (
    ReplayCorpus(
        "C1",
        "data/forecastbench/2026-05-24",
        FORECASTBENCH_LEDGER,
        LEDGERS / "m5-replay-c1-prices.sqlite",
        None,
    ),
    ReplayCorpus(
        "C2",
        "data/forecastbench/2026-06-21",
        LEDGERS / "m5-replay-c2.sqlite",
        LEDGERS / "m5-replay-c2-prices.sqlite",
        LEDGERS / "m5-replay-c2-prices.sqlite",
    ),
)
ARM_NAMES = {
    "a": "(a) price only",
    "b": "(b) Augur, no price",
    "c": "(c) blend, ⅓ Augur in log-odds",
    "d": "(d) Augur, price shown",
}


def _held(ok: bool) -> str:
    return "**held**" if ok else "**failed**"


def _model_of(path: Path, tournament: str) -> tuple[str, int]:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT model, settings_json FROM runs WHERE mode = 'backtest' AND tournament = ?",
            (tournament,),
        ).fetchall()
    models = sorted({str(r[0]) for r in rows})
    draws = max(int(json.loads(r[1]).get("forecasters") or 1) for r in rows)
    return ", ".join(models), draws


def render_replay(root: Path) -> str:
    """M5's arms on ForecastBench market questions, with P1–P6 marked."""
    scores: dict[str, replay.CorpusScore] = {}
    models: dict[str, tuple[str, int]] = {}
    for c in REPLAY_CORPORA:
        if not (root / c.forecasts).exists() or not (root / c.prices).exists():
            return f"`{c.forecasts}` or `{c.prices}` is missing: run the replay evaluation.\n"
        qs = root / f"{c.stem}-llm.subset.json"
        rs = root / f"{c.stem}_resolution_set.subset.json"
        loaded = load_set(qs, rs, released=date.min)
        tournament = f"forecastbench:{loaded.question_set}"
        scores[c.label] = replay.score_corpus(
            label=c.label,
            questions=loaded.kept,
            tournament=tournament,
            forecasts=root / c.forecasts,
            prices=root / c.prices,
            with_prices=None if c.with_prices is None else root / c.with_prices,
        )
        models[c.label] = _model_of(root / c.forecasts, tournament)
    out = [
        "| Corpus | Model (arm b) | Draws | Questions | Exact price as of the freeze "
        "| Within 0.01 of ForecastBench's freeze value |",
        "|---|---|---|---|---|---|",
    ]
    for label, sc in scores.items():
        model, draws = models[label]
        platforms = ", ".join(
            f"{p} {got}/{n}" for p, (got, n) in sorted(sc.priced_by_platform.items())
        )
        match, priced = sc.freeze_match
        out.append(
            f"| {label} | `{model}` | {draws} | {sc.questions} | {sc.priced} ({platforms}) "
            f"| {match}/{priced} ({match / priced:.0%}) |"
        )
    out += ["", "| Arm | " + " | ".join(scores) + " |", "|---|" + "---|" * len(scores)]
    for arm, name in ARM_NAMES.items():
        cells = []
        for sc in scores.values():
            found = [a for a in sc.arms if a.name == arm]
            cells.append(
                f"{_est(found[0].brier, 4)}, log {found[0].log:.3f} (n={found[0].brier.n})"
                if found
                else "not run"
            )
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    cells = [
        "" if sc.all_b is None else f"{_est(sc.all_b.brier, 4)} (n={sc.all_b.brier.n})"
        for sc in scores.values()
    ]
    out.append("| (b) on every question, priced or not | " + " | ".join(cells) + " |")
    out += [
        "",
        "| Paired Brier difference [95% CI] (MDE) | " + " | ".join(scores) + " |",
        "|---|" + "---|" * len(scores),
    ]
    names = {
        "b-a": "(b) − (a): Augur minus the price",
        "c-b": "(c) − (b): blend minus Augur",
        "c-a": "(c) − (a): blend minus the price",
        "d-b": "(d) − (b): showing the price",
        "d-a": "(d) − (a): Augur shown the price, minus the price",
    }
    for key, name in names.items():
        cells = []
        for sc in scores.values():
            hits = [d for d in sc.diffs if d.name == key]
            cells.append(f"{_signed(hits[0].brier, 4)} ({hits[0].brier.mde:.4f})" if hits else "")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    c2 = scores["C2"]
    out += [
        "",
        f"Settings arms on C2, re-aggregated from (b)'s own draws ({c2.settings_n} questions "
        "with three parsed draws; no new calls):",
        "",
        "| Paired Brier difference against the median of 3 [95% CI] (MDE) | C2 |",
        "|---|---|",
    ]
    out += [f"| {d.name} | {_signed(d.brier, 4)} ({d.brier.mde:.4f}) |" for d in c2.settings]
    live = scorecard.load(_ledgers(root), modes=["live", "dry_run"])
    out += [
        "",
        f"C3, live and dry-run questions in the committed ledgers with a resolution: "
        f"{len(live.scored)}. Research depth and the replay of live research wait on these.",
    ]
    out += ["", *_replay_verdicts(scores), ""]
    out.append(
        "Method: arms (a)–(d) are paired on the questions with an exact price (C1 has no "
        "arm (d)); every probability is clipped to [0.01, 0.99] before scoring; log is the "
        "mean natural log of the probability given to the outcome (higher is better). "
        f"Intervals: paired bootstrap over questions, {c2.diffs[0].brier.resamples:,} "
        f"resamples, seed {c2.diffs[0].brier.seed}; MDE is the true difference this n would "
        "detect with 80% power at the 5% level."
    )
    return "\n".join(out) + "\n"


def _diff(sc: replay.CorpusScore, name: str) -> Estimate:
    return next(d.brier for d in [*sc.diffs, *sc.settings] if d.name == name)


def _replay_verdicts(scores: dict[str, replay.CorpusScore]) -> list[str]:
    c1, c2 = scores["C1"], scores["C2"]
    lines = ["Predictions stated in PLAN.md before the run:", ""]
    p1 = [_diff(s, "b-a").low > 0 for s in (c1, c2)]
    lines.append(
        f"- P1 (price alone beats Augur, interval above 0): C1 {_held(p1[0])}, C2 {_held(p1[1])}."
    )
    p2 = [_diff(s, "c-b").high < 0 for s in (c1, c2)]
    lines.append(
        f"- P2 (blend beats Augur, interval below 0): C1 {_held(p2[0])}, C2 {_held(p2[1])}."
    )
    ca1, ca2 = _diff(c1, "c-a"), _diff(c2, "c-a")
    lines.append(
        f"- P3 (blend worse than the price on C1; within ±0.01 on C2, interval including 0): "
        f"C1 {_held(ca1.mean > 0)}, "
        f"C2 {_held(abs(ca2.mean) <= 0.01 and ca2.low <= 0 <= ca2.high)}."
    )
    if any(d.name == "d-b" for d in c2.diffs):
        db, da = _diff(c2, "d-b"), _diff(c2, "d-a")
        lines.append(
            f"- P4 (showing the price improves Brier by at least 0.03, interval below 0; "
            "the model still under-uses it): first part "
            f"{_held(db.mean <= -0.03 and db.high < 0)}, second part {_held(da.mean > 0)}."
        )
    match, priced = c2.freeze_match
    exact = [c2.priced_by_platform.get(p, (0, 0)) for p in ("polymarket", "manifold")]
    none = [c2.priced_by_platform.get(p, (0, 0)) for p in ("metaculus", "infer")]
    got, total = sum(g for g, _ in exact), sum(n for _, n in exact)
    lines.append(
        f"- P5, first two parts (freeze value matched on ≥90% of priced C2 questions; an exact "
        f"price for ≥90% of Polymarket and Manifold, 0% of Metaculus and INFER): "
        f"{match}/{priced} matched {_held(match >= 0.9 * priced)}; "
        f"{got}/{total} priced {_held(got >= 0.9 * total)}; "
        f"{sum(g for g, _ in none)} of {sum(n for _, n in none)} Metaculus and INFER priced "
        f"{_held(sum(g for g, _ in none) == 0)}. The third part (MiniBench) is below."
    )
    one, logodds, ext = (
        _diff(c2, "one draw - median of 3"),
        _diff(c2, "mean log-odds - median"),
        _diff(c2, f"extremized x{replay.EXTREMIZE} - median"),
    )
    lines.append(
        f"- P6 (one draw within 0.01 of three, interval including 0; mean of log-odds within "
        f"0.005 of the median; extremizing worse, interval including 0): "
        f"{_held(abs(one.mean) < 0.01 and one.low <= 0 <= one.high)}, "
        f"{_held(abs(logodds.mean) < 0.005)}, "
        f"{_held(ext.mean > 0 and ext.low <= 0 <= ext.high)}."
    )
    return lines


def _has(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _count(path: Path, sql: str) -> int:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        return int(conn.execute(sql).fetchone()[0])


BLOCKS: dict[str, Block] = {
    b.name: b
    for b in (
        Block("forecastbench", render_forecastbench),
        Block("live-scores", render_live),
        Block("ledgers", render_ledgers),
        Block("standings", render_standings),
        Block("replay", render_replay),
    )
}


# -- the file -------------------------------------------------------------------


def render(text: str, root: Path = REPO, only: Sequence[str] | None = None) -> str:
    """``text`` with every known block re-rendered; an unknown block name is an error."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        block = BLOCKS.get(name)
        if block is None:
            raise ReportError(f"RESULTS.md has a block {name!r} that augur report cannot render")
        if only is not None and name not in only:
            return match.group(0)
        return match.group(1) + block.render(root) + match.group(4)

    return _BLOCK.sub(replace, text)


def check(path: Path, root: Path = REPO, required: Sequence[str] | None = None) -> list[str]:
    """The names of the blocks in ``path`` that differ from a fresh render.

    A block named in ``required`` but absent, or present more than once, is
    reported too, so deleting a block cannot make the check pass.

    Refuses while a committed ledger has unsaved write-ahead-log content: a
    reader sees it, git does not, so the check would pass here and fail on a
    clean checkout.
    """
    pending = [p.name for p in sorted((root / LEDGERS).glob("*.sqlite-wal")) if p.stat().st_size]
    if pending:
        raise ReportError(
            f"{', '.join(pending)} hold rows not yet in the committed files; finish or close "
            "the writer (sqlite3 FILE 'PRAGMA wal_checkpoint(TRUNCATE)') and run again"
        )
    text = path.read_text(encoding="utf-8")
    names = [m.group("name") for m in _BLOCK.finditer(text)]
    problems = {f"{n} (missing)" for n in required or () if n not in names}
    problems |= {f"{n} (twice)" for n in names if names.count(n) > 1}
    fresh = render(text, root)
    if fresh != text:
        olds = [(m.group("name"), m.group("body")) for m in _BLOCK.finditer(text)]
        news = [m.group("body") for m in _BLOCK.finditer(fresh)]
        problems |= {n for (n, body), now in zip(olds, news, strict=True) if body != now}
    return sorted(problems)
