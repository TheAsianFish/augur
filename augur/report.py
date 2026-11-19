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
from pathlib import Path

from augur import scorecard, standings
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


def check(path: Path, root: Path = REPO) -> list[str]:
    """The names of the blocks in ``path`` that differ from a fresh render.

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
    fresh = render(text, root)
    if fresh == text:
        return []
    old = {m.group("name"): m.group("body") for m in _BLOCK.finditer(text)}
    new = {m.group("name"): m.group("body") for m in _BLOCK.finditer(fresh)}
    return sorted(n for n in old if old[n] != new.get(n))
