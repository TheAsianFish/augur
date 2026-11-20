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

from augur import q2, replay, scorecard, standings
from augur.clock import to_iso
from augur.config import Settings
from augur.forecastbench import load_set
from augur.llm.catalog import Catalog
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


TEMPLATES_SNAPSHOT = STANDINGS / "minibench-2026-11-02.json"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


def render_templates(root: Path) -> str:
    """Template bots by model, agent against template, and the proposed template-based floor."""
    from augur.llm import catalog as llm_catalog

    round_ = standings.load_snapshot(root / TEMPLATES_SNAPSHOT)
    models = llm_catalog.load_snapshot(root / "augur" / "llm" / "openrouter_models.json")
    return templates_markdown(round_, models)


def templates_markdown(round_: standings.RoundStandings, models: Catalog) -> str:
    """The three template tables for one saved round (also printed by ``augur standings``)."""
    from augur.llm.registry import FRONTIER_TIER_NAME

    board = round_.board()
    rows = standings.template_table(round_)
    plain = [r for r in rows if not r.variant]
    nonneg = [r for r in plain if (r.average or 0.0) >= 0]
    out = [
        f"Source: {round_.source_url}, fetched {to_iso(round_.fetched_at)}; round "
        f"`{round_.round.slug}`, score type {board.score_type}. The snapshot's finalized flag is "
        f"{'yes' if board.finalized else 'no'}. Averages are Metaculus's summed spot peer score "
        "divided by questions covered; intervals are Metaculus's, rescaled the same way.",
        "",
        f"**1. Template bots (`metac-<model>+asknews`).** {len(rows)} entries "
        f"({len(plain)} plain, {len(rows) - len(plain)} experiment variants); "
        f"{len(nonneg)} plain templates average >= 0.",
        "",
        "| Model | Bot | Rank | Avg spot peer score / question [interval] | Questions |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.label}{' ' + r.variant if r.variant else ''} | `{r.name}` | "
            f"{r.rank if r.rank is not None else '-'} | {standings._fmt_avg(r.entry)} | "
            f"{standings._fmt_questions(r.entry.coverage)} |"
        )
    pairs, lone_agents, lone_templates = standings.agent_vs_template(round_)
    out += [
        "",
        "**2. Agent harness against template, same model.** `metac-agent-X` is matched to the "
        "plain template `metac-X+asknews` (or `metac-X-<effort>+asknews`). Each difference is a "
        "difference of two averages, with no paired interval: the two bots' per-question scores "
        "are not in the snapshot.",
        "",
        "| Model | Agent avg | Template avg | Agent minus template |",
        "|---|---|---|---|",
    ]
    for p in pairs:
        out.append(
            f"| {p.model} | {p.agent.average:.2f} | {p.template.average:.2f} | "
            f"{p.difference:+.2f} |"
        )
    out += [
        "",
        f"Unmatched agents: {', '.join(lone_agents) or 'none'}. "
        f"Templates with no agent: {', '.join(lone_templates) or 'none'}.",
    ]
    admitted, rejected, frontier = standings.admitted_by_template(round_, models)
    draws = int(Settings.model_fields["forecasters"].default)
    t_in, t_out = standings.question_tokens(draws)
    out += [
        "",
        f"**3. Proposed floor: admit a model iff its template's average spot peer score is >= 0.** "
        f"{len(admitted)} admitted, {len(rejected)} not, of {len(admitted) + len(rejected)} "
        "models (a model with templates at two effort settings counts once, by its unsuffixed "
        "bot). Worst case per question, from the constants in the agents: one planner "
        f"call, one brief and {draws} draws, {t_in:,} input and {t_out:,} output tokens in all, "
        f"priced at the OpenRouter list ({models.fetched_at:%Y-%m-%d}); a round is "
        f"{standings.ROUND_QUESTIONS} questions. Where several listings share the name, the "
        "dearest prices it.",
        "",
        "| Model | Template avg | OpenRouter id (match by name) | $/M in | $/M out | "
        f"Per question | Per {standings.ROUND_QUESTIONS}-question round |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in admitted:
        listing = (
            f"`{a.priced_id}`"
            + (f" (+{len(a.listed_ids) - 1} more)" if len(a.listed_ids) > 1 else "")
            if a.priced_id
            else "not listed"
        )
        pin = f"{a.usd_per_million_input:g}" if a.usd_per_million_input is not None else "n/a"
        pout = f"{a.usd_per_million_output:g}" if a.usd_per_million_output is not None else "n/a"
        out.append(
            f"| {a.template.key} | {a.template.average:.2f} | {listing} | {pin} | {pout} | "
            f"{_usd(a.per_question)} | {_usd(a.per_round)} |"
        )
    out += [
        "",
        "Not admitted (average < 0): "
        + (", ".join(f"{r.key} ({r.average:.1f})" for r in rejected) or "none")
        + ".",
        "",
        f"Current frontier tier (`{FRONTIER_TIER_NAME}`) without template evidence in this "
        "round: "
        + (", ".join(f"`{n}`" for n, r in frontier if r is None) or "none")
        + ". With evidence: "
        + (
            ", ".join(
                f"`{n}` ({r.average:+.1f})"
                for n, r in frontier
                if r is not None and r.average is not None
            )
            or "none"
        )
        + ".",
    ]
    return "\n".join(out) + "\n"


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
    first_match, first_priced = c2.first_measure
    if first_priced != priced:
        lines.append(
            "  As first measured, with the seven-day staleness cut the price run applied "
            f"before Manifold was exempted (a change made after seeing the data): "
            f"{first_match}/{first_priced} matched "
            f"{_held(first_match >= 0.9 * first_priced)}; "
            f"{first_priced}/{total} priced {_held(first_priced >= 0.9 * total)}. "
            "So P5's first two parts failed as first measured and hold only after that change."
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


# -- Q2: what leakage is worth, and the research loop (M6) -------------------------

Q2_SET = "data/forecastbench/2026-06-21"
Q2_TOURNAMENT = "forecastbench:2026-06-21-llm.json"
Q2_SAMPLE = 64
Q2_LOOP_SAMPLE = 40
Q2_LABELS = Path("data/judge/m6-reference-labels.json")
Q2_ARMS = {
    "N": (LEDGERS / "m5-replay-c2.sqlite", Q2_TOURNAMENT),
    "O": (LEDGERS / "m6-c2-open.sqlite", Q2_TOURNAMENT + "+onepass-asof-open"),
    "T": (LEDGERS / "m6-c2-now.sqlite", Q2_TOURNAMENT + "+onepass-asof-now"),
    "L": (LEDGERS / "m6-c2-loop.sqlite", Q2_TOURNAMENT + "+loop3-asof-open"),
}
# El Lahib et al. (ACL 2026), gpt-oss-120b, 93 questions with a score-4 document.
PUBLISHED = {"no retrieval": 0.244, "leak-free (score 0)": 0.242, "leaky (scores 3-4)": 0.108}


def _q2(root: Path) -> q2.Q2 | None:
    arms = {
        a: q2.ArmSource(root / path, tournament)
        for a, (path, tournament) in Q2_ARMS.items()
        if (root / path).exists()
    }
    if not {"N", "O", "T"} <= arms.keys():
        return None
    report = load_set(
        root / f"{Q2_SET}-llm.subset.json",
        root / f"{Q2_SET}_resolution_set.subset.json",
        released=date.min,
    )
    return q2.score_q2(
        questions=report.kept,
        sample=Q2_SAMPLE,
        loop_sample=Q2_LOOP_SAMPLE,
        arms=arms,
        prices=q2.ArmSource(root / LEDGERS / "m5-replay-c2-prices.sqlite", Q2_TOURNAMENT),
        labels=root / Q2_LABELS,
    )


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def render_leakage(root: Path) -> str:
    s = _q2(root)
    if s is None:
        return "The M6 arms' ledgers are not committed yet.\n"
    arm = {r.arm: r for r in s.arms}
    out = [
        f"C2's fixed sample: the first {Q2_SAMPLE} eligible questions by question id; "
        f"{s.sample} were forecast by every arm. Model `nvidia/nemotron-3-ultra-550b-a55b:free` "
        "(listed 2026-06-04, before the 2026-06-11 freeze) in every role, 3 draws, median. "
        "Lower Brier is better; a negative difference favours the first arm.",
        "",
        "| Arm | Brier [95% interval] | Log score | n |",
        "|---|---|---|---|",
    ]
    out += [
        f"| {r.arm}: {q2.ARM_NAMES[r.arm]} | {_est(r.brier, 4)} | {r.log:.3f} | {r.n} |"
        for r in s.arms
    ]
    out += ["", "| Paired difference | Brier [95% interval] (MDE) |", "|---|---|"]
    out += [f"| {d.label} | {_signed(d.brier, 4)} ({d.brier.mde:.4f}) |" for d in s.diffs]
    out += [
        "",
        f"**Edge over the price** (the as-of price from M5, on the {s.priced} priced questions "
        "of the sample; a negative difference would be an edge):",
        "",
        "| Arm minus the price | Brier [95% interval] (MDE) |",
        "|---|---|",
    ]
    out += [f"| {d.label} | {_signed(d.brier, 4)} ({d.brier.mde:.4f}) |" for d in s.edge]
    if s.unpriced:
        n = s.unpriced[0].n
        out += [
            "",
            f"Unpriced questions of the sample (Metaculus and INFER, n = {n}; no price to set "
            "beside): " + "; ".join(f"{r.arm} {_est(r.brier, 4)}" for r in s.unpriced) + ".",
        ]
    if s.loop_arms:
        out += [
            "",
            f"**The research loop** (the first {Q2_LOOP_SAMPLE} of the sample; {s.loop_n} "
            "forecast by the loop and the one pass):",
            "",
            "| Arm | Brier [95% interval] | Log score | n |",
            "|---|---|---|---|",
        ]
        out += [
            f"| {r.arm}: {q2.ARM_NAMES[r.arm]} | {_est(r.brier, 4)} | {r.log:.3f} | {r.n} |"
            for r in s.loop_arms
        ]
        out += [
            f"| {d.label} | {_signed(d.brier, 4)} (MDE {d.brier.mde:.4f}) | | |"
            for d in s.loop_diffs
        ]
    if s.costs:
        out += [
            "",
            "Cost per question forecast ($0: free model): successful model calls (of which "
            "research), output tokens, wall seconds.",
            "",
            "| Arm | Questions | Calls (research) | Output tokens | Seconds |",
            "|---|---|---|---|---|",
        ]
        out += [
            f"| {c.arm} | {c.questions} | {c.calls:.1f} ({c.research_calls:.1f}) | "
            f"{c.output_tokens:,.0f} | {c.seconds:.0f} |"
            for c in s.costs
        ]
    if s.judges:
        out += [
            "",
            "**The judge** (the same free model, El Lahib et al.'s 0 to 4 rubric, against each "
            "question's open instant, shown the resolution). Major leakage is a score of 3 or 4.",
            "",
            "| Arm | Questions judged | Sources | Scores 0/1/2/3/4 | Sources at 3+ | "
            "Questions with a source at 3+ |",
            "|---|---|---|---|---|---|",
        ]
        out += [
            f"| {j.arm} | {j.questions} | {j.sources} | {'/'.join(map(str, j.by_score))} | "
            f"{_pct(j.major_sources, j.sources)} | {_pct(j.major_questions, j.questions)} |"
            for j in s.judges
        ]
    if s.agree is not None:
        a = s.agree
        out += [
            "",
            (
                f"**Judge against the hand labels** ({a.n} sources): exact agreement with "
                if s.agree_by_hand
                else f"**Judge against the provisional reference labels** ({a.n} sources; a "
                "second model's rating, not hand labels: see To measure): exact agreement with "
            )
            + f"0 and 1 merged {a.exact_merged:.1%}, quadratic weighted kappa {a.qwk:.2f}; "
            f"on major leakage (3+ against 0 to 2) {a.major_agree:.1%} agreement, "
            f"kappa {a.major_kappa:.2f} (labels {a.human_major}, judge {a.judge_major} at 3+). "
            "The paper's judge against two annotators: 76.1% and 0.85 on 134 documents.",
        ]
    if s.flagged:
        out += [
            "",
            f"Questions where the judge found a source at 3+ in T (n = {s.flagged_n}), the "
            "nearest to the paper's selection: "
            + "; ".join(f"{r.arm} {_est(r.brier, 4)}" for r in s.flagged)
            + ".",
        ]
    out += [
        "",
        "Beside the published experiment (not the same protocol): "
        + "; ".join(f"{k} {v:.3f}" for k, v in PUBLISHED.items())
        + f" (gap {PUBLISHED['no retrieval'] - PUBLISHED['leaky (scores 3-4)']:.3f}). "
        f"Here: N {arm['N'].brier.mean:.4f}, O {arm['O'].brier.mean:.4f}, "
        f"T {arm['T'].brier.mean:.4f} (gap N − T {arm['N'].brier.mean - arm['T'].brier.mean:.4f}).",
        "",
        *_q2_verdicts(s),
    ]
    return "\n".join(out) + "\n"


def _q2_verdicts(s: q2.Q2) -> list[str]:
    d = {x.label.split(" ")[0]: x.brier for x in s.diffs}
    lines = ["Predictions stated in PLAN.md before the run:", ""]
    tn, on = d["T"], d["O"]
    lines.append(
        f"- P7 (Q2): retrieval as of today improves Brier by at least 0.05 with the interval "
        f"below 0: {_held(-tn.mean >= 0.05 and tn.high < 0)}; as of open no better than none by "
        f"more than 0.02: {_held(-on.mean <= 0.02)}."
    )
    j = {x.arm: x for x in s.judges}
    if "O" in j and "T" in j:
        t_rate = j["T"].major_questions / j["T"].questions
        o_rate = j["O"].major_questions / j["O"].questions
        lines.append(
            f"- P8 (the judge flags at least 30% of T's questions and under 10% of O's): "
            f"T {t_rate:.0%} {_held(t_rate >= 0.30)}, O {o_rate:.0%} {_held(o_rate < 0.10)}."
        )
    if s.agree is not None:
        if s.agree_by_hand:
            lines.append(
                "- P9 (judge against the hand labels: kappa at least 0.6, "
                f"major-leak agreement at least 85%): {_held(s.agree.qwk >= 0.6)}, "
                f"{_held(s.agree.major_agree >= 0.85)}."
            )
        else:
            lines.append(
                "- P9 (judge against the hand labels: kappa at least 0.6, major-leak agreement "
                "at least 85%): **not yet tested**; the labels are not hand labels. Against "
                f"the provisional reference labels: kappa {s.agree.qwk:.2f}, "
                f"{s.agree.major_agree:.1%}."
            )
    if s.loop_diffs:
        lo = s.loop_diffs[0].brier
        costs = {c.arm: c for c in s.costs}
        more = "L" in costs and "O" in costs and costs["L"].calls > costs["O"].calls
        lines.append(
            f"- P10 (loop within ±0.02 of the one pass, interval including 0; more calls): "
            f"{_held(abs(lo.mean) <= 0.02 and lo.low <= 0 <= lo.high)}, {_held(more)}."
        )
    edge = {x.label.split(" ")[0]: x.brier for x in s.edge}
    worse = [a for a in ("N", "O", "L") if a in edge]
    lines.append(
        "- P11 (N, O and L all worse than the price, interval above 0; T's point estimate "
        "above 0): "
        + ", ".join(f"{a} {_held(edge[a].low > 0)}" for a in worse)
        + (f", T {_held(edge['T'].mean > 0)}" if "T" in edge else "")
        + "."
    )
    return lines


BLOCKS: dict[str, Block] = {
    b.name: b
    for b in (
        Block("forecastbench", render_forecastbench),
        Block("live-scores", render_live),
        Block("ledgers", render_ledgers),
        Block("standings", render_standings),
        Block("templates", render_templates),
        Block("replay", render_replay),
        Block("leakage", render_leakage),
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
