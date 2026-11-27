"""M18: a day-of-year arm for ForecastBench's Meteo-France temperature questions.

PLAN.md's ``m18-registration`` block in code. A Meteo-France question asks
whether a station's daily mean temperature on each resolution date r will be
higher than on the set's due date d. No model is called anywhere here.

* :func:`needed_days` lists the days the arms read for a question, all in
  earlier years and all dated at or before its freeze instant F (an
  observation for day D is dated 00:00 UTC of D + 1, item 2);
* :func:`record_observations` fetches each station once from DBnomics' keyless
  API and writes, per question, the admitted observations on those days to a
  backtest ledger as one dated source, admitted through the leakage gate with
  F as the cutoff; :func:`ledger_observations` reads back only what the gate
  admitted, refusing any line dated after its question's cutoff;
* the arms: S (:func:`arm_s`, item 4), S′ (:func:`arm_s_prime`, item 5) and the
  calendar control K (:func:`arm_k`, item 6); anything they cannot reach falls
  back to M15's arm B, and is counted.

The scoring is :mod:`augur.fbseason_eval`.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from augur.clock import ensure_utc, parse_iso
from augur.errors import SearchError
from augur.fbdataset import DatasetError, DQuestion, parse_closes
from augur.leakage import RunMode, admit_source
from augur.ledger import Ledger
from augur.questions import Question
from augur.research.dbnomics import DBnomicsFetcher, DSeries, code_from_id, to_source

M18_DIR = Path("data/forecastbench/m18")
MANIFEST = M18_DIR / "confirmatory.json"
LEDGER = Path("data/ledgers/m18-dbnomics.sqlite")
TOURNAMENT = "forecastbench-dbnomics:m18"
PREFIX = "meteofrance_TEMPERATURE_"
READING_DAY = date(2027, 1, 11)
"""The registered reading (item 8): earlier readings are progress readings."""
WINDOW_S = 3
MIN_DAYS_S = 4
"""Each of S's windows needs at least this many of its 2·3 + 1 days (item 3)."""
WINDOW_S_PRIME = 10
K_UP, K_DOWN = 0.9, 0.1
MAX_SHIFT = 30
"""Year shifts tried; DBnomics' station series start in 2012, so later ones find nothing."""


class SeasonError(DatasetError):
    """M18's committed files or ledger are not what the registration describes."""


def is_meteo(q: DQuestion) -> bool:
    return q.source == "dbnomics" and q.id.startswith(PREFIX)


# -- the calendar ---------------------------------------------------------------------


def shift(t: date, k: int) -> date:
    """``t`` with its year less ``k`` (29 February becomes 28 February)."""
    try:
        return t.replace(year=t.year - k)
    except ValueError:
        return t.replace(year=t.year - k, day=28)


def window(t: date, w: int, k: int) -> list[date]:
    """W(t, w, k): the days within ``w`` of ``t`` shifted back ``k`` years."""
    c = shift(t, k)
    return [c + timedelta(days=j) for j in range(-w, w + 1)]


def last_admitted_day(freeze: datetime) -> date:
    """The last day D whose observation (dated 00:00 UTC of D + 1) is at or before F."""
    return ensure_utc(freeze).date() - timedelta(days=1)


def needed_days(q: DQuestion, dates: Iterable[date]) -> set[date]:
    """Every admitted day S, S′ and K read for ``q`` at the resolution dates ``dates``."""
    last = last_admitted_day(q.freeze)
    out: set[date] = set()
    for r in dates:
        for k in range(1, MAX_SHIFT + 1):
            for day in window(r, WINDOW_S_PRIME, k) + window(q.due, WINDOW_S, k):
                if day <= last:
                    out.add(day)
    return out


# -- the arms -------------------------------------------------------------------------


@dataclass(frozen=True)
class Shift:
    k: int
    m_r: float
    m_d: float


def usable_shifts(obs: Mapping[date, float], due: date, r: date, freeze: datetime) -> list[Shift]:
    """The year shifts S can use (item 3), with both window means."""
    last = last_admitted_day(freeze)
    out: list[Shift] = []
    for k in range(1, MAX_SHIFT + 1):
        wr, wd = window(r, WINDOW_S, k), window(due, WINDOW_S, k)
        if max(wr[-1], wd[-1]) > last:
            continue
        vr = [obs[d] for d in wr if d in obs]
        vd = [obs[d] for d in wd if d in obs]
        if len(vr) < MIN_DAYS_S or len(vd) < MIN_DAYS_S:
            continue
        out.append(Shift(k, statistics.fmean(vr), statistics.fmean(vd)))
    return out


def arm_s(shifts: Sequence[Shift], b: float) -> float:
    """(wins + 1) / (n + 2) over the usable shifts; B with none (item 4)."""
    if not shifts:
        return b
    wins = sum(1 for s in shifts if s.m_r > s.m_d)
    return (wins + 1) / (len(shifts) + 2)


@dataclass(frozen=True)
class Exceedance:
    above: int
    n: int


def exceedance(
    obs: Mapping[date, float], r: date, freeze: datetime, v: float | None
) -> Exceedance | None:
    """S′'s counts: admitted observations within 10 days of r in earlier years above v."""
    if v is None:
        return None
    last = last_admitted_day(freeze)
    days = {d for k in range(1, MAX_SHIFT + 1) for d in window(r, WINDOW_S_PRIME, k)}
    xs = [obs[d] for d in sorted(days) if d <= last and d in obs]
    if not xs:
        return None
    return Exceedance(sum(1 for x in xs if x > v), len(xs))


def arm_s_prime(e: Exceedance | None, b: float) -> float:
    """(#{x > v} + 1) / (N + 2); B with no observation or no freeze value (item 5)."""
    return b if e is None else (e.above + 1) / (e.n + 2)


def station_change(shifts: Sequence[Shift]) -> float | None:
    """A station's mean of m_r − m_d over its usable shifts."""
    if not shifts:
        return None
    return statistics.fmean(s.m_r - s.m_d for s in shifts)


def arm_k(changes: Iterable[float | None]) -> tuple[float, float | None]:
    """K from the stations' changes for one (set, r), and the pooled change (item 6)."""
    got = [c for c in changes if c is not None]
    if not got:
        return 0.5, None
    pooled = statistics.fmean(got)
    return (K_UP if pooled > 0 else K_DOWN if pooled < 0 else 0.5), pooled


# -- the confirmatory manifest ---------------------------------------------------------


@dataclass(frozen=True)
class Planned:
    """One confirmatory instance as the manifest lists it."""

    question_set: str
    id: str
    resolution_date: date
    horizon: int

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.question_set, "dbnomics", self.id, self.resolution_date.isoformat())


@dataclass(frozen=True)
class Manifest:
    instances: tuple[Planned, ...]
    left_out: int
    window: tuple[date, date]
    question_sets: dict[str, Any]


def load_manifest(root: Path) -> Manifest:
    path = root / MANIFEST
    if not path.exists():
        raise SeasonError(f"no confirmatory manifest at {path}; it is committed with M18")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = tuple(
        Planned(
            str(i["question_set"]),
            str(i["id"]),
            date.fromisoformat(str(i["resolution_date"])),
            int(i["horizon"]),
        )
        for i in doc["instances"]
    )
    first, last = (date.fromisoformat(str(x)) for x in doc["window"])
    if any(not (first <= p.resolution_date <= last) for p in rows):
        raise SeasonError(f"{path} lists an instance outside its window")
    return Manifest(rows, len(doc["left_out"]), (first, last), dict(doc["question_sets"]))


# -- the observations, written to the ledger first ----------------------------------------


def wanted(
    questions: Iterable[DQuestion], manifest: Manifest, corpus_sets: Iterable[str]
) -> list[tuple[DQuestion, list[date]]]:
    """Each Meteo-France question the arms read, with the resolution dates they read it at.

    A corpus set's question is read at its resolved dates; any question the
    manifest names, at its confirmatory dates as well.
    """
    planned: dict[tuple[str, str], set[date]] = {}
    for p in manifest.instances:
        planned.setdefault((p.question_set, p.id), set()).add(p.resolution_date)
    corpus = set(corpus_sets)
    out: list[tuple[DQuestion, list[date]]] = []
    for q in questions:
        if not is_meteo(q):
            continue
        dates = set(planned.get((q.question_set, q.id), set()))
        if q.question_set in corpus:
            dates.update(d for d, _ in q.resolutions)
        if dates:
            out.append((q, sorted(dates)))
    missing = set(planned) - {(q.question_set, q.id) for q, _ in out}
    if missing:
        raise SeasonError(f"the manifest names {len(missing)} questions the extract lacks")
    return sorted(out, key=lambda t: (t[0].question_set, t[0].id))


def ledger_question(q: DQuestion) -> Question:
    return q.to_question().model_copy(
        update={
            "title": (
                f"Will {q.id}'s daily mean on each resolution date be higher than on "
                f"{q.question_set}? (ForecastBench {q.question_set}, dbnomics)"
            ),
            "url": q.url or "https://db.nomics.world",
        }
    )


def recorded_ids(path: Path) -> set[int]:
    """Questions the ledger already holds an outcome for (so a rerun resumes)."""
    if not path.exists():
        return set()
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT o.question_id FROM question_outcomes o JOIN runs r ON r.run_id = o.run_id "
            "WHERE r.tournament = ?",
            (TOURNAMENT,),
        ).fetchall()
    return {int(r[0]) for r in rows}


@dataclass
class ObsReport:
    run_id: str
    recorded: int = 0
    stations: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    refused: int = 0
    """Points the server sent dated after the freeze instant of the question they were for."""

    def line(self) -> str:
        skipped = ", ".join(f"{n} {why}" for why, n in sorted(self.skipped.items())) or "none"
        return (
            f"{self.stations} stations fetched; {self.recorded} questions' observations "
            f"recorded; {self.refused} points dated after their freeze refused; "
            f"skipped: {skipped}"
        )


async def record_observations(
    *,
    todo: Sequence[tuple[DQuestion, list[date]]],
    fetcher: DBnomicsFetcher,
    ledger: Ledger,
    code_version: str,
    out: Callable[[str], None] = print,
) -> ObsReport:
    """Fetch each station once and record each question's admitted observations (no model)."""
    run_id = ledger.start_run(
        mode=RunMode.BACKTEST,
        tournament=TOURNAMENT,
        provider="dbnomics-v22",
        model="none",
        settings={"questions": len(todo), "window_s": WINDOW_S, "window_s_prime": WINDOW_S_PRIME},
        code_version=code_version,
    )
    report = ObsReport(run_id)
    by_code: dict[str, list[tuple[DQuestion, list[date]]]] = {}
    for q, dates in todo:
        by_code.setdefault(code_from_id(q.id), []).append((q, dates))
    for n, (code, group) in enumerate(sorted(by_code.items()), start=1):
        latest = max(q.freeze for q, _ in group)
        try:
            series: DSeries | None = await fetcher.fetch(code, latest)
            report.stations += 1
        except SearchError as exc:
            series = None
            problem = str(exc)
        for q, dates in group:
            started = time.monotonic()
            lq = ledger_question(q)
            ledger.record_question(
                run_id, lq, q.freeze, {"question_set": q.question_set, "id": q.id}
            )

            def skip(why: str, detail: str, lq: Question = lq, started: float = started) -> None:
                report.skipped[why] = report.skipped.get(why, 0) + 1
                ledger.record_outcome(
                    run_id=run_id,
                    question_id=lq.question_id,
                    post_id=lq.post_id,
                    question_type="binary",
                    status="skipped",
                    reason=detail[:300],
                    submitted=False,
                    closes_at=None,
                    minutes_left=None,
                    cost_usd=0.0,
                    seconds=time.monotonic() - started,
                )

            if series is None:
                skip("the DBnomics lookup failed", f"lookup failed: {problem}")
                continue
            view = series.until(q.freeze)
            report.refused += view.refused
            try:
                source = to_source(view, q.series, needed_days(q, dates))
            except SearchError as exc:
                skip("no observation on the days the arms read", str(exc))
                continue
            ledger.record_source(source)
            admission = admit_source(source, q.freeze, RunMode.BACKTEST)
            ledger.record_admission(run_id, lq.question_id, admission)
            if not admission.admitted:
                skip("refused by the leakage gate", admission.reason)
                continue
            report.recorded += 1
            ledger.record_outcome(
                run_id=run_id,
                question_id=lq.question_id,
                post_id=lq.post_id,
                question_type="binary",
                status="forecast",
                reason=f"{source.excerpt.count(chr(10))} observations, {view.refused} refused",
                submitted=False,
                closes_at=None,
                minutes_left=None,
                cost_usd=0.0,
                seconds=time.monotonic() - started,
            )
        out(f"[{n}/{len(by_code)}] {code}: {len(group)} questions")
    return report


def ledger_observations(path: Path) -> dict[int, dict[date, float]]:
    """Each question's admitted observations, read back from the ledger.

    Only a source the gate admitted for that question is read, and every line
    is refused here too unless its day is dated (00:00 UTC of the next day) at
    or before the question's cutoff.
    """
    if not path.exists():
        raise SeasonError(f"no observations ledger at {path}; run augur backtest dbnomics-series")
    out: dict[int, dict[date, float]] = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT q.question_id, q.cutoff, s.excerpt, s.published_at FROM questions q "
            "JOIN runs r ON r.run_id = q.run_id "
            "JOIN source_uses u ON u.run_id = q.run_id AND u.question_id = q.question_id "
            "JOIN sources s ON s.source_id = u.source_id "
            "WHERE r.tournament = ? AND u.admitted = 1 AND s.searcher = 'series'",
            (TOURNAMENT,),
        ).fetchall()
    for qid, cutoff, excerpt, published in rows:
        cut = ensure_utc(parse_iso(str(cutoff)))
        if published is None or ensure_utc(parse_iso(str(published))) > cut:
            raise SeasonError(f"question {qid}: an admitted source dated after its cutoff")
        lines = parse_closes(str(excerpt))
        last = last_admitted_day(cut)
        if any(d > last for d, _ in lines):
            raise SeasonError(f"question {qid}: an observation dated after its cutoff")
        if int(qid) in out:
            raise SeasonError(f"question {qid} has two admitted observation sources")
        out[int(qid)] = dict(lines)
    return out
