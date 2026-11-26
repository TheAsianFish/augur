"""M15: ForecastBench's dataset questions with no model in the forecast.

ForecastBench asks, of a series (a stock's close, a FRED or DBnomics series,
an ACLED count, a Wikipedia ranking), whether its value on each of several
resolution dates will be higher than on the forecast due date. Every Augur
backtest before M15 dropped these questions. This module is PLAN.md's
``m15-registration`` block in code:

* :func:`load` reads the committed extract (``data/forecastbench/dataset``,
  written by ``scripts/extract_dataset_questions.py``): the history sets
  from 2026-01-04 and the seven corpus sets from 2026-07-05 to 2026-09-27;
* an *instance* is one question at one resolution date resolved to 0 or 1;
* :class:`History` is the walk-forward history: an outcome dated D is known to
  a question frozen at F only if D + :data:`LAG_DAYS` is on or before F's day;
* the arms: A always 50%, B the source's base rate, C the series' own rate
  shrunk toward B, and D on ``yfinance`` the share of the ticker's past
  same-horizon changes that were positive, shrunk toward B;
* :func:`record_closes` fetches each ``yfinance`` question's closes as of its
  freeze instant and writes them to a backtest ledger as dated sources,
  admitted through the leakage gate; :func:`ledger_closes` reads back only
  what the gate admitted, so the scored arm never sees anything else.

No model is called anywhere here. The scoring against ForecastBench's own
entrants on the same instances is :mod:`augur.fbdataset_eval`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any

from augur.clock import ensure_utc, parse_iso
from augur.errors import AugurError, SearchError
from augur.leakage import RunMode, admit_source
from augur.ledger import Ledger
from augur.questions import Question
from augur.research.series import SeriesFetcher, to_source

DATASET_DIR = Path("data/forecastbench/dataset")
LEDGER = Path("data/ledgers/m15-dataset-closes.sqlite")
TOURNAMENT = "forecastbench-dataset:2026-07-05..2026-09-27"
SOURCES: tuple[str, ...] = ("acled", "dbnomics", "fred", "wikipedia", "yfinance")
CORPUS_SETS: tuple[str, ...] = (
    "2026-07-05",
    "2026-07-19",
    "2026-08-02",
    "2026-08-16",
    "2026-08-30",
    "2026-09-13",
    "2026-09-27",
)
HISTORY_FIRST = "2026-01-04"

LAG_DAYS = 14
"""An outcome dated D is known at F only if D + 14 days is on or before F's day (item 2)."""
MIN_SOURCE = 20
"""Arm B is 0.5 while its source has fewer known outcomes (item 4)."""
SERIES_PSEUDO = 4.0
"""Arm C's pseudo-count of arm B (item 5)."""
SERIES_WEIGHT = 0.5
"""Arm D's weight on the share of positive changes; the rest is arm B (item 6)."""
MIN_CHANGES = 100
LOOKBACK_DAYS = 730
RATE_WINDOW_DAYS = 365
FETCH_DAYS = 800
"""How far before the freeze instant the closes are asked for (the lookback plus slack)."""
ARMS: tuple[str, ...] = ("A", "B", "C", "D")
ARM_NAMES: dict[str, str] = {
    "A": "always 50%",
    "B": "source base rate",
    "C": "series' own rate",
    "D": "series arm (yfinance)",
}


class DatasetError(AugurError):
    """The committed extract or ledger is not what M15 expects."""


# -- the corpus ---------------------------------------------------------------------


@dataclass(frozen=True)
class DQuestion:
    """One single-series dataset question of one set, with its resolved outcomes."""

    question_set: str
    """The set's due date, ``2026-08-30``."""
    source: str
    id: str
    freeze: datetime
    freeze_value: float | None
    url: str
    resolutions: tuple[tuple[date, int], ...]

    @property
    def due(self) -> date:
        return date.fromisoformat(self.question_set)

    @property
    def series(self) -> str:
        return f"{self.source}:{self.id}"

    @property
    def question_id(self) -> int:
        """A stable integer id: 52 bits of the SHA-256 of set, source and id."""
        digest = hashlib.sha256(f"{self.question_set}:{self.series}".encode()).hexdigest()
        return int(digest[:13], 16)

    def to_question(self) -> Question:
        """The question as the ledger records it (it opened at its freeze instant)."""
        return Question(
            post_id=self.question_id,
            question_id=self.question_id,
            title=(
                f"Will {self.id}'s close on each resolution date be higher than on "
                f"{self.question_set}? (ForecastBench {self.question_set}, {self.source})"
            ),
            type="binary",
            status="open",
            url=self.url or "https://www.forecastbench.org",
            open_time=self.freeze,
            scheduled_close_time=datetime.combine(self.due, dtime(0, 0), tzinfo=UTC),
            scheduled_resolve_time=None,
            description="",
            resolution_criteria="",
        )


@dataclass(frozen=True)
class Instance:
    """One question at one resolution date: what is scored."""

    question: DQuestion
    resolution_date: date
    outcome: int

    @property
    def source(self) -> str:
        return self.question.source

    @property
    def key(self) -> tuple[str, str, str, str]:
        q = self.question
        return (q.question_set, q.source, q.id, self.resolution_date.isoformat())


@dataclass
class Loaded:
    questions: list[DQuestion]
    """Every set's questions, history and corpus."""
    corpus: list[Instance]
    """The corpus sets' instances, in set, source, id and date order."""
    removed: dict[str, int] = field(default_factory=dict)
    """Corpus sets only: what the extract left out, by reason."""
    dataset_questions: int = 0
    """Corpus sets only: dataset questions before anything was left out."""
    provenance: dict[str, Any] = field(default_factory=dict)


def load(root: Path) -> Loaded:
    """Read every committed set of the extract; the corpus sets must all be there."""
    directory = root / DATASET_DIR
    files = sorted(directory.glob("2026-*.json"))
    if not files:
        raise DatasetError(
            f"no dataset extract in {directory}; run scripts/extract_dataset_questions.py"
        )
    loaded = Loaded([], [])
    seen: set[str] = set()
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        s = path.stem
        if doc.get("forecast_due_date") != s or s < HISTORY_FIRST:
            raise DatasetError(f"{path.name} is not the {s} set of the extract")
        role = "corpus" if s in CORPUS_SETS else "history"
        if doc.get("role") != role:
            raise DatasetError(f"{path.name} is marked {doc.get('role')!r}, expected {role!r}")
        seen.add(s)
        prov = doc.get("provenance", {})
        loaded.provenance[s] = {
            "questions_sha256": prov.get("questions", {}).get("sha256"),
            "resolutions_sha256": prov.get("resolutions", {}).get("sha256"),
            "fetched": prov.get("fetched"),
        }
        for raw in doc["questions"]:
            q = DQuestion(
                question_set=s,
                source=str(raw["source"]),
                id=str(raw["id"]),
                freeze=ensure_utc(parse_iso(str(raw["freeze_datetime"]))),
                freeze_value=raw.get("freeze_value"),
                url=str(raw.get("url") or ""),
                resolutions=tuple(
                    (date.fromisoformat(d), int(y)) for d, y in sorted(raw["resolutions"].items())
                ),
            )
            if q.source not in SOURCES:
                raise DatasetError(f"{path.name}: {q.series} is not a dataset source")
            loaded.questions.append(q)
            if role == "corpus":
                loaded.corpus.extend(Instance(q, d, y) for d, y in q.resolutions)
        if role == "corpus":
            loaded.dataset_questions += int(doc["dataset_questions"])
            for reason, n in doc["removed"].items():
                loaded.removed[reason] = loaded.removed.get(reason, 0) + int(n)
    missing = [s for s in CORPUS_SETS if s not in seen]
    if missing:
        raise DatasetError(f"the extract lacks the corpus sets {', '.join(missing)}")
    loaded.corpus.sort(key=lambda i: i.key)
    return loaded


# -- the walk-forward history ---------------------------------------------------------


@dataclass(frozen=True)
class Known:
    """Outcome counts known at one instant: by source and by series, ``(yes, n)``."""

    by_source: Mapping[str, tuple[int, int]]
    by_series: Mapping[str, tuple[int, int]]
    latest: date | None
    """The latest outcome date counted (for the time-order check)."""


class History:
    """Every outcome of every set, read walk-forward."""

    def __init__(self, questions: Iterable[DQuestion]) -> None:
        self._outcomes = sorted(
            (d, q.source, q.series, y) for q in questions for d, y in q.resolutions
        )
        self._cache: dict[date, Known] = {}

    @staticmethod
    def known_on(day: date) -> date:
        """The latest outcome date known on ``day``."""
        return day - timedelta(days=LAG_DAYS)

    def at(self, freeze: datetime) -> Known:
        """The counts of the outcomes known at ``freeze`` (item 2)."""
        day = ensure_utc(freeze).date()
        cached = self._cache.get(day)
        if cached is not None:
            return cached
        bound = self.known_on(day)
        by_source: dict[str, tuple[int, int]] = {}
        by_series: dict[str, tuple[int, int]] = {}
        latest: date | None = None
        for d, source, series, y in self._outcomes:
            if d > bound:
                break
            yes, n = by_source.get(source, (0, 0))
            by_source[source] = (yes + y, n + 1)
            yes, n = by_series.get(series, (0, 0))
            by_series[series] = (yes + y, n + 1)
            latest = d
        known = Known(by_source, by_series, latest)
        self._cache[day] = known
        return known


def arm_b(known: Known, source: str) -> float:
    """The source's walk-forward base rate (item 4)."""
    yes, n = known.by_source.get(source, (0, 0))
    return 0.5 if n < MIN_SOURCE else (yes + 1) / (n + 2)


def arm_c(known: Known, series: str, b: float) -> float:
    """The series' own walk-forward rate, shrunk toward arm B (item 5)."""
    yes, n = known.by_series.get(series, (0, 0))
    return (yes + SERIES_PSEUDO * b) / (n + SERIES_PSEUDO)


# -- arm D: the series arm --------------------------------------------------------------


@dataclass(frozen=True)
class SeriesShare:
    p_up: float
    h: int
    changes: int


def observations_per_day(closes: Sequence[tuple[date, float]]) -> float:
    """The series' observations per calendar day over its last :data:`RATE_WINDOW_DAYS`."""
    last = closes[-1][0]
    window = [d for d, _ in closes if (last - d).days < RATE_WINDOW_DAYS]
    span = (last - window[0]).days
    return (len(window) - 1) / span if span > 0 else 1.0


def share_up(
    closes: Sequence[tuple[date, float]], due: date, resolution: date, *, freeze: date
) -> SeriesShare | None:
    """The share of past ``h``-observation changes above 0, or ``None`` (item 6).

    ``closes`` are the admitted closes, oldest first; ``h`` is the calendar
    days from the due date to the resolution date times the observation rate.
    The window is the :data:`LOOKBACK_DAYS` before ``freeze`` (the freeze
    instant's day), as registered, not before the last close.
    """
    if not closes:
        return None
    h = max(1, round((resolution - due).days * observations_per_day(closes)))
    values = [v for d, v in closes if (freeze - d).days <= LOOKBACK_DAYS]
    n = len(values) - h
    if n < MIN_CHANGES:
        return None
    up = sum(1 for i in range(n) if values[i + h] > values[i])
    return SeriesShare(up / n, h, n)


def arm_d(share: SeriesShare | None, b: float) -> float:
    """``0.5·p_series + 0.5·B``, or B where the series rule has no share."""
    if share is None:
        return b
    return SERIES_WEIGHT * share.p_up + (1.0 - SERIES_WEIGHT) * b


# -- the closes, written to the ledger first -------------------------------------------


def yfinance_questions(loaded: Loaded) -> list[DQuestion]:
    """The corpus sets' ``yfinance`` questions, in set and ticker order."""
    return sorted(
        (q for q in loaded.questions if q.source == "yfinance" and q.question_set in CORPUS_SETS),
        key=lambda q: (q.question_set, q.id),
    )


def recorded_ids(path: Path) -> set[int]:
    """Questions the closes ledger already holds (so a rerun resumes)."""
    if not path.exists():
        return set()
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT q.question_id FROM questions q JOIN runs r ON r.run_id = q.run_id "
            "WHERE r.tournament = ?",
            (TOURNAMENT,),
        ).fetchall()
    return {int(r[0]) for r in rows}


@dataclass
class ClosesReport:
    run_id: str
    recorded: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    refused: int = 0
    """Closes the server sent that were dated after their question's freeze instant."""

    def line(self) -> str:
        skipped = ", ".join(f"{n} {why}" for why, n in sorted(self.skipped.items())) or "none"
        return (
            f"{self.recorded} questions' closes recorded; {self.refused} closes dated after "
            f"the freeze refused; skipped: {skipped}"
        )


async def record_closes(
    *,
    questions: Sequence[DQuestion],
    fetcher: SeriesFetcher,
    ledger: Ledger,
    code_version: str,
    out: Callable[[str], None] = print,
) -> ClosesReport:
    """Fetch and record each question's closes as of its freeze instant (no model call).

    Each question gets one source, the admitted closes of the
    :data:`FETCH_DAYS` before its freeze instant, written to the ledger and
    then admitted through the gate with the freeze instant as the cutoff.
    """
    run_id = ledger.start_run(
        mode=RunMode.BACKTEST,
        tournament=TOURNAMENT,
        provider="yahoo-chart",
        model="none",
        settings={"questions": len(questions), "fetch_days": FETCH_DAYS},
        code_version=code_version,
    )
    report = ClosesReport(run_id)
    for i, dq in enumerate(questions, start=1):
        started = time.monotonic()
        q = dq.to_question()
        payload = {"question_set": dq.question_set, "source": dq.source, "id": dq.id}
        ledger.record_question(run_id, q, dq.freeze, payload)

        def skip(why: str, detail: str, q: Question = q, started: float = started) -> None:
            report.skipped[why] = report.skipped.get(why, 0) + 1
            ledger.record_outcome(
                run_id=run_id,
                question_id=q.question_id,
                post_id=q.post_id,
                question_type="binary",
                status="skipped",
                reason=detail[:300],
                submitted=False,
                closes_at=None,
                minutes_left=None,
                cost_usd=0.0,
                seconds=time.monotonic() - started,
            )

        try:
            series = await fetcher.fetch(
                dq.id, dq.freeze, start=dq.freeze - timedelta(days=FETCH_DAYS)
            )
        except SearchError as exc:
            skip("the chart lookup failed", f"chart lookup failed: {exc}")
            out(f"[{i}/{len(questions)}] {dq.question_set} {dq.id}: lookup failed")
            continue
        report.refused += series.refused
        if not series.observations:
            skip("no close at or before the freeze", "no close dated at or before the freeze")
            out(f"[{i}/{len(questions)}] {dq.question_set} {dq.id}: no close")
            continue
        source = to_source(series, dq.id, keep=len(series.observations))
        ledger.record_source(source)
        admission = admit_source(source, dq.freeze, RunMode.BACKTEST)
        ledger.record_admission(run_id, q.question_id, admission)
        if not admission.admitted:
            skip("refused by the leakage gate", admission.reason)
            continue
        report.recorded += 1
        ledger.record_outcome(
            run_id=run_id,
            question_id=q.question_id,
            post_id=q.post_id,
            question_type="binary",
            status="forecast",
            reason=(
                f"{len(series.observations)} closes to {series.observations[-1].day}, "
                f"{series.refused} later ones refused"
            ),
            submitted=False,
            closes_at=None,
            minutes_left=None,
            cost_usd=0.0,
            seconds=time.monotonic() - started,
        )
        out(
            f"[{i}/{len(questions)}] {dq.question_set} {dq.id}: {len(series.observations)} "
            f"closes to {series.observations[-1].day} ({series.refused} refused)"
        )
    return report


_CLOSE_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) ([0-9.eE+-]+)$")


def parse_closes(excerpt: str) -> list[tuple[date, float]]:
    """The ``day close`` lines of a series source's excerpt, oldest first."""
    out: list[tuple[date, float]] = []
    for line in excerpt.splitlines():
        m = _CLOSE_LINE.match(line)
        if m:
            out.append((date.fromisoformat(m.group(1)), float(m.group(2))))
    if any(b[0] <= a[0] for a, b in itertools.pairwise(out)):
        raise DatasetError("a series source's closes are not in day order")
    return out


def ledger_closes(path: Path) -> dict[int, list[tuple[date, float]]]:
    """Each question's admitted closes, read back from the closes ledger.

    Only a source the gate admitted for that question is read, and a close
    is refused here too if its day is not before the question's cutoff (the
    close of day D is dated 00:00 UTC of D + 1).
    """
    if not path.exists():
        raise DatasetError(f"no closes ledger at {path}; run augur backtest dataset-closes")
    out: dict[int, list[tuple[date, float]]] = {}
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
            raise DatasetError(f"question {qid}: an admitted source dated after its cutoff")
        closes = parse_closes(str(excerpt))
        if any(
            datetime.combine(d + timedelta(days=1), dtime(0, 0), tzinfo=UTC) > cut
            for d, _ in closes
        ):
            raise DatasetError(f"question {qid}: a close dated after its cutoff")
        if int(qid) in out:
            raise DatasetError(f"question {qid} has two admitted series sources")
        out[int(qid)] = closes
    return out


# -- every arm on every instance ------------------------------------------------------------


@dataclass(frozen=True)
class Forecasts:
    """Each arm's forecast on one instance, and what it rested on."""

    instance: Instance
    a: float
    b: float
    c: float
    d: float | None
    """``yfinance`` only."""
    source_n: int
    series_n: int
    share: SeriesShare | None
    latest_known: date | None

    def get(self, arm: str) -> float:
        value = {"A": self.a, "B": self.b, "C": self.c, "D": self.d}[arm]
        if value is None:
            raise DatasetError(f"arm {arm} has no forecast on {self.instance.key}")
        return value


def forecast_all(
    loaded: Loaded, closes: Mapping[int, Sequence[tuple[date, float]]]
) -> list[Forecasts]:
    """Arms A to D on every corpus instance, walk-forward; refuses any later outcome."""
    history = History(loaded.questions)
    out: list[Forecasts] = []
    for inst in loaded.corpus:
        q = inst.question
        known = history.at(q.freeze)
        if known.latest is not None and known.latest > History.known_on(q.freeze.date()):
            raise DatasetError(f"{inst.key}: an outcome known after its freeze was counted")
        b = arm_b(known, q.source)
        c = arm_c(known, q.series, b)
        share: SeriesShare | None = None
        d: float | None = None
        if q.source == "yfinance":
            got = closes.get(q.question_id)
            share = (
                share_up(got, q.due, inst.resolution_date, freeze=q.freeze.date()) if got else None
            )
            d = arm_d(share, b)
        out.append(
            Forecasts(
                instance=inst,
                a=0.5,
                b=b,
                c=c,
                d=d,
                source_n=known.by_source.get(q.source, (0, 0))[1],
                series_n=known.by_series.get(q.series, (0, 0))[1],
                share=share,
                latest_known=known.latest,
            )
        )
    return out


def brier(p: float, y: int) -> float:
    if not 0.0 <= p <= 1.0 or math.isnan(p):
        raise DatasetError(f"a forecast must be in [0, 1], got {p}")
    return (p - y) ** 2
