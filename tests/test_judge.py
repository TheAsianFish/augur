"""The leakage judge: its parser, its agreement statistics, and a judged arm in a ledger."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from augur.backtest import Retrieval
from augur.clock import FixedClock
from augur.config import Settings
from augur.errors import AugurError, LedgerError
from augur.forecastbench import FBQuestion
from augur.judge import (
    JUDGE_PREFIX,
    JUDGE_SYSTEM,
    agreement,
    cohen_kappa,
    judge_arm,
    judge_request,
    judgments,
    parse_scores,
    quadratic_weighted_kappa,
    shown_sources,
)
from augur.leakage import RunMode, scan_ledger
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, LlmResponse
from augur.llm.endpoints import PRESETS, Route
from augur.llm.router import Router

from .fakes.forecastbench import LOCAL, small_set
from .fakes.llm import ScriptedProvider
from .helpers import catalog_of
from .test_backtest_arms import (
    LATE_TITLE,
    TOURNAMENT,
    WALL,
    DatedSearcher,
    _run,
    _settings,
)

# -- parse_scores -----------------------------------------------------------------------


def test_parse_scores_reads_labelled_scores() -> None:
    text = '[{"source": "S1", "score": 0, "reason": "noise"}, {"source": "S2", "score": 4}]'
    assert parse_scores(text, 2) == {1: (0, "noise"), 2: (4, "")}


def test_parse_scores_tolerates_prose_fences_and_label_styles() -> None:
    text = (
        'Here you go:\n```json\n[{"source": "[S2]", "score": 3, "reason": "r"}, '
        '{"source": "s1", "score": "2"}]\n```'
    )
    assert parse_scores(text, 2) == {2: (3, "r"), 1: (2, "")}


def test_parse_scores_uses_the_position_when_the_label_is_missing_or_odd() -> None:
    text = '[{"score": 1}, {"source": "second", "score": 2}, {"source": null, "score": 3}]'
    assert parse_scores(text, 3) == {1: (1, ""), 2: (2, ""), 3: (3, "")}


@pytest.mark.parametrize(
    "score", ["5", "-1", "2.5", "high", "null", "[]", "true", "{}", '"nan"', "1e1"]
)
def test_parse_scores_drops_a_score_it_would_have_to_guess(score: str) -> None:
    assert parse_scores('[{"source": "S1", "score": ' + score + "}]", 1) == {}


def test_parse_scores_accepts_integral_floats() -> None:
    assert parse_scores('[{"source": "S1", "score": 3.0}]', 1) == {1: (3, "")}


def test_parse_scores_drops_out_of_range_labels() -> None:
    text = (
        '[{"source": "S0", "score": 1}, {"source": "S3", "score": 1}, {"source": "S1", "score": 2}]'
    )
    assert parse_scores(text, 2) == {1: (2, "")}


def test_parse_scores_skips_non_objects_but_counts_their_place() -> None:
    assert parse_scores('[1, "x", {"score": 4}]', 3) == {3: (4, "")}


@pytest.mark.parametrize("text", ["", "no array here", "[not json]", '{"score": 3}', "[[1, 2"])
def test_parse_scores_returns_nothing_for_an_unusable_reply(text: str) -> None:
    assert parse_scores(text, 2) == {}


def test_parse_scores_truncates_the_reason() -> None:
    [(_, reason)] = parse_scores(
        '[{"source": "S1", "score": 1, "reason": "%s"}]' % ("x" * 500), 1
    ).values()
    assert len(reason) == 300


def test_parse_scores_later_duplicates_win() -> None:
    text = '[{"source": "S1", "score": 1}, {"source": "S1", "score": 3}]'
    assert parse_scores(text, 1) == {1: (3, "")}


@pytest.mark.parametrize("score", ["Infinity", "-Infinity", "1e999"])
def test_parse_scores_drops_an_infinite_score_instead_of_raising(score: str) -> None:
    """Python's json accepts ``Infinity``; ``int(inf)`` raises OverflowError."""
    assert parse_scores('[{"source": "S1", "score": ' + score + "}]", 1) == {}


# -- agreement statistics ---------------------------------------------------------------


def test_qwk_is_one_for_identical_scores_and_matches_a_hand_computation() -> None:
    assert quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == pytest.approx(1.0)
    # One 4 judged 3: 1 - (1/16) / (17/16).
    assert quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 3]) == pytest.approx(16 / 17)


def test_qwk_is_zero_for_chance_and_negative_for_reversed_scores() -> None:
    assert quadratic_weighted_kappa([0, 0, 0], [4, 4, 4]) == pytest.approx(0.0)
    assert quadratic_weighted_kappa([0, 4, 0, 4], [4, 0, 4, 0]) == pytest.approx(-1.0)


def test_qwk_with_no_variation_at_all_is_defined() -> None:
    assert quadratic_weighted_kappa([2, 2], [2, 2]) == 1.0


@pytest.mark.parametrize(("a", "b"), [([], []), ([1], [1, 2])])
def test_the_kappas_refuse_unequal_or_empty_input(a: list[int], b: list[int]) -> None:
    with pytest.raises(ValueError, match="equal"):
        quadratic_weighted_kappa(a, b)
    with pytest.raises(ValueError, match="equal"):
        cohen_kappa([bool(x) for x in a], [bool(x) for x in b])


def test_cohen_kappa_matches_a_hand_computation() -> None:
    # po = 3/4, pa = 1/2, pb = 1/4, pe = 1/2.
    assert cohen_kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(
        0.5
    )
    assert cohen_kappa([True, True], [True, True]) == 1.0
    assert cohen_kappa([True, True], [False, False]) == pytest.approx(0.0)
    assert cohen_kappa([True, False], [False, True]) == pytest.approx(-1.0)


def test_agreement_merges_zero_and_one_and_counts_major_leakage() -> None:
    pairs = [(0, 1), (1, 0), (3, 4), (2, 2), (4, 2)]
    got = agreement(pairs)
    assert got.n == 5
    assert got.exact_merged == pytest.approx(3 / 5)  # (0,1), (1,0), (2,2)
    assert got.major_agree == pytest.approx(4 / 5)
    assert (got.human_major, got.judge_major) == (2, 1)
    # po = .8, pa = .4, pb = .2, pe = .56.
    assert got.major_kappa == pytest.approx(0.24 / 0.44)
    assert got.qwk == pytest.approx(quadratic_weighted_kappa([0, 1, 3, 2, 4], [1, 0, 4, 2, 2]))


def test_agreement_with_no_pairs_is_an_error() -> None:
    with pytest.raises(ValueError, match="no hand label"):
        agreement([])


# -- the request ------------------------------------------------------------------------


def test_the_judge_request_states_the_freeze_cutoff_and_the_outcome(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    fb = next(q for q in questions if q.fb_id == "a")  # resolved Yes
    wrong = next(q for q in questions if q.fb_id == "b")  # resolved No
    src = DatedSearcher()._page("early", "Some page", datetime(2025, 2, 1, tzinfo=UTC), WALL)
    request = judge_request(fb, [src])
    assert request.messages[0].content == JUDGE_SYSTEM
    user = request.messages[1].content
    assert "Information cutoff date: 2025-03-01 00:00 UTC" in user
    assert "How it resolved: Yes" in user
    assert "[S1] Some page (dated, dated 2025-02-01, https://example.org/early)" in user
    assert "How it resolved: No" in judge_request(wrong, [src]).messages[1].content


# -- a judged arm -----------------------------------------------------------------------


class JudgeProvider(ScriptedProvider):
    """Scores the page whose title is the late one 4 and every other source 1."""

    def __init__(self, reply: str | None = None) -> None:
        super().__init__((30,))
        self.reply = reply

    async def complete(self, request: LlmRequest) -> LlmResponse:
        if request.messages[0].content != JUDGE_SYSTEM:
            return await super().complete(request)
        self.requests.append(request)
        if self.reply is not None:
            text = self.reply
        else:
            blocks = re.findall(r"^\[S(\d+)\] (.*?) \(", request.messages[1].content, re.M)
            items = [
                f'{{"source": "S{n}", "score": {4 if title == LATE_TITLE else 1}, "reason": "r"}}'
                for n, title in blocks
            ]
            text = "[" + ", ".join(items) + "]"
        return LlmResponse(text=text, model=request.model, input_tokens=1, output_tokens=1)


NEW = Route(endpoint=PRESETS["openrouter"], model="acme/judge-2026")


def judge_router(provider: ScriptedProvider) -> Router:
    """A judge released after every question opened: allowed, as it never forecasts."""
    catalog = catalog_of((NEW.model, 0.0, 0.0), listed=datetime(2026, 1, 1, tzinfo=UTC))
    return Router(
        {"research": [NEW], "forecast": [LOCAL]},
        {"openrouter": provider, "ollama": provider},
        catalog,
        FixedClock(WALL),
    )


async def _judge(
    db: Path, questions: list[FBQuestion], provider: ScriptedProvider, **kw: Any
) -> Any:
    settings: Settings = _settings()
    with Ledger.open(db, FixedClock(datetime(2026, 7, 1, tzinfo=UTC))) as ledger:
        return await judge_arm(
            ledger=ledger,
            tournament=TOURNAMENT,
            questions=questions,
            settings=settings,
            router=judge_router(provider),
            clock=FixedClock(WALL),
            code_version="test",
            **kw,
        )


def _rows(path: Path, sql: str) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(path)) as conn:
        return [tuple(r) for r in conn.execute(sql)]


async def test_a_judged_now_arm_is_recorded_as_unrestricted_with_the_judge_prefix(
    tmp_path: Path,
) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(
        db,
        questions[:2],
        Retrieval([DatedSearcher()], as_of="now"),
        wall_clock=FixedClock(WALL),
    )
    provider = JudgeProvider()
    result = await _judge(db, questions, provider)
    assert (result.questions, result.judged, result.resumed, result.failed) == (2, 2, 0, [])
    runs = _rows(db, "SELECT mode, tournament, model FROM runs ORDER BY started_at, rowid")
    assert runs[-1] == ("unrestricted", JUDGE_PREFIX + TOURNAMENT, NEW.model)
    stored = _rows(
        db,
        "SELECT s.title, j.score, j.judge_model, j.rationale FROM leak_judgments j "
        "JOIN sources s USING (source_id) ORDER BY j.question_id, s.title",
    )
    assert len(stored) == 4  # two sources per question
    assert {(t, score) for t, score, *_ in stored} == {
        ("Background before the freeze", 1),
        (LATE_TITLE, 4),
    }
    assert {m for _, _, m, _ in stored} == {NEW.model}
    got = judgments(db, TOURNAMENT)
    assert sorted(got.values()) == [1, 1, 4, 4]
    assert {q for q, _ in got} == {q.question_id for q in questions[:2]}
    # Every judge call is role 'judge', by a model released after the questions opened,
    # and the scan still passes.
    assert _rows(db, "SELECT DISTINCT role FROM llm_calls WHERE model = 'acme/judge-2026'") == [
        ("judge",)
    ]
    assert scan_ledger(db) == []
    # The judge is told how each question resolved.
    users = [
        r.messages[1].content for r in provider.requests if r.messages[0].content == JUDGE_SYSTEM
    ]
    assert len(users) == 2
    assert all("How it resolved:" in u for u in users)


async def test_a_judged_open_arm_is_a_backtest_judge_run(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(db, questions[:1], Retrieval([DatedSearcher()]))
    result = await _judge(db, questions, JudgeProvider())
    assert result.judged == 1
    assert _rows(db, "SELECT mode, tournament FROM runs WHERE tournament LIKE 'judge:%'") == [
        ("backtest", JUDGE_PREFIX + TOURNAMENT)
    ]
    # Only the page the gate admitted was shown, so only it is judged.
    assert _rows(
        db, "SELECT s.title, j.score FROM leak_judgments j JOIN sources s USING (source_id)"
    ) == [("Background before the freeze", 1)]
    assert scan_ledger(db) == []


async def test_a_judge_run_is_resumable(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(db, questions[:2], Retrieval([DatedSearcher()]))
    first = await _judge(db, questions, JudgeProvider())
    assert first.judged == 2
    provider = JudgeProvider()
    again = await _judge(db, questions, provider)
    assert (again.questions, again.judged, again.resumed) == (2, 0, 2)
    assert provider.requests == []
    assert _rows(db, "SELECT COUNT(*) FROM leak_judgments") == [(2,)]


async def test_a_question_outside_the_given_set_is_skipped(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(db, questions[:2], Retrieval([DatedSearcher()]))
    result = await _judge(db, questions[:1], JudgeProvider())
    assert (result.questions, result.judged) == (1, 1)


async def test_an_unusable_reply_is_a_failure_not_a_crash(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(db, questions[:1], Retrieval([DatedSearcher()]))
    result = await _judge(db, questions, JudgeProvider("I cannot rate these."))
    assert result.judged == 0
    [failure] = result.failed
    assert "no usable scores" in failure
    assert _rows(db, "SELECT COUNT(*) FROM leak_judgments") == [(0,)]
    # Not recorded as judged, so a rerun tries again.
    again = await _judge(db, questions, JudgeProvider())
    assert again.judged == 1


async def test_partial_scores_store_only_the_scored_sources(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(
        db,
        questions[:1],
        Retrieval([DatedSearcher()], as_of="now"),
        wall_clock=FixedClock(WALL),
    )
    reply = '[{"source": "S2", "score": 3, "reason": "partial"}, {"source": "S1", "score": 9}]'
    result = await _judge(db, questions, JudgeProvider(reply))
    assert result.judged == 1
    assert _rows(db, "SELECT score, rationale FROM leak_judgments") == [(3, "partial")]


async def test_judging_an_arm_that_cited_nothing_is_an_error(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(db, questions, None)  # no retrieval: no source was shown
    with pytest.raises(AugurError, match="nothing to judge"):
        await _judge(db, questions, JudgeProvider())


async def test_judging_a_tournament_that_mixes_modes_is_an_error(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(db, questions[:1], Retrieval([DatedSearcher()]))
    await _run(
        db,
        questions[1:2],
        Retrieval([DatedSearcher()], as_of="now"),
        wall_clock=FixedClock(WALL),
    )
    with pytest.raises(AugurError, match="mixes run modes"):
        await _judge(db, questions, JudgeProvider())


async def test_shown_sources_is_what_the_forecasters_cited_in_order(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(
        db,
        questions[:1],
        Retrieval([DatedSearcher()], as_of="now"),
        wall_clock=FixedClock(WALL),
    )
    with closing(sqlite3.connect(db)) as conn:
        [shown] = shown_sources(conn, TOURNAMENT)
        titles = [
            conn.execute("SELECT title FROM sources WHERE source_id = ?", (sid,)).fetchone()[0]
            for sid in shown.source_ids
        ]
        assert shown_sources(conn, "forecastbench:other.json") == []
    assert shown.mode is RunMode.UNRESTRICTED
    assert shown.cutoff == WALL
    assert shown.question_id == questions[0].question_id
    assert titles == ["Background before the freeze", LATE_TITLE]


async def test_the_latest_run_of_a_question_wins_in_shown_sources(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    one = questions[:1]
    await _run(db, one, Retrieval([DatedSearcher()]))
    # A later run of the same tournament, with no retrieval, is not resumed (it is a new
    # tournament in no way); force it by naming the same tournament with a different mode.
    await _run(db, one, Retrieval([DatedSearcher()], as_of="now"), wall_clock=FixedClock(WALL))
    with closing(sqlite3.connect(db)) as conn:
        [shown] = shown_sources(conn, TOURNAMENT)
    assert shown.mode is RunMode.UNRESTRICTED
    assert len(shown.source_ids) == 2


# -- the ledger table -------------------------------------------------------------------


async def test_judgments_are_append_only_and_bounded(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "arm.sqlite"
    await _run(db, questions[:1], Retrieval([DatedSearcher()]))
    await _judge(db, questions, JudgeProvider())
    with Ledger.open(db, FixedClock(datetime(2026, 7, 1, tzinfo=UTC))) as ledger:
        conn = ledger.connection
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE leak_judgments SET score = 0")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM leak_judgments")
        run_id, qid, sid, call_id = conn.execute(
            "SELECT run_id, question_id, source_id, call_id FROM leak_judgments"
        ).fetchone()
        for bad in (-1, 5):
            with pytest.raises(LedgerError, match="score BETWEEN 0 AND 4"):
                ledger.record_judgment(
                    run_id=run_id,
                    question_id=qid,
                    source_id=sid,
                    call_id=call_id,
                    judge_model="m",
                    score=bad,
                    rationale="",
                )
        ledger.record_judgment(
            run_id=run_id,
            question_id=qid,
            source_id=sid,
            call_id=call_id,
            judge_model="m",
            score=2,
            rationale="y" * 5000,
        )
        [(longest,)] = conn.execute("SELECT MAX(LENGTH(rationale)) FROM leak_judgments")
        assert longest == 1000
    # The latest judgment of a pair wins.
    assert list(judgments(db, TOURNAMENT).values()) == [2]
