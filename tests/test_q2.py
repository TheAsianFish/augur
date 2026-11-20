"""Q2's scoring from small, real ledgers: pairing, truncation, the price edge, the judge."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from augur.clock import FixedClock
from augur.forecastbench import FBQuestion
from augur.judge import JUDGE_PREFIX
from augur.leakage import RunMode, admit_source
from augur.ledger import Ledger
from augur.q2 import (
    Q2,
    ArmSource,
    forecasts,
    judge_row,
    major_questions,
    reference_agreement,
    score_q2,
)
from augur.replay import PRICES_SUFFIX

from .helpers import market_source, source

TOURNAMENT = "forecastbench:test-set"
T0 = datetime(2026, 6, 11, tzinfo=UTC)


def _fb(fb_id: str, outcome: int) -> FBQuestion:
    return FBQuestion(
        fb_id=fb_id,
        source="manifold",
        question=f"Will {fb_id} happen?",
        resolution_criteria="c",
        background="b",
        source_intro="i",
        url=f"https://example.org/{fb_id}",
        freeze_datetime=T0,
        freeze_value=0.5,
        market_open=T0 - timedelta(days=30),
        market_close=T0 + timedelta(days=30),
        resolution_date=date(2026, 8, 1),
        outcome=outcome,
    )


def _questions() -> list[FBQuestion]:
    """Five questions in question-id order, outcomes 1, 0, 1, 0, 1."""
    raw = sorted((_fb(f"q{i}", 0) for i in range(5)), key=lambda q: q.question_id)
    return [q.model_copy(update={"outcome": 1 - i % 2}) for i, q in enumerate(raw)]


QS = _questions()
IDS = [q.question_id for q in QS]
Q0, Q1, Q2_, Q3, Q4 = IDS


def _open(path: Path) -> tuple[Ledger, FixedClock]:
    clock = FixedClock(T0)
    return Ledger.open(path, clock), clock


def _start(ledger: Ledger, tournament: str, mode: RunMode) -> str:
    run_id = ledger.start_run(
        mode=mode,
        tournament=tournament,
        provider="replay",
        model="m",
        settings={},
        code_version="test",
    )
    for q in QS:
        ledger.record_question(run_id, q.to_question(), T0, {})
    return run_id


def _arm(
    path: Path,
    mode: RunMode,
    aggregates: list[tuple[int, float]],
    *,
    tournament: str = TOURNAMENT,
    members: list[tuple[int, float]] | None = None,
) -> None:
    """A ledger with one run; the aggregates are recorded in order, a second apart."""
    ledger, clock = _open(path)
    with ledger:
        run_id = _start(ledger, tournament, mode)
        for qid, p in members or []:
            ledger.record_forecast(
                run_id=run_id,
                question_id=qid,
                kind="member",
                member=0,
                method="draw",
                probability=p,
                reasoning="",
                call_id=None,
            )
        for qid, p in aggregates:
            clock.advance(1)
            ledger.record_forecast(
                run_id=run_id,
                question_id=qid,
                kind="aggregate",
                member=None,
                method="median",
                probability=p,
                reasoning="",
                call_id=None,
            )


def _judge(
    path: Path, entries: list[tuple[int, str, int]], tournament: str = TOURNAMENT
) -> dict[str, str]:
    """Judgments in order, a second apart; ``tag`` names a source. Returns tag -> source id."""
    ledger, clock = _open(path)
    ids: dict[str, str] = {}
    with ledger:
        run_id = _start(ledger, JUDGE_PREFIX + tournament, RunMode.BACKTEST)
        for qid, tag, score in entries:
            if tag not in ids:
                s = source(
                    T0 - timedelta(days=1),
                    retrieved_at=T0,
                    text=f"text {tag}",
                    url=f"https://x/{tag}",
                )
                ledger.record_source(s)
                ids[tag] = s.source_id
            call_id = ledger.record_call(
                run_id=run_id,
                question_id=qid,
                role="judge",
                member=None,
                provider="replay",
                model="m",
                model_release_date=None,
                request_hash="h",
                request={},
                response_text="",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                error=None,
                started_at=T0,
                source_ids=[],
            )
            clock.advance(1)
            ledger.record_judgment(
                run_id=run_id,
                question_id=qid,
                source_id=ids[tag],
                call_id=call_id,
                judge_model="m",
                score=score,
                rationale="",
            )
    return ids


def _prices(path: Path, quotes: dict[int, float]) -> None:
    ledger, _ = _open(path)
    with ledger:
        run_id = _start(ledger, TOURNAMENT + PRICES_SUFFIX, RunMode.BACKTEST)
        for qid, p in quotes.items():
            s = market_source(T0 - timedelta(hours=1), as_of=T0, probability=p)
            ledger.record_source(s)
            ledger.record_admission(run_id, qid, admit_source(s, T0, RunMode.BACKTEST))


def _costs(path: Path) -> None:
    ledger, _ = _open(path)
    with ledger:
        run_id = _start(ledger, TOURNAMENT + "+onepass-asof-open", RunMode.BACKTEST)
        for qid, secs in ((Q0, 10.0), (Q1, 20.0), (Q2_, 30.0), (Q3, 99.0)):
            ledger.record_outcome(
                run_id=run_id,
                question_id=qid,
                post_id=qid,
                question_type="binary",
                status="forecast",
                reason="",
                submitted=False,
                closes_at=None,
                minutes_left=None,
                cost_usd=0.0,
                seconds=secs,
            )
        for qid, role, error in (
            (Q0, "researcher", None),
            (Q0, "forecaster", None),
            (Q0, "forecaster", None),
            (Q1, "forecaster", None),
            (Q2_, "forecaster", "boom"),
            (Q3, "forecaster", None),
        ):
            ledger.record_call(
                run_id=run_id,
                question_id=qid,
                role=role,
                member=None,
                provider="replay",
                model="m",
                model_release_date=None,
                request_hash="h",
                request={},
                response_text="",
                input_tokens=1,
                output_tokens=10,
                cost_usd=0.0,
                latency_ms=1,
                error=error,
                started_at=T0,
                source_ids=[],
            )


@pytest.fixture
def arms(tmp_path: Path) -> dict[str, ArmSource]:
    """N, O, T, L over five questions; the sample of 4 leaves out Q4, O never forecast Q3."""
    _arm(tmp_path / "n.sqlite", RunMode.BACKTEST, [(q, 0.5) for q in IDS])
    # Q1 forecast twice in O: the later aggregate (0.2) wins over the first (0.4). The member
    # forecast of Q3 is not an aggregate and does not make O a forecaster of Q3.
    _arm(
        tmp_path / "o.sqlite",
        RunMode.BACKTEST,
        [(Q0, 0.8), (Q1, 0.4), (Q2_, 0.6), (Q4, 0.5), (Q1, 0.2)],
        tournament=TOURNAMENT + "+onepass-asof-open",
        members=[(Q3, 0.5)],
    )
    _arm(
        tmp_path / "t.sqlite",
        RunMode.UNRESTRICTED,
        [(q, p) for q, p in zip(IDS, (0.9, 0.1, 0.9, 0.1, 0.9), strict=True)],
        tournament=TOURNAMENT + "+onepass-asof-now",
    )
    _arm(
        tmp_path / "l.sqlite",
        RunMode.BACKTEST,
        [(Q0, 0.7), (Q2_, 0.5), (Q3, 0.5)],
        tournament=TOURNAMENT + "+loop3-asof-open",
    )
    _prices(tmp_path / "prices.sqlite", {Q0: 0.7, Q2_: 0.4})
    out = {"N": "", "O": "+onepass-asof-open", "T": "+onepass-asof-now", "L": "+loop3-asof-open"}
    return {
        a: ArmSource(tmp_path / f"{a.lower()}.sqlite", TOURNAMENT + sfx) for a, sfx in out.items()
    }


def _score(tmp_path: Path, arms: dict[str, ArmSource], **kw: object) -> Q2:
    params: dict[str, object] = {
        "questions": QS[::-1],
        "sample": 4,
        "loop_sample": 3,
        "arms": arms,
        "prices": ArmSource(tmp_path / "prices.sqlite", TOURNAMENT),
        "labels": tmp_path / "none.json",
    }
    return score_q2(**(params | kw))  # type: ignore[arg-type]


def test_the_questions_are_in_id_order_with_known_outcomes() -> None:
    assert sorted(IDS) == IDS
    assert [q.outcome for q in QS] == [1, 0, 1, 0, 1]


def test_arms_pair_on_questions_every_main_arm_forecast(
    tmp_path: Path, arms: dict[str, ArmSource]
) -> None:
    s = _score(tmp_path, arms)
    # The sample keeps Q0..Q3 (Q4 is cut by id order); O never forecast Q3.
    assert s.sample == 3
    assert [r.arm for r in s.arms] == ["N", "O", "T"]
    assert [r.n for r in s.arms] == [3, 3, 3]
    by = {r.arm: r for r in s.arms}
    # Outcomes 1, 0, 1. N 0.5 each: 0.25. O 0.8, 0.2 (latest), 0.6: (0.04+0.04+0.16)/3.
    assert by["N"].brier.mean == pytest.approx(0.25)
    assert by["O"].brier.mean == pytest.approx(0.08)
    assert by["T"].brier.mean == pytest.approx(0.01)
    assert by["N"].log == pytest.approx(-0.6931, abs=1e-3)


def test_sample_truncation_follows_question_id_order_not_input_order(
    tmp_path: Path, arms: dict[str, ArmSource]
) -> None:
    assert _score(tmp_path, arms, questions=QS).sample == _score(tmp_path, arms).sample == 3
    # A sample of 2 keeps Q0 and Q1 only; one of 5 keeps all, but O lacks Q3 so Q4 joins.
    assert _score(tmp_path, arms, sample=2).sample == 2
    assert _score(tmp_path, arms, sample=5).sample == 4


def test_differences_have_the_documented_sign(tmp_path: Path, arms: dict[str, ArmSource]) -> None:
    s = _score(tmp_path, arms)
    d = {x.label.split(" ")[0] + x.label.split(" ")[2]: x.brier.mean for x in s.diffs}
    assert [x.label.split(" (")[0] for x in s.diffs] == ["T − N", "O − N", "T − O"]
    # Negative: the first arm is better.
    assert d["TN"] == pytest.approx(0.01 - 0.25)
    assert d["ON"] == pytest.approx(0.08 - 0.25)
    assert d["TO"] == pytest.approx(0.01 - 0.08)


def test_the_latest_aggregate_wins_and_members_do_not_count(
    tmp_path: Path, arms: dict[str, ArmSource]
) -> None:
    got = forecasts(arms["O"].path, arms["O"].tournament)
    assert got[Q1] == 0.2
    assert Q3 not in got
    assert got[Q4] == 0.5


def test_live_runs_are_not_read(tmp_path: Path) -> None:
    _arm(tmp_path / "live.sqlite", RunMode.LIVE, [(Q0, 0.3)])
    assert forecasts(tmp_path / "live.sqlite", TOURNAMENT) == {}


def test_edge_over_the_price_uses_priced_questions_and_unpriced_are_separate(
    tmp_path: Path, arms: dict[str, ArmSource]
) -> None:
    s = _score(tmp_path, arms)
    assert s.priced == 2  # Q0 and Q2; Q1 has no price
    edge = {x.label.split(" ")[0]: x.brier.mean for x in s.edge if " − price" in x.label}
    # Prices 0.7 and 0.4 on outcomes 1 and 1: Brier 0.09 and 0.36.
    assert edge["N"] == pytest.approx((0.25 - 0.09 + 0.25 - 0.36) / 2)
    assert edge["O"] == pytest.approx((0.04 - 0.09 + 0.16 - 0.36) / 2)
    assert edge["T"] == pytest.approx((0.01 - 0.09 + 0.01 - 0.36) / 2)
    # Q1 (outcome 0) alone: N 0.25, O 0.2 -> 0.04, T 0.1 -> 0.01.
    assert [(r.arm, r.n) for r in s.unpriced] == [("N", 1), ("O", 1), ("T", 1)]
    assert [r.brier.mean for r in s.unpriced] == pytest.approx([0.25, 0.04, 0.01])


def test_with_no_prices_there_is_no_edge_and_every_question_is_unpriced(
    tmp_path: Path, arms: dict[str, ArmSource]
) -> None:
    _prices(tmp_path / "empty.sqlite", {})
    s = _score(tmp_path, arms, prices=ArmSource(tmp_path / "empty.sqlite", TOURNAMENT))
    assert s.priced == 0
    assert s.edge == []
    assert s.unpriced[0].n == 3


def test_loop_rows_cover_only_loop_sample_questions_both_l_and_o_forecast(
    tmp_path: Path, arms: dict[str, ArmSource]
) -> None:
    s = _score(tmp_path, arms)
    # loop_sample 3 keeps Q0..Q2; L forecast Q0, Q2 (and Q3, outside the loop sample and not in O).
    assert s.loop_n == 2
    assert [r.arm for r in s.loop_arms] == ["N", "O", "L"]
    assert all(r.n == 2 for r in s.loop_arms)
    by = {r.arm: r.brier.mean for r in s.loop_arms}
    assert by["O"] == pytest.approx(0.10)
    assert by["L"] == pytest.approx((0.09 + 0.25) / 2)
    [diff] = s.loop_diffs
    assert diff.label.startswith("L − O")
    assert diff.brier.mean == pytest.approx(0.07)
    loop_edge = [x for x in s.edge if x.label.startswith("L − price")]
    assert [x.label for x in loop_edge] == ["L − price (loop sample, n = 2)"]
    assert loop_edge[0].brier.mean == pytest.approx((0.09 - 0.09 + 0.25 - 0.36) / 2)


def test_without_a_loop_arm_there_are_no_loop_rows(
    tmp_path: Path, arms: dict[str, ArmSource]
) -> None:
    s = _score(tmp_path, {a: v for a, v in arms.items() if a != "L"})
    assert (s.loop_n, s.loop_arms, s.loop_diffs) == (0, [], [])
    assert not any(x.label.startswith("L") for x in s.edge)


def test_judge_row_counts_by_score_and_restricts_to_the_sample(tmp_path: Path) -> None:
    path = tmp_path / "o.sqlite"
    _arm(path, RunMode.BACKTEST, [(Q0, 0.5)], tournament=TOURNAMENT + "+x")
    _judge(
        path,
        [(Q0, "a", 0), (Q0, "b", 3), (Q1, "c", 4), (Q1, "c", 1), (Q2_, "d", 2), (Q3, "e", 4)],
    )
    row = judge_row("O", path, TOURNAMENT, {Q0, Q1, Q2_})
    assert row is not None
    assert row.questions == 3
    assert row.sources == 4  # Q3's source is outside the sample
    # a=0, b=3, c=1 (the later judgment wins), d=2.
    assert row.by_score == (1, 1, 1, 1, 0)
    assert row.major_sources == 1
    assert row.major_questions == 1
    assert judge_row("O", path, TOURNAMENT, {Q4}) is None
    assert judge_row("O", path, "forecastbench:other", {Q0}) is None


def test_major_questions_are_distinct_questions_with_a_score_of_three_or_more(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.sqlite"
    _arm(path, RunMode.UNRESTRICTED, [(Q0, 0.5)], tournament=TOURNAMENT + "+x")
    _judge(path, [(Q0, "a", 3), (Q0, "b", 4), (Q1, "c", 2), (Q3, "d", 4)])
    assert major_questions(path, TOURNAMENT) == {Q0, Q3}


def _labels(path: Path, items: list[tuple[str, int, str, int]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "labels": [
                    {"arm": a, "question_id": q, "source_id": s, "score": v} for a, q, s, v in items
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_reference_agreement_pairs_by_arm_question_and_source(tmp_path: Path) -> None:
    o, t = tmp_path / "o.sqlite", tmp_path / "t.sqlite"
    _arm(o, RunMode.BACKTEST, [(Q0, 0.5)], tournament=TOURNAMENT + "+o")
    _arm(t, RunMode.UNRESTRICTED, [(Q0, 0.5)], tournament=TOURNAMENT + "+t")
    so = _judge(o, [(Q0, "a", 0), (Q0, "b", 3), (Q1, "c", 2), (Q1, "c", 1)], TOURNAMENT + "+o")
    st = _judge(t, [(Q1, "x", 4)], TOURNAMENT + "+t")
    labels = _labels(
        tmp_path / "labels.json",
        [
            ("O", Q0, so["a"], 0),  # judged 0: agrees
            ("O", Q0, so["b"], 4),  # judged 3: both major
            ("O", Q1, so["c"], 1),  # the later judgment (1) is the one paired
            ("T", Q1, st["x"], 0),  # judged 4: disagrees, in a different arm's ledger
            ("O", Q1, st["x"], 4),  # no judgment of this source in O: ignored
            ("O", Q2_, so["a"], 4),  # right source, wrong question: ignored
            ("L", Q0, so["a"], 4),  # an arm with no ledger: ignored
        ],
    )
    arms = {"O": (o, TOURNAMENT + "+o"), "T": (t, TOURNAMENT + "+t")}
    a = reference_agreement(labels, arms)
    assert a is not None
    assert a.n == 4
    # Pairs (hand, judge): (0,0), (4,3), (1,1), (0,4).
    assert a.exact_merged == pytest.approx(2 / 4)  # (4, 3) and (0, 4) differ
    assert (a.human_major, a.judge_major) == (1, 2)
    assert a.major_agree == pytest.approx(3 / 4)


def test_reference_agreement_is_none_without_a_file_or_a_matching_judgment(tmp_path: Path) -> None:
    o = tmp_path / "o.sqlite"
    _arm(o, RunMode.BACKTEST, [(Q0, 0.5)], tournament=TOURNAMENT + "+o")
    arms = {"O": (o, TOURNAMENT + "+o")}
    assert reference_agreement(tmp_path / "missing.json", arms) is None
    assert reference_agreement(_labels(tmp_path / "l.json", [("O", Q0, "nope", 2)]), arms) is None


def test_score_q2_reports_the_judge_the_flagged_questions_and_the_agreement(
    tmp_path: Path, arms: dict[str, ArmSource]
) -> None:
    o = _judge(arms["O"].path, [(Q0, "a", 0), (Q1, "b", 1)], arms["O"].tournament)
    t = _judge(arms["T"].path, [(Q1, "c", 4), (Q4, "d", 4), (Q2_, "e", 0)], arms["T"].tournament)
    labels = _labels(
        tmp_path / "labels.json", [("O", Q0, o["a"], 0), ("T", Q1, t["c"], 4), ("N", Q0, "z", 4)]
    )
    s = _score(tmp_path, arms, labels=labels)
    assert [(j.arm, j.questions, j.sources) for j in s.judges] == [("O", 2, 2), ("T", 2, 2)]
    # T's Q4 is outside the 3 compared questions: it is not flagged and not counted.
    assert s.flagged_n == 1
    assert [r.arm for r in s.flagged] == ["N", "O", "T"]
    assert [r.brier.mean for r in s.flagged] == pytest.approx([0.25, 0.04, 0.01])
    assert s.agree is not None
    assert s.agree.n == 2


def test_cost_rows_count_successful_calls_per_question_forecast(
    tmp_path: Path, arms: dict[str, ArmSource]
) -> None:
    _costs(arms["O"].path)
    s = _score(tmp_path, arms)
    [row] = s.costs
    assert row.arm == "O"
    assert row.questions == 3  # Q3 is outside the compared questions
    assert row.calls == pytest.approx(4 / 3)  # the errored call and Q3's are not counted
    assert row.research_calls == pytest.approx(1 / 3)
    assert row.output_tokens == pytest.approx(40 / 3)
    assert row.seconds == pytest.approx(20.0)
