"""The ForecastBench loader: what it keeps, what it removes, and what the model may see."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from augur.forecastbench import FBQuestion, base_rate, load_set, subset_files

from .fakes.forecastbench import (
    QUESTION_SET,
    RELEASED,
    raw_question,
    raw_resolution,
    write_pair,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "forecastbench"


def _mixed(tmp_path: Path) -> tuple[Path, Path]:
    questions = [
        raw_question("keep-1", source="manifold"),
        raw_question("keep-2", source="polymarket"),
        raw_question("keep-3", source="infer"),
        raw_question("keep-4", source="metaculus"),
        raw_question("fred-1", source="fred"),  # a dataset question
        raw_question("acled-1", source="acled"),
        raw_question("open-1"),  # not resolved
        raw_question("frac-1"),  # resolved to a price, not an outcome
        raw_question("noreso-1"),  # absent from the resolution set
        raw_question("early-1", freeze="2024-09-19T00:00:00+00:00"),  # the release day
        raw_question("early-2", freeze="2024-09-18T12:00:00+00:00"),  # before it
        raw_question("nofreeze-1", freeze=None),
    ]
    resolutions = [
        raw_resolution("keep-1", 1.0),
        raw_resolution("keep-2", 0, source="polymarket"),
        raw_resolution("keep-3", 0.0, source="infer"),
        raw_resolution("keep-4", 1, source="metaculus"),
        raw_resolution("fred-1", 1.0, source="fred"),
        raw_resolution("acled-1", 1.0, source="acled"),
        raw_resolution("open-1", 0.0, resolved=False),
        raw_resolution("frac-1", 0.37),
        raw_resolution("early-1", 1.0),
        raw_resolution("early-2", 1.0),
        raw_resolution("nofreeze-1", 1.0),
    ]
    return write_pair(tmp_path, questions, resolutions)


def test_load_set_counts_each_removal_by_reason(tmp_path: Path) -> None:
    qs, rs = _mixed(tmp_path)
    report = load_set(qs, rs, released=RELEASED)
    assert report.total == 12
    assert report.question_set == QUESTION_SET
    assert report.forecast_due_date == "2025-03-02"
    assert [q.fb_id for q in report.kept] == ["keep-1", "keep-2", "keep-3", "keep-4"]
    assert sorted(report.removed.values()) == [1, 2, 2, 3]
    by_reason = {k.split()[0] + " " + k.split()[1]: n for k, n in report.removed.items()}
    assert by_reason["not a"] == 2
    assert by_reason["not resolved"] == 3  # open, fractional, no resolution record
    assert by_reason["posed on"] == 2
    assert by_reason["no freeze"] == 1
    assert len(report.kept) + sum(report.removed.values()) == report.total
    lines = report.lines()
    assert lines[0] == f"{QUESTION_SET}: 12 questions"
    assert lines[-1].endswith("kept       4")


def test_kept_questions_carry_outcomes_and_market_fields(tmp_path: Path) -> None:
    qs, rs = _mixed(tmp_path)
    kept = {q.fb_id: q for q in load_set(qs, rs, released=RELEASED).kept}
    assert [kept[k].outcome for k in ("keep-1", "keep-2", "keep-3", "keep-4")] == [1, 0, 0, 1]
    first = kept["keep-1"]
    assert first.freeze_datetime == datetime(2025, 3, 1, tzinfo=UTC)
    assert first.freeze_value == pytest.approx(0.7123)
    assert first.market_close == datetime(2025, 6, 1, tzinfo=UTC)
    assert first.resolution_date == date(2025, 4, 1)
    assert first.background == "Some background."


def test_a_question_a_day_after_the_release_is_kept(tmp_path: Path) -> None:
    qs, rs = write_pair(
        tmp_path,
        [raw_question("late", freeze="2024-09-20T00:00:00+00:00")],
        [raw_resolution("late", 1.0)],
    )
    assert len(load_set(qs, rs, released=RELEASED).kept) == 1


def test_limit_takes_the_first_questions_and_counts_the_rest(tmp_path: Path) -> None:
    qs, rs = _mixed(tmp_path)
    report = load_set(qs, rs, released=RELEASED, limit=2)
    assert [q.fb_id for q in report.kept] == ["keep-1", "keep-2"]
    assert report.removed["over the limit"] == 2
    assert load_set(qs, rs, released=RELEASED, limit=4).kept[-1].fb_id == "keep-4"
    assert "over the limit" not in load_set(qs, rs, released=RELEASED, limit=10).removed
    assert load_set(qs, rs, released=RELEASED, limit=0).kept == []


def test_a_mismatched_pair_is_refused(tmp_path: Path) -> None:
    qs, rs = _mixed(tmp_path)
    other = json.loads(rs.read_text(encoding="utf-8"))
    other["question_set"] = "2025-02-16-llm.json"
    rs.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="pass the matching pair"):
        load_set(qs, rs, released=RELEASED)


def test_missing_and_na_text_fields_become_empty_or_defaults(tmp_path: Path) -> None:
    qs, rs = write_pair(
        tmp_path,
        [
            raw_question(
                "x",
                background="N/A",
                url="",
                value="N/A",
                market_info_close_datetime=None,
                market_info_open_datetime="garbage",
            )
        ],
        [raw_resolution("x", 1.0)],
    )
    [q] = load_set(qs, rs, released=RELEASED).kept
    assert q.background == ""
    assert q.url == "https://www.forecastbench.org"
    assert q.freeze_value is None
    assert q.market_close is None
    assert q.market_open is None


def test_a_price_outside_the_unit_interval_is_not_kept_as_a_price(tmp_path: Path) -> None:
    qs, rs = write_pair(tmp_path, [raw_question("x", value="1.5")], [raw_resolution("x", 1.0)])
    [q] = load_set(qs, rs, released=RELEASED).kept
    assert q.freeze_value is None


def test_question_id_is_stable_and_distinct_per_source(tmp_path: Path) -> None:
    qs, rs = write_pair(
        tmp_path,
        [raw_question("1", source="manifold"), raw_question("1", source="infer")],
        [raw_resolution("1", 1.0), raw_resolution("1", 0.0, source="infer")],
    )
    a, b = load_set(qs, rs, released=RELEASED).kept
    again = load_set(qs, rs, released=RELEASED).kept
    assert a.question_id != b.question_id
    assert [a.question_id, b.question_id] == [q.question_id for q in again]
    digest = hashlib.sha256(b"manifold:1").hexdigest()
    assert a.question_id == int(digest[:13], 16)
    assert 0 < a.question_id < 2**52


def test_to_question_is_a_binary_question_opened_at_the_freeze(tmp_path: Path) -> None:
    qs, rs = _mixed(tmp_path)
    fb = load_set(qs, rs, released=RELEASED).kept[0]
    q = fb.to_question()
    assert q.type == "binary"
    assert q.status == "open"
    assert q.open_time == fb.freeze_datetime == datetime(2025, 3, 1, tzinfo=UTC)
    assert q.question_id == q.post_id == fb.question_id
    assert q.title == "Will event keep-1 happen?"
    assert q.scheduled_close_time == datetime(2025, 6, 1, tzinfo=UTC)
    assert q.url == "https://example.org/manifold/keep-1"


def test_nothing_the_forecaster_must_not_see_reaches_the_question(tmp_path: Path) -> None:
    qs, rs = write_pair(
        tmp_path,
        [
            raw_question(
                "x",
                value="0.7123",
                freeze_datetime_value_explanation="SECRET-EXPLANATION 0.7123",
                market_info_resolution_criteria="SECRET-MARKET-CRITERIA",
            )
        ],
        [raw_resolution("x", 1.0)],
    )
    [fb] = load_set(qs, rs, released=RELEASED).kept
    q = fb.to_question()
    shown = "\n".join([q.title, q.description, q.resolution_criteria])
    assert "0.7123" not in shown
    assert "SECRET" not in shown
    dumped = q.model_dump_json()
    assert "0.7123" not in dumped
    assert "SECRET" not in dumped
    assert "Some background." in q.description
    assert "predict the outcome of a prediction market" in q.description
    assert q.description.index("predict") < q.description.index("Some background.")


@pytest.mark.parametrize(
    ("market_close", "expected"),
    [
        (datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 6, 1, tzinfo=UTC)),
        (datetime(2025, 2, 1, tzinfo=UTC), datetime(2025, 3, 1, tzinfo=UTC)),  # before the freeze
        (None, datetime(2025, 4, 1, tzinfo=UTC)),  # the resolution date, midnight UTC
    ],
)
def test_close_time_never_precedes_the_freeze(
    market_close: datetime | None, expected: datetime
) -> None:
    fb = FBQuestion(
        fb_id="x",
        source="manifold",
        question="Q?",
        resolution_criteria="",
        background="",
        source_intro="",
        url="https://example.org",
        freeze_datetime=datetime(2025, 3, 1, tzinfo=UTC),
        freeze_value=0.5,
        market_open=None,
        market_close=market_close,
        resolution_date=date(2025, 4, 1),
        outcome=1,
    )
    assert fb.to_question().scheduled_close_time == expected


def test_subset_files_keep_exactly_the_kept_questions_with_every_field(tmp_path: Path) -> None:
    qs, rs = _mixed(tmp_path)
    report = load_set(qs, rs, released=RELEASED)
    sub_q, sub_r = subset_files(report, qs, rs)
    full_q = json.loads(qs.read_text(encoding="utf-8"))
    full_r = json.loads(rs.read_text(encoding="utf-8"))
    kept_ids = ["keep-1", "keep-2", "keep-3", "keep-4"]
    assert [q["id"] for q in sub_q["questions"]] == kept_ids
    assert [r["id"] for r in sub_r["resolutions"]] == kept_ids
    for q in sub_q["questions"]:
        assert q in full_q["questions"]  # every field, unchanged
    for r in sub_r["resolutions"]:
        assert r in full_r["resolutions"]
    assert {k: v for k, v in sub_q.items() if k != "questions"} == {
        k: v for k, v in full_q.items() if k != "questions"
    }
    # The subset loads to the same questions.
    qs2, rs2 = tmp_path / "sub_q.json", tmp_path / "sub_r.json"
    qs2.write_text(json.dumps(sub_q), encoding="utf-8")
    rs2.write_text(json.dumps(sub_r), encoding="utf-8")
    assert load_set(qs2, rs2, released=RELEASED).kept == report.kept


def test_subset_matches_on_source_as_well_as_id(tmp_path: Path) -> None:
    qs, rs = write_pair(
        tmp_path,
        [raw_question("1", source="manifold"), raw_question("1", source="infer")],
        [raw_resolution("1", 1.0), raw_resolution("1", 0.0, source="infer", resolved=False)],
    )
    report = load_set(qs, rs, released=RELEASED)
    assert [q.source for q in report.kept] == ["manifold"]
    sub_q, sub_r = subset_files(report, qs, rs)
    assert [q["source"] for q in sub_q["questions"]] == ["manifold"]
    assert [r["source"] for r in sub_r["resolutions"]] == ["manifold"]


def test_base_rate() -> None:
    assert base_rate([_fb(1), _fb(0), _fb(0), _fb(0)]) == 0.25
    with pytest.raises(ValueError, match="no questions"):
        base_rate([])


def _fb(outcome: int) -> FBQuestion:
    return FBQuestion(
        fb_id=str(outcome),
        source="manifold",
        question="Q?",
        resolution_criteria="",
        background="",
        source_intro="",
        url="u",
        freeze_datetime=datetime(2025, 3, 1, tzinfo=UTC),
        freeze_value=None,
        market_open=None,
        market_close=None,
        resolution_date=date(2025, 4, 1),
        outcome=outcome,
    )


def test_the_committed_subset_loads() -> None:
    report = load_set(
        DATA / "2026-05-24-llm.subset.json",
        DATA / "2026-05-24_resolution_set.subset.json",
        released=date(2024, 9, 19),
    )
    assert len(report.kept) == 139
    assert report.total == 139
    assert report.removed == {}
    assert all(q.freeze_datetime.date() > date(2024, 9, 19) for q in report.kept)
    assert {q.freeze_datetime for q in report.kept} == {datetime(2026, 5, 14, tzinfo=UTC)}
    assert len({q.question_id for q in report.kept}) == 139
    assert all(q.to_question().type == "binary" for q in report.kept)
