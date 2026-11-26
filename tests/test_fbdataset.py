"""M15's no-model arms, the extract's cut and the registered predictions, offline."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from augur import fbdataset
from augur.fbdataset import (
    DatasetError,
    DQuestion,
    History,
    Known,
    SeriesShare,
    arm_b,
    arm_c,
    arm_d,
    forecast_all,
    load,
    parse_closes,
    share_up,
)
from augur.fbdataset_eval import hypotheses

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "extract_dataset_questions.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("augur_extract_dataset_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


extract = _load_script()


def dq(
    source: str = "fred",
    id: str = "X",
    resolutions: tuple[tuple[date, int], ...] = (),
    freeze: datetime = datetime(2026, 6, 25, tzinfo=UTC),
    question_set: str = "2026-07-05",
) -> DQuestion:
    return DQuestion(question_set, source, id, freeze, None, "", resolutions)


# -- the walk-forward history -------------------------------------------------------------


def test_history_lag_boundary() -> None:
    freeze = datetime(2026, 6, 25, 15, 30, tzinfo=UTC)
    on_edge = freeze.date() - timedelta(days=14)
    just_inside = freeze.date() - timedelta(days=13)
    history = History([dq(resolutions=((on_edge, 1), (just_inside, 1)))])
    known = history.at(freeze)
    assert known.by_source == {"fred": (1, 1)}
    assert known.by_series == {"fred:X": (1, 1)}
    assert known.latest == on_edge
    assert History.known_on(freeze.date()) == on_edge


def test_history_only_just_inside_is_not_counted() -> None:
    freeze = datetime(2026, 6, 25, tzinfo=UTC)
    history = History([dq(resolutions=((date(2026, 6, 12), 1),))])
    known = history.at(freeze)
    assert known.by_source == {}
    assert known.latest is None


def test_arm_b_needs_twenty_known_outcomes() -> None:
    nineteen = Known({"fred": (19, 19)}, {}, None)
    twenty = Known({"fred": (15, 20)}, {}, None)
    assert arm_b(nineteen, "fred") == 0.5
    assert arm_b(twenty, "fred") == pytest.approx(16 / 22)
    assert arm_b(twenty, "acled") == 0.5


def test_arm_c_shrinks_toward_b() -> None:
    known = Known({}, {"fred:X": (3, 5)}, None)
    assert arm_c(known, "fred:Y", 0.6) == pytest.approx(0.6)
    assert arm_c(known, "fred:X", 0.6) == pytest.approx((3 + 4 * 0.6) / (5 + 4))


# -- arm D -------------------------------------------------------------------------------


FREEZE_DAY = date(2026, 7, 1)


def _daily(n: int, value: Any) -> list[tuple[date, float]]:
    start = date(2025, 1, 1)
    return [(start + timedelta(days=i), float(value(i))) for i in range(n)]


def test_share_up_increasing_series() -> None:
    share = share_up(
        _daily(200, lambda i: i), date(2026, 7, 5), date(2026, 7, 12), freeze=FREEZE_DAY
    )
    assert share == SeriesShare(1.0, 7, 193)


def test_share_up_flat_series_is_not_higher() -> None:
    share = share_up(
        _daily(200, lambda i: 5.0), date(2026, 7, 5), date(2026, 7, 12), freeze=FREEZE_DAY
    )
    assert share is not None
    assert share.p_up == 0.0
    assert share.h == 7


def test_share_up_needs_one_hundred_changes() -> None:
    due, res = date(2026, 7, 5), date(2026, 7, 12)
    assert share_up(_daily(106, lambda i: i), due, res, freeze=FREEZE_DAY) is None  # 99 changes
    got = share_up(_daily(107, lambda i: i), due, res, freeze=FREEZE_DAY)  # 100 changes
    assert got is not None
    assert got.changes == 100
    assert share_up([], due, res, freeze=FREEZE_DAY) is None


def test_share_up_horizon_follows_observation_rate() -> None:
    weekly = [(date(2024, 1, 1) + timedelta(days=7 * i), float(i)) for i in range(110)]
    # one observation every 7 days: a 14-day horizon is 2 observations
    got = share_up(weekly, date(2026, 7, 5), date(2026, 7, 19), freeze=date(2026, 2, 16))
    assert got is not None
    assert got.h == 2
    # a very short horizon never drops below one observation
    short = share_up(
        _daily(200, lambda i: i), date(2026, 7, 5), date(2026, 7, 5), freeze=FREEZE_DAY
    )
    assert short is not None
    assert short.h == 1


def test_arm_d_blend_and_fallback() -> None:
    assert arm_d(None, 0.42) == 0.42
    assert arm_d(SeriesShare(1.0, 7, 193), 0.5) == pytest.approx(0.75)
    assert arm_d(SeriesShare(0.2, 7, 193), 0.6) == pytest.approx(0.5 * 0.2 + 0.5 * 0.6)


def test_parse_closes() -> None:
    excerpt = "AAPL daily closes, as of today\n2026-01-01 10.5\n2026-01-02 1e1\nnoise\n"
    assert parse_closes(excerpt) == [(date(2026, 1, 1), 10.5), (date(2026, 1, 2), 10.0)]


@pytest.mark.parametrize("excerpt", ["2026-01-02 1\n2026-01-01 2", "2026-01-01 1\n2026-01-01 2"])
def test_parse_closes_refuses_disorder(excerpt: str) -> None:
    with pytest.raises(DatasetError):
        parse_closes(excerpt)


# -- load and forecast_all ----------------------------------------------------------------


def _doc(due: str, role: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "forecast_due_date": due,
        "question_set": f"{due}-llm.json",
        "role": role,
        "dataset_questions": len(questions),
        "removed": {"a question with no resolved date": 1} if role == "corpus" else {},
        "provenance": {
            "questions": {"sha256": "q" + due},
            "resolutions": {"sha256": "r" + due},
            "fetched": "2026-11-26",
        },
        "questions": questions,
    }


def _raw(
    source: str, id: str, freeze: str, resolutions: dict[str, int], value: float | None = None
) -> dict[str, Any]:
    return {
        "source": source,
        "id": id,
        "freeze_datetime": freeze,
        "freeze_value": value,
        "url": "",
        "resolutions": resolutions,
    }


def _write_root(root: Path) -> None:
    directory = root / fbdataset.DATASET_DIR
    directory.mkdir(parents=True)
    # 20 yes outcomes known (D + 14 <= 2026-06-25), then 5 no outcomes that are not yet
    yes = {(date(2026, 5, 23) + timedelta(days=i)).isoformat(): 1 for i in range(20)}
    no = {(date(2026, 6, 12) + timedelta(days=i)).isoformat(): 0 for i in range(5)}
    history = _doc(
        "2026-01-04",
        "history",
        [_raw("fred", "X", "2026-01-01T00:00:00+00:00", {**yes, **no})],
    )
    (directory / "2026-01-04.json").write_text(json.dumps(history))
    for s in fbdataset.CORPUS_SETS:
        questions: list[dict[str, Any]] = []
        if s == "2026-07-05":
            questions = [
                _raw("fred", "X", "2026-06-25T00:00:00+00:00", {"2026-10-03": 1}),
                _raw("yfinance", "T", "2026-06-25T00:00:00+00:00", {"2026-07-12": 1}),
            ]
        elif s == "2026-07-19":
            questions = [_raw("fred", "X", "2026-06-29T00:00:00+00:00", {"2026-10-17": 0})]
        (directory / f"{s}.json").write_text(json.dumps(_doc(s, "corpus", questions)))


def test_load_and_forecast_all_honours_the_lag(tmp_path: Path) -> None:
    _write_root(tmp_path)
    loaded = load(tmp_path)
    assert len(loaded.corpus) == 3
    assert [i.key[0] for i in loaded.corpus] == ["2026-07-05", "2026-07-05", "2026-07-19"]
    assert loaded.dataset_questions == 3
    assert loaded.removed == {"a question with no resolved date": 7}
    assert loaded.provenance["2026-07-05"]["fetched"] == "2026-11-26"
    yf = fbdataset.yfinance_questions(loaded)
    assert len(yf) == 1
    closes = {yf[0].question_id: _daily(200, lambda i: i)}
    by_key = {f.instance.key: f for f in forecast_all(loaded, closes)}

    first = by_key[("2026-07-05", "fred", "X", "2026-10-03")]
    assert first.source_n == 20
    assert first.b == pytest.approx(21 / 22)  # the five later "no" outcomes are not known yet
    assert first.latest_known == date(2026, 6, 11)
    assert first.c == pytest.approx((20 + 4 * first.b) / (20 + 4))
    assert first.d is None
    assert first.a == 0.5

    later = by_key[("2026-07-19", "fred", "X", "2026-10-17")]
    assert later.source_n == 24  # freeze 06-29 knows outcomes to 06-15: 20 yes, 4 no
    assert later.b == pytest.approx(21 / 26)
    assert later.latest_known == date(2026, 6, 15)

    stock = by_key[("2026-07-05", "yfinance", "T", "2026-07-12")]
    assert stock.share == SeriesShare(1.0, 7, 193)
    assert stock.b == 0.5
    assert stock.d == pytest.approx(0.75)
    assert stock.get("D") == stock.d
    assert first.get("B") == first.b
    with pytest.raises(DatasetError):
        first.get("D")

    # without closes, arm D is B exactly
    bare = {f.instance.key: f for f in forecast_all(loaded, {})}
    assert bare[("2026-07-05", "yfinance", "T", "2026-07-12")].d == 0.5


def test_load_refuses_missing_corpus_set(tmp_path: Path) -> None:
    _write_root(tmp_path)
    (tmp_path / fbdataset.DATASET_DIR / "2026-09-27.json").unlink()
    with pytest.raises(DatasetError, match="2026-09-27"):
        load(tmp_path)


# -- the extract's cut --------------------------------------------------------------------


def _cut_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    questions = {
        "question_set": "2026-07-05-llm.json",
        "forecast_due_date": "2026-07-05",
        "questions": [
            {
                "source": "yfinance",
                "id": "AAPL",
                "freeze_datetime": "2026-06-25T00:00:00+00:00",
                "freeze_datetime_value": "187.5",
                "url": "https://finance.yahoo.com/AAPL",
                "resolution_dates": ["2026-07-12", "2026-08-04", "2026-09-01", "2026-12-01"],
            },
            {"source": "fred", "id": ["A", "B"], "resolution_dates": ["2026-07-12"]},
            {"source": "serpapi", "id": "s1", "resolution_dates": ["2026-07-12"]},
            {"source": "kalshi", "id": "k1", "resolution_dates": ["2026-07-12"]},
            {"source": "fred", "id": "Z", "resolution_dates": []},
        ],
    }
    resolutions = {
        "question_set": "2026-07-05-llm.json",
        "resolutions": [
            {
                "source": "yfinance",
                "id": "AAPL",
                "resolution_date": "2026-07-12",
                "resolved": True,
                "resolved_to": 1.0,
            },
            {
                "source": "yfinance",
                "id": "AAPL",
                "resolution_date": "2026-08-04T00:00:00",
                "resolved": True,
                "resolved_to": 0,
            },
            {
                "source": "yfinance",
                "id": "AAPL",
                "resolution_date": "2026-09-01",
                "resolved": False,
                "resolved_to": 0,
            },
            {
                "source": "fred",
                "id": "ghost",
                "resolution_date": "2026-07-12",
                "resolved": True,
                "resolved_to": 1,
            },
            {
                "source": "kalshi",
                "id": "k1",
                "resolution_date": "2026-07-12",
                "resolved": True,
                "resolved_to": 1,
            },
        ],
    }
    return questions, resolutions


def test_cut_counts_every_reason() -> None:
    questions, resolutions = _cut_fixture()
    doc = extract.cut(questions, resolutions, "2026-11-26")
    assert doc["dataset_questions"] == 4  # kalshi is not counted
    assert doc["forecast_due_date"] == "2026-07-05"
    assert doc["questions"] == [
        {
            "source": "yfinance",
            "id": "AAPL",
            "freeze_datetime": "2026-06-25T00:00:00+00:00",
            "freeze_value": 187.5,
            "url": "https://finance.yahoo.com/AAPL",
            "resolutions": {"2026-07-12": 1, "2026-08-04": 0},
        }
    ]
    assert doc["removed"] == {
        "a dated row not resolved to 0 or 1": 1,
        "a listed resolution date after the fetch day": 1,
        "a listed past resolution date with no row resolved to 0 or 1": 1,
        "a pair question (a list of ids)": 1,
        "another dataset source (serpapi)": 1,
        "a question with no resolved date": 1,
        "a resolution row whose question is not in the question set": 1,
    }


def test_cut_refuses_mismatched_sets() -> None:
    questions, resolutions = _cut_fixture()
    resolutions["question_set"] = "2026-07-19-llm.json"
    with pytest.raises(extract.ExtractError):
        extract.cut(questions, resolutions, "2026-11-26")


def test_cut_refuses_a_date_resolved_twice() -> None:
    questions, resolutions = _cut_fixture()
    resolutions["resolutions"].append(copy.deepcopy(resolutions["resolutions"][0]))
    with pytest.raises(extract.ExtractError):
        extract.cut(questions, resolutions, "2026-11-26")


# -- the registered predictions --------------------------------------------------------------


def _comp(
    group: str,
    first: str,
    second: str,
    mean: float,
    s_low: float,
    s_high: float,
    d_low: float,
    d_high: float,
) -> dict[str, Any]:
    return {
        "group": group,
        "first": first,
        "second": second,
        "n": 1000,
        "mean": mean,
        "by_series": {"low": s_low, "high": s_high, "clusters": 300},
        "by_date": {"low": d_low, "high": d_high, "clusters": 40},
    }


def _holding() -> list[dict[str, Any]]:
    return [
        _comp("pooled", "B", "A", -0.05, -0.09, -0.01, -0.15, 0.05),
        _comp("pooled", "B", "Naive", -0.03, -0.05, -0.01, -0.06, 0.0),
        _comp("pooled", "C", "B", -0.02, -0.03, -0.005, -0.04, 0.0),
        _comp("yfinance", "D", "B", 0.01, -0.01, 0.03, -0.02, 0.04),
        _comp("model-4", "model-4", "B", -0.03, -0.05, -0.01, -0.06, 0.0),
        _comp("hybrid ensemble 1", "hybrid ensemble 1", "B", -0.03, -0.05, -0.01, -0.06, 0.0),
        _comp("stat-v1", "B", "stat-v1", 0.02, 0.0, 0.02, -0.01, 0.05),
    ]


def _verdicts(cs: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    out, follow = hypotheses(cs)
    return {h["id"]: h["verdict"] for h in out}, follow


def test_hypotheses_all_hold() -> None:
    verdicts, follow = _verdicts(_holding())
    assert verdicts == dict.fromkeys(["P52", "P53", "P54", "P55", "P56", "P57"], "held")
    assert follow["opens"] is True
    assert follow["p52_and_p53"] is True
    assert follow["b_no_worse_than_stat_v1"] is False


def _flip(name: str) -> list[dict[str, Any]]:
    cs = copy.deepcopy(_holding())
    if name == "P52":
        cs[0]["mean"] = -0.02
    elif name == "P52-interval":
        cs[0]["by_series"]["high"] = 0.0
    elif name == "P53":
        cs[1]["mean"] = -0.01
    elif name == "P54":
        cs[0]["by_date"] = {"low": -0.09, "high": 0.0, "clusters": 40}  # 0.09 < 2 * 0.08
    elif name == "P55":
        cs[3]["mean"] = -0.01
    elif name == "P56":
        cs[5]["mean"] = -0.01
    elif name == "P57":
        cs[2]["by_series"]["high"] = 0.0
    return cs


@pytest.mark.parametrize(
    ("flip", "pid"),
    [
        ("P52", "P52"),
        ("P52-interval", "P52"),
        ("P53", "P53"),
        ("P54", "P54"),
        ("P55", "P55"),
        ("P56", "P56"),
        ("P57", "P57"),
    ],
)
def test_hypotheses_each_fails_alone(flip: str, pid: str) -> None:
    verdicts, _ = _verdicts(_flip(flip))
    assert verdicts[pid] == "failed"
    assert [k for k, v in verdicts.items() if v == "failed"] == [pid]


def test_hypotheses_boundaries() -> None:
    cs = _holding()
    cs[0]["mean"] = -0.03  # P52 holds at exactly -0.03
    cs[3]["mean"] = -0.005  # -(D - B) = 0.005 is not below 0.005
    verdicts, _ = _verdicts(cs)
    assert verdicts["P52"] == "held"
    assert verdicts["P55"] == "failed"
    cs = _holding()
    cs[0]["by_date"] = {"low": -0.10, "high": 0.06, "clusters": 40}  # width 0.16 = 2 * 0.08
    assert _verdicts(cs)[0]["P54"] == "held"


def test_follow_up_rule() -> None:
    # P52 fails and stat-v1 is not matched: closed
    cs = _flip("P52")
    _, follow = _verdicts(cs)
    assert follow["opens"] is False
    # P53 fails likewise
    assert _verdicts(_flip("P53"))[1]["opens"] is False
    # B no worse than stat-v1 opens it whatever P52 and P53 did
    cs[6]["by_series"]["high"] = 0.009
    _, follow = _verdicts(cs)
    assert follow["opens"] is True
    assert follow["p52_and_p53"] is False
    assert follow["b_no_worse_than_stat_v1"] is True
    cs[6]["by_series"]["high"] = 0.01  # strictly below 0.01 is required
    assert _verdicts(cs)[1]["opens"] is False


def test_a_mean_that_rounds_to_the_threshold_is_not_at_the_threshold() -> None:
    # The verdict reads the unrounded mean: -0.029989 is not "at most -0.03",
    # though it prints as -0.0300 to four places.
    cs = _holding()
    cs[0]["mean"] = -0.029989229767720307
    assert _verdicts(cs)[0]["P52"] == "failed"


def test_share_up_window_is_the_730_days_before_the_freeze() -> None:
    # Closes every day up to the day before the freeze: the one 730 days before
    # the freeze is in the window and the one 731 days before is not.
    freeze = date(2026, 7, 1)
    closes = [(freeze - timedelta(days=k), float(1000 - k)) for k in range(800, 0, -1)]
    got = share_up(closes, date(2026, 7, 5), date(2026, 7, 6), freeze=freeze)
    assert got is not None
    assert got.changes == 730 - got.h
