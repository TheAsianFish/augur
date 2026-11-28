"""M18's scoring: the reading rule, the interval a verdict reads, and the committed rows."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from augur import fbseason_eval
from augur.fbseason import SeasonError

REPO = Path(__file__).resolve().parent.parent


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_m18_{name}", REPO / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _strata(**published: int) -> list[dict[str, Any]]:
    brier = {"A": 0.25, "B": 0.40, "S": 0.10, "S′": 0.20, "K": 0.11}
    out = []
    for g in ("all", "90d", "180d", "365d"):
        n = published.get(g.replace("d", ""), published.get(g, 0))
        out.append(
            {
                "stratum": g,
                "published": n,
                "brier": dict(brier) if n else {},
                "brier_exact": dict(brier) if n else {},
            }
        )
    return out


def _diff(
    group: str, first: str, second: str, mean: float, high: float, clusters: int = 7
) -> dict[str, Any]:
    return {
        "group": group,
        "first": first,
        "second": second,
        "n": 50,
        "mean": mean,
        "low": mean - 0.05,
        "high": high,
        "clusters": clusters,
        "question_level": {"low": mean - 0.01, "high": mean + 0.01, "n": 50},
    }


def _diffs(sk90: float = 0.0, sk180: float = 0.01) -> list[dict[str, Any]]:
    return [
        _diff("all", "S", "B", -0.30, -0.10),
        _diff("90d", "S", "K", sk90, sk90 + 0.02, 3),
        _diff("180d", "S", "K", sk180, sk180 + 0.02, 3),
    ]


NO_GATE: dict[str, Any] = {"scored": 0, "planned": 248, "comparisons": []}


def test_every_hypothesis_waits_for_the_reading_day() -> None:
    strata = _strata(all=313, **{"90": 121, "180": 143, "365": 49})
    for as_of in (None, "2026-12-30", "2027-01-10"):
        got = fbseason_eval.hypotheses(strata, _diffs(), NO_GATE, as_of)
        assert [h["verdict"] for h in got] == ["not yet readable"] * 4


def test_on_the_reading_day_each_is_held_or_failed() -> None:
    strata = _strata(all=313, **{"90": 121, "180": 143, "365": 49})
    strata[3]["brier"] = {"A": 0.25, "B": 0.3, "S": 0.26, "S′": 0.24, "K": 0.5}
    strata[3]["brier_exact"] = dict(strata[3]["brier"])
    gate = {
        "scored": 200,
        "planned": 248,
        "comparisons": [_diff("gate", "B+S", "stat-v1", -0.01, 0.009)],
    }
    got = {
        h["id"]: h["verdict"]
        for h in fbseason_eval.hypotheses(strata, _diffs(), gate, "2027-01-11")
    }
    assert got == {"P62": "held", "P63": "held", "P64": "held", "P65": "held"}
    bad = fbseason_eval.hypotheses(strata, _diffs(sk90=0.021), gate, "2027-01-11")
    assert bad[1]["verdict"] == "failed"


def test_a_stratum_with_nothing_published_stays_unreadable() -> None:
    strata = _strata(all=121, **{"90": 121})
    got = fbseason_eval.hypotheses(strata, _diffs(), NO_GATE, "2027-01-11")
    assert [h["verdict"] for h in got] == [
        "held",
        "not yet readable",
        "not yet readable",
        "not yet readable",
    ]


def test_one_cluster_reads_the_question_level_interval() -> None:
    one = _diff("365d", "S", "K", -0.05, 0.10, clusters=1)
    assert fbseason_eval._ci(one) == pytest.approx((-0.06, -0.04))
    seven = _diff("all", "S", "K", -0.05, 0.10, clusters=7)
    assert fbseason_eval._ci(seven) == pytest.approx((-0.10, 0.10))


def test_horizons() -> None:
    assert fbseason_eval.horizon_of(date(2026, 9, 13), date(2026, 9, 20)) == 7
    assert fbseason_eval.horizon_of(date(2026, 1, 4), date(2027, 1, 4)) == 365
    assert fbseason_eval.horizon_of(date(2026, 6, 7), date(2026, 12, 4)) == 180
    with pytest.raises(SeasonError):
        fbseason_eval.horizon_of(date(2026, 1, 4), date(2026, 2, 20))


def test_later_outcomes_read_only_later_recuts(tmp_path: Path) -> None:
    d = tmp_path / fbseason_eval.LATER_DIR
    d.mkdir(parents=True)
    doc = {
        "role": "later",
        "forecast_due_date": "2026-09-13",
        "provenance": {"fetched": "2027-01-11"},
        "questions": [{"source": "dbnomics", "id": "x", "resolutions": {"2026-12-12": 1}}],
    }
    (d / "2026-09-13.json").write_text(json.dumps(doc), encoding="utf-8")
    got, as_of = fbseason_eval.later_outcomes(tmp_path)
    assert got == {("2026-09-13", "dbnomics", "x", "2026-12-12"): 1}
    assert as_of == "2027-01-11"
    doc["role"] = "corpus"
    (d / "2026-09-13.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SeasonError):
        fbseason_eval.later_outcomes(tmp_path)
    assert fbseason_eval.later_outcomes(tmp_path / "nowhere") == ({}, None)


def test_the_manifest_script_splits_on_the_extract(tmp_path: Path) -> None:
    m = _script("m18_confirmatory")
    raw, ext = tmp_path / "raw", tmp_path / "ext"
    raw.mkdir()
    ext.mkdir()
    qs = {
        "forecast_due_date": "2026-09-13",
        "questions": [
            {
                "source": "dbnomics",
                "id": f"meteofrance_TEMPERATURE_celsius.{n}.D",
                "resolution_dates": ["2026-09-20", "2026-12-12", "2027-03-12"],
            }
            for n in ("07005", "07015")
        ]
        + [{"source": "yfinance", "id": "NVDA", "resolution_dates": ["2026-12-12"]}],
    }
    (raw / "2026-09-13-llm.json").write_text(json.dumps(qs), encoding="utf-8")
    held = {"questions": [{"id": "meteofrance_TEMPERATURE_celsius.07005.D"}]}
    (ext / "2026-09-13.json").write_text(json.dumps(held), encoding="utf-8")
    doc = m.build(raw, ext, "2026-11-27")
    assert [(i["id"][-7:], i["resolution_date"], i["horizon"]) for i in doc["instances"]] == [
        ("07005.D", "2026-12-12", 90)
    ]
    assert len(doc["left_out"]) == 1


def test_the_extract_counts_two_ticker_questions_out() -> None:
    ex = _script("extract_dataset_questions")
    questions = {
        "question_set": "2026-10-11-llm.json",
        "forecast_due_date": "2026-10-11",
        "questions": [
            {
                "source": "yfinance",
                "id": "WDC_PEP",
                "question": "Will WDC have a higher rate of return than PEP between ...",
                "resolution_dates": ["2026-10-18"],
                "freeze_datetime": "2026-10-09T00:00:00+00:00",
            }
        ],
    }
    resolutions = {
        "question_set": "2026-10-11-llm.json",
        "resolutions": [
            {
                "source": "yfinance",
                "id": "WDC_PEP",
                "resolution_date": "2026-10-18",
                "resolved": True,
                "resolved_to": 1,
            }
        ],
    }
    doc = ex.cut(questions, resolutions, "2026-11-27")
    assert doc["questions"] == []
    assert doc["removed"] == {"a two-ticker return question (the form from 2026-10-11)": 1}


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    out: dict[str, Any] = json.loads((REPO / fbseason_eval.SNAPSHOT).read_text(encoding="utf-8"))
    return out


def test_the_committed_rows_match_the_registration(committed: dict[str, Any]) -> None:
    rep = committed["replication"]
    assert rep["instances"] == 799
    assert {h["horizon"]: h["n"] for h in rep["by_horizon"]} == {
        "all": 799,
        7: 307,
        30: 304,
        90: 188,
    }
    conf = committed["confirmatory"]
    assert conf["planned"] == 313
    assert conf["left_out"] == 12
    assert {s["stratum"]: s["planned"] for s in conf["strata"]} == {
        "all": 313,
        "90d": 121,
        "180d": 143,
        "365d": 49,
    }
    assert len(conf["forecasts"]) == 313
    assert all(0.0 < f["S"] < 1.0 and f["K"] in (0.1, 0.9, 0.5) for f in conf["forecasts"])
    assert conf["gate"]["planned"] == 248
    assert committed["holdout"]["temperature_questions"] == 0


def test_verdicts_read_unrounded_brier_scores() -> None:
    """A Brier of 0.120049 displays as 0.1200 but misses P62's 0.12; a 0.00003 lead still counts."""
    strata = _strata(all=313, **{"90": 121, "180": 143, "365": 49})
    strata[0]["brier"]["S"] = 0.12
    strata[0]["brier_exact"]["S"] = 0.120049
    strata[3]["brier"] = {"A": 0.25, "B": 0.3, "S": 0.2503, "S′": 0.2503, "K": 0.5}
    strata[3]["brier_exact"] = {"A": 0.25, "B": 0.3, "S": 0.25034, "S′": 0.25031, "K": 0.5}
    got = {
        h["id"]: h["verdict"]
        for h in fbseason_eval.hypotheses(strata, _diffs(), NO_GATE, "2027-01-11")
    }
    assert got["P62"] == "failed"
    assert got["P64"] == "held"


def test_a_re_cut_after_the_reading_day_is_a_late_reading() -> None:
    assert fbseason_eval._kind(None) == "progress reading"
    assert fbseason_eval._kind("2027-01-10") == "progress reading"
    assert fbseason_eval._kind("2027-01-11") == "registered reading"
    assert fbseason_eval._kind("2027-01-12") == "late reading"
