"""M10's same-question comparison on hand-built forecast sets and arms."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from augur import cli, fbcompare, q4
from augur.fbcompare import (
    Corpus,
    FBCompareError,
    ForecastSet,
    Item,
    compare,
    group_summary,
    match,
    verdicts,
)

SET = "2026-06-21"
PROVENANCE = {
    "url": "https://example.org/sets.tar.gz",
    "bytes": 1,
    "sha256": "0" * 64,
    "last_modified": "Tue, 17 Nov 2026 08:27:21 GMT",
    "fetched": "2026-11-23",
    "licence": "CC BY-SA 4.0",
    "attribution": "ForecastBench, Forecasting Research Institute",
    "extracted_by": "scripts/extract_forecast_sets.py",
}


def _entrant(model: str, group: str, org: str = "ForecastBench") -> dict[str, Any]:
    return {
        "file": f"{model}.json",
        "organization": org,
        "model": model,
        "model_organization": org,
        "leaderboard_eligible": True,
        "group": group,
        "web_search": False,
    }


def _set(
    entrants: list[dict[str, Any]],
    rows: list[tuple[str, float, list[float | None], list[int]]],
    question_set: str = SET,
) -> ForecastSet:
    return ForecastSet.model_validate(
        {
            "question_set": question_set,
            "provenance": PROVENANCE,
            "entrants": entrants,
            "questions": [
                {
                    "source": "polymarket",
                    "id": qid,
                    "resolution_date": "2026-07-01",
                    "resolved": True,
                    "resolved_to": outcome,
                    "market_value_on_due_date": 0.5,
                    "forecasts": forecasts,
                    "imputed": imputed,
                }
                for qid, outcome, forecasts, imputed in rows
            ],
        }
    )


def _item(qid: str, outcome: int, p: float, question_set: str = SET) -> Item:
    return Item(question_set, "polymarket", qid, outcome, date(2026, 7, 1), p, 0.4)


ENTRANTS = [
    _entrant("Z (zero shot)", "zero-shot"),
    _entrant("Naive Forecaster", "baseline"),
    _entrant("bot", "external", org="Team"),
]
ROWS: list[tuple[str, float, list[float | None], list[int]]] = [
    # question, outcome, forecasts (Z, naive, bot), imputed entrant indices
    ("a", 1.0, [0.9, 0.6, 0.8], [1]),
    ("b", 0.0, [0.5, 0.4, 0.1], [0, 1]),
    ("c", 1.0, [0.7, 0.5, None], [1]),
]
ITEMS = [_item("a", 1, 0.5), _item("b", 0, 0.5), _item("c", 1, 0.5)]


def _series(m: fbcompare.Matched) -> dict[str, fbcompare.Series]:
    return {s.name: s for s in m.series}


def test_match_excludes_imputed_forecasts_except_baselines() -> None:
    m = match(Corpus("T", "test", "m", False, ITEMS), {SET: _set(ENTRANTS, ROWS)})
    s = _series(m)
    z = s["ForecastBench / Z (zero shot)"]
    assert z.forecasts == {0: 0.9, 2: 0.7}  # b is imputed, so Z is compared on two
    assert z.imputed == 1
    # The naive forecaster is imputed by construction: its flag is ignored.
    naive = s["ForecastBench / Naive Forecaster"]
    assert naive.forecasts == {0: 0.6, 1: 0.4, 2: 0.5}
    assert naive.imputed == 0
    bot = s["Team / bot"]
    assert bot.forecasts == {0: 0.8, 1: 0.1}
    assert bot.missing == 1
    assert s[fbcompare.MARKET_DUE].forecasts == {0: 0.5, 1: 0.5, 2: 0.5}
    assert s[fbcompare.MARKET_FREEZE].forecasts == {0: 0.4, 1: 0.4, 2: 0.4}
    assert m.outcome_disagreements == []


def test_match_uses_augur_outcome_and_counts_disagreements() -> None:
    items = [_item("a", 0, 0.5), *ITEMS[1:]]
    m = match(Corpus("T", "test", "m", False, items), {SET: _set(ENTRANTS, ROWS)})
    assert m.outcome_disagreements == [f"{SET}:polymarket:a"]


def test_match_refuses_an_instance_missing_from_the_extract() -> None:
    items = [*ITEMS, _item("zz", 1, 0.5)]
    with pytest.raises(FBCompareError, match="polymarket:zz"):
        match(Corpus("T", "test", "m", False, items), {SET: _set(ENTRANTS, ROWS)})


def test_pooling_over_sets_keys_entrants_by_organization_and_model() -> None:
    later = "2026-07-05"
    sets = {
        SET: _set(ENTRANTS, ROWS),
        later: _set(ENTRANTS[2:], [("d", 0.0, [0.2], [])], later),
    }
    items = [*ITEMS, _item("d", 0, 0.5, later)]
    s = _series(match(Corpus("T", "test", "m", False, items), sets))
    assert s["Team / bot"].forecasts == {0: 0.8, 1: 0.1, 3: 0.2}
    assert s["Team / bot"].sets == {SET, later}
    assert s["ForecastBench / Z (zero shot)"].sets == {SET}


def test_compare_pairs_each_entrant_on_its_own_questions() -> None:
    out = compare(Corpus("T", "test", "m", False, ITEMS), {SET: _set(ENTRANTS, ROWS)})
    rows = {e["name"]: e for e in out["entrants"]}
    z = rows["ForecastBench / Z (zero shot)"]
    # Augur 0.5 everywhere: Brier 0.25 on a and c. Z: 0.01 and 0.09.
    assert z["n"] == 2
    assert z["imputed_excluded"] == 1
    assert z["brier"] == pytest.approx(0.05)
    assert z["augur_brier"] == pytest.approx(0.25)
    assert z["diff"]["mean"] == pytest.approx(0.2)
    assert out["n"] == 3
    assert out["augur_brier"]["mean"] == pytest.approx(0.25)
    assert out["groups"]["zero-shot"]["augur_rank"] == 2


def test_compare_with_an_entrant_left_with_nothing() -> None:
    rows = [
        (q, y, [None if j == 0 else p for j, p in enumerate(fs)], imp) for q, y, fs, imp in ROWS
    ]
    out = compare(Corpus("T", "test", "m", False, ITEMS), {SET: _set(ENTRANTS, rows)})
    assert [u["name"] for u in out["unscored"]] == ["ForecastBench / Z (zero shot)"]
    assert out["groups"]["zero-shot"] == {"count": 0}


def _e(
    name: str, mean: float, low: float | None, high: float | None, n: int = 100
) -> dict[str, Any]:
    return {
        "name": name,
        "n": n,
        "brier": 0.2 - mean,
        "diff": {"mean": mean, "low": low, "high": high, "n": n, "clusters": n},
    }


def test_group_summary_rank_median_and_bottom_quarter() -> None:
    entrants = [
        _e("a", 0.05, 0.01, 0.09),
        _e("b", 0.02, -0.01, 0.05),
        _e("c", 0.01, -0.02, 0.04),
        _e("d", -0.03, -0.05, -0.01),
        _e("tiny", 0.30, None, None, n=1),
    ]
    s = group_summary(entrants)
    assert s["count"] == 5
    assert s["beat_augur"] == 4
    assert s["augur_rank"] == 5
    assert s["ranked"] == 6
    assert s["bottom_quarter"] is True  # rank 5 > 0.75 x 6 = 4.5
    assert s["median_diff"] == pytest.approx(0.02)
    assert s["beat_augur_interval"] == 1
    assert s["augur_beats_interval"] == 1
    assert s["no_interval"] == 1
    # Best and worst are named among entrants with at least half the median n.
    assert (s["best"], s["worst"]) == ("a", "d")
    s = group_summary(entrants[1:4])
    assert (s["augur_rank"], s["ranked"], s["bottom_quarter"]) == (3, 4, False)


def _group(
    count: int,
    better: int,
    beat_interval: int = 0,
    median: float = 0.0,
) -> dict[str, Any]:
    ranked = count + 1
    rank = better + 1
    return {
        "count": count,
        "beat_augur": better,
        "beat_augur_interval": beat_interval,
        "augur_beats_interval": 0,
        "augur_rank": rank,
        "ranked": ranked,
        "bottom_quarter": rank > 0.75 * ranked,
        "median_diff": median,
        "median_n": 100,
    }


def _corpora(
    c2_zero: dict[str, Any],
    c4_zero: dict[str, Any],
    c2_freeze: dict[str, Any],
    o_external: dict[str, Any],
    externals: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    def corpus(groups: dict[str, Any], entrants: list[dict[str, Any]]) -> dict[str, Any]:
        base = {g: _group(1, 0) for g in fbcompare.GROUPS}
        return {"groups": {**base, **groups}, "entrants": entrants}

    return {
        "C1": corpus({}, []),
        "C2": corpus({"zero-shot": c2_zero, "zero-shot-freeze": c2_freeze}, []),
        "O": corpus({"external": o_external}, []),
        "C4": corpus({"zero-shot": c4_zero}, externals),
    }


def test_verdicts_hold_on_numbers_built_to_hold() -> None:
    v = verdicts(
        _corpora(
            c2_zero=_group(20, 16, median=0.03),
            c4_zero=_group(40, 35, median=0.02),
            c2_freeze=_group(20, 20, beat_interval=18),
            o_external=_group(10, 9, median=0.05),
            externals=[{**_e("x", -0.02, -0.04, 0.01), "group": "external"}],
        )
    )
    assert [(x["id"], x["held"]) for x in v] == [
        ("P30", True),
        ("P31", True),
        ("P32", True),
        ("P33", True),
    ]
    assert all(x["prediction"] == fbcompare.PREDICTIONS[x["id"]] for x in v)


def test_verdicts_fail_on_the_literal_reading() -> None:
    v = verdicts(
        _corpora(
            # Worse than the median on C2 but rank 13 of 23: not the bottom quarter.
            c2_zero=_group(22, 12, median=0.002),
            c4_zero=_group(40, 35, median=0.02),
            # 17 of 20 is 85%, short of 90%.
            c2_freeze=_group(20, 20, beat_interval=17),
            # A median of exactly 0 is not worse.
            o_external=_group(10, 5, median=0.0),
            externals=[
                {**_e("beaten", -0.03, -0.06, -0.001), "group": "external"},
                # One market: no interval, so it cannot count.
                {**_e("one", -0.3, None, None, n=1), "group": "external"},
            ],
        )
    )
    held = {x["id"]: x["held"] for x in v}
    assert held == {"P30": False, "P31": False, "P32": False, "P33": False}
    p33 = next(x for x in v if x["id"] == "P33")
    assert "beaten" in p33["numbers"]
    assert "C4: one" not in p33["numbers"]


def test_dumps_is_sorted_and_byte_stable() -> None:
    snap = {"b": [1, {"z": 1, "a": None}], "a": "−"}
    text = fbcompare.dumps(snap)
    assert text == fbcompare.dumps(json.loads(text))
    assert text.index('"a"') < text.index('"b"')
    assert text.endswith("}\n")


def test_check_fails_on_a_tampered_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snap = {"corpora": [], "verdicts": [{"id": "P30", "held": False}]}
    monkeypatch.setattr(fbcompare, "snapshot", lambda root: snap)
    out = tmp_path / "snap.json"
    assert cli.main(["fbcompare", "--out", str(out)]) == 0
    assert cli.main(["fbcompare", "--out", str(out), "--check"]) == 0
    out.write_text(out.read_text(encoding="utf-8").replace("false", "true"), encoding="utf-8")
    assert cli.main(["fbcompare", "--out", str(out), "--check"]) == 1
    assert "differs" in capsys.readouterr().out
    out.unlink()
    assert cli.main(["fbcompare", "--out", str(out), "--check"]) == 1


def test_committed_extract_loads_and_covers_every_corpus_set() -> None:
    repo = Path(__file__).resolve().parent.parent
    sets = fbcompare.load_sets(repo / fbcompare.SETS_DIR)
    assert {"2026-05-24", *q4.SETS} <= sets.keys()
    assert len({fs.provenance.sha256 for fs in sets.values()}) == 1
