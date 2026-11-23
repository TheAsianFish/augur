"""scripts/extract_forecast_sets.py on a tiny hand-built forecast set."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ex = _load("extract_forecast_sets")
SET = "2026-06-21"


def _row(source: str, id_: str, p: float | None, *, imputed: bool = False) -> dict[str, Any]:
    return {
        "id": id_,
        "source": source,
        "direction": [],
        "forecast_due_date": f"{SET}T00:00:00.000",
        "market_value_on_due_date": 0.3,
        "resolution_date": "2026-07-01T00:00:00.000",
        "resolved_to": 1.0,
        "resolved": True,
        "forecast": p,
        "imputed": imputed,
    }


def _entrant(directory: Path, file: str, org: str, model: str, rows: list[dict[str, Any]]) -> None:
    doc = {
        "organization": org,
        "model": model,
        "model_organization": org,
        "forecast_due_date": SET,
        "question_set": f"{SET}-llm.json",
        "leaderboard_eligible": True,
        "forecasts": rows,
    }
    (directory / f"{SET}.{file}").write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    sets = tmp_path / "sets" / SET
    sets.mkdir(parents=True)
    _entrant(
        sets,
        "OpenAI.gpt-x_zero_shot.json",
        "ForecastBench",
        "GPT-X (zero shot)",
        [_row("polymarket", "a", 0.6), _row("manifold", "b", 0.2, imputed=True)],
    )
    _entrant(
        sets,
        "ForecastBench.Google.Google DeepMind.g-run-variant-01-zero-shot-with-freeze-values.json",
        "ForecastBench",
        "g-high-web-search-zero-shot-with-freeze-values",
        [_row("polymarket", "a", 0.7), _row("manifold", "b", 0.1)],
    )
    _entrant(
        sets,
        "external.Team.1.json",
        "Team",
        "bot",
        # A dataset question and one Augur never forecast: both left out.
        [_row("polymarket", "a", 0.9), _row("acled", "x", 0.5), _row("manifold", "zz", 0.5)],
    )
    _entrant(
        sets,
        "ForecastBench.naive-forecaster.json",
        "ForecastBench",
        "Naive Forecaster",
        [_row("polymarket", "a", 0.3, imputed=True), _row("manifold", "b", 0.3, imputed=True)],
    )
    subsets = tmp_path / "subsets"
    subsets.mkdir()
    subset = {
        "question_set": f"{SET}-llm.json",
        "questions": [
            {"source": "polymarket", "id": "a"},
            {"source": "manifold", "id": "b"},
            {"source": "acled", "id": "x"},
        ],
    }
    (subsets / f"{SET}-llm.subset.json").write_text(json.dumps(subset), encoding="utf-8")
    (tmp_path / "archive.tar.gz").write_bytes(b"not really an archive")
    return tmp_path


def test_groups_from_file_names() -> None:
    assert ex.group_of("Anthropic.claude_zero_shot.json") == "zero-shot"
    assert ex.group_of("Anthropic.claude_zero_shot_with_freeze_values.json") == "zero-shot-freeze"
    assert ex.group_of("ForecastBench.OpenAI.OpenAI.m-run-variant-01-zero-shot.json") == "zero-shot"
    assert ex.group_of("external.Anyone.2.json") == "external"
    assert ex.group_of("ForecastBench.always-0.5.json") == "baseline"
    with pytest.raises(ex.ExtractError, match="classify"):
        ex.group_of("ForecastBench.something-new.json")


def test_extract_keeps_only_augur_market_questions(tree: Path) -> None:
    keys = ex.subset_keys(tree / "subsets" / f"{SET}-llm.subset.json")
    assert keys == [("polymarket", "a"), ("manifold", "b")]
    doc = ex.extract_set(tree / "sets" / SET, SET, keys)
    names = [e["model"] for e in doc["entrants"]]
    groups = {e["model"]: e["group"] for e in doc["entrants"]}
    assert groups == {
        "Naive Forecaster": "baseline",
        "g-high-web-search-zero-shot-with-freeze-values": "zero-shot-freeze",
        "GPT-X (zero shot)": "zero-shot",
        "bot": "external",
    }
    assert [e["web_search"] for e in doc["entrants"]] == ["web-search" in n for n in names]
    a, b = doc["questions"]
    assert (a["source"], a["id"], b["source"], b["id"]) == ("polymarket", "a", "manifold", "b")
    assert a["resolved_to"] == 1.0
    assert a["resolution_date"] == "2026-07-01"
    assert a["market_value_on_due_date"] == 0.3
    by_name = dict(zip(names, zip(a["forecasts"], b["forecasts"], strict=True), strict=True))
    assert by_name["bot"] == (0.9, None)  # the external entry did not forecast b
    assert by_name["GPT-X (zero shot)"] == (0.6, 0.2)
    gpt = names.index("GPT-X (zero shot)")
    naive = names.index("Naive Forecaster")
    assert b["imputed"] == sorted([gpt, naive])
    assert a["imputed"] == [naive]


def test_disagreeing_outcomes_refuse(tree: Path) -> None:
    _entrant(
        tree / "sets" / SET,
        "external.Other.1.json",
        "Other",
        "x",
        [{**_row("polymarket", "a", 0.5), "resolved_to": 0.0}],
    )
    with pytest.raises(ex.ExtractError, match="resolved_to"):
        ex.extract_set(tree / "sets" / SET, SET, [("polymarket", "a")])


def test_main_writes_a_byte_stable_file_with_provenance(tree: Path) -> None:
    out = tree / "out"
    argv = [
        str(tree / "sets"),
        str(tree / "archive.tar.gz"),
        str(out),
        "--subsets",
        str(tree / "subsets"),
        "--last-modified",
        "Tue, 17 Nov 2026 08:27:21 GMT",
        "--fetched",
        "2026-11-23",
    ]
    assert ex.main(argv) == 0
    first = (out / f"{SET}.json").read_text(encoding="utf-8")
    assert ex.main(argv) == 0
    assert (out / f"{SET}.json").read_text(encoding="utf-8") == first
    doc = json.loads(first)
    pv = doc["provenance"]
    assert pv["bytes"] == len(b"not really an archive")
    assert len(pv["sha256"]) == 64
    assert pv["licence"] == "CC BY-SA 4.0"
    assert "Forecasting Research Institute" in pv["attribution"]
    assert doc["question_set"] == SET
    # One entrant and one question per line.
    assert sum(1 for line in first.splitlines() if line.startswith('  {"file"')) == 4
