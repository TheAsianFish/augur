"""The live calibrator file: loading, the model check, and the committed copy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from augur import livecal, q4
from augur.calibrate import Platt
from augur.errors import ConfigError
from augur.report import REPO


def _write(path: Path, params: dict[str, object], model: str = "m") -> Path:
    path.write_text(
        json.dumps({"model": model, "fitted_on": "a test", "params": params}), encoding="utf-8"
    )
    return path


def test_load_round_trips_a_fitted_map(tmp_path: Path) -> None:
    fitted = Platt().fit([0.1, 0.2, 0.7, 0.9, 0.4], [0, 1, 1, 1, 0])
    cal = livecal.load_live(_write(tmp_path / "c.json", fitted.params()))
    assert cal.name == "platt"
    assert cal(0.3) == round(fitted(0.3), 3)
    assert cal.applies(["m", "m"])
    assert not cal.applies(["m", "other"])
    assert not cal.applies([])


def test_raw_never_applies(tmp_path: Path) -> None:
    cal = livecal.load_live(_write(tmp_path / "c.json", {"name": "raw"}))
    assert not cal.applies(["m"])


@pytest.mark.parametrize(
    "content", ["not json", "{}", '{"model": "m", "fitted_on": "x", "params": {"name": "?"}}']
)
def test_a_broken_file_is_a_configuration_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "c.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="--no-calibrate"):
        livecal.load_live(path)


def test_the_committed_calibrator_loads_and_names_c4_s_model() -> None:
    cal = livecal.load_live()
    assert cal.model == q4.MODEL
    assert "C4" in cal.fitted_on


# -- source-aware maps and withheld sources ------------------------------------


def _write_full(path: Path, params: dict[str, object], **extra: object) -> Path:
    path.write_text(
        json.dumps({"model": "m", "fitted_on": "a test", "params": params, **extra}),
        encoding="utf-8",
    )
    return path


def test_withheld_source_does_not_apply_but_others_do(tmp_path: Path) -> None:
    fitted = Platt().fit([0.1, 0.2, 0.7, 0.9, 0.4], [0, 1, 1, 1, 0])
    cal = livecal.load_live(
        _write_full(tmp_path / "c.json", fitted.params(), withheld_from=["metaculus"])
    )
    assert cal.withheld_from == frozenset({"metaculus"})
    assert livecal.LIVE_SOURCE == "metaculus"
    assert not cal.applies(["m"])
    assert not cal.applies(["m"], "metaculus")
    assert cal.applies(["m"], "polymarket")
    assert not cal.applies(["other"], "polymarket")
    # the unsourced adapter ignores the source
    assert cal(0.3, "polymarket") == cal(0.3, "metaculus") == round(fitted(0.3), 3)


def test_withheld_from_defaults_to_nothing(tmp_path: Path) -> None:
    cal = livecal.load_live(_write(tmp_path / "c.json", {"name": "raw"}))
    assert cal.withheld_from == frozenset()


def test_a_partial_map_loads_and_uses_the_sources_offset(tmp_path: Path) -> None:
    params = {
        "name": "platt-partial",
        "a": 0.5,
        "b": -0.2,
        "offsets": {"metaculus": -0.7, "polymarket": 0.4},
        "prior_sd": 0.5,
        "n": 100,
        "iterations": 5,
    }
    cal = livecal.load_live(_write_full(tmp_path / "c.json", params))
    assert cal.name == "platt-partial"
    x = math.log(0.3 / 0.7)

    def sig(z: float) -> float:
        return 1 / (1 + math.exp(-z))

    assert cal(0.3, "metaculus") == round(sig(0.5 * x - 0.2 - 0.7), 3)
    assert cal(0.3, "polymarket") == round(sig(0.5 * x - 0.2 + 0.4), 3)
    assert cal(0.3) == cal(0.3, "metaculus")  # the live source is the default
    assert cal(0.3, "unseen") == round(sig(0.5 * x - 0.2), 3)
    assert cal.applies(["m"])


def test_a_by_source_map_loads(tmp_path: Path) -> None:
    pooled = Platt().fit([0.1, 0.8, 0.5, 0.4], [0, 1, 1, 0]).params()
    own = Platt().fit([0.2, 0.9, 0.6, 0.3], [0, 1, 0, 0]).params()
    params = {
        "name": "platt-by-source",
        "min_n": 30,
        "pooled": pooled,
        "by_source": {"metaculus": own},
    }
    cal = livecal.load_live(_write_full(tmp_path / "c.json", params))
    assert cal(0.5, "metaculus") != cal(0.5, "polymarket")


def test_a_malformed_sourced_map_is_a_configuration_error(tmp_path: Path) -> None:
    path = _write_full(tmp_path / "c.json", {"name": "platt-partial", "a": 1.0})
    with pytest.raises(ConfigError, match="--no-calibrate"):
        livecal.load_live(path)


def test_the_committed_calibrator_is_consistent_with_the_source_decision() -> None:
    cal = livecal.load_live()
    snap_path = REPO / q4.SNAPSHOT
    if not snap_path.exists():
        pytest.skip("augur calibrate has not been run")
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    assert q4.source_choice(snap["source_aware"]) == snap["source_aware"]["choice"]
    if snap["source_aware"]["choice"] is None:
        assert cal.name != "raw"
        assert cal.withheld_from == frozenset({"metaculus"})
        assert not cal.applies([cal.model])
        assert cal.applies([cal.model], "polymarket")
    else:
        assert cal.name == snap["source_aware"]["choice"]
        assert cal.withheld_from == frozenset()
