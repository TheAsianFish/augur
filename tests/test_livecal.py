"""The live calibrator file: loading, the model check, and the committed copy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from augur import livecal, q4
from augur.calibrate import Platt
from augur.errors import ConfigError


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
