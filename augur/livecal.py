"""The live calibrator: the map M8 chose, applied only to the model it was fitted on.

Calibration is a property of one forecaster. A map fitted on one model's
resolved history says nothing about another model's forecasts, so the live
calibrator names its model and is applied to a binary aggregate only when
every member that answered came from that model; otherwise the forecast is
left raw. It is applied after the members are combined and before the
exact-price blend. The aggregate row then holds the calibrated probability,
with the raw one in its ``method``; a later fit on live history must use
the raw value (or recompute it from the member rows), never the
calibrated one.

The fitted map is committed as JSON beside this module
(``live_calibrator.json``), written by ``augur calibrate`` from every
resolution in the corpus, so a run needs no history at hand. When no
calibrator passed M8's replacement rule the file names ``raw``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from augur.calibrate import CalibrationError, CalibrationMap, from_params
from augur.errors import ConfigError

LIVE_PATH = Path(__file__).resolve().parent / "live_calibrator.json"


@dataclass(frozen=True)
class LiveCalibrator:
    model: str
    """The model (as the route names it) whose forecasts the map was fitted on."""
    fitted_on: str
    """What it was fitted on, for the ledger and the comment."""
    map: CalibrationMap

    @property
    def name(self) -> str:
        return self.map.name

    def applies(self, models: Sequence[str]) -> bool:
        """True when there is a map to apply and every answering model is the fitted one."""
        return self.name != "raw" and bool(models) and all(m == self.model for m in models)

    def __call__(self, p: float) -> float:
        return round(self.map(p), 3)


def load_live(path: Path = LIVE_PATH) -> LiveCalibrator:
    """The committed live calibrator; a broken file is a configuration error."""
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return LiveCalibrator(
            model=str(data["model"]),
            fitted_on=str(data["fitted_on"]),
            map=from_params(data["params"]),
        )
    except (OSError, ValueError, KeyError, TypeError, CalibrationError) as exc:
        raise ConfigError(
            f"cannot read the live calibrator {path} ({exc}); re-create it with augur calibrate "
            "or run with --no-calibrate"
        ) from None
