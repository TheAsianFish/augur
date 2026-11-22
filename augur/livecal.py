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

A map is also a property of the sources it was shown to work on. The file's
``withheld_from`` lists the question sources the map must not touch: M8's
pooled map is 0.035 Brier worse than raw on Metaculus questions, and until
a map passes the source-aware rule (PLAN.md, 2026-11-22) it is withheld
from them. Every question the bot forecasts live is a Metaculus question
(:data:`LIVE_SOURCE`), so a withheld map changes dry runs and backtests
only where a caller names another source. A source-aware map (``augur.sourcecal``)
receives the question's source.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from augur.calibrate import CalibrationError, CalibrationMap, from_params
from augur.errors import ConfigError
from augur.sourcecal import SourcedMap, sourced_from_params

LIVE_PATH = Path(__file__).resolve().parent / "live_calibrator.json"
LIVE_SOURCE = "metaculus"
"""The source of every question the bot forecasts live."""
SOURCED_NAMES = frozenset({"platt-by-source", "platt-partial"})


@dataclass(frozen=True)
class LiveCalibrator:
    model: str
    """The model (as the route names it) whose forecasts the map was fitted on."""
    fitted_on: str
    """What it was fitted on, for the ledger and the comment."""
    map: SourcedMap
    withheld_from: frozenset[str] = frozenset()
    """Question sources the map is not applied to (not shown to be no worse than raw there)."""

    @property
    def name(self) -> str:
        return self.map.name

    def applies(self, models: Sequence[str], source: str = LIVE_SOURCE) -> bool:
        """True when there is a map to apply, the source is not withheld, and every
        answering model is the fitted one."""
        return (
            self.name != "raw"
            and source not in self.withheld_from
            and bool(models)
            and all(m == self.model for m in models)
        )

    def __call__(self, p: float, source: str = LIVE_SOURCE) -> float:
        return round(self.map(p, source), 3)


@dataclass(frozen=True)
class Unsourced:
    """A map that ignores the source (raw, extremizing, pooled Platt, isotonic)."""

    inner: CalibrationMap

    @property
    def name(self) -> str:
        return self.inner.name

    def __call__(self, p: float, source: str) -> float:
        return self.inner(p)

    def params(self) -> dict[str, Any]:
        return self.inner.params()


def load_live(path: Path = LIVE_PATH) -> LiveCalibrator:
    """The committed live calibrator; a broken file is a configuration error."""
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        params = data["params"]
        return LiveCalibrator(
            model=str(data["model"]),
            fitted_on=str(data["fitted_on"]),
            map=(
                sourced_from_params(params)
                if params.get("name") in SOURCED_NAMES
                else Unsourced(from_params(params))
            ),
            withheld_from=frozenset(str(s) for s in data.get("withheld_from", [])),
        )
    except (OSError, ValueError, KeyError, TypeError, CalibrationError) as exc:
        raise ConfigError(
            f"cannot read the live calibrator {path} ({exc}); re-create it with augur calibrate "
            "or run with --no-calibrate"
        ) from None
