"""Tiny ForecastBench question and resolution sets, and a backtest ledger built from them."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from augur.backtest import BacktestResult, PinnedClock, run_forecastbench
from augur.clock import FixedClock
from augur.config import Settings
from augur.forecastbench import FBQuestion, load_set
from augur.ledger import Ledger
from augur.llm.catalog import Catalog
from augur.llm.endpoints import PRESETS, Route
from augur.llm.router import Router

from .llm import ScriptedProvider

QUESTION_SET = "2025-03-02-llm.json"
FREEZE = "2025-03-01T00:00:00+00:00"
LOCAL = Route(endpoint=PRESETS["ollama"], model="qwen2.5:7b")  # released 2024-09-19
RELEASED = date(2024, 9, 19)


def raw_question(
    fb_id: str,
    *,
    source: str = "manifold",
    freeze: str | None = FREEZE,
    value: object = "0.7123",
    **extra: object,
) -> dict[str, Any]:
    """One question as ForecastBench publishes it (field names copied from the real set)."""
    raw: dict[str, Any] = {
        "background": "Some background.",
        "freeze_datetime": freeze,
        "freeze_datetime_value": value,
        "freeze_datetime_value_explanation": "The market value.",
        "id": fb_id,
        "market_info_close_datetime": "2025-06-01T00:00:00+00:00",
        "market_info_open_datetime": "2024-01-01T00:00:00+00:00",
        "market_info_resolution_criteria": "N/A",
        "question": f"Will event {fb_id} happen?",
        "resolution_criteria": f"Resolves to the outcome of market {fb_id}.",
        "resolution_dates": "N/A",
        "source": source,
        "source_intro": "We would like you to predict the outcome of a prediction market.",
        "url": f"https://example.org/{source}/{fb_id}",
    }
    raw.update(extra)
    return raw


def raw_resolution(
    fb_id: str,
    resolved_to: object = 1.0,
    *,
    source: str = "manifold",
    resolved: bool = True,
    on: str = "2025-04-01",
) -> dict[str, Any]:
    return {
        "direction": None,
        "id": fb_id,
        "resolution_date": on,
        "resolved": resolved,
        "resolved_to": resolved_to,
        "source": source,
    }


def write_pair(
    tmp_path: Path, questions: list[dict[str, Any]], resolutions: list[dict[str, Any]]
) -> tuple[Path, Path]:
    qs = tmp_path / "qs.json"
    rs = tmp_path / "rs.json"
    qs.write_text(
        json.dumps(
            {
                "forecast_due_date": "2025-03-02",
                "question_set": QUESTION_SET,
                "questions": questions,
            }
        ),
        encoding="utf-8",
    )
    rs.write_text(
        json.dumps(
            {
                "forecast_due_date": "2025-03-02",
                "question_set": QUESTION_SET,
                "resolutions": resolutions,
            }
        ),
        encoding="utf-8",
    )
    return qs, rs


def small_set(tmp_path: Path) -> tuple[Path, Path, list[FBQuestion]]:
    """Three eligible questions: outcomes 1, 0 and 0, market prices 0.8, 0.3 and 0.6."""
    qs, rs = write_pair(
        tmp_path,
        [
            raw_question("a", value="0.8"),
            raw_question("b", source="polymarket", value="0.3"),
            raw_question("c", source="infer", value="0.6"),
        ],
        [
            raw_resolution("a", 1.0),
            raw_resolution("b", 0.0, source="polymarket"),
            raw_resolution("c", 0.0, source="infer"),
        ],
    )
    return qs, rs, load_set(qs, rs, released=RELEASED).kept


def free_local_catalog() -> Catalog:
    return Catalog.from_api({"data": []}, datetime(2026, 11, 18, tzinfo=UTC))


async def run_backtest(
    ledger_path: Path,
    questions: list[FBQuestion],
    *,
    route: Route = LOCAL,
    catalog: Catalog | None = None,
    percents: tuple[float | None, ...] = (30,),
    tournament: str = f"forecastbench:{QUESTION_SET}",
    forecasters: int = 3,
) -> BacktestResult:
    """Run the backtest with a scripted provider; the ledger is closed on return."""
    clock = PinnedClock(datetime(2026, 1, 1, tzinfo=UTC))
    router = Router.single(
        route,
        ScriptedProvider(percents),
        catalog or free_local_catalog(),
        FixedClock(clock.now()),
    )
    settings = Settings.model_validate(
        {"provider": "replay", "model": route.model, "forecasters": forecasters}
    )
    with Ledger.open(ledger_path, clock) as ledger:
        return await run_forecastbench(
            questions=questions,
            tournament=tournament,
            settings=settings,
            router=router,
            ledger=ledger,
            clock=clock,
            code_version="test",
            out=lambda _line: None,
        )
