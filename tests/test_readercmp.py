"""M16: a set forecast before its outcomes exist, the release gate on both routes, and the rule."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from augur import readercmp
from augur.backtest import PinnedClock, run_forecastbench
from augur.clock import FixedClock
from augur.config import Settings
from augur.forecastbench import FBMarketQuestion, FBQuestion, load_open_set
from augur.leakage import RunMode, admit_model
from augur.ledger import Ledger
from augur.llm.catalog import Catalog, load_snapshot, release_bound
from augur.llm.endpoints import PRESETS, Route, endpoints_from_env, parse_route
from augur.llm.router import Router
from augur.resolve import resolve_forecastbench
from augur.score import Estimate

from .fakes.forecastbench import raw_question, raw_resolution, write_pair
from .fakes.llm import ScriptedProvider

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "forecastbench"
NEWER = Route(endpoint=PRESETS["openrouter"], model="nvidia/nemotron-3.5-super-vl:free")
CURRENT = Route(endpoint=PRESETS["openrouter"], model="nvidia/nemotron-3-ultra-550b-a55b:free")
REGISTERED_SHA = {
    "2026-11-08": "9a5a6d891a372d065399f3849a3c0f42e8e41ac07ebfbff97ca33cedd8fc00fb",
    "2026-11-22": "4bb4cc57450dfe27032ee85fb75ee24d7bb4906fc6f762c5a4de2c45cbb7c952",
}


def _catalog(listed: dict[str, date]) -> Catalog:
    rows = [
        {
            "id": model,
            "created": int(datetime.combine(day, datetime.min.time(), UTC).timestamp()),
            "pricing": {"prompt": "0", "completion": "0"},
        }
        for model, day in listed.items()
    ]
    return Catalog.from_api({"data": rows}, datetime(2026, 11, 26, tzinfo=UTC))


def _open_set(tmp_path: Path) -> Path:
    qs, _ = write_pair(
        tmp_path,
        [
            raw_question("a", value="0.8"),
            raw_question("b", source="polymarket", value="0.3"),
            raw_question("c", source="metaculus", value="0.6"),
            raw_question("fred-1", source="fred"),
            raw_question("kalshi-1", source="kalshi"),
            raw_question("noclose", market_info_close_datetime="N/A"),
        ],
        [],
    )
    return qs


def _resolutions(tmp_path: Path, question_set: str, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / f"{question_set}-res.json"
    path.write_text(
        json.dumps({"question_set": f"{question_set}-llm.json", "resolutions": rows}),
        encoding="utf-8",
    )
    return path


async def _forecast(
    path: Path,
    route: Route,
    catalog: Catalog,
    percent: float,
    qs: Path,
    question_set: str = "2026-11-08",
) -> int:
    report = load_open_set(qs, released=date(2024, 1, 1))
    clock = PinnedClock(datetime(2026, 11, 26, tzinfo=UTC))
    provider = ScriptedProvider((percent,))
    router = Router.single(route, provider, catalog, FixedClock(clock.now()))
    settings = Settings.model_validate({"provider": "replay", "model": route.model})
    settings = settings.model_copy(update={"forecasters": 1})
    with Ledger.open(path, clock) as ledger:
        await run_forecastbench(
            questions=report.kept,
            tournament=f"forecastbench:{question_set}-llm.json",
            settings=settings,
            router=router,
            ledger=ledger,
            clock=clock,
            code_version="test",
            out=lambda _line: None,
        )
    return len(provider.requests)


def test_an_open_set_keeps_market_questions_with_no_outcome(tmp_path: Path) -> None:
    report = load_open_set(_open_set(tmp_path), released=date(2024, 1, 1))
    assert [q.fb_id for q in report.kept] == ["a", "b", "c"]
    assert report.removed == {
        "not a market question (dataset questions are M6's)": 2,
        "no market close date": 1,
    }
    for q in report.kept:
        assert not isinstance(q, FBQuestion)
        assert "outcome" not in q.model_dump()


def test_the_release_gate_refuses_either_route_on_a_set_posed_before_its_listing() -> None:
    snapshot = load_snapshot()
    endpoints = endpoints_from_env({})
    for route, listed in (
        (readercmp.CURRENT, date(2026, 6, 4)),
        (readercmp.NEWER, date(2026, 10, 15)),
    ):
        released, basis = release_bound(parse_route(route, endpoints), snapshot)
        assert (released, basis) == (listed, "OpenRouter listing date")
        before = datetime.combine(listed, datetime.min.time(), UTC).replace(hour=12)
        after = datetime(2026, 10, 29, tzinfo=UTC)  # the 2026-11-08 set's freeze
        assert not admit_model(released, before, RunMode.BACKTEST).admitted
        assert admit_model(released, after, RunMode.BACKTEST).admitted
    # The 2026-10-25 set froze on 2026-10-09: the newer route may not forecast it.
    newer, _ = release_bound(parse_route(readercmp.NEWER, endpoints), snapshot)
    assert newer is not None
    assert not admit_model(newer, datetime(2026, 10, 9, tzinfo=UTC), RunMode.BACKTEST).admitted


def test_the_loader_and_the_run_both_refuse_a_route_listed_after_the_freeze(
    tmp_path: Path,
) -> None:
    qs = _open_set(tmp_path)  # frozen 2025-03-01
    late = date(2025, 3, 1)
    assert load_open_set(qs, released=late).kept == []
    # The run's own gate, on questions loaded for an earlier route, makes no call.
    catalog = _catalog({NEWER.model: late, CURRENT.model: late})
    calls = asyncio.run(_forecast(tmp_path / "l.sqlite", NEWER, catalog, 30, qs))
    assert calls == 0
    with Ledger.open(tmp_path / "l.sqlite", PinnedClock(datetime(2026, 1, 1, tzinfo=UTC))) as led:
        reasons = [
            r[0] for r in led.connection.execute("SELECT reason FROM question_outcomes").fetchall()
        ]
    assert len(reasons) == 3
    assert all("released" in r or "listed" in r or "after" in r for r in reasons), reasons


def test_the_committed_open_sets_are_the_registered_ones_and_postdate_both_listings() -> None:
    for day, sha in REGISTERED_SHA.items():
        meta = json.loads((DATA / f"{day}.open.source.json").read_text(encoding="utf-8"))
        assert meta["sha256"]["questions"] == sha
        assert "resolutions" not in meta["urls"]
        report = load_open_set(DATA / f"{day}-llm.open.json", released=date(2026, 10, 15))
        assert len(report.kept) == meta["kept"] == 146
        assert all(q.freeze_datetime.date() > date(2026, 10, 15) for q in report.kept)
        assert {q.source for q in report.kept} == {"metaculus", "polymarket"}


def test_paired_scoring_on_two_route_ledgers_matches_a_hand_computation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "data" / "ledgers").mkdir(parents=True)
    qs = _open_set(tmp_path)
    catalog = _catalog({NEWER.model: date(2024, 1, 1), CURRENT.model: date(2024, 1, 1)})
    asyncio.run(_forecast(root / readercmp.LEDGERS["newer"], NEWER, catalog, 30, qs))
    asyncio.run(_forecast(root / readercmp.LEDGERS["current"], CURRENT, catalog, 60, qs))
    rs = _resolutions(
        tmp_path,
        "2026-11-08",
        [
            raw_resolution("a", 1.0),
            raw_resolution("b", 0.0, source="polymarket"),
            raw_resolution("c", 0.0, source="metaculus", resolved=False),
        ],
    )
    for name in ("newer", "current"):
        with Ledger.open(root / readercmp.LEDGERS[name], FixedClock(datetime.now(UTC))) as led:
            resolve_forecastbench(led, rs)
    ps, left = readercmp.pairs(root)
    # a: 0.7² − 0.4² = 0.33; b: 0.3² − 0.6² = −0.27; c is not resolved.
    assert sorted(round(p.diff, 6) for p in ps) == [-0.27, 0.33]
    assert left["not resolved to 0 or 1"] == 1
    snap = readercmp.snapshot(root)
    assert math.isclose(snap["backtest"]["mean"], 0.03)
    assert snap["sets"]["2026-11-08"]["paired"] == 2
    assert snap["usage"]["newer"]["calls"] == 3
    assert snap["reading"]["kind"] == "progress"
    assert snap["verdict"].startswith("not decidable at this n (progress reading")


def _est(low: float, high: float) -> Estimate:
    return Estimate((low + high) / 2, low, high, 100, 100, 0.01, 0.03, 10_000, 1)


@pytest.mark.parametrize(
    ("low", "high", "newer_ms", "expected"),
    [
        (-0.03, -0.001, 9_000.0, "switched"),  # wholly below 0, even if slower
        (-0.02, 0.009, 1_000.0, "switched"),  # inside the margin and no slower
        (-0.02, 0.009, 2_001.0, "not decidable at this n"),  # inside the margin but slower
        (-0.02, 0.011, 1_000.0, "not decidable at this n"),  # outside the margin
        (0.001, 0.04, 1_000.0, "kept"),
    ],
)
def test_the_registered_rule(low: float, high: float, newer_ms: float, expected: str) -> None:
    latency: dict[str, float | None] = {"newer": newer_ms, "current": 2_000.0}
    assert readercmp.verdict(_est(low, high), latency, "registered") == expected
    progress = readercmp.verdict(_est(low, high), latency, "progress")
    assert progress.startswith(expected)
    assert "progress reading" in progress


def test_the_reading_date_decides_whether_the_rule_binds() -> None:
    assert readercmp.reading_kind(None) == "progress"
    assert readercmp.reading_kind(date(2027, 1, 10)) == "progress"
    assert readercmp.reading_kind(date(2027, 1, 11)) == "registered"
    assert readercmp.reading_kind(date(2027, 1, 12)) == "late"


def test_n_needed_from_the_measured_spread() -> None:
    # (1.96 + 0.84)² · 0.1² / 0.01² ≈ 785
    assert readercmp.n_needed(0.1, 0.01) == 785


def test_open_question_types_stay_separate() -> None:
    assert issubclass(FBQuestion, FBMarketQuestion)
    assert "outcome" not in FBMarketQuestion.model_fields


def test_registration_hashes_match_the_plan() -> None:
    plan = (REPO / "PLAN.md").read_text(encoding="utf-8")
    for sha in REGISTERED_SHA.values():
        assert f"`{sha[:8]}…{sha[-6:]}`" in plan or f"`{sha[:8]}…{sha[-5:]}`" in plan


def test_a_question_posed_in_both_sets_is_paired_per_set_and_resampled_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / "data" / "ledgers").mkdir(parents=True)
    qs = _open_set(tmp_path)
    catalog = _catalog({NEWER.model: date(2024, 1, 1), CURRENT.model: date(2024, 1, 1)})
    for day, (newer, current) in (("2026-11-08", (30, 60)), ("2026-11-22", (90, 20))):
        asyncio.run(_forecast(root / readercmp.LEDGERS["newer"], NEWER, catalog, newer, qs, day))
        asyncio.run(
            _forecast(root / readercmp.LEDGERS["current"], CURRENT, catalog, current, qs, day)
        )
    # Only the 2026-11-08 file exists: the 2026-11-22 rows of the same questions stay open.
    rs = _resolutions(tmp_path, "2026-11-08", [raw_resolution("a", 1.0)])
    for name in ("newer", "current"):
        with Ledger.open(root / readercmp.LEDGERS[name], FixedClock(datetime.now(UTC))) as led:
            resolve_forecastbench(led, rs)
    ps, left = readercmp.pairs(root)
    assert [(p.question_set, round(p.p_newer, 2), round(p.p_current, 2)) for p in ps] == [
        ("2026-11-08", 0.3, 0.6)
    ]
    assert left["not resolved to 0 or 1"] == 5  # b and c in the first set, all three in the second
    # Now the second set's file: question a twice, one cluster.
    rs2 = _resolutions(tmp_path, "2026-11-22", [raw_resolution("a", 1.0)])
    for name in ("newer", "current"):
        with Ledger.open(root / readercmp.LEDGERS[name], FixedClock(datetime.now(UTC))) as led:
            resolve_forecastbench(led, rs2)
    snap = readercmp.snapshot(root)
    assert snap["sets"]["2026-11-22"]["paired"] == 1
    assert snap["backtest"]["n"] == 2
    # One cluster: the interval collapses to the mean of a's two rows.
    assert snap["backtest"]["low"] == snap["backtest"]["high"] == snap["backtest"]["mean"]
    assert snap["usage"]["newer"]["forecast"] == {"2026-11-08": 3, "2026-11-22": 3}
