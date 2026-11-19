"""``augur resolve``: resolution values, the Metaculus reader, ForecastBench and manual."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from augur.clock import FixedClock
from augur.errors import AugurError
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.metaculus import MetaculusClient
from augur.questions import Question, question_from_post
from augur.ratelimit import RateLimiter
from augur.resolve import (
    community_values,
    resolution_value,
    resolve_forecastbench,
    resolve_manual,
    resolve_metaculus,
)

from .conftest import T0
from .fakes.forecastbench import raw_resolution, run_backtest, small_set
from .fakes.metaculus import (
    FAKE_TOKEN,
    MC_DETAIL,
    NUMERIC_DETAIL,
    FakeMetaculus,
    binary_post,
    load_fixture,
)


def _q(name: str | None = None) -> Question:
    post = load_fixture(name) if name else binary_post(1, 2)
    return question_from_post(post)


BINARY = _q()
MC = _q(MC_DETAIL)
NUMERIC = _q(NUMERIC_DETAIL)


# -- resolution_value ----------------------------------------------------------


@pytest.mark.parametrize("raw", [None, ""])
def test_no_resolution_is_none(raw: object) -> None:
    assert resolution_value(BINARY, raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("yes", ("resolved", "yes")),
        ("No", ("resolved", "no")),
        (" YES ", ("resolved", "yes")),
        ("annulled", ("annulled", None)),
        ("Ambiguous", ("ambiguous", None)),
    ],
)
def test_binary_resolutions(raw: str, expected: tuple[str, str | None]) -> None:
    assert resolution_value(BINARY, raw) == expected


def test_binary_rejects_anything_else() -> None:
    for bad in ("maybe", "1", "true", "Democratic Party"):
        with pytest.raises(AugurError, match="resolved to"):
            resolution_value(BINARY, bad)


def test_multiple_choice_resolves_to_an_option_exactly() -> None:
    assert MC.options
    option = MC.options[1]
    assert resolution_value(MC, option) == ("resolved", option)
    assert resolution_value(MC, f"  {option}  ") == ("resolved", option)
    assert resolution_value(MC, "annulled") == ("annulled", None)
    with pytest.raises(AugurError, match="not an option"):
        resolution_value(MC, option.lower())
    with pytest.raises(AugurError, match="not an option"):
        resolution_value(MC, "yes")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12.5", ("resolved", "12.5")),
        ("-3", ("resolved", "-3")),
        (7, ("resolved", "7")),
        ("below_lower_bound", ("resolved", "below_lower_bound")),
        ("Above_Upper_Bound", ("resolved", "above_upper_bound")),
        ("annulled", ("annulled", None)),
    ],
)
def test_numeric_resolutions(raw: object, expected: tuple[str, str | None]) -> None:
    assert resolution_value(NUMERIC, raw) == expected


def test_numeric_rejects_text() -> None:
    with pytest.raises(AugurError, match="resolved to 'lots'"):
        resolution_value(NUMERIC, "lots")


def test_a_question_type_that_is_not_scored_raises() -> None:
    other = BINARY.model_copy(update={"type": "date"})
    with pytest.raises(AugurError, match="not scored"):
        resolution_value(other, "2026-01-01")
    # ... but an annulment needs no type
    assert resolution_value(other, "annulled") == ("annulled", None)


# -- community_values ----------------------------------------------------------


def test_community_values_binary_takes_the_yes_probability() -> None:
    assert community_values(BINARY, {"forecast_values": [0.7, 0.3]}) == {"probability": 0.3}


def test_community_values_multiple_choice_pairs_options() -> None:
    assert MC.options
    n = len(MC.options)
    values = [round(1 / n, 6)] * n
    assert community_values(MC, {"forecast_values": values}) == {
        "probabilities": dict(zip(MC.options, values, strict=True))
    }
    assert community_values(MC, {"forecast_values": values[:-1]}) is None


def test_community_values_continuous_needs_the_full_cdf() -> None:
    assert NUMERIC.scaling is not None
    n = NUMERIC.scaling.inbound_outcome_count + 1
    cdf = [i / (n - 1) for i in range(n)]
    assert community_values(NUMERIC, {"forecast_values": cdf}) == {"cdf": cdf}
    assert community_values(NUMERIC, {"forecast_values": cdf[:-1]}) is None


@pytest.mark.parametrize(
    "latest",
    [
        None,
        "x",
        [],
        {},
        {"forecast_values": None},
        {"forecast_values": []},
        {"forecast_values": "0.5"},
        {"forecast_values": [0.5, "0.5"]},
        {"forecast_values": [0.5, None]},
        {"forecast_values": [0.2, 0.3, 0.5]},  # three values for a binary question
        {"forecast_values": [1.0]},
    ],
)
def test_malformed_community_values_are_none(latest: object) -> None:
    assert community_values(BINARY, latest) is None


# -- resolve_metaculus ---------------------------------------------------------


def _fast_limiter() -> RateLimiter:
    return RateLimiter(1000, burst=1000, max_concurrent=4)


@pytest.fixture
def fake() -> FakeMetaculus:
    return FakeMetaculus(clock=FixedClock(T0))


@pytest.fixture
async def client(fake: FakeMetaculus) -> AsyncIterator[MetaculusClient]:
    async with fake.client() as http:
        yield MetaculusClient(
            "https://www.metaculus.com/api", SecretStr(FAKE_TOKEN), http, _fast_limiter()
        )


@pytest.fixture
def live(tmp_path: Path, clock: FixedClock) -> Iterator[Ledger]:
    with Ledger.open(tmp_path / "live.sqlite", clock) as ledger:
        yield ledger


def _add(
    fake: FakeMetaculus,
    live: Ledger,
    run_id: str,
    post_id: int,
    *,
    status: str = "resolved",
    resolution: Any = None,
    **question_fields: Any,
) -> int:
    """A post on the fake server, and the same question recorded in the ledger as open."""
    post = binary_post(post_id, post_id + 1000)
    live.record_question(run_id, question_from_post(post), T0, post)
    stored = binary_post(post_id, post_id + 1000, status=status)
    stored["question"]["resolution"] = resolution
    stored["question"].update(question_fields)
    fake.add_post(stored)
    return post_id + 1000


def _run(live: Ledger, mode: RunMode = RunMode.DRY_RUN) -> str:
    return live.start_run(
        mode=mode,
        tournament="minibench",
        provider="replay",
        model="m",
        settings={},
        code_version="t",
    )


def _count(ledger: Ledger, table: str) -> int:
    return ledger.count(table)


async def test_resolve_metaculus_sorts_questions_into_each_outcome(
    fake: FakeMetaculus, client: MetaculusClient, live: Ledger
) -> None:
    run_id = _run(live)
    q_yes = _add(
        fake,
        live,
        run_id,
        1,
        resolution="yes",
        actual_resolve_time="2026-11-01T12:00:00Z",
    )
    q_hidden = _add(fake, live, run_id, 2, resolution=None)  # resolved, but hidden from bots
    _add(fake, live, run_id, 3, status="open")
    q_annulled = _add(fake, live, run_id, 4, resolution="annulled")
    _add(fake, live, run_id, 5, resolution="maybe")  # malformed
    missing = binary_post(6, 1006)
    live.record_question(run_id, question_from_post(missing), T0, missing)  # not on the server
    out: list[str] = []
    report = await resolve_metaculus(live, client, out=out.append)
    assert (report.checked, report.resolved, report.annulled) == (6, 1, 1)
    assert (report.hidden, report.open, report.unchanged, report.failed) == (1, 1, 0, 2)
    assert "hides the resolution" in " ".join(out)
    rows = live.connection.execute(
        "SELECT question_id, status, value, source, resolved_at, evidence FROM resolutions "
        "ORDER BY question_id"
    ).fetchall()
    assert [tuple(r)[:4] for r in rows] == [
        (q_yes, "resolved", "yes", "metaculus_api"),
        (q_annulled, "annulled", None, "metaculus_api"),
    ]
    assert rows[0][4] == "2026-11-01T12:00:00.000000Z"
    assert rows[1][4] is None
    assert "GET /api/posts/1/" in rows[0][5]
    assert live.latest_resolution(q_hidden) is None
    assert "1 resolved, 1 annulled, 1 resolved but hidden" in report.line()


async def test_resolving_twice_adds_no_second_resolution(
    fake: FakeMetaculus, client: MetaculusClient, live: Ledger
) -> None:
    run_id = _run(live)
    _add(fake, live, run_id, 1, resolution="no")
    first = await resolve_metaculus(live, client, out=lambda _s: None)
    second = await resolve_metaculus(live, client, out=lambda _s: None)
    assert (first.resolved, first.unchanged) == (1, 0)
    assert (second.resolved, second.unchanged) == (0, 1)
    assert _count(live, "resolutions") == 1


async def test_a_changed_resolution_is_a_new_row_and_the_latest_wins(
    fake: FakeMetaculus, client: MetaculusClient, live: Ledger
) -> None:
    run_id = _run(live)
    qid = _add(fake, live, run_id, 1, resolution="no")
    await resolve_metaculus(live, client, out=lambda _s: None)
    fake.posts[1]["question"]["resolution"] = "yes"
    again = await resolve_metaculus(live, client, out=lambda _s: None)
    assert again.resolved == 1
    assert _count(live, "resolutions") == 2
    assert live.latest_resolution(qid) == ("resolved", "yes")


async def test_backtest_questions_are_not_read_from_metaculus(
    fake: FakeMetaculus, client: MetaculusClient, live: Ledger
) -> None:
    _add(fake, live, _run(live, RunMode.BACKTEST), 1, resolution="yes")
    report = await resolve_metaculus(live, client, out=lambda _s: None)
    assert report.checked == 0
    assert fake.requests == []


async def test_the_community_aggregate_is_kept_as_a_reference(
    fake: FakeMetaculus, client: MetaculusClient, live: Ledger
) -> None:
    run_id = _run(live)
    qid = _add(
        fake,
        live,
        run_id,
        1,
        resolution="yes",
        aggregations={
            "recency_weighted": {
                "history": None,
                "latest": {
                    "start_time": 1_794_941_427.5,
                    "end_time": None,
                    "forecast_values": [0.25, 0.75],
                },
                "score_data": {"baseline_score": 41.5, "peer_score": 9.0, "coverage": 1.0},
            }
        },
    )
    report = await resolve_metaculus(live, client, out=lambda _s: None)
    assert (report.references, report.scores) == (1, 2)
    [ref] = live.connection.execute(
        "SELECT question_id, forecaster, probability, values_json, as_of, source, detail "
        "FROM reference_forecasts"
    ).fetchall()
    assert ref[:3] == (qid, "community", 0.75)
    assert ref[3] == '{"probability": 0.75}'
    assert ref[4] == datetime.fromtimestamp(1_794_941_427.5, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    assert "recency_weighted" in ref[5]
    assert "recency_weighted aggregate" in ref[6]
    scores = live.connection.execute(
        "SELECT forecaster, score_type, score FROM metaculus_scores ORDER BY score_type"
    ).fetchall()
    assert [tuple(s) for s in scores] == [
        ("community", "recency_weighted:baseline_score", 41.5),
        ("community", "recency_weighted:peer_score", 9.0),
    ]


async def test_resolving_twice_keeps_one_copy_of_each_reference_and_score(
    fake: FakeMetaculus, client: MetaculusClient, live: Ledger
) -> None:
    run_id = _run(live)
    _add(
        fake,
        live,
        run_id,
        1,
        resolution=None,
        aggregations={
            "recency_weighted": {
                "latest": {"start_time": 1_794_941_427.5, "forecast_values": [0.4, 0.6]},
                "score_data": {"baseline_score": 12.0},
            }
        },
    )
    await resolve_metaculus(live, client, out=lambda _s: None)
    again = await resolve_metaculus(live, client, out=lambda _s: None)
    assert (again.references, again.scores) == (0, 0)
    assert _count(live, "reference_forecasts") == 1
    assert _count(live, "metaculus_scores") == 1


async def test_unweighted_is_used_when_recency_weighted_is_absent(
    fake: FakeMetaculus, client: MetaculusClient, live: Ledger
) -> None:
    _add(
        fake,
        live,
        _run(live),
        1,
        resolution="yes",
        aggregations={
            "unweighted": {
                "latest": {"start_time": "2026-10-01T00:00:00Z", "forecast_values": [0.6, 0.4]}
            }
        },
    )
    await resolve_metaculus(live, client, out=lambda _s: None)
    [(p, as_of)] = live.connection.execute(
        "SELECT probability, as_of FROM reference_forecasts"
    ).fetchall()
    assert p == 0.4
    assert as_of == "2026-10-01T00:00:00.000000Z"


async def test_this_accounts_metaculus_scores_are_kept(
    fake: FakeMetaculus, client: MetaculusClient, live: Ledger
) -> None:
    qid = _add(
        fake,
        live,
        _run(live),
        1,
        resolution="yes",
        my_forecasts={
            "history": [],
            "latest": None,
            "score_data": {
                "baseline_score": 22.0,
                "spot_peer_score": -3.5,
                "spot_baseline_score": 20.0,
                "coverage": 0.9,  # not a score
                "note": "text",
            },
        },
    )
    report = await resolve_metaculus(live, client, out=lambda _s: None)
    assert report.scores == 3
    rows = live.connection.execute(
        "SELECT question_id, forecaster, score_type, score FROM metaculus_scores "
        "ORDER BY score_type"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        (qid, "augur", "baseline_score", 22.0),
        (qid, "augur", "spot_baseline_score", 20.0),
        (qid, "augur", "spot_peer_score", -3.5),
    ]


async def test_references_are_kept_even_before_the_question_resolves(
    fake: FakeMetaculus, client: MetaculusClient, live: Ledger
) -> None:
    _add(
        fake,
        live,
        _run(live),
        1,
        status="open",
        aggregations={"recency_weighted": {"latest": {"forecast_values": [0.5, 0.5]}}},
    )
    report = await resolve_metaculus(live, client, out=lambda _s: None)
    assert (report.open, report.resolved, report.references) == (1, 0, 1)
    assert _count(live, "resolutions") == 0
    [(as_of,)] = live.connection.execute("SELECT as_of FROM reference_forecasts").fetchall()
    assert as_of.endswith("Z")  # falls back to the close time when there is no start time


# -- resolve_forecastbench -----------------------------------------------------


async def _backtest(tmp_path: Path) -> tuple[Path, Path, list[int]]:
    _qs, rs, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    await run_backtest(db, questions)
    return db, rs, [q.question_id for q in questions]


async def test_resolve_forecastbench_writes_resolutions_and_market_references(
    tmp_path: Path, clock: FixedClock
) -> None:
    db, rs, ids = await _backtest(tmp_path)
    with Ledger.open(db, clock) as ledger:
        report = resolve_forecastbench(ledger, rs)
        assert (report.checked, report.resolved, report.open, report.unchanged) == (3, 3, 0, 0)
        assert (report.references, report.failed) == (3, 0)
        assert [ledger.latest_resolution(i) for i in ids] == [
            ("resolved", "yes"),
            ("resolved", "no"),
            ("resolved", "no"),
        ]
        rows = ledger.connection.execute(
            "SELECT question_id, forecaster, probability, as_of, detail, source "
            "FROM reference_forecasts ORDER BY rowid"
        ).fetchall()
        # Questions of one freeze date share a recorded_at, so row order is not defined.
        assert {r[0]: (r[1], r[2]) for r in rows} == {
            ids[0]: ("market", 0.8),
            ids[1]: ("market", 0.3),
            ids[2]: ("market", 0.6),
        }
        assert {r[3] for r in rows} == {"2025-03-01T00:00:00.000000Z"}
        details = {r[0]: r[4] for r in rows}
        assert "polymarket price at the freeze date" in details[ids[1]]
        assert "not shown to the forecaster" in details[ids[0]]
        res = ledger.connection.execute(
            "SELECT question_id, source, resolved_at, evidence FROM resolutions"
        ).fetchall()
        assert {r[1] for r in res} == {"forecastbench"}
        assert {r[2] for r in res} == {"2025-04-01T00:00:00.000000Z"}
        evidence = {r[0]: r[3] for r in res}
        assert "manifold a resolved_to 1.0 on 2025-04-01" in evidence[ids[0]]


async def test_resolve_forecastbench_twice_adds_nothing(tmp_path: Path, clock: FixedClock) -> None:
    db, rs, _ = await _backtest(tmp_path)
    with Ledger.open(db, clock) as ledger:
        resolve_forecastbench(ledger, rs)
        again = resolve_forecastbench(ledger, rs)
        assert (again.resolved, again.unchanged, again.references) == (0, 3, 0)
        assert ledger.count("resolutions") == 3
        assert ledger.count("reference_forecasts") == 3


async def test_resolve_forecastbench_leaves_unresolved_questions_open(
    tmp_path: Path, clock: FixedClock
) -> None:
    db, rs, ids = await _backtest(tmp_path)
    import json

    data = json.loads(rs.read_text(encoding="utf-8"))
    data["resolutions"][0] = raw_resolution("a", 0.4)  # a price, not an outcome
    data["resolutions"][1]["resolved"] = False
    data["resolutions"].pop(2)  # not in the set at all
    rs.write_text(json.dumps(data), encoding="utf-8")
    with Ledger.open(db, clock) as ledger:
        report = resolve_forecastbench(ledger, rs)
        assert (report.checked, report.resolved, report.open) == (3, 0, 3)
        assert ledger.count("resolutions") == 0
        assert ledger.latest_resolution(ids[0]) is None


async def test_resolve_forecastbench_ignores_non_forecastbench_questions(
    tmp_path: Path, clock: FixedClock
) -> None:
    db, rs, _ = await _backtest(tmp_path)
    with Ledger.open(db, clock) as ledger:
        run_id = _run(ledger, RunMode.BACKTEST)
        post = binary_post(9, 90)
        ledger.record_question(run_id, question_from_post(post), T0, post)
        report = resolve_forecastbench(ledger, rs)
        assert report.checked == 3


async def test_resolve_forecastbench_without_a_price_records_no_reference(
    tmp_path: Path, clock: FixedClock
) -> None:
    from augur.forecastbench import load_set

    from .fakes.forecastbench import RELEASED, raw_question, write_pair

    qs, rs = write_pair(tmp_path, [raw_question("n", value="N/A")], [raw_resolution("n", 1.0)])
    db = tmp_path / "np.sqlite"
    await run_backtest(db, load_set(qs, rs, released=RELEASED).kept)
    with Ledger.open(db, clock) as ledger:
        report = resolve_forecastbench(ledger, rs)
        assert (report.resolved, report.references) == (1, 0)


# -- resolve_manual ------------------------------------------------------------


def test_manual_resolution_records_value_and_evidence(live: Ledger) -> None:
    run_id = _run(live, RunMode.LIVE)
    post = binary_post(7, 70)
    live.record_question(run_id, question_from_post(post), T0, post)
    assert resolve_manual(live, post_id=7, value="Yes", evidence="https://example.org/q") == (
        "resolved",
        "yes",
    )
    [row] = live.connection.execute(
        "SELECT question_id, post_id, status, value, source, evidence, resolved_at FROM resolutions"
    ).fetchall()
    assert tuple(row) == (70, 7, "resolved", "yes", "manual", "https://example.org/q", None)
    assert resolve_manual(live, post_id=7, value="annulled", evidence="the page") == (
        "annulled",
        None,
    )
    assert live.latest_resolution(70) == ("annulled", None)


def test_manual_resolution_needs_evidence(live: Ledger) -> None:
    run_id = _run(live, RunMode.LIVE)
    post = binary_post(7, 70)
    live.record_question(run_id, question_from_post(post), T0, post)
    for blank in ("", "   "):
        with pytest.raises(AugurError, match="needs --evidence"):
            resolve_manual(live, post_id=7, value="yes", evidence=blank)
    assert _count(live, "resolutions") == 0


def test_manual_resolution_needs_the_post_in_the_ledger(live: Ledger) -> None:
    with pytest.raises(AugurError, match="post 7 is not in this ledger"):
        resolve_manual(live, post_id=7, value="yes", evidence="x")


def test_manual_resolution_rejects_a_bad_value(live: Ledger) -> None:
    run_id = _run(live, RunMode.LIVE)
    post = binary_post(7, 70)
    live.record_question(run_id, question_from_post(post), T0, post)
    with pytest.raises(AugurError, match="resolved to"):
        resolve_manual(live, post_id=7, value="maybe", evidence="x")
    with pytest.raises(AugurError, match="give the value"):
        resolve_manual(live, post_id=7, value="", evidence="x")
    assert _count(live, "resolutions") == 0


async def test_manual_resolution_of_a_forecastbench_question_is_a_clear_error(
    tmp_path: Path, clock: FixedClock
) -> None:
    db, _, ids = await _backtest(tmp_path)
    with Ledger.open(db, clock) as ledger, closing(sqlite3.connect(db)) as conn:
        [(post_id,)] = conn.execute(
            "SELECT post_id FROM questions WHERE question_id = ?", (ids[0],)
        ).fetchall()
        # A backtest question's payload is not a Metaculus post: this must be an AugurError
        # (the CLI turns those into exit code 2), not a bare ValueError traceback.
        with pytest.raises(AugurError):
            resolve_manual(ledger, post_id=post_id, value="yes", evidence="x")
