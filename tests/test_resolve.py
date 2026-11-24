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

from augur.cli import main
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
    resolve_manual_file,
    resolve_metaculus,
)
from augur.standings import snapshot_text

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
from .helpers import record_aggregate, record_template, round_snapshot


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


# -- resolve_manual_file -------------------------------------------------------

ROUND = [11, 12, 13, 14]
"""The round's posts: Augur and the template forecast 11 and 12, the template alone 13;
14 is in the round but was never forecast; 15 is in the ledger but not in the round."""
URL = "https://www.metaculus.com/questions/{}/"


def _round_ledger(live: Ledger) -> None:
    run_id = _run(live, RunMode.DRY_RUN)
    for post_id in (11, 12, 13, 14, 15):
        post = binary_post(post_id, post_id + 1000)
        q = question_from_post(post)
        live.record_question(run_id, q, T0, post)
        if post_id in (11, 12, 15):
            record_aggregate(live, run_id, q, {"probability": 0.6})
        if post_id in (11, 12, 13):
            record_template(live, q, {"probability": 0.5})


def _csv(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "resolutions.csv"
    path.write_text("\n".join(["post_id,value,evidence", *lines]) + "\n", encoding="utf-8")
    return path


def test_a_complete_file_writes_every_row(live: Ledger, tmp_path: Path) -> None:
    _round_ledger(live)
    path = _csv(
        tmp_path, f"11,yes,{URL.format(11)}", f"12,No,{URL.format(12)}", f"13,annulled,{URL}"
    )
    report = resolve_manual_file(live, path, ROUND)
    assert [(m.post_id, m.status, m.value) for m in report.written] == [
        (11, "resolved", "yes"),
        (12, "resolved", "no"),
        (13, "annulled", None),
    ]
    assert (report.round_questions, report.already_resolved) == (3, 0)
    rows = live.connection.execute(
        "SELECT post_id, status, value, source, evidence FROM resolutions ORDER BY post_id"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        (11, "resolved", "yes", "manual", URL.format(11)),
        (12, "resolved", "no", "manual", URL.format(12)),
        (13, "annulled", None, "manual", URL),
    ]


def test_a_partial_file_writes_nothing_and_names_the_missing_posts(
    live: Ledger, tmp_path: Path
) -> None:
    _round_ledger(live)
    path = _csv(tmp_path, f"12,yes,{URL}")
    with pytest.raises(AugurError, match=r"post\(s\) 11, 13; add a line for each"):
        resolve_manual_file(live, path, ROUND)
    assert _count(live, "resolutions") == 0


def test_a_bad_value_on_line_3_writes_nothing(live: Ledger, tmp_path: Path) -> None:
    _round_ledger(live)
    path = _csv(tmp_path, f"11,yes,{URL}", f"12,maybe,{URL}", f"13,no,{URL}")
    with pytest.raises(AugurError, match=r"line 3: .*resolved to 'maybe'.*nothing was written"):
        resolve_manual_file(live, path, ROUND)
    assert _count(live, "resolutions") == 0


@pytest.mark.parametrize("evidence", ["", "  ", "read it on the page"])
def test_a_line_without_an_evidence_url_is_refused(
    live: Ledger, tmp_path: Path, evidence: str
) -> None:
    _round_ledger(live)
    path = _csv(tmp_path, f"11,yes,{URL}", f"12,yes,{evidence}", f"13,no,{URL}")
    with pytest.raises(AugurError, match="line 3: post 12 needs an evidence URL"):
        resolve_manual_file(live, path, ROUND)
    assert _count(live, "resolutions") == 0


def test_a_post_already_resolved_need_not_be_in_the_file(live: Ledger, tmp_path: Path) -> None:
    _round_ledger(live)
    resolve_manual(live, post_id=11, value="yes", evidence=URL)
    path = _csv(tmp_path, f"12,no,{URL}", f"13,no,{URL}")
    report = resolve_manual_file(live, path, ROUND)
    assert (len(report.written), report.round_questions, report.already_resolved) == (2, 3, 1)
    assert _count(live, "resolutions") == 3


def test_a_post_twice_is_refused(live: Ledger, tmp_path: Path) -> None:
    _round_ledger(live)
    path = _csv(tmp_path, f"11,yes,{URL}", f"12,no,{URL}", f"13,no,{URL}", f"11,no,{URL}")
    with pytest.raises(AugurError, match="line 5: post 11 is already on line 2"):
        resolve_manual_file(live, path, ROUND)
    assert _count(live, "resolutions") == 0


def test_a_post_outside_the_round_or_the_ledger_is_refused(live: Ledger, tmp_path: Path) -> None:
    _round_ledger(live)
    full = [f"11,yes,{URL}", f"12,no,{URL}", f"13,no,{URL}"]
    with pytest.raises(AugurError, match="post 15 is not in the round's standings snapshot"):
        resolve_manual_file(live, _csv(tmp_path, *full, f"15,no,{URL}"), ROUND)
    with pytest.raises(AugurError, match="post 99 is not in this ledger"):
        resolve_manual_file(live, _csv(tmp_path, *full, f"99,no,{URL}"), [*ROUND, 99])
    assert _count(live, "resolutions") == 0


def test_a_file_without_the_header_is_refused(live: Ledger, tmp_path: Path) -> None:
    _round_ledger(live)
    path = tmp_path / "bad.csv"
    path.write_text(f"11,yes,{URL}\n", encoding="utf-8")
    with pytest.raises(AugurError, match="header post_id,value,evidence"):
        resolve_manual_file(live, path, ROUND)


def test_cli_manual_file_needs_standings_and_excludes_manual(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _csv(tmp_path, f"11,yes,{URL}")
    with pytest.raises(SystemExit):
        main(["resolve", "--manual-file", str(path)])
    assert "--manual-file and --standings go together" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["resolve", "--manual-file", str(path), "--manual", "11", "yes"])
    assert "not allowed with argument" in capsys.readouterr().err


def test_cli_manual_file_writes_the_round(
    tmp_path: Path, clock: FixedClock, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "shadow.sqlite"
    with Ledger.open(db, clock) as ledger:
        _round_ledger(ledger)
    snapshot = tmp_path / "round.json"
    snapshot.write_text(snapshot_text(round_snapshot(ROUND)), encoding="utf-8")
    partial = _csv(tmp_path, f"11,yes,{URL}")
    args = ["resolve", "--ledger", str(db), "--standings", str(snapshot), "--manual-file"]
    assert main([*args, str(partial)]) == 2
    assert "post(s) 12, 13" in capsys.readouterr().err
    full = _csv(tmp_path, f"11,yes,{URL}", f"12,no,{URL}", f"13,annulled,{URL}")
    assert main([*args, str(full)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == ["post 11: resolved yes", "post 12: resolved no", "post 13: annulled"]
    assert out[3].startswith("3 resolution(s) written from ")
    with Ledger.open(db, clock) as ledger:
        assert _count(ledger, "resolutions") == 3
