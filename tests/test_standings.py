from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from augur import standings
from augur.clock import FixedClock
from augur.config import Settings
from augur.errors import ConfigError
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.llm.catalog import Catalog
from augur.metaculus import MetaculusClient
from augur.ratelimit import RateLimiter
from tests.fakes.metaculus import FAKE_TOKEN, FIXTURES, FakeMetaculus, binary_post

SLUG = "minibench-2026-11-02"
NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _leaderboards() -> list[dict[str, Any]]:
    data = _fixture("leaderboard_minibench_2026-11-02_trimmed.json")
    assert isinstance(data, list)
    return data


def _round(
    username: str | None = None,
    extra: list[dict[str, Any]] | None = None,
    questions: list[standings.RoundQuestion] | None = None,
) -> standings.RoundStandings:
    board = _leaderboards()[0]
    board["entries"] = [*board["entries"], *(extra or [])]
    return standings.RoundStandings(
        round=standings.RoundInfo.model_validate(_fixture("tournament_minibench_2026-11-02.json")),
        leaderboard_raw=board,
        questions=questions or [],
        fetched_at=NOW,
        source_url="https://example.test/api/leaderboards/project/33135/",
        augur_username=username,
    )


def _entry(username: str, score: float, rank: int, coverage: float = 59) -> dict[str, Any]:
    return {
        "user": {"id": 1, "username": username, "is_bot": True},
        "aggregation_method": None,
        "score": score,
        "ci_lower": score - 59,
        "ci_upper": score + 118,
        "rank": rank,
        "excluded": False,
        "exclusion_status": 0,
        "coverage": coverage,
        "contribution_count": coverage,
    }


def test_average_and_interval_are_the_sum_divided_by_coverage() -> None:
    board = _round().board()
    top = standings.rank_one(board)
    assert top is not None
    assert top.name == "takeoff.watch"
    assert top.rank == 1
    assert top.coverage == 59
    assert top.average == pytest.approx(1343.428126301855 / 59)
    assert top.average == pytest.approx(22.7699, abs=1e-4)
    lo, hi = top.average_interval or (0.0, 0.0)
    assert lo == pytest.approx(1024.3336914727163 / 59)
    assert hi == pytest.approx(1713.873517834211 / 59)
    assert (round(lo, 2), round(hi, 2)) == (17.36, 29.05)


def test_fractional_and_zero_coverage() -> None:
    board = standings.Leaderboard.model_validate(
        {"entries": [_entry("a", 10.0, 1, coverage=2.5), _entry("b", 0.0, 2, coverage=0)]}
    )
    a, b = board.entries
    assert a.average == pytest.approx(4.0)
    assert a.average_interval == pytest.approx((-49.0 / 2.5, 128.0 / 2.5))
    assert b.average is None
    assert b.average_interval is None


def test_classification() -> None:
    cases = {
        "metac-claude-opus-5-5-high+asknews": "Metaculus template bot",
        "metac-gpt-4-1+asknews[optimized-prompt]": "Metaculus template bot (experiment variant)",
        "metac-deepseek-v3-1+asknews(variance-test-1)": (
            "Metaculus template bot (experiment variant)"
        ),
        "metac-agent-claude-opus-5-5": "Metaculus house bot (not the template)",
        "metac-autocast": "Metaculus house bot (not the template)",
        "metac-claude-code-latest": "Metaculus house bot (not the template)",
        "metac-asknews-deepnews[research-only]": "Metaculus house bot (not the template)",
        "metac-uniform-probability-bot": "Metaculus house bot (not the template)",
        "metaculus-superforecaster": "Metaculus house bot (not the template)",
        "nostreambot": "open-source bot",
        "Panshul42": "open-source bot",
        "Augur-bot-1": "Augur",
        "takeoff.watch": "bot",
    }
    for name, kind in cases.items():
        entry = standings.Entry.model_validate(_entry(name, 1.0, 1))
        assert standings.classify(entry, "Augur-bot-1") == kind, name


def test_picks_from_the_real_trimmed_leaderboard() -> None:
    board = _round().board()
    template = standings.best_template(board)
    house = standings.best_house_bot(board)
    assert template is not None
    assert template.name == "metac-claude-opus-5-5-high+asknews"
    assert template.rank == 107
    assert house is not None
    assert house.name == "metac-autocast-experimental"
    assert house.rank == 3
    assert standings.best_open_source(board) is None
    methods = [e.aggregation_method for e in standings.community_aggregates(board)]
    assert methods == ["unweighted", "recency_weighted", "reputation_weighted"]
    unweighted = standings.community_aggregates(board)[0]
    assert unweighted.average == pytest.approx(298.63263246413703 / 59)


def test_open_source_and_augur_rows() -> None:
    extra = [_entry("nostreambot", 500.0, 40), _entry("Augur-bot-1", 100.0, 150)]
    text = standings.render(_round("Augur-bot-1", extra))
    board = _round("Augur-bot-1", extra).board()
    best = standings.best_open_source(board)
    assert best is not None
    assert best.name == "nostreambot"
    mine = standings.augur_entry(board, "Augur-bot-1")
    assert mine is not None
    assert mine.rank == 150
    assert "| Augur | 150 | Augur-bot-1 | Augur | 59 | 1.69 [0.69, 3.69] | 100.0 |" in text
    assert "no listed open-source bot" not in text


def test_missing_augur_account_and_unknown_username() -> None:
    absent = standings.render(_round("Augur-bot-1"))
    assert "Augur (Augur-bot-1) is not on this round's leaderboard" in absent
    assert "no listed open-source bot in this round" in absent
    assert "| Rank 1 | 1 | takeoff.watch | bot | 59 | 22.77 [17.36, 29.05] | 1343.4 |" in absent
    unknown = standings.render(_round(None))
    assert "account unknown" in unknown
    assert FAKE_TOKEN not in absent


def test_empty_leaderboard_renders() -> None:
    board = _leaderboards()[0]
    board["entries"] = []
    board["max_coverage"] = 0
    round_ = _round().model_copy(update={"leaderboard_raw": board})
    text = standings.render(round_)
    assert "no entries yet" in text


def _outcome(
    ledger: Ledger, run: str, qid: int, status: str, reason: str, submitted: bool = False
) -> None:
    ledger.record_outcome(
        run_id=run,
        question_id=qid,
        post_id=qid - 1000,
        question_type="binary",
        status=status,
        reason=reason,
        submitted=submitted,
        closes_at=None,
        minutes_left=None,
        cost_usd=0.0,
        seconds=None,
    )


def test_missed_questions_are_explained_from_the_ledger(tmp_path: Path) -> None:
    clock = FixedClock(NOW)
    path = tmp_path / "l.sqlite"
    with Ledger.open(path, clock) as ledger:
        run = ledger.start_run(
            mode=RunMode.LIVE,
            tournament="minibench",
            provider="p",
            model="m",
            settings={},
            code_version="t",
        )
        _outcome(ledger, run, 2001, "forecast", "ok", submitted=True)
        _outcome(ledger, run, 2002, "skipped", "below the submit floor")
        clock.advance(60)
        _outcome(ledger, run, 2002, "missed", "closed before the run reached it")
        _outcome(ledger, run, 2003, "forecast", "ok", submitted=False)
        # a later "already forecast" skip must not hide the submitted forecast
        _outcome(ledger, run, 2001, "skipped", "already forecast")
    qs = [
        standings.RoundQuestion(post_id=1001 + i, question_id=2001 + i, title=f"Q{i}", status="x")
        for i in range(4)
    ]
    outcomes = standings.read_outcomes([path])
    text = standings.render(_round("Augur-bot-1", questions=qs), outcomes)
    assert "1 of 4 questions forecast and submitted; 3 not forecast." in text
    assert "Q1: missed: closed before the run reached it" in text
    assert "Q2: forecast in a dry run, not submitted" in text
    assert "Q3: no run saw it" in text
    assert "Q0" not in text
    assert "No ledger given" in standings.render(_round(questions=qs))


def test_read_outcomes_refuses_a_missing_ledger(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        standings.read_outcomes([tmp_path / "nope.sqlite"])


def test_snapshot_round_trips_byte_stable(tmp_path: Path) -> None:
    round_ = _round(
        "Augur-bot-1",
        questions=[standings.RoundQuestion(post_id=1, question_id=2, title="t", status="resolved")],
    )
    text = standings.snapshot_text(round_)
    path = tmp_path / "s.json"
    path.write_text(text, encoding="utf-8")
    loaded = standings.load_snapshot(path)
    assert standings.snapshot_text(loaded) == text
    assert standings.render(loaded) == standings.render(round_)
    assert text.endswith("}\n")
    assert '\n "fetched_at"' in text
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError):
        standings.load_snapshot(path)


class _Server(FakeMetaculus):
    """The fake plus the leaderboard, tournament and ``users/me`` routes."""

    def __init__(self, username: str | None = "Augur-bot-1") -> None:
        super().__init__()
        self.username = username

    def handle(self, request: httpx.Request) -> httpx.Response:
        route = request.url.path.removeprefix("/api")
        if route in (
            "/users/me/",
            f"/projects/tournaments/{SLUG}/",
            "/leaderboards/project/33135/",
        ):
            self.requests.append(request)
            if request.headers.get("Authorization") != f"Token {FAKE_TOKEN}":
                return httpx.Response(403, json={"detail": "no"})
            if route == "/users/me/":
                if self.username is None:
                    return httpx.Response(404, json={"detail": "no"})
                return httpx.Response(200, json={"id": 9, "username": self.username})
            if route.startswith("/projects"):
                return httpx.Response(200, json=_fixture("tournament_minibench_2026-11-02.json"))
            return httpx.Response(200, json=_leaderboards())
        return super().handle(request)


def _client(server: FakeMetaculus) -> tuple[MetaculusClient, httpx.AsyncClient]:
    http = server.client()
    limiter = RateLimiter(rate_per_second=1000.0, burst=10, max_concurrent=4)
    return (
        MetaculusClient("https://www.metaculus.com/api", SecretStr(FAKE_TOKEN), http, limiter),
        http,
    )


async def test_fetch_round_over_fake_http() -> None:
    server = _Server()
    for i in range(3):
        post = binary_post(1001 + i, 2001 + i, title=f"Q{i}", status="resolved")
        server.add_post(post, SLUG)
    client, http = _client(server)
    async with http:
        user = await standings.resolve_username(client, {})
        round_ = await standings.fetch_round(client, SLUG, FixedClock(NOW), augur_username=user)
    assert user == "Augur-bot-1"
    assert round_.fetched_at == NOW
    assert round_.round.id == 33135
    assert round_.source_url == "https://www.metaculus.com/api/leaderboards/project/33135/"
    assert [q.post_id for q in round_.questions] == [1001, 1002, 1003]
    assert round_.questions[0].question_id == 2001
    assert len(round_.board().entries) == len(_leaderboards()[0]["entries"])
    assert FAKE_TOKEN not in standings.snapshot_text(round_)


async def test_username_falls_back_to_the_environment() -> None:
    server = _Server(username=None)
    client, http = _client(server)
    async with http:
        assert await standings.resolve_username(client, {"AUGUR_BOT_USERNAME": "me-bot"}) == (
            "me-bot"
        )
        assert await standings.resolve_username(client, {}) is None


async def test_main_saves_and_renders_a_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = _Server()
    server.add_post(binary_post(1001, 2001, title="Q0", status="resolved"), SLUG)
    real = httpx.AsyncClient

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=server.transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    settings = Settings(metaculus_token=SecretStr(FAKE_TOKEN))
    out = tmp_path / "s.json"
    args = argparse.Namespace(round_slug=SLUG, from_path=None, ledger=[], out=out)
    assert await standings.main(args, settings) == 0
    printed = capsys.readouterr().out
    assert "takeoff.watch" in printed
    assert "Augur (Augur-bot-1) is not on" in printed
    again = argparse.Namespace(round_slug=None, from_path=out, ledger=[], out=None)
    assert await standings.main(again, settings) == 0
    assert capsys.readouterr().out == printed
    assert FAKE_TOKEN not in out.read_text(encoding="utf-8")


# -- template bots, the agent harness and the template-based floor -----------------------


def _board(entries: list[tuple[str, float, float]]) -> standings.RoundStandings:
    raw = {
        "entries": [
            {
                "user": {"id": i, "username": name, "is_bot": True},
                "score": score,
                "coverage": cov,
                "rank": i,
                "ci_lower": score - 10,
                "ci_upper": score + 10,
            }
            for i, (name, score, cov) in enumerate(entries, start=1)
        ]
    }
    return standings.RoundStandings(
        round=standings.RoundInfo(id=1, slug="r", name="R"),
        leaderboard_raw=raw,
        questions=[],
        fetched_at=NOW,
        source_url="https://example.test/lb",
    )


def _catalog() -> Catalog:
    payload = {
        "data": [
            {
                "id": "anthropic/claude-opus-5.5",
                "created": 1_771_804_800,
                "pricing": {"prompt": "0.000004", "completion": "0.00002"},
            },
            {
                "id": "anthropic/claude-opus-5.5:batch",
                "created": 1_771_804_800,
                "pricing": {"prompt": "0.000002", "completion": "0.00001"},
            },
        ]
    }
    return Catalog.from_api(payload, NOW, "snapshot")


_SMALL = [
    ("metac-agent-claude-opus-5-5", 600.0, 60),
    ("metac-claude-opus-5-5-high+asknews", 120.0, 60),
    ("metac-claude-code-latest", 500.0, 60),
    ("metac-agent-grok-4-7", 300.0, 60),
    ("metac-grok-4-7-high+asknews", -60.0, 60),
    ("metac-agent-orphan", 100.0, 50),
    ("metac-gpt-4-1+asknews[optimized-prompt]", 10.0, 60),
    ("metac-gpt-4-1+asknews", 0.0, 60),
    ("metac-autocast", 900.0, 60),
]


def test_template_names_exclude_agents_and_other_house_bots() -> None:
    assert standings.template_parts("metac-claude-opus-5-5-high+asknews") == (
        "claude-opus-5-5-high",
        "",
    )
    assert standings.template_parts("metac-gpt-4-1+asknews[optimized-prompt]") == (
        "gpt-4-1",
        "[optimized-prompt]",
    )
    for name in ("metac-agent-grok-4-7", "metac-claude-code-latest", "metac-autocast"):
        assert standings.template_parts(name) is None


def test_template_table_sorted_by_average_with_variants_marked() -> None:
    rows = standings.template_table(_board([(n, s, c) for n, s, c in _SMALL]))
    assert [r.name for r in rows] == [
        "metac-claude-opus-5-5-high+asknews",
        "metac-gpt-4-1+asknews[optimized-prompt]",
        "metac-gpt-4-1+asknews",
        "metac-grok-4-7-high+asknews",
    ]
    assert rows[0].average == 2.0
    assert rows[0].key == "claude-opus-5-5"
    assert [r.variant for r in rows] == ["", "[optimized-prompt]", "", ""]


def test_agent_vs_template_matches_through_the_effort_suffix_and_names_the_rest() -> None:
    pairs, lone_agents, lone_templates = standings.agent_vs_template(
        _board([(n, s, c) for n, s, c in _SMALL])
    )
    assert [(p.model, round(p.difference, 6)) for p in pairs] == [
        ("claude-opus-5-5", 8.0),  # 10.0 - 2.0
        ("grok-4-7", 6.0),  # 5.0 - (-1.0)
    ]
    assert lone_agents == ["metac-agent-orphan"]
    assert lone_templates == ["metac-gpt-4-1+asknews"]  # the variant never pairs or lists


def test_floor_admits_at_zero_and_above_and_rejects_below() -> None:
    admitted, rejected, frontier = standings.admitted_by_template(
        _board([(n, s, c) for n, s, c in _SMALL]), _catalog(), draws=5
    )
    assert [a.template.key for a in admitted] == ["claude-opus-5-5", "gpt-4-1"]  # 0.0 admitted
    assert [r.key for r in rejected] == ["grok-4-7"]
    by_name = dict(frontier)
    assert by_name["claude-opus-5-5"] is not None
    assert by_name["gpt-5"] is None


def test_worst_case_cost_arithmetic_and_unlisted_models() -> None:
    from augur.agents.forecaster import FORECAST_MAX_TOKENS
    from augur.agents.researcher import BRIEF_MAX_TOKENS, PLAN_MAX_TOKENS

    tokens_in, tokens_out = standings.question_tokens(5)
    assert tokens_in == 1_500 + 6_000 + 5 * 3_000
    assert tokens_out == PLAN_MAX_TOKENS + BRIEF_MAX_TOKENS + 5 * FORECAST_MAX_TOKENS
    admitted, _, _ = standings.admitted_by_template(
        _board([(n, s, c) for n, s, c in _SMALL]), _catalog(), draws=5
    )
    opus, gpt = admitted
    # The dearest of the two listings prices it (the :batch one is half).
    assert opus.priced_id == "anthropic/claude-opus-5.5"
    assert opus.listed_ids == ("anthropic/claude-opus-5.5", "anthropic/claude-opus-5.5:batch")
    expected = (tokens_in * 4.0 + tokens_out * 20.0) / 1e6
    assert opus.per_question == pytest.approx(expected)
    assert opus.per_round == pytest.approx(expected * 60)
    assert gpt.listed_ids == ()
    assert gpt.per_question is None
    assert gpt.per_round is None


def test_season_row_shows_the_same_model_template_and_augur() -> None:
    extra = [_entry("Augur-bot-1", 118.0, 40)]
    q = standings.RoundQuestion(post_id=7, question_id=70, title="q", status="open")
    text = standings.season_markdown([_round("Augur-bot-1", extra, [q])], {})
    row = text.splitlines()[2]
    assert "| -7.66 [-14.51, -0.73], rank 295 |" in row  # the template on the free model
    assert "| 2.00 [1.00, 4.00], rank 40 |" in row  # Augur's own row
    assert row.endswith("| 0 of 1 |")


def test_season_without_augur_or_ledgers_says_so() -> None:
    text = standings.season_markdown([_round()], None)
    row = text.splitlines()[2]
    assert row.endswith("| no entry | not read |")


def test_season_rows_are_ordered_by_start_and_empty_is_explicit() -> None:
    assert standings.season_markdown([], None) == "No standings snapshot committed.\n"
