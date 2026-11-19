from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from augur.baseline import (
    BaselineUnavailableError,
    TemplateFailedError,
    TemplateForecast,
    build_template_runner,
    from_template_prediction,
    litellm_model,
    questions_to_shadow,
    research_mode,
    run_shadow,
    template_values,
)
from augur.cli import main
from augur.clock import FixedClock
from augur.config import Settings
from augur.distributions import cdf_from_percentiles
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.llm.endpoints import parse_route
from augur.questions import Question, question_from_post

from .conftest import T0
from .helpers import binary_question

FIXTURES = Path(__file__).parent / "fixtures" / "metaculus"


def _fixture(name: str) -> Question:
    post: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return question_from_post(post)


def _binary() -> Question:
    return binary_question(T0, post_id=43327, question_id=42808)[0]


def _choice() -> Question:
    return _fixture("post_46223_bot_testing_multiple_choice_detail.json")


def _numeric() -> Question:
    return _fixture("post_43323_bot_testing_numeric_detail.json")


def _discrete() -> Question:
    return _fixture("post_43321_bot_testing_discrete_detail.json")


def _cdf(q: Question, percentiles: list[tuple[float, float]]) -> list[float]:
    assert q.scaling is not None
    return cdf_from_percentiles(percentiles, q.scaling)


class FakeRunner:
    """Answers from a table by post id; an exception in the table is raised."""

    def __init__(self, answers: dict[int, TemplateForecast | Exception]) -> None:
        self.answers = answers
        self.calls: list[int] = []

    async def forecast(self, post_id: int) -> TemplateForecast:
        self.calls.append(post_id)
        answer = self.answers[post_id]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _template_rows(ledger: Ledger) -> list[tuple[int, str, str, str, float | None]]:
    rows = ledger.connection.execute(
        "SELECT post_id, question_type, values_json, source, probability "
        "FROM reference_forecasts WHERE forecaster = 'template' ORDER BY rowid"
    ).fetchall()
    return [tuple(r) for r in rows]


async def test_every_type_is_converted_and_recorded(ledger: Ledger, clock: FixedClock) -> None:
    binary, choice, numeric, discrete = _binary(), _choice(), _numeric(), _discrete()
    num_pcts = [(0.1, -20.0), (0.5, -12.0), (0.9, -5.0)]
    dis_pcts = [(0.1, 1.0), (0.5, 3.0), (0.9, 7.0)]
    runner = FakeRunner(
        {
            binary.post_id: TemplateForecast(detail="d", probability=0.23),
            choice.post_id: TemplateForecast(
                detail="d",
                probabilities={"Other": 0.02, "Republican Party": 0.48, "Democratic Party": 0.5},
            ),
            numeric.post_id: TemplateForecast(
                detail="d", cdf=_cdf(numeric, num_pcts), percentiles=num_pcts
            ),
            discrete.post_id: TemplateForecast(
                detail="d", cdf=_cdf(discrete, dis_pcts), percentiles=dis_pcts
            ),
        }
    )
    result = await run_shadow(ledger, runner, [binary, choice, numeric, discrete], clock)
    assert (result.recorded, result.skipped, result.failed) == (4, 0, 0)
    rows = _template_rows(ledger)
    assert [r[1] for r in rows] == ["binary", "multiple_choice", "numeric", "discrete"]
    assert json.loads(rows[0][2]) == {"probability": 0.23}
    assert rows[0][4] == 0.23
    assert set(json.loads(rows[1][2])["probabilities"]) == set(choice.options or [])
    numeric_values = json.loads(rows[2][2])
    assert len(numeric_values["cdf"]) == 201
    assert numeric_values["percentiles"] == [list(p) for p in num_pcts]
    assert len(json.loads(rows[3][2])["cdf"]) == 17
    assert all(r[3] == "forecasting-tools TemplateBot, unpublished" for r in rows)


async def test_a_question_with_a_template_row_is_not_run_again(
    ledger: Ledger, clock: FixedClock
) -> None:
    q = _binary()
    runner = FakeRunner({q.post_id: TemplateForecast(detail="d", probability=0.4)})
    await run_shadow(ledger, runner, [q], clock)
    again = await run_shadow(ledger, runner, [q], clock)
    assert (again.recorded, again.skipped, again.failed) == (0, 1, 0)
    assert runner.calls == [q.post_id]
    assert len(_template_rows(ledger)) == 1


async def test_one_failure_does_not_stop_the_batch(ledger: Ledger, clock: FixedClock) -> None:
    first = binary_question(T0, post_id=1, question_id=11)[0]
    second = binary_question(T0, post_id=2, question_id=12)[0]
    numeric = _numeric()
    runner = FakeRunner(
        {
            1: RuntimeError("rate limited"),
            # A CDF Metaculus would refuse is a failure, not a row.
            numeric.post_id: TemplateForecast(detail="d", cdf=[0.5] * 201),
            2: TemplateForecast(detail="d", probability=0.7),
        }
    )
    result = await run_shadow(ledger, runner, [first, numeric, second], clock)
    assert (result.recorded, result.skipped, result.failed) == (1, 0, 2)
    assert "rate limited" in result.lines[0]
    assert "CDF invalid" in result.lines[1]
    assert [r[0] for r in _template_rows(ledger)] == [2]


def test_values_reject_the_wrong_shape() -> None:
    with pytest.raises(TemplateFailedError, match="no probability"):
        template_values(_binary(), TemplateForecast(detail="d", cdf=[0.1]))
    with pytest.raises(TemplateFailedError, match="invalid"):
        template_values(_choice(), TemplateForecast(detail="d", probabilities={"Other": 1.0}))


@dataclass
class _Option:
    option_name: str
    probability: float


@dataclass
class _OptionList:
    predicted_options: list[_Option]


@dataclass
class _Percentile:
    percentile: float
    value: float


class _Distribution:
    """Shaped like forecasting-tools' ``NumericDistribution``."""

    def __init__(self, cdf: list[float], declared: list[_Percentile]) -> None:
        self._cdf = cdf
        self.declared_percentiles = declared

    def get_cdf(self) -> list[_Percentile]:
        return [_Percentile(p, float(i)) for i, p in enumerate(self._cdf)]


def test_template_predictions_are_read_by_shape() -> None:
    assert from_template_prediction(0.31, "d").probability == 0.31
    choice = from_template_prediction(_OptionList([_Option("A", 0.6), _Option("B", 0.4)]), "d")
    assert choice.probabilities == {"A": 0.6, "B": 0.4}
    numeric = _numeric()
    cdf = _cdf(numeric, [(0.1, -20.0), (0.9, -5.0)])
    dist = from_template_prediction(
        _Distribution(cdf, [_Percentile(0.1, -20.0), _Percentile(0.9, -5.0)]), "d"
    )
    assert dist.cdf == cdf
    assert dist.percentiles == [(0.1, -20.0), (0.9, -5.0)]
    assert template_values(numeric, dist)["cdf"] == cdf
    with pytest.raises(TemplateFailedError):
        from_template_prediction(True, "d")
    with pytest.raises(TemplateFailedError):
        from_template_prediction("0.3", "d")


def test_questions_to_shadow_are_those_with_an_aggregate_and_no_template_row(
    ledger: Ledger,
) -> None:
    run = ledger.start_run(
        mode=RunMode.DRY_RUN,
        tournament="minibench",
        provider="replay",
        model="m",
        settings={},
        code_version="test",
    )
    done, todo, unforecast = (binary_question(T0, post_id=p, question_id=p + 10) for p in (1, 2, 3))
    for q, post in (done, todo, unforecast):
        ledger.record_question(run, q, T0, post)
    for q, _ in (done, todo):
        ledger.record_forecast(
            run_id=run,
            question_id=q.question_id,
            kind="aggregate",
            member=None,
            method="median",
            probability=0.5,
            reasoning="",
            call_id=None,
        )
    ledger.record_reference(
        question_id=done[0].question_id,
        post_id=done[0].post_id,
        question_type="binary",
        forecaster="template",
        detail="d",
        values={"probability": 0.5},
        as_of=T0,
        source="s",
    )
    assert [q.post_id for q in questions_to_shadow(ledger)] == [2]


def test_routes_map_to_litellm_without_keys() -> None:
    settings = Settings.from_env({})
    endpoints = settings._endpoints()
    model, extra = litellm_model(parse_route("openrouter:openai/gpt-5.4@high", endpoints))
    assert (model, extra) == ("openrouter/openai/gpt-5.4", {"reasoning_effort": "high"})
    model, extra = litellm_model(parse_route("groq:llama-x", endpoints))
    assert model == "openai/llama-x"
    assert extra == {"base_url": "https://api.groq.com/openai/v1"}
    assert litellm_model(parse_route("ollama:qwen2.5:7b", endpoints))[0] == "ollama_chat/qwen2.5:7b"


def test_research_is_asknews_with_keys_and_none_without() -> None:
    assert research_mode(Settings.from_env({}))[0] == "no_research"
    with_keys = Settings.from_env({"ASKNEWS_CLIENT_ID": "i", "ASKNEWS_SECRET": "s"})
    assert research_mode(with_keys)[0] == "asknews/news-summaries"


def test_without_the_extra_the_command_says_how_to_install_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # ``None`` in sys.modules makes ``import forecasting_tools`` raise ImportError.
    monkeypatch.setitem(sys.modules, "forecasting_tools", None)
    # main() would point the logs at this test's captured stderr, closed after it.
    monkeypatch.setattr("augur.cli.configure_logging", lambda *_a, **_k: None)
    with pytest.raises(BaselineUnavailableError, match="uv sync --extra baseline"):
        build_template_runner(Settings.from_env({}))
    code = main(["baseline", "--ledger", str(tmp_path / "l.sqlite")])
    assert code == 2
    assert "uv sync --extra baseline" in capsys.readouterr().err
    assert not (tmp_path / "l.sqlite").exists()


@pytest.mark.integration
def test_the_real_template_bot_is_built_unpublished_with_the_route_model() -> None:
    pytest.importorskip("forecasting_tools")
    settings = Settings.from_env(
        {
            "METACULUS_TOKEN": "t",
            "OPENROUTER_API_KEY": "k",
            "AUGUR_FORECAST_MODELS": "openrouter:google/gemma-4-31b-it:free",
        }
    )
    runner = build_template_runner(settings)
    bot = runner._bot
    assert bot.publish_reports_to_metaculus is False
    assert bot.get_llm("default", "llm").model == "openrouter/google/gemma-4-31b-it:free"
    assert bot.get_llm("parser", "llm").model == "openrouter/google/gemma-4-31b-it:free"
    assert bot.get_llm("researcher") == "no_research"
    assert "unpublished" in runner._detail
    assert "k" not in runner._detail.split()
