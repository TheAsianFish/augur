"""M17: the run's mode is decided once, and the default can never post.

``scripts/run_mode.py`` is exercised over every combination of event, branch,
visibility, variables and secrets. The forecast job's two posting conditions are
mirrored here and the mirror is pinned to the workflow's text, so the exhaustive
check below reads the same expressions GitHub evaluates.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_mode.py"
WORKFLOW = ROOT / ".github" / "workflows" / "tournament.yml"
TOKEN = "tok_mode_0123456789abcdef"
KEY = "sk-or-v1-mode-0123456789abcdef"
SECRETS = {"METACULUS_TOKEN": TOKEN, "OPENROUTER_API_KEY": KEY}
MAIN = "refs/heads/main"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("augur_run_mode_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_mode = _load()


def _scheduled(**extra: str) -> dict[str, str]:
    return {"EVENT": "schedule", "REF": MAIN, "REPO_PRIVATE": "false", **SECRETS, **extra}


def test_no_variable_set_is_a_chained_minibench_shadow_tick() -> None:
    decision = run_mode.decide(_scheduled())
    assert (decision.mode, decision.schedule, decision.chain_minutes) == (
        "shadow",
        "minibench",
        "30",
    )


def test_a_chained_run_takes_the_same_default() -> None:
    env = _scheduled(EVENT="workflow_dispatch", CHAINED="true", INPUT_SHADOW="false")
    decision = run_mode.decide(env)
    assert (decision.mode, decision.chain_minutes) == ("shadow", "30")


@pytest.mark.parametrize(
    ("missing", "named"),
    [
        (("METACULUS_TOKEN",), "METACULUS_TOKEN"),
        (("OPENROUTER_API_KEY",), "OPENROUTER_API_KEY"),
        (("METACULUS_TOKEN", "OPENROUTER_API_KEY"), "METACULUS_TOKEN and OPENROUTER_API_KEY"),
    ],
)
def test_a_missing_secret_is_named_and_stops_the_run_and_the_chain(
    missing: tuple[str, ...], named: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _scheduled(GITHUB_OUTPUT=str(tmp_path / "out"))
    for name in missing:
        env[name] = "  " if name == "OPENROUTER_API_KEY" else ""
    assert run_mode.main(env) == 0
    printed = capsys.readouterr().out
    assert f"secret {named} missing" in printed
    assert (tmp_path / "out").read_text() == "mode=off\nschedule=off\nchain_minutes=\n"
    assert TOKEN not in printed
    assert KEY not in printed


def test_secret_values_are_never_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_mode.main(_scheduled(GITHUB_OUTPUT=str(tmp_path / "out")))
    printed = capsys.readouterr().out + (tmp_path / "out").read_text()
    assert TOKEN not in printed
    assert KEY not in printed
    assert "mode=shadow" in printed


@pytest.mark.parametrize(
    "env",
    [
        {"AUGUR_SCHEDULE": "off"},
        {"AUGUR_SCHEDULE": "OFF "},
        {"AUGUR_SCHEDULE": "sometimes"},
        {"AUGUR_SHADOW": "false"},
        {"AUGUR_SHADOW": "false", "AUGUR_CHAIN_MINUTES": "30"},
        {"AUGUR_SCHEDULE": "off", "AUGUR_SUBMIT": "true"},
        {"AUGUR_SCHEDULE": "off", "AUGUR_SHADOW": "true", "AUGUR_CHAIN_MINUTES": "30"},
    ],
)
def test_the_off_switches_run_nothing_and_chain_nothing(env: dict[str, str]) -> None:
    decision = run_mode.decide(_scheduled(**env))
    assert (decision.mode, decision.chain_minutes) == ("off", "")


def test_all_adds_futureeval_as_before() -> None:
    assert run_mode.decide(_scheduled(AUGUR_SCHEDULE="all")).schedule == "all"


def test_submit_turns_the_default_into_the_forecast_job_without_a_default_chain() -> None:
    decision = run_mode.decide(_scheduled(AUGUR_SUBMIT="true"))
    assert (decision.mode, decision.chain_minutes) == ("forecast", "")
    chained = run_mode.decide(_scheduled(AUGUR_SUBMIT="true", AUGUR_CHAIN_MINUTES="45"))
    assert chained.chain_minutes == "45"


def test_shadow_true_wins_over_submit() -> None:
    decision = run_mode.decide(_scheduled(AUGUR_SUBMIT="true", AUGUR_SHADOW="true"))
    assert decision.mode == "shadow"


def test_a_private_repository_never_chains() -> None:
    decision = run_mode.decide(_scheduled(REPO_PRIVATE="true"))
    assert (decision.mode, decision.chain_minutes) == ("shadow", "")


def test_unknown_visibility_leaves_the_check_to_the_chain_job() -> None:
    assert run_mode.decide(_scheduled(REPO_PRIVATE="")).chain_minutes == "30"


def test_scheduled_and_chained_runs_act_on_main_only() -> None:
    assert run_mode.decide(_scheduled(REF="refs/heads/dev/m17")).mode == "off"


def test_a_manual_run_is_what_was_dispatched_and_never_chains() -> None:
    manual = {"EVENT": "workflow_dispatch", "REF": MAIN, **SECRETS}
    shadow = run_mode.decide({**manual, "INPUT_SHADOW": "true"})
    forecast = run_mode.decide({**manual, "INPUT_SHADOW": "false"})
    assert (shadow.mode, shadow.schedule, shadow.chain_minutes) == ("shadow", "all", "")
    assert (forecast.mode, forecast.chain_minutes) == ("forecast", "")
    no_key = run_mode.decide({**manual, "INPUT_SHADOW": "true", "OPENROUTER_API_KEY": ""})
    assert no_key.mode == "off"


def test_another_event_is_off() -> None:
    assert run_mode.decide(_scheduled(EVENT="push")).mode == "off"


# -- every path to a post, exhaustively ----------------------------------------------

FORECAST_IF = (
    "needs.poll.outputs.has_work == 'true' && needs.poll.outputs.mode == 'forecast' &&\n"
    "      ((github.event_name == 'workflow_dispatch' && !inputs.chained && !inputs.shadow) ||\n"
    "       (vars.AUGUR_SUBMIT == 'true' && vars.AUGUR_SHADOW != 'true'))"
)
SUBMIT_EXPR = (
    "SUBMIT: ${{ github.ref == 'refs/heads/main' && ((github.event_name == 'workflow_dispatch'"
    " && !inputs.chained && inputs.submit) || ((github.event_name == 'schedule' ||"
    " inputs.chained) && vars.AUGUR_SUBMIT == 'true')) }}"
)


def test_the_mirrored_conditions_are_the_workflows() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert FORECAST_IF in text
    assert SUBMIT_EXPR in text
    code = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))
    assert code.count("--submit") == 1, "one place adds --submit: the forecast job's step"
    assert 'if [ "$SUBMIT" = "true" ]; then args+=(--submit); fi' in text


def _posts(
    event: str, chained: bool, shadow_in: bool, submit_in: bool, ref: str, v: dict[str, str]
) -> bool:
    """Would the run pass ``--submit``? The mode step, then the forecast job's ``if``,
    then its SUBMIT expression, as written in the workflow (pinned above)."""
    env = {
        "EVENT": event,
        "CHAINED": "true" if chained else "",
        "INPUT_SHADOW": "true" if shadow_in else "false",
        "REF": ref,
        "REPO_PRIVATE": "false",
        **SECRETS,
        **v,
    }
    mode = run_mode.decide(env).mode

    def eq(name: str, value: str) -> bool:
        # GitHub's expression == compares strings ignoring case.
        return v.get(name, "").lower() == value

    manual = event == "workflow_dispatch" and not chained
    runs = mode == "forecast" and (
        (manual and not shadow_in)
        or (eq("AUGUR_SUBMIT", "true") and not eq("AUGUR_SHADOW", "true"))
    )
    submit = ref == MAIN and (
        (manual and submit_in) or ((event == "schedule" or chained) and eq("AUGUR_SUBMIT", "true"))
    )
    return runs and submit


VALUES = {
    "AUGUR_SCHEDULE": [None, "off", "minibench", "all", "bogus"],
    "AUGUR_SHADOW": [None, "true", "false", "TRUE"],
    "AUGUR_SUBMIT": [None, "", "false", "1", "yes", "True", "true"],
    "AUGUR_CHAIN_MINUTES": [None, "30"],
}


def _variable_sets() -> list[dict[str, str]]:
    names = list(VALUES)
    out = []
    for combo in itertools.product(*(VALUES[n] for n in names)):
        out.append({n: x for n, x in zip(names, combo, strict=True) if x is not None})
    return out


def test_no_unattended_run_posts_unless_augur_submit_is_true() -> None:
    checked = 0
    for v in _variable_sets():
        for event, chained in (("schedule", False), ("workflow_dispatch", True)):
            for ref in (MAIN, "refs/heads/dev/m17"):
                posts = _posts(event, chained, False, False, ref, v)
                checked += 1
                submit = v.get("AUGUR_SUBMIT", "").lower() == "true"
                if not submit or v.get("AUGUR_SHADOW", "").lower() == "true":
                    assert not posts, (event, chained, ref, v)
    assert checked == len(_variable_sets()) * 4


def test_with_every_variable_unset_no_path_posts() -> None:
    for event, chained, shadow_in, submit_in, ref in itertools.product(
        ("schedule", "workflow_dispatch", "push"),
        (False, True),
        (False, True),
        (False, True),
        (MAIN, "refs/heads/dev/m17"),
    ):
        manual = event == "workflow_dispatch" and not chained
        if manual and submit_in:
            continue  # a person asked to post: the submit floor in augur forecast decides
        assert not _posts(event, chained, shadow_in, submit_in, ref, {}), (
            event,
            chained,
            shadow_in,
            ref,
        )
