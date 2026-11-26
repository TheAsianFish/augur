"""The tournament workflow's shadow job (M11) can never post.

Read as text (no YAML parser is a dependency): jobs are the two-space-indented
keys under ``jobs:``, and each job's block runs to the next one.
"""

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "tournament.yml"


def _code(text: str) -> str:
    """The workflow without comments, so prose cannot satisfy or break a check."""
    return "\n".join(
        line.split(" #", 1)[0] for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _jobs() -> dict[str, str]:
    text = _code(WORKFLOW.read_text(encoding="utf-8"))
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def test_the_shadow_job_never_passes_submit_or_reads_the_submit_switch() -> None:
    shadow = _jobs()["shadow"]
    assert "--submit" not in shadow
    assert "SUBMIT" not in shadow
    assert "AUGUR_SUBMIT" not in shadow
    assert shadow.count("--shadow-ledger shadow.sqlite") >= 3


def test_shadow_and_forecast_jobs_never_both_run() -> None:
    jobs = _jobs()
    assert "needs.poll.outputs.shadow == 'false'" in jobs["forecast"]
    assert "vars.AUGUR_SHADOW != 'true'" in jobs["forecast"]
    assert "!inputs.shadow" in jobs["forecast"]
    assert "needs.poll.outputs.shadow == 'true'" in jobs["shadow"]


def test_the_shadow_job_pushes_after_each_contestant_whatever_happened() -> None:
    shadow = _jobs()["shadow"]
    steps = re.split(r"^      - ", shadow, flags=re.M)
    names = [s.split("\n", 1)[0] for s in steps]
    forecast = next(i for i, n in enumerate(names) if n.startswith("name: Forecast"))
    template = next(i for i, n in enumerate(names) if n.startswith("name: Template"))
    reader = next(i for i, n in enumerate(names) if n.startswith("name: Newer free reader"))
    pushes = [i for i, s in enumerate(steps) if "push_shadow.sh" in s]
    assert len(pushes) == 3
    assert forecast < pushes[0] < template < pushes[1] < reader < pushes[2]
    for i in (*pushes, template, reader):
        assert "if: always()" in steps[i]
    # One name per tick, so each push replaces the one before instead of adding a file.
    assert all('"$SHADOW_NAME"' in steps[i] for i in pushes)
    for i in (forecast, template, reader):
        minutes = re.search(r"timeout-minutes: (\d+)", steps[i])
        assert minutes is not None
    total = sum(int(m) for m in re.findall(r"^        timeout-minutes: (\d+)", shadow, re.M))
    job = int(re.search(r"^    timeout-minutes: (\d+)", shadow, re.M).group(1))  # type: ignore[union-attr]
    assert total + 10 <= job, "every step fits the job with time to push"


def test_the_reader_runs_last_only_behind_its_variable_and_never_posts() -> None:
    shadow = _jobs()["shadow"]
    steps = re.split(r"^      - ", shadow, flags=re.M)
    reader = [s for s in steps if "augur reader" in s]
    assert len(reader) == 1
    step = reader[0]
    assert step.startswith("name: Newer free reader (reported arm, M16)")
    assert "if: always() && vars.AUGUR_SHADOW_READER == 'true'" in step
    assert "continue-on-error: true" in step
    assert "AUGUR_SHADOW_READER: 'true'" in step
    assert "AUGUR_SHADOW_DAILY_CALLS: ${{ vars.AUGUR_SHADOW_DAILY_CALLS }}" in step
    assert "--ledger run.sqlite --shadow-ledger shadow.sqlite" in step
    assert "--max-minutes 8" in step
    assert "--submit" not in step
    assert "METACULUS_TOKEN" not in step, "the reader reads the ledger, not Metaculus"
    last_push = [s for s in steps if "push_shadow.sh" in s][-1]
    assert "vars.AUGUR_SHADOW_READER == 'true'" in last_push


def test_the_shadow_job_installs_the_template_and_uses_the_free_route() -> None:
    shadow = _jobs()["shadow"]
    assert "--extra baseline" in shadow
    assert "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free" in shadow
    assert "AUGUR_MAX_COST_PER_RUN: '0.00'" in shadow


def test_the_poll_reads_the_shadow_ledger_only_in_shadow_mode() -> None:
    poll = _jobs()["poll"]
    assert "scripts/shadow_fetch.sh" in poll
    assert "if: env.AUGUR_SHADOW == 'true'" in poll
    assert "vars.AUGUR_SHADOW == 'true'" in poll


def test_every_job_has_a_timeout() -> None:
    for name, body in _jobs().items():
        assert re.search(r"^    timeout-minutes: \d+", body, re.M), name


def test_a_chained_shadow_tick_takes_the_shadow_cap_not_the_dispatch_default() -> None:
    shadow = _jobs()["shadow"]
    line = next(x for x in shadow.splitlines() if "AUGUR_MAX_QUESTIONS_PER_RUN:" in x)
    assert "!inputs.chained && inputs.max_questions" in line
    assert "vars.AUGUR_SHADOW_MAX_QUESTIONS || '4'" in line
