#!/usr/bin/env python3
"""Decide a Tournament run's mode once, in the poll job. Standard library only.

Every later job reads this decision (``mode``) instead of re-deriving it from
repository variables, so a variable changed in one place and not another can
never start both the shadow job and the forecast job, or neither by accident.

Modes:

- ``shadow``: Augur and the template on the same free route, unpublished (M11).
- ``forecast``: the ordinary forecast job. A scheduled or chained run posts
  whatever it forecasts, so it is ``forecast`` only when ``AUGUR_SUBMIT`` is
  ``true`` (and ``AUGUR_SHADOW`` is not); the forecast job checks
  ``AUGUR_SUBMIT`` again itself.
- ``off``: nothing runs and nothing is chained.

The default (M17): a scheduled or chained run on ``main`` with no variable set
is a MiniBench shadow tick, chained every 30 minutes on a public repository.
``AUGUR_SHADOW=false`` or ``AUGUR_SCHEDULE=off`` turns it off. A secret the
mode needs is checked here (its value is never printed): a missing one is
named and the mode is ``off``, so the run exits 0 and the chain stops.

Environment: ``EVENT`` (``schedule``, ``workflow_dispatch``), ``CHAINED``,
``INPUT_SHADOW`` (``true`` or ``false``), ``REF`` (``refs/heads/main``),
``REPO_PRIVATE`` (``true``, ``false`` or empty when the event does not say),
``AUGUR_SCHEDULE``, ``AUGUR_SHADOW``, ``AUGUR_SUBMIT``, ``AUGUR_CHAIN_MINUTES``,
``METACULUS_TOKEN``, ``OPENROUTER_API_KEY``, ``GITHUB_OUTPUT`` (receives
``mode``, ``schedule`` and ``chain_minutes``; empty minutes: no chain).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

SHADOW, FORECAST, OFF = "shadow", "forecast", "off"
SCHEDULES = ("off", "minibench", "all")
DEFAULT_SCHEDULE = "minibench"
DEFAULT_SHADOW_CHAIN_MINUTES = "30"
MAIN = "refs/heads/main"


@dataclass(frozen=True)
class Decision:
    mode: str
    schedule: str
    """What ``scripts/poll.py`` polls: ``all`` on a manual run (it names its tournament)."""
    chain_minutes: str
    """Whole minutes before the next chained run, or empty: no chained run follows."""
    lines: tuple[str, ...]


def _flag(env: Mapping[str, str], name: str) -> str:
    return env.get(name, "").strip().lower()


def _missing(env: Mapping[str, str], names: tuple[str, ...]) -> list[str]:
    return [n for n in names if not env.get(n, "").strip()]


def decide(env: Mapping[str, str]) -> Decision:
    lines: list[str] = []
    event = _flag(env, "EVENT")
    chained = _flag(env, "CHAINED") == "true"
    manual = event == "workflow_dispatch" and not chained
    submit = _flag(env, "AUGUR_SUBMIT") == "true"
    shadow_var = _flag(env, "AUGUR_SHADOW")

    def off(reason: str, schedule: str = "off") -> Decision:
        lines.append(f"mode=off: {reason}")
        return Decision(OFF, schedule, "", tuple(lines))

    if manual:
        mode = SHADOW if _flag(env, "INPUT_SHADOW") == "true" else FORECAST
        schedule = "all"  # a manual run polls the tournament it was asked for
        lines.append(f"a manual run: mode {mode}, as dispatched")
    else:
        if event != "schedule" and not chained:
            return off(f"event {event or 'unknown'!r} is neither scheduled nor chained")
        if env.get("REF", "").strip() != MAIN:
            return off(f"scheduled and chained runs act on main only, not {env.get('REF')!r}")
        raw = _flag(env, "AUGUR_SCHEDULE")
        schedule = raw or DEFAULT_SCHEDULE
        if schedule not in SCHEDULES:
            return off(f"AUGUR_SCHEDULE={raw!r} is not off, minibench or all")
        if schedule == "off":
            return off("AUGUR_SCHEDULE is off")
        if shadow_var == "true":
            mode = SHADOW
            lines.append("AUGUR_SHADOW is true: a shadow tick")
        elif submit:
            mode = FORECAST
            lines.append("AUGUR_SUBMIT is true and AUGUR_SHADOW is not: the forecast job posts")
        elif shadow_var == "false":
            return off("AUGUR_SHADOW is false and AUGUR_SUBMIT is not true", schedule)
        else:
            mode = SHADOW
            lines.append("no AUGUR_SUBMIT: a shadow tick, the default (nothing is posted)")
        lines.append(
            f"schedule {schedule}" + ("" if raw else " (AUGUR_SCHEDULE unset: the default)")
        )

    needs = ("METACULUS_TOKEN", "OPENROUTER_API_KEY") if mode == SHADOW else ("METACULUS_TOKEN",)
    missing = _missing(env, needs)
    if missing:
        return off(
            f"secret {' and '.join(missing)} missing in this repository; add it under "
            "Settings > Secrets and variables > Actions"
        )

    chain = ""
    if not manual:
        configured = env.get("AUGUR_CHAIN_MINUTES", "").strip()
        chain = configured or (DEFAULT_SHADOW_CHAIN_MINUTES if mode == SHADOW else "")
        private = _flag(env, "REPO_PRIVATE")
        if chain and private == "true":
            lines.append("a private repository: no chained run (its minutes would be billed)")
            chain = ""
        elif chain:
            source = "AUGUR_CHAIN_MINUTES" if configured else "the shadow default"
            lines.append(f"the next run is chained after {chain} minutes ({source})")
        else:
            lines.append("no chained run (AUGUR_CHAIN_MINUTES unset outside shadow mode)")
    return Decision(mode, schedule, chain, tuple(lines))


def main(env: Mapping[str, str] | None = None) -> int:
    environ = os.environ if env is None else env
    decision = decide(environ)
    for line in decision.lines:
        print(line)
    print(
        f"mode={decision.mode} schedule={decision.schedule} chain_minutes={decision.chain_minutes}"
    )
    output = environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"mode={decision.mode}\n")
            fh.write(f"schedule={decision.schedule}\n")
            fh.write(f"chain_minutes={decision.chain_minutes}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
