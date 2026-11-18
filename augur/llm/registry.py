"""Hand-checked release dates for local models, and the submit floor's frontier tier.

Prices and the listing dates of hosted models come from OpenRouter's model
list (:mod:`augur.llm.catalog`); only the local models used for development
and backtests keep release dates here, taken from the publisher's release
announcement.
"""

from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from augur.llm.endpoints import Route

_QWEN25 = "https://qwenlm.github.io/blog/qwen2.5/"
_LOCAL_RELEASES: dict[str, tuple[date, str]] = {
    "qwen2.5:7b": (date(2024, 9, 19), _QWEN25),
    "qwen2.5:3b": (date(2024, 9, 19), _QWEN25),
    "qwen2.5:1.5b": (date(2024, 9, 19), _QWEN25),
    "llama3.1:8b": (date(2024, 7, 23), "https://ai.meta.com/blog/meta-llama-3-1/"),
}


def local_release_date(model: str) -> date | None:
    found = _LOCAL_RELEASES.get(model)
    return found[0] if found else None


# The submit floor. A forecast posted to a live tournament is scored against
# about 110 bots, most on frontier models, and a weak forecast loses peer
# points that skipping the question would not. So ``--submit`` outside the
# practice tournaments needs every forecast to come from a hosted model in
# this named tier; small and local models are for tests, dry runs, backtests
# and the bot-testing-area. The tier is the frontier models used by the
# Spring 2026 leaders and the Metaculus template bots, by bare name (see
# ``bare_model_name``); it is reviewed whenever the hosted model list changes,
# and never widened to get a run through.
FRONTIER_TIER_NAME = "frontier-2026-11"
FRONTIER_TIER: frozenset[str] = frozenset(
    {
        "gpt-5",
        "gpt-5-1",
        "o3",
        "claude-opus-4-1",
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "gemini-2-5-pro",
        "gemini-3-pro-preview",
        "grok-4",
    }
)
# Hosts that serve these models under these names. A model called ``gpt-5``
# on any other host (a local server, a custom endpoint) is not trusted to be
# it, whatever the endpoint is named.
FRONTIER_HOSTS: frozenset[str] = frozenset(
    {
        "openrouter.ai",
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "api.x.ai",
    }
)
PRACTICE_TOURNAMENTS = frozenset({"bot-testing-area"})

_SEPARATORS = re.compile(r"[._\s]+")


def bare_model_name(model: str) -> str:
    """``openai/gpt-5.1:free`` -> ``gpt-5-1``: the name the frontier tier is keyed on.

    Vendor prefix and ``:variant`` suffix dropped, lower case, and dots and
    underscores read as dashes, because hosts spell versions both ways
    (OpenRouter lists ``claude-opus-5.5``; Anthropic's id is ``claude-opus-5-5``).
    """
    bare = model.strip().lower().rsplit("/", 1)[-1].split(":", 1)[0]
    return _SEPARATORS.sub("-", bare)


def in_frontier_tier(route: Route) -> bool:
    """Whether a forecast from ``route`` may be submitted to a live tournament."""
    endpoint = route.endpoint
    return (
        not endpoint.is_local
        and endpoint.host in FRONTIER_HOSTS
        and bare_model_name(route.model) in FRONTIER_TIER
    )
