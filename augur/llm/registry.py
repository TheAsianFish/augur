"""Models Augur knows, with the release date that gates backtests.

A release date is a hard upper bound on what a model can remember: a model
may forecast a backtest question only if it was released before the question
opened. Dates are from the publisher's release announcement.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    release_date: date | None
    usd_per_million_input: float = 0.0
    usd_per_million_output: float = 0.0
    source: str = ""
    known: bool = True

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> float:
        return (
            (input_tokens or 0) * self.usd_per_million_input
            + (output_tokens or 0) * self.usd_per_million_output
        ) / 1_000_000


_QWEN25 = "https://qwenlm.github.io/blog/qwen2.5/"
_REGISTRY: dict[str, ModelInfo] = {
    m.name: m
    for m in (
        ModelInfo(name="qwen2.5:7b", release_date=date(2024, 9, 19), source=_QWEN25),
        ModelInfo(name="qwen2.5:3b", release_date=date(2024, 9, 19), source=_QWEN25),
        ModelInfo(name="qwen2.5:1.5b", release_date=date(2024, 9, 19), source=_QWEN25),
        ModelInfo(
            name="llama3.1:8b",
            release_date=date(2024, 7, 23),
            source="https://ai.meta.com/blog/meta-llama-3-1/",
        ),
    )
}


# What an unpriced model is assumed to cost on a paid or sponsored provider, so
# that budget caps still bind: above the list price of current frontier models.
UNPRICED_INPUT_PER_MILLION = 15.0
UNPRICED_OUTPUT_PER_MILLION = 75.0
LOCAL_PROVIDERS = frozenset({"ollama", "replay"})


# The submit floor. A forecast posted to a live tournament is scored against
# about 110 bots, most on frontier models, and a weak forecast loses peer
# points that skipping the question would not. So ``--submit`` outside the
# practice tournaments needs a hosted model in this named tier; small and local
# models are for tests, dry runs, backtests and the bot-testing-area. The tier
# is the frontier models used by the Spring 2026 leaders and the Metaculus
# template bots, by bare name (vendor prefix and ``:variant`` suffix dropped);
# it is reviewed whenever the hosted model list changes.
FRONTIER_TIER_NAME = "frontier-2026-11"
FRONTIER_TIER: frozenset[str] = frozenset(
    {
        "gpt-5",
        "gpt-5.1",
        "o3",
        "claude-opus-4-1",
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "gemini-2.5-pro",
        "gemini-3-pro-preview",
        "grok-4",
    }
)
HOSTED_PROVIDERS = frozenset({"metaculus_proxy"})
PRACTICE_TOURNAMENTS = frozenset({"bot-testing-area"})


def bare_model_name(model: str) -> str:
    """``openai/gpt-5:free`` -> ``gpt-5``: the name the frontier tier is keyed on."""
    return model.strip().lower().rsplit("/", 1)[-1].split(":", 1)[0]


def in_frontier_tier(model: str, provider: str) -> bool:
    """Whether ``model`` served by ``provider`` may submit to a live tournament."""
    return provider in HOSTED_PROVIDERS and bare_model_name(model) in FRONTIER_TIER


def lookup(model: str) -> ModelInfo:
    """The registry entry for ``model``; an unknown model has no release date and costs 0."""
    return _REGISTRY.get(model, ModelInfo(name=model, release_date=None, known=False))


def priced(model: str, provider: str) -> ModelInfo:
    """``lookup``, but an unknown model on a non-local provider gets conservative prices."""
    info = lookup(model)
    if info.known or provider in LOCAL_PROVIDERS:
        return info
    return info.model_copy(
        update={
            "usd_per_million_input": UNPRICED_INPUT_PER_MILLION,
            "usd_per_million_output": UNPRICED_OUTPUT_PER_MILLION,
        }
    )


def known_models() -> list[ModelInfo]:
    return sorted(_REGISTRY.values(), key=lambda m: m.name)
