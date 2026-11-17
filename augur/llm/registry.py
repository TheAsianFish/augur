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


def lookup(model: str) -> ModelInfo:
    """The registry entry for ``model``; an unknown model has no release date and costs 0."""
    return _REGISTRY.get(model, ModelInfo(name=model, release_date=None))


def known_models() -> list[ModelInfo]:
    return sorted(_REGISTRY.values(), key=lambda m: m.name)
