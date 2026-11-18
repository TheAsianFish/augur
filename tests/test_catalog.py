from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from augur.errors import ProviderError
from augur.llm.catalog import (
    Catalog,
    Price,
    fetch_catalog,
    load_snapshot,
    price_for,
    release_bound,
    snapshot_text,
)
from augur.llm.endpoints import Endpoint, Route, endpoints_from_env

NOW = datetime(2026, 11, 1, tzinfo=UTC)
TS_2025_03_01 = int(datetime(2025, 3, 1, 12, 0, tzinfo=UTC).timestamp())
TS_2025_06_01 = int(datetime(2025, 6, 1, tzinfo=UTC).timestamp())
EPS = endpoints_from_env({})


def _row(model_id: str, created: int, prompt: str, completion: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "created": created,
        "pricing": {"prompt": prompt, "completion": completion},
        "context_length": 8192,
    }


def _payload() -> dict[str, Any]:
    return {
        "data": [
            _row("openai/gpt-5.1", TS_2025_03_01, "0.00000125", "0.00001"),
            _row("google/gemma-4-31b-it:free", TS_2025_03_01, "0", "0"),
            _row("openrouter/auto", TS_2025_03_01, "-1", "-1"),
            _row("meta-llama/llama-3.3-70b-versatile", TS_2025_03_01, "0.000001", "0.000002"),
            _row("other/llama-3.3-70b-versatile:free", TS_2025_06_01, "0", "0"),
        ]
    }


def _catalog() -> Catalog:
    return Catalog.from_api(_payload(), NOW)


def _route(endpoint: Endpoint, model: str) -> Route:
    return Route(endpoint=endpoint, model=model)


def test_from_api_converts_prices_and_dates() -> None:
    cat = _catalog()
    gpt = cat.get("openai/gpt-5.1")
    assert gpt is not None
    assert gpt.usd_per_million_input == pytest.approx(1.25)
    assert gpt.usd_per_million_output == pytest.approx(10.0)
    assert gpt.listed_on == date(2025, 3, 1)
    auto = cat.get("openrouter/auto")
    assert auto is not None
    assert auto.usd_per_million_input is None
    assert not auto.priced
    assert cat.origin == "live"


def test_from_api_skips_bad_rows() -> None:
    payload = {
        "data": [
            _row("ok/model", TS_2025_03_01, "0", "0"),
            {"id": "no/created", "pricing": {}},
            {"id": "bad/created", "created": "yesterday", "pricing": {}},
            {"created": TS_2025_03_01},
        ]
    }
    cat = Catalog.from_api(payload, NOW)
    assert list(cat.entries) == ["ok/model"]


@pytest.mark.parametrize("payload", [{}, {"data": "nope"}, {"data": None}])
def test_from_api_without_data_list_raises(payload: dict[str, Any]) -> None:
    with pytest.raises(ProviderError):
        Catalog.from_api(payload, NOW)


def test_price_for_local_and_free() -> None:
    cat = _catalog()
    local = price_for(_route(EPS["ollama"], "qwen2.5:7b"), cat)
    assert local is not None
    assert local.basis == "local"
    assert local.cost(1_000_000, 1_000_000) == 0
    free = price_for(_route(EPS["groq"], "llama-3.3-70b-versatile"), cat)
    assert free is not None
    assert free.cost(5_000, 5_000) == 0


def test_a_frontier_model_on_a_free_host_is_priced_from_the_list() -> None:
    """A billing-enabled Gemini key on the "free" preset must still count against the caps."""
    cat = _catalog()
    on_gemini = price_for(_route(EPS["gemini"], "gpt-5.1"), cat)
    assert on_gemini is not None
    assert on_gemini.cost(1_000_000, 1_000_000) == pytest.approx(11.25)
    assert price_for(_route(EPS["gemini"], "gemini-2.5-pro"), cat) is None, "unlisted: not called"


def test_price_for_openrouter() -> None:
    cat = _catalog()
    price = price_for(_route(EPS["openrouter"], "openai/gpt-5.1"), cat)
    assert isinstance(price, Price)
    assert price.usd_per_million_input == pytest.approx(1.25)
    assert price.cost(1_000_000, 1_000_000) == pytest.approx(11.25)
    free = price_for(_route(EPS["openrouter"], "google/gemma-4-31b-it:free"), cat)
    assert free is not None
    assert free.cost(1_000_000, 1_000_000) == 0


def test_price_for_unknown_or_unpriced_is_none() -> None:
    cat = _catalog()
    assert price_for(_route(EPS["openrouter"], "not/listed"), cat) is None
    assert price_for(_route(EPS["openrouter"], "openrouter/auto"), cat) is None
    custom = Endpoint(name="mine", base_url="https://example.com/v1", billing="unpriced")
    assert price_for(_route(custom, "openai/gpt-5.1"), cat) is None


def test_release_bound() -> None:
    cat = _catalog()
    assert release_bound(_route(EPS["ollama"], "qwen2.5:7b"), cat) == (
        date(2024, 9, 19),
        "release announcement",
    )
    exact, source = release_bound(_route(EPS["openrouter"], "openai/gpt-5.1"), cat)
    assert exact == date(2025, 3, 1)
    assert "listing" in source
    later, _ = release_bound(_route(EPS["groq"], "llama-3.3-70b-versatile"), cat)
    assert later == date(2025, 6, 1)
    assert release_bound(_route(EPS["groq"], "mystery-model"), cat) == (None, "unknown")


def test_snapshot_round_trip() -> None:
    cat = _catalog()
    again = Catalog.from_api(cat.to_snapshot(), NOW, "snapshot")
    assert set(again.entries) == set(cat.entries)
    for model_id, entry in cat.entries.items():
        other = again.entries[model_id]
        assert other.listed_on == entry.listed_on
        assert other.usd_per_million_input == entry.usd_per_million_input
        assert other.usd_per_million_output == entry.usd_per_million_output


def test_snapshot_text_is_json_one_model_per_line() -> None:
    cat = _catalog()
    text = snapshot_text(cat)
    parsed = json.loads(text)
    assert len(parsed["data"]) == len(cat.entries)
    assert sum(1 for line in text.splitlines() if line.startswith("  {")) == len(cat.entries)


def test_committed_snapshot_loads() -> None:
    cat = load_snapshot()
    assert cat.origin == "snapshot"
    assert len(cat.entries) > 100
    assert any(
        m.endswith(":free") and e.usd_per_million_input == 0 and e.usd_per_million_output == 0
        for m, e in cat.entries.items()
    )


def test_missing_snapshot_is_empty(tmp_path: Path) -> None:
    cat = load_snapshot(tmp_path / "missing.json")
    assert cat.origin == "empty"
    assert cat.entries == {}


async def test_fetch_catalog_live() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        cat = await fetch_catalog(http, NOW)
    assert cat.origin == "live"
    assert "openai/gpt-5.1" in cat.entries


async def test_fetch_catalog_falls_back_to_snapshot_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        cat = await fetch_catalog(http, NOW)
    assert cat.origin == "snapshot"
    assert len(cat.entries) > 100
