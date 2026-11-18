"""Live checks against OpenRouter. Free models only: these never spend credit."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import SecretStr

from augur.llm.catalog import fetch_catalog, price_for
from augur.llm.endpoints import PRESETS, Route
from augur.llm.openai_compat import OpenAICompatProvider
from augur.llm.probe import probe_routes

FREE_MODEL = os.environ.get("AUGUR_LIVE_FREE_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")


@pytest.mark.integration
async def test_live_model_list_prices_free_models_at_zero() -> None:
    async with httpx.AsyncClient() as http:
        catalog = await fetch_catalog(http, datetime.now(UTC))
    assert catalog.origin == "live"
    free = [e for e in catalog.entries.values() if e.id.endswith(":free")]
    assert free, "OpenRouter lists no :free model"
    route = Route(endpoint=PRESETS["openrouter"], model=free[0].id)
    price = price_for(route, catalog)
    assert price is not None
    assert price.cost(1_000_000, 1_000_000) == 0.0


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="needs OPENROUTER_API_KEY")
async def test_live_probe_of_a_free_model() -> None:
    endpoint = PRESETS["openrouter"]
    route = Route(endpoint=endpoint, model=FREE_MODEL)
    key = SecretStr(os.environ["OPENROUTER_API_KEY"])
    async with httpx.AsyncClient() as http:
        catalog = await fetch_catalog(http, datetime.now(UTC))
        provider = OpenAICompatProvider(http, endpoint, key, max_retries=0, timeout=30.0)
        [result] = await probe_routes(
            [route], {route: ["probe"]}, {"openrouter": provider}, catalog
        )
    # A free model may be busy upstream; what must hold is that the probe says which.
    assert result.outcome in {"answered", "empty", "rate_limited", "unavailable"}
    assert result.cost_usd == 0.0
    assert key.get_secret_value() not in result.detail
