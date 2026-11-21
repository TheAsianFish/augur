"""Build the providers and the router a run uses."""

from __future__ import annotations

import httpx

import augur
from augur.clock import Clock
from augur.config import Settings
from augur.errors import ConfigError
from augur.llm.base import Provider
from augur.llm.catalog import Catalog
from augur.llm.endpoints import Endpoint
from augur.llm.ollama import OllamaProvider
from augur.llm.openai_compat import OpenAICompatProvider
from augur.llm.replay import RecordingProvider, ReplayProvider
from augur.llm.router import Router


def default_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=10.0),
        headers={
            "User-Agent": f"augur-forecast/{augur.__version__} "
            "(+https://github.com/TheAsianFish/augur)"
        },
    )


def build_provider(
    endpoint: Endpoint,
    settings: Settings,
    http: httpx.AsyncClient,
    *,
    max_retries: int = 3,
    timeout: float | None = None,
) -> Provider:
    """The provider for one endpoint (never a replay; see :func:`build_providers`)."""
    if endpoint.api == "ollama":
        return OllamaProvider(http, endpoint.base_url)
    key = settings.keys.get(endpoint.key_env) if endpoint.key_env else None
    return OpenAICompatProvider(http, endpoint, key, max_retries=max_retries, timeout=timeout)


def build_providers(
    settings: Settings,
    http: httpx.AsyncClient,
    *,
    max_retries: int = 3,
    record: bool = True,
    timeout: float | None = None,
) -> dict[str, Provider]:
    """One provider per endpoint the chains (and the members' own chains) use, by endpoint name."""
    chains = (*settings.chains().values(), *settings.member_chains())
    names = {r.endpoint.name: r.endpoint for c in chains for r in c}
    if settings.is_replay:
        if settings.replay_path is None:
            raise ConfigError(
                "AUGUR_PROVIDER=replay needs AUGUR_REPLAY_PATH set to a recorded JSONL file; "
                "record one with AUGUR_RECORD_PATH and a live provider"
            )
        replay = ReplayProvider(settings.replay_path)
        return dict.fromkeys(names, replay)
    providers: dict[str, Provider] = {}
    for name, endpoint in names.items():
        provider = build_provider(
            endpoint, settings, http, max_retries=max_retries, timeout=timeout
        )
        if record and settings.record_path is not None:
            provider = RecordingProvider(provider, settings.record_path)
        providers[name] = provider
    return providers


def build_router(
    settings: Settings, http: httpx.AsyncClient, catalog: Catalog, clock: Clock
) -> Router:
    return Router(settings.chains(), build_providers(settings, http), catalog, clock)
