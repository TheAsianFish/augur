"""Build the configured provider."""

from __future__ import annotations

import httpx

import augur
from augur.config import Settings
from augur.errors import ConfigError, MissingCredentialsError
from augur.llm.base import Provider
from augur.llm.metaculus_proxy import MetaculusProxyProvider
from augur.llm.ollama import OllamaProvider
from augur.llm.replay import RecordingProvider, ReplayProvider


def default_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=10.0),
        headers={
            "User-Agent": f"augur-forecast/{augur.__version__} "
            "(+https://github.com/TheAsianFish/augur)"
        },
    )


def build_provider(settings: Settings, http: httpx.AsyncClient) -> Provider:
    provider: Provider
    if settings.provider == "ollama":
        provider = OllamaProvider(http, settings.ollama_base_url)
    elif settings.provider == "metaculus_proxy":
        if settings.metaculus_token is None:
            raise MissingCredentialsError("the metaculus_proxy provider", ("METACULUS_TOKEN",))
        provider = MetaculusProxyProvider(http, settings.proxy_base_url, settings.metaculus_token)
    else:
        if settings.replay_path is None:
            raise ConfigError(
                "AUGUR_PROVIDER=replay needs AUGUR_REPLAY_PATH set to a recorded JSONL file; "
                "record one with AUGUR_RECORD_PATH and a live provider"
            )
        return ReplayProvider(settings.replay_path)
    if settings.record_path is not None:
        provider = RecordingProvider(provider, settings.record_path)
    return provider
