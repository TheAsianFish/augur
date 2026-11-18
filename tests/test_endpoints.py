from __future__ import annotations

import pytest

from augur.errors import ConfigError
from augur.llm.endpoints import (
    PRESETS,
    Endpoint,
    Route,
    endpoints_from_env,
    parse_chain,
    parse_route,
)


def test_parse_route_splits_at_first_colon() -> None:
    eps = endpoints_from_env({})
    route = parse_route("openrouter:google/gemma-4-31b-it:free", eps)
    assert route.endpoint.name == "openrouter"
    assert route.model == "google/gemma-4-31b-it:free"
    assert route.key == "openrouter:google/gemma-4-31b-it:free"
    assert parse_route(route.key, eps) == route
    assert parse_route("ollama:qwen2.5:7b", eps).model == "qwen2.5:7b"


@pytest.mark.parametrize("text", ["openrouter", "openrouter:", "openrouter:   "])
def test_parse_route_needs_endpoint_and_model(text: str) -> None:
    with pytest.raises(ConfigError, match="endpoint:model"):
        parse_route(text, endpoints_from_env({}))


def test_parse_route_unknown_endpoint_names_known_ones() -> None:
    with pytest.raises(ConfigError) as info:
        parse_route("nowhere:model", endpoints_from_env({}))
    message = str(info.value)
    assert "openrouter" in message
    assert "groq" in message
    assert "AUGUR_ENDPOINT_<NAME>_URL" in message


def test_parse_chain_keeps_order_drops_duplicates_and_blanks() -> None:
    eps = endpoints_from_env({})
    chain = parse_chain("groq:a, ollama:b:c,, groq:a ,openrouter:x/y", eps)
    assert [r.key for r in chain] == ["groq:a", "ollama:b:c", "openrouter:x/y"]


@pytest.mark.parametrize("text", ["", " , ,"])
def test_parse_chain_empty_is_error(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_chain(text, endpoints_from_env({}))


def test_presets_and_key_envs() -> None:
    eps = endpoints_from_env({})
    assert {"openrouter", "groq", "gemini", "github", "ollama"} <= set(eps)
    assert eps["openrouter"].key_env == "OPENROUTER_API_KEY"
    assert eps["groq"].key_env == "GROQ_API_KEY"
    assert eps["gemini"].key_env == "GEMINI_API_KEY"
    assert eps["github"].key_env == "GITHUB_TOKEN"
    assert eps["ollama"].key_env is None


def test_ollama_host_sets_base_url() -> None:
    eps = endpoints_from_env({"OLLAMA_HOST": "gpu-box:11434"})
    assert eps["ollama"].base_url == "http://gpu-box:11434"
    assert PRESETS["ollama"].base_url == "http://localhost:11434"


def test_custom_endpoint_from_env() -> None:
    eps = endpoints_from_env(
        {
            "AUGUR_ENDPOINT_MYHOST_URL": "https://example.com/v1",
            "AUGUR_ENDPOINT_MYHOST_KEY_ENV": "MYHOST_API_KEY",
            "AUGUR_ENDPOINT_MYHOST_BILLING": "free",
            "AUGUR_ENDPOINT_MYHOST_RPM": "30",
        }
    )
    ep = eps["myhost"]
    assert ep.name == "myhost"
    assert ep.base_url == "https://example.com/v1"
    assert ep.key_env == "MYHOST_API_KEY"
    assert ep.billing == "free"
    assert ep.rpm == 30


def test_invalid_billing_is_config_error() -> None:
    with pytest.raises(ConfigError, match="BILLING"):
        endpoints_from_env(
            {
                "AUGUR_ENDPOINT_MYHOST_URL": "https://example.com/v1",
                "AUGUR_ENDPOINT_MYHOST_BILLING": "sometimes",
            }
        )


def test_invalid_endpoint_name_is_config_error() -> None:
    with pytest.raises(ConfigError, match="endpoint name"):
        endpoints_from_env({"AUGUR_ENDPOINT_MY.HOST_URL": "https://example.com/v1"})


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:8080/v1", True),
        ("http://localhost/v1", True),
        ("http://[::1]:8000", True),
        ("https://openrouter.ai/api/v1", False),
    ],
)
def test_is_local(url: str, expected: bool) -> None:
    assert Endpoint(name="x", base_url=url).is_local is expected


def test_is_local_for_ollama_api_and_host_lowercase() -> None:
    assert Endpoint(name="x", base_url="https://example.com", api="ollama").is_local
    assert Endpoint(name="x", base_url="https://OpenRouter.AI/v1").host == "openrouter.ai"


def _route(endpoint: Endpoint, model: str) -> Route:
    return Route(endpoint=endpoint, model=model)
