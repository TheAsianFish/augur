from __future__ import annotations

import json
from datetime import date

import pytest

from augur.config import Settings
from augur.errors import ConfigError
from augur.llm.endpoints import (
    PRESETS,
    Endpoint,
    Route,
    endpoints_from_env,
    parse_chain,
    parse_route,
)
from augur.llm.registry import bare_model_name, in_frontier_tier, local_release_date


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


def test_bare_model_name() -> None:
    assert bare_model_name("openai/gpt-5.1:free") == "gpt-5-1"
    assert bare_model_name("anthropic/claude-opus-5.5") == "claude-opus-5-5"
    assert bare_model_name(" Claude-Opus-4-5 ") == "claude-opus-4-5"
    assert bare_model_name("qwen2.5:7b") == "qwen2-5"


def _route(endpoint: Endpoint, model: str) -> Route:
    return Route(endpoint=endpoint, model=model)


def test_in_frontier_tier() -> None:
    eps = endpoints_from_env({})
    assert in_frontier_tier(_route(eps["openrouter"], "openai/gpt-5.1"))
    assert in_frontier_tier(_route(eps["openrouter"], "anthropic/claude-opus-5.5"))
    assert not in_frontier_tier(_route(eps["openrouter"], "openai/gpt-5-mini"))
    assert in_frontier_tier(_route(eps["openrouter"], "openai/gpt-5.4"))
    assert in_frontier_tier(_route(eps["openrouter"], "anthropic/claude-opus-5"))
    assert not in_frontier_tier(_route(eps["openrouter"], "openai/gpt-5.4-mini"))
    assert not in_frontier_tier(_route(eps["openrouter"], "openai/gpt-6.2-astra"))
    assert not in_frontier_tier(_route(eps["groq"], "openai/gpt-5.1"))
    assert not in_frontier_tier(_route(eps["ollama"], "gpt-5"))


def test_frontier_name_on_local_endpoint_named_openrouter_is_not_frontier() -> None:
    fake = Endpoint(name="openrouter", base_url="http://127.0.0.1:8080/v1", billing="openrouter")
    assert not in_frontier_tier(_route(fake, "openai/gpt-5.1"))


def test_local_release_date() -> None:
    assert local_release_date("qwen2.5:7b") == date(2024, 9, 19)
    assert local_release_date("mystery:1b") is None


def test_metaculus_proxy_provider_is_rejected() -> None:
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings.from_env({"AUGUR_PROVIDER": "metaculus_proxy"})


def test_research_chain_defaults_to_forecast_chain() -> None:
    env = {"AUGUR_FORECAST_MODELS": "openrouter:openai/gpt-5,openrouter:google/gemma-4-31b-it:free"}
    chains = Settings.from_env(env).chains()
    assert chains["research"] == chains["forecast"]
    assert [r.key for r in chains["forecast"]] == [
        "openrouter:openai/gpt-5",
        "openrouter:google/gemma-4-31b-it:free",
    ]


def test_research_chain_can_differ() -> None:
    env = {
        "AUGUR_FORECAST_MODELS": "openrouter:openai/gpt-5",
        "AUGUR_RESEARCH_MODELS": "groq:llama-3.3-70b-versatile",
    }
    chains = Settings.from_env(env).chains()
    assert chains["research"] != chains["forecast"]
    assert chains["research"][0].key == "groq:llama-3.3-70b-versatile"


def test_default_and_replay_chains_use_ollama() -> None:
    default = Settings.from_env({}).chains()
    assert [r.key for r in default["forecast"]] == ["ollama:qwen2.5:7b"]
    assert default["research"] == default["forecast"]
    replay = Settings.from_env({"AUGUR_PROVIDER": "replay"}).chains()
    assert all(r.endpoint.name == "ollama" for r in replay["forecast"])


def test_keys_only_hold_endpoint_key_envs() -> None:
    settings = Settings.from_env(
        {"OPENROUTER_API_KEY": "sk-or-secret", "SOME_OTHER_SECRET": "other"}
    )
    assert list(settings.keys) == ["OPENROUTER_API_KEY"]
    assert settings.keys["OPENROUTER_API_KEY"].get_secret_value() == "sk-or-secret"


def test_public_dict_has_no_secrets() -> None:
    secret = "sk-or-very-secret-value"
    settings = Settings.from_env(
        {
            "OPENROUTER_API_KEY": secret,
            "METACULUS_TOKEN": "meta-token-secret",
            "ASKNEWS_CLIENT_SECRET": "asknews-secret",
            "AUGUR_FORECAST_MODELS": "openrouter:openai/gpt-5",
        }
    )
    data = settings.public_dict()
    text = json.dumps(data)
    for value in (secret, "meta-token-secret", "asknews-secret"):
        assert value not in text
    assert data["keys_set"] == ["OPENROUTER_API_KEY"]
    assert data["chains"] == {
        "research": ["openrouter:openai/gpt-5"],
        "forecast": ["openrouter:openai/gpt-5"],
    }


def test_bad_model_entry_fails_at_from_env() -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({"AUGUR_FORECAST_MODELS": "openrouter:ok,nonsense"})


def test_probe_flag() -> None:
    assert Settings.from_env({"AUGUR_PROBE": "0"}).probe is False
    assert Settings.from_env({}).probe is True


def test_a_preset_or_alias_with_a_frontier_name_is_not_frontier() -> None:
    """``@preset/gpt-5`` or ``small:x/gpt-5`` can route to any model behind the name."""
    eps = endpoints_from_env({})
    assert not in_frontier_tier(_route(eps["openrouter"], "@preset/gpt-5"))
    assert not in_frontier_tier(_route(eps["openrouter"], "meta-llama/llama-3-8b:x/gpt-5"))
    assert in_frontier_tier(_route(eps["openrouter"], "openai/gpt-5:free"))


# Reasoning effort per route


def test_parse_route_reads_a_trailing_effort() -> None:
    eps = endpoints_from_env({})
    route = parse_route("openrouter:openai/gpt-5.4@high", eps)
    assert (route.model, route.reasoning_effort) == ("openai/gpt-5.4", "high")
    assert route.key == "openrouter:openai/gpt-5.4@high"
    assert parse_route(route.key, eps) == route
    free = parse_route("openrouter:google/gemma-4-31b-it:free@low", eps)
    assert (free.model, free.reasoning_effort) == ("google/gemma-4-31b-it:free", "low")
    local = parse_route("ollama:qwen2.5:7b@minimal", eps)
    assert (local.model, local.reasoning_effort) == ("qwen2.5:7b", "minimal")
    assert parse_route("ollama:qwen2.5:7b", eps).reasoning_effort is None


def test_a_leading_preset_is_a_model_not_an_effort() -> None:
    eps = endpoints_from_env({})
    bare = parse_route("openrouter:@preset/x", eps)
    assert (bare.model, bare.reasoning_effort) == ("@preset/x", None)
    with_effort = parse_route("openrouter:@preset/x@xhigh", eps)
    assert (with_effort.model, with_effort.reasoning_effort) == ("@preset/x", "xhigh")
    assert not in_frontier_tier(bare)
    assert not in_frontier_tier(with_effort)


def test_an_unknown_effort_names_the_allowed_ones() -> None:
    with pytest.raises(ConfigError) as info:
        parse_route("openrouter:openai/gpt-5.4@extreme", endpoints_from_env({}))
    for allowed in ("minimal", "low", "medium", "high", "xhigh"):
        assert allowed in str(info.value)


def test_efforts_make_distinct_routes_of_one_model() -> None:
    eps = endpoints_from_env({})
    chain = parse_chain("openrouter:openai/gpt-5.4,openrouter:openai/gpt-5.4@high", eps)
    assert [r.key for r in chain] == ["openrouter:openai/gpt-5.4", "openrouter:openai/gpt-5.4@high"]
    assert chain[0] != chain[1]
    same = parse_chain("openrouter:openai/gpt-5.4@high,openrouter:openai/gpt-5.4@high", eps)
    assert len(same) == 1


def test_effort_does_not_change_tier_membership() -> None:
    eps = endpoints_from_env({})
    plain = parse_route("openrouter:openai/gpt-5.1", eps)
    high = parse_route("openrouter:openai/gpt-5.1@high", eps)
    assert in_frontier_tier(plain)
    assert in_frontier_tier(high)
    assert not in_frontier_tier(parse_route("openrouter:openai/gpt-5-mini@high", eps))
    assert not in_frontier_tier(parse_route("groq:openai/gpt-5.1@high", eps))
