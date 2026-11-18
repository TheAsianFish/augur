"""Where models live: endpoints, routes and the ordered model chain per role.

An *endpoint* is an OpenAI-compatible host (a base URL and the name of the
environment variable that holds its key) or a local Ollama server. A *route*
is one model on one endpoint, written ``endpoint:model`` (split at the first
colon, so ``openrouter:google/gemma-4-31b-it:free`` and ``ollama:qwen2.5:7b``
both parse). Each role has an ordered chain of routes; the router tries them
in order and falls to the next on rate-limit, out-of-credit and similar
errors.

Adding a host is configuration, not code::

    AUGUR_ENDPOINT_MYHOST_URL=https://example.com/v1
    AUGUR_ENDPOINT_MYHOST_KEY_ENV=MYHOST_API_KEY
    AUGUR_ENDPOINT_MYHOST_BILLING=free     # or: openrouter, unpriced (default)
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from augur.errors import ConfigError
from augur.llm.base import EFFORTS, Effort

Api = Literal["openai", "ollama"]
# How a call is priced. ``openrouter``: from OpenRouter's model list.
# ``free``: the host does not bill (a free tier, a local server).
# ``unpriced``: the host bills but Augur has no price, so its models are not
# used until priced; they are never assumed to cost frontier rates.
Billing = Literal["openrouter", "free", "unpriced"]
Role = Literal["research", "forecast"]
ROLES: tuple[Role, ...] = ("research", "forecast")

# Which chain each call role draws from: planning and the brief are cheap,
# the forecasts are the answer.
CALL_ROLE_CHAIN: dict[str, Role] = {
    "planner": "research",
    "researcher": "research",
    "forecaster": "forecast",
}

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class Endpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    base_url: str
    key_env: str | None = None
    api: Api = "openai"
    billing: Billing = "unpriced"
    # Client-side ceiling, requests per minute. The probe measures the real one.
    rpm: float = Field(default=60.0, gt=0)
    max_concurrent: int = Field(default=4, ge=1)

    @property
    def host(self) -> str:
        return (urlsplit(self.base_url).hostname or "").lower()

    @property
    def is_local(self) -> bool:
        """A server on this machine: never a hosted model, never billed."""
        if self.api == "ollama":
            return True
        host = self.host
        if host in {"localhost", "host.docker.internal"} or host.endswith(".localhost"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False


class Route(BaseModel):
    model_config = ConfigDict(frozen=True)

    endpoint: Endpoint
    # The bare model id, as sent to the host.
    model: str
    reasoning_effort: Effort | None = None

    @property
    def effort_applied(self) -> bool | None:
        """Whether the host is sent the effort: ``None`` if none is set, ``False`` for Ollama."""
        if self.reasoning_effort is None:
            return None
        return self.endpoint.api != "ollama"

    @property
    def key(self) -> str:
        suffix = f"@{self.reasoning_effort}" if self.reasoning_effort else ""
        return f"{self.endpoint.name}:{self.model}{suffix}"

    def __str__(self) -> str:
        return self.key


PRESETS: dict[str, Endpoint] = {
    e.name: e
    for e in (
        # Metaculus's sponsored credits arrive as an OpenRouter key; `:free`
        # models cost nothing on any key. OpenRouter documents 20 requests a
        # minute for free models; the probe measures it.
        Endpoint(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            key_env="OPENROUTER_API_KEY",
            billing="openrouter",
            rpm=20,
        ),
        Endpoint(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            key_env="GROQ_API_KEY",
            billing="free",
            rpm=30,
        ),
        Endpoint(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            key_env="GEMINI_API_KEY",
            billing="free",
            rpm=10,
        ),
        Endpoint(
            name="github",
            base_url="https://models.github.ai/inference",
            key_env="GITHUB_TOKEN",
            billing="free",
            rpm=10,
            max_concurrent=2,
        ),
        # Ollama's native API, because its OpenAI-compatible one cannot set the
        # context window per request and silently drops the start of a long
        # prompt. Point a custom endpoint at http://localhost:11434/v1 to use
        # the generic path instead.
        Endpoint(
            name="ollama",
            base_url="http://localhost:11434",
            api="ollama",
            billing="free",
            rpm=600,
            max_concurrent=1,
        ),
    )
}


def endpoints_from_env(environ: Mapping[str, str]) -> dict[str, Endpoint]:
    """The presets, ``OLLAMA_HOST`` applied, plus every ``AUGUR_ENDPOINT_<NAME>_URL``."""

    def get(name: str) -> str | None:
        value = environ.get(name, "").strip()
        return value or None

    found = dict(PRESETS)
    host = get("OLLAMA_HOST")
    if host:
        url = host if host.startswith("http") else f"http://{host}"
        found["ollama"] = found["ollama"].model_copy(update={"base_url": url})
    prefix, suffix = "AUGUR_ENDPOINT_", "_URL"
    for var in sorted(environ):
        if not (var.startswith(prefix) and var.endswith(suffix)):
            continue
        raw = var[len(prefix) : -len(suffix)]
        name = raw.lower()
        if not _NAME.match(name):
            raise ConfigError(
                f"{var}: an endpoint name is letters, digits, '-' or '_' (got {raw!r})"
            )
        found_url = get(var)
        if found_url is None:
            continue
        stem = f"{prefix}{raw}"
        billing = (get(f"{stem}_BILLING") or "unpriced").lower()
        api = (get(f"{stem}_API") or "openai").lower()
        try:
            found[name] = Endpoint.model_validate(
                {
                    "name": name,
                    "base_url": found_url,
                    "key_env": get(f"{stem}_KEY_ENV"),
                    "api": api,
                    "billing": billing,
                    "rpm": get(f"{stem}_RPM") or 60.0,
                }
            )
        except ValueError:
            raise ConfigError(
                f"invalid endpoint {name}: check {stem}_API (openai or ollama), "
                f"{stem}_BILLING (openrouter, free or unpriced) and {stem}_RPM"
            ) from None
    return found


def parse_route(text: str, endpoints: Mapping[str, Endpoint]) -> Route:
    """``openrouter:openai/gpt-5@high`` -> Route; the endpoint must be known.

    A trailing ``@minimal|low|medium|high|xhigh`` is the route's reasoning effort.
    """
    name, sep, model = text.strip().partition(":")
    name = name.strip().lower()
    model = model.strip()
    if not sep or not model:
        raise ConfigError(
            f"model entry {text.strip()!r} needs the form endpoint:model, "
            "for example openrouter:google/gemma-4-31b-it:free or ollama:qwen2.5:7b"
        )
    if name not in endpoints:
        raise ConfigError(
            f"unknown endpoint {name!r} in {text.strip()!r}; known: "
            f"{', '.join(sorted(endpoints))} (add one with AUGUR_ENDPOINT_<NAME>_URL)"
        )
    effort: Effort | None = None
    head, at, tail = model.rpartition("@")
    # Only a trailing ``@<effort>``: a leading ``@preset/...`` id is a model.
    if at and head:
        if tail.lower() not in EFFORTS:
            raise ConfigError(
                f"unknown reasoning effort {tail!r} in {text.strip()!r}; "
                f"allowed: {', '.join(EFFORTS)} (for example {name}:{head}@high)"
            )
        model, effort = head.strip(), cast(Effort, tail.lower())
    return Route(endpoint=endpoints[name], model=model, reasoning_effort=effort)


def parse_chain(text: str, endpoints: Mapping[str, Endpoint]) -> tuple[Route, ...]:
    """A comma-separated, ordered, de-duplicated chain of routes."""
    routes: list[Route] = []
    for part in text.split(","):
        if part.strip():
            route = parse_route(part, endpoints)
            if route not in routes:
                routes.append(route)
    if not routes:
        raise ConfigError("an empty model list; give at least one endpoint:model")
    return tuple(routes)
