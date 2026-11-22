"""Settings, read from the environment only. Secrets are ``SecretStr`` and never logged."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from augur.errors import ConfigError
from augur.llm.endpoints import Endpoint, Role, Route, endpoints_from_env, parse_chain

REPLAY = "replay"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    metaculus_token: SecretStr | None = None
    metaculus_base_url: str = "https://www.metaculus.com/api"
    asknews_client_id: SecretStr | None = None
    asknews_client_secret: SecretStr | None = None
    # One endpoint and model for both roles, or ``replay``. The chains below,
    # when set, take precedence.
    provider: str = "ollama"
    model: str = "qwen2.5:7b"
    forecast_models: str | None = None
    research_models: str | None = None
    endpoints: dict[str, Endpoint] = Field(default_factory=dict)
    # Endpoint keys by the name of the variable that holds them.
    keys: dict[str, SecretStr] = Field(default_factory=dict)
    probe: bool = True
    replay_path: Path | None = None
    record_path: Path | None = None
    ledger_path: Path = Path("augur.db")
    max_questions_per_run: int = Field(default=10, ge=0)
    max_cost_per_question: float = Field(default=0.50, ge=0)
    max_cost_per_run: float = Field(default=5.00, ge=0)
    forecasters: int = Field(default=5, ge=1, le=25)
    # Deadlines. A question stops being worked on this long before it closes;
    # one with less than ``min_forecast_minutes`` left before that is missed,
    # not started; and the whole run stops starting questions after
    # ``run_deadline_minutes`` (the scheduled job's timeout is above it).
    close_margin_minutes: float = Field(default=5.0, ge=0)
    min_forecast_minutes: float = Field(default=4.0, gt=0)
    run_deadline_minutes: float = Field(default=30.0, gt=0)
    # The fleet (M7). The defaults are the single-researcher, single-prompt,
    # one-question-at-a-time run every earlier milestone made.
    research_passes: int = Field(default=1, ge=1, le=2)
    # Forecaster prompt variants (names in ``augur.agents.forecaster.VARIANTS``);
    # member m takes variant m % len.
    forecaster_variants: tuple[str, ...] = ("base",)
    # Per-member model chains, ``;``-separated, each in AUGUR_FORECAST_MODELS's
    # syntax; member m takes chain m % len. ``None``: the forecast chain.
    member_models: str | None = None
    question_concurrency: int = Field(default=1, ge=1, le=8)
    log_level: str = "INFO"

    @field_validator("forecaster_variants")
    @classmethod
    def _known_variants(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [v for v in value if v not in _variant_names()]
        if not value or unknown:
            raise ValueError(f"unknown forecaster variant(s) {unknown}")
        return value

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Settings:
        def get(name: str) -> str | None:
            value = environ.get(name, "").strip()
            return value or None

        if (get("AUGUR_PROVIDER") or "").lower() == "metaculus_proxy":
            raise ConfigError(
                "AUGUR_PROVIDER=metaculus_proxy is gone: llm-proxy.metaculus.com no longer "
                "resolves and Metaculus now issues credits as an OpenRouter key. Set "
                "OPENROUTER_API_KEY and AUGUR_FORECAST_MODELS=openrouter:<model> instead"
            )
        endpoints = endpoints_from_env(environ)
        variants = _parse_variants(get("AUGUR_FORECASTER_VARIANTS"))
        keys = {
            e.key_env: SecretStr(value)
            for e in endpoints.values()
            if e.key_env and (value := get(e.key_env)) is not None
        }
        raw: dict[str, object] = {
            "metaculus_token": get("METACULUS_TOKEN"),
            "metaculus_base_url": get("AUGUR_METACULUS_BASE_URL"),
            "asknews_client_id": get("ASKNEWS_CLIENT_ID"),
            "asknews_client_secret": get("ASKNEWS_CLIENT_SECRET") or get("ASKNEWS_SECRET"),
            "provider": (get("AUGUR_PROVIDER") or "").lower() or None,
            "model": get("AUGUR_MODEL"),
            "forecast_models": get("AUGUR_FORECAST_MODELS"),
            "research_models": get("AUGUR_RESEARCH_MODELS"),
            "endpoints": endpoints,
            "keys": keys,
            "probe": None
            if get("AUGUR_PROBE") is None
            else (get("AUGUR_PROBE") or "").lower() not in {"0", "false", "no", "off"},
            "replay_path": get("AUGUR_REPLAY_PATH"),
            "record_path": get("AUGUR_RECORD_PATH"),
            "ledger_path": get("AUGUR_LEDGER"),
            "max_questions_per_run": get("AUGUR_MAX_QUESTIONS_PER_RUN"),
            "max_cost_per_question": get("AUGUR_MAX_COST_PER_QUESTION"),
            "max_cost_per_run": get("AUGUR_MAX_COST_PER_RUN"),
            "forecasters": get("AUGUR_FORECASTERS"),
            "close_margin_minutes": get("AUGUR_CLOSE_MARGIN_MINUTES"),
            "min_forecast_minutes": get("AUGUR_MIN_FORECAST_MINUTES"),
            "run_deadline_minutes": get("AUGUR_RUN_DEADLINE_MINUTES"),
            "research_passes": get("AUGUR_RESEARCH_PASSES"),
            "forecaster_variants": variants,
            "member_models": get("AUGUR_MEMBER_MODELS"),
            "question_concurrency": get("AUGUR_QUESTION_CONCURRENCY"),
            "log_level": get("AUGUR_LOG_LEVEL"),
        }
        try:
            settings = cls.model_validate({k: v for k, v in raw.items() if v is not None})
        except ValidationError as exc:
            fields = ", ".join(str(err["loc"][0]) for err in exc.errors())
            raise ConfigError(
                f"invalid settings ({fields}); check the AUGUR_* variables in CONTRIBUTING.md"
            ) from None
        settings.chains()  # fail now, not mid-run, on a bad model list
        settings.member_chains()
        return settings

    @property
    def is_replay(self) -> bool:
        return self.provider == REPLAY

    def _endpoints(self) -> dict[str, Endpoint]:
        return self.endpoints or endpoints_from_env({})

    def chains(self) -> dict[Role, tuple[Route, ...]]:
        """The ordered model chain for each role."""
        endpoints = self._endpoints()
        if self.forecast_models:
            forecast = parse_chain(self.forecast_models, endpoints)
        else:
            endpoint = "ollama" if self.is_replay else self.provider
            forecast = parse_chain(f"{endpoint}:{self.model}", endpoints)
        research = (
            parse_chain(self.research_models, endpoints) if self.research_models else forecast
        )
        return {"research": research, "forecast": forecast}

    def member_chains(self) -> tuple[tuple[Route, ...], ...]:
        """Each member's own model chain, in order; empty: members use the forecast chain."""
        if not self.member_models:
            return ()
        endpoints = self._endpoints()
        parts = self.member_models.split(";")
        if any(not part.strip() for part in parts):
            raise ConfigError(
                f"AUGUR_MEMBER_MODELS {self.member_models!r} has an empty member chain; give "
                "one endpoint:model,... list per member, separated by ';' "
                "(for example openrouter:openai/gpt-5;openrouter:anthropic/claude-sonnet-4.5)"
            )
        return tuple(parse_chain(part, endpoints) for part in parts)

    @property
    def has_metaculus_token(self) -> bool:
        return self.metaculus_token is not None

    @property
    def has_asknews(self) -> bool:
        return self.asknews_client_id is not None and self.asknews_client_secret is not None

    def public_dict(self) -> dict[str, object]:
        """Settings safe to write to the ledger: secrets reduced to whether they are set."""
        data = self.model_dump(mode="json", exclude={"keys", "endpoints"})
        for name in ("metaculus_token", "asknews_client_id", "asknews_client_secret"):
            data[name] = getattr(self, name) is not None
        data["keys_set"] = sorted(self.keys)
        chains = self.chains()
        data["chains"] = {role: [r.key for r in routes] for role, routes in chains.items()}
        members = self.member_chains()
        data["member_chains"] = [[r.key for r in routes] for routes in members]
        used = {r.endpoint.name for routes in (*chains.values(), *members) for r in routes}
        data["endpoints"] = {
            name: e.model_dump(mode="json") for name, e in self._endpoints().items() if name in used
        }
        return data


def _variant_names() -> tuple[str, ...]:
    # Imported here: the agents import half the package, and settings load first.
    from augur.agents.forecaster import VARIANTS

    return tuple(VARIANTS)


def _parse_variants(raw: str | None) -> tuple[str, ...] | None:
    """``AUGUR_FORECASTER_VARIANTS``: a comma list of known variant names, or ``None`` (unset)."""
    if raw is None:
        return None
    names = tuple(v.strip().lower() for v in raw.split(","))
    known = _variant_names()
    unknown = [v for v in names if v not in known]
    if unknown or not names:
        raise ConfigError(
            f"AUGUR_FORECASTER_VARIANTS {raw!r}: unknown variant(s) "
            f"{', '.join(repr(v) for v in unknown)}; use a comma list of {', '.join(known)}"
        )
    return names
