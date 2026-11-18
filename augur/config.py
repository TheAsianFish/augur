"""Settings, read from the environment only. Secrets are ``SecretStr`` and never logged."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from augur.errors import ConfigError

ProviderName = Literal["ollama", "metaculus_proxy", "replay"]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    metaculus_token: SecretStr | None = None
    metaculus_base_url: str = "https://www.metaculus.com/api"
    asknews_client_id: SecretStr | None = None
    asknews_client_secret: SecretStr | None = None
    provider: ProviderName = "ollama"
    model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    proxy_base_url: str = "https://llm-proxy.metaculus.com/proxy/openai/v1"
    replay_path: Path | None = None
    record_path: Path | None = None
    ledger_path: Path = Path("augur.db")
    max_questions_per_run: int = Field(default=10, ge=0)
    max_cost_per_question: float = Field(default=0.50, ge=0)
    max_cost_per_run: float = Field(default=5.00, ge=0)
    forecasters: int = Field(default=5, ge=1, le=25)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Settings:
        def get(name: str) -> str | None:
            value = environ.get(name, "").strip()
            return value or None

        raw: dict[str, object] = {
            "metaculus_token": get("METACULUS_TOKEN"),
            "metaculus_base_url": get("AUGUR_METACULUS_BASE_URL"),
            "asknews_client_id": get("ASKNEWS_CLIENT_ID"),
            "asknews_client_secret": get("ASKNEWS_CLIENT_SECRET") or get("ASKNEWS_SECRET"),
            "provider": get("AUGUR_PROVIDER"),
            "model": get("AUGUR_MODEL"),
            "ollama_base_url": get("OLLAMA_HOST") and _ollama_url(get("OLLAMA_HOST") or ""),
            "proxy_base_url": get("AUGUR_PROXY_BASE_URL"),
            "replay_path": get("AUGUR_REPLAY_PATH"),
            "record_path": get("AUGUR_RECORD_PATH"),
            "ledger_path": get("AUGUR_LEDGER"),
            "max_questions_per_run": get("AUGUR_MAX_QUESTIONS_PER_RUN"),
            "max_cost_per_question": get("AUGUR_MAX_COST_PER_QUESTION"),
            "max_cost_per_run": get("AUGUR_MAX_COST_PER_RUN"),
            "forecasters": get("AUGUR_FORECASTERS"),
            "log_level": get("AUGUR_LOG_LEVEL"),
        }
        try:
            return cls.model_validate({k: v for k, v in raw.items() if v is not None})
        except ValidationError as exc:
            fields = ", ".join(str(err["loc"][0]) for err in exc.errors())
            raise ConfigError(
                f"invalid settings ({fields}); check the AUGUR_* variables in CONTRIBUTING.md"
            ) from None

    @property
    def has_metaculus_token(self) -> bool:
        return self.metaculus_token is not None

    @property
    def has_asknews(self) -> bool:
        return self.asknews_client_id is not None and self.asknews_client_secret is not None

    def public_dict(self) -> dict[str, object]:
        """Settings safe to write to the ledger: secrets reduced to whether they are set."""
        data = self.model_dump(mode="json")
        for name in ("metaculus_token", "asknews_client_id", "asknews_client_secret"):
            data[name] = getattr(self, name) is not None
        return data


def _ollama_url(host: str) -> str:
    return host if host.startswith("http") else f"http://{host}"
