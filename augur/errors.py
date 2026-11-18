"""Typed exceptions. Each message says what to do next."""

from __future__ import annotations


class AugurError(Exception):
    """Base class for every error Augur raises on purpose."""


class ConfigError(AugurError):
    """A setting is missing or invalid."""


class MissingCredentialsError(ConfigError):
    """A component needs a credential that is not in the environment."""

    def __init__(self, component: str, variables: tuple[str, ...]) -> None:
        names = ", ".join(variables)
        super().__init__(
            f"{component} needs {names} in the environment; set it (see .env.example) "
            f"or run without {component}."
        )
        self.component = component
        self.variables = variables


class SubmitFloorError(ConfigError):
    """``--submit`` to a live tournament with a model below the frontier tier."""


class MetaculusError(AugurError):
    """The Metaculus API refused or failed a request."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderError(AugurError):
    """A model provider failed a call."""


class ReplayMissError(ProviderError):
    """The replay provider has no recording for a request."""


class FallbackError(ProviderError):
    """This route cannot answer now; the router moves to the next one.

    ``kind`` names the reason in the ledger's ``model_switches`` rows.
    """

    kind = "unavailable"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        limit_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.limit_headers = dict(limit_headers or {})


class RateLimitedError(FallbackError):
    """HTTP 429 that retries within the provider did not clear."""

    kind = "rate_limited"


class OutOfCreditError(FallbackError):
    """The key's balance or spending limit is used up (HTTP 402, or a 403 that says so)."""

    kind = "out_of_credit"


class EndpointAuthError(FallbackError):
    """The endpoint refused the key: missing, expired or revoked."""

    kind = "auth"


class ModelUnavailableError(FallbackError):
    """The endpoint does not serve this model, or kept failing (5xx, unreachable)."""

    kind = "unavailable"


class EmptyAnswerError(FallbackError):
    """The model returned no text (a reasoning model can spend ``max_tokens`` thinking).

    The call moves to the next model, but the route stays in use: a request
    with more room may get an answer.
    """

    kind = "empty"


class UnpricedModelError(FallbackError):
    """The endpoint bills, and Augur has no price for the model, so it is not called."""

    kind = "unpriced"


class BelowFloorError(ProviderError):
    """The next route in a scored run is below the frontier tier; the question is skipped."""


class ModelsExhaustedError(ProviderError):
    """Every route in a role's chain failed or is cooling down."""


class SearchError(AugurError):
    """A searcher failed; the question continues with the other searchers."""


class ForecastParseError(AugurError):
    """A model's answer did not contain a usable probability."""


class LeakageError(AugurError):
    """A source or model would let a forecast see past its cutoff."""


class LedgerError(AugurError):
    """The ledger refused a write or failed its integrity rules."""
