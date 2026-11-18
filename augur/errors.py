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


class SearchError(AugurError):
    """A searcher failed; the question continues with the other searchers."""


class ForecastParseError(AugurError):
    """A model's answer did not contain a usable probability."""


class LeakageError(AugurError):
    """A source or model would let a forecast see past its cutoff."""


class LedgerError(AugurError):
    """The ledger refused a write or failed its integrity rules."""
