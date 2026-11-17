"""Structured JSON logs to stderr, with secrets scrubbed from every event.

Redaction runs as the last processor before rendering, so nothing a caller
binds (a URL, a header dict, an exception message) can carry a token out.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, TextIO

import structlog

SECRET_ENV_VARS: tuple[str, ...] = (
    "METACULUS_TOKEN",
    "ASKNEWS_CLIENT_ID",
    "ASKNEWS_CLIENT_SECRET",
    "ASKNEWS_SECRET",
    "ASKNEWS_API_KEY",
)
REDACTED = "[redacted]"
_SENSITIVE_KEY = re.compile(r"(authorization|token|secret|password|api[_-]?key|cookie)", re.I)
_AUTH_VALUE = re.compile(r"\b(Token|Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}", re.I)


def _secret_values(environ: Mapping[str, str]) -> list[str]:
    # Settings strip whitespace, so a padded value must be matched stripped too.
    values = [environ.get(name, "").strip() for name in SECRET_ENV_VARS]
    # Short values would redact ordinary words; real tokens are long.
    return sorted({v for v in values if len(v) >= 8}, key=len, reverse=True)


def redact_text(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, REDACTED)
    return _AUTH_VALUE.sub(lambda m: f"{m.group(1)} {REDACTED}", text)


def _redact(value: Any, secrets: list[str], key: str | None = None) -> Any:
    if key is not None and _SENSITIVE_KEY.search(key) and isinstance(value, str | bytes):
        return REDACTED
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, bytes):
        return redact_text(value.decode("utf-8", "replace"), secrets)
    if isinstance(value, Mapping):
        return {str(k): _redact(v, secrets, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_redact(v, secrets) for v in value]
    if isinstance(value, int | float | bool) or value is None:
        return value
    return redact_text(repr(value), secrets)


class Redactor:
    """structlog processor that removes secrets from every field."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ

    def __call__(
        self, _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        secrets = _secret_values(self._environ if self._environ is not None else os.environ)
        return {k: _redact(v, secrets, k) for k, v in event_dict.items()}


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """JSON lines to ``stream`` (stderr by default) at ``level``."""
    numeric = logging.getLevelNamesMapping().get(level.upper())
    if numeric is None:
        raise ValueError(f"AUGUR_LOG_LEVEL={level!r} is not a level; use DEBUG, INFO or WARNING")
    # httpx logs each request URL at INFO; keep it quiet unless debugging.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            Redactor(),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger
