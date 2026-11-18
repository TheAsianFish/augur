from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from augur.clock import FixedClock, ensure_utc, parse_iso, to_iso
from augur.config import Settings
from augur.errors import ConfigError


def test_naive_datetimes_are_refused() -> None:
    with pytest.raises(ValueError, match="naive"):
        ensure_utc(datetime(2026, 1, 1))  # noqa: DTZ001


def test_iso_round_trip_is_fixed_width_and_ordered() -> None:
    a = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=-8)))
    b = datetime(2026, 1, 1, 8, 0, 0, 1, tzinfo=UTC)
    assert to_iso(a) == "2026-01-01T08:00:00.000000Z"
    assert to_iso(a) < to_iso(b)
    assert parse_iso(to_iso(b)) == b
    assert parse_iso("2026-11-17T18:32:14Z") == datetime(2026, 11, 17, 18, 32, 14, tzinfo=UTC)


def test_fixed_clock_advances() -> None:
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    clock.advance(90)
    assert clock.now() == datetime(2026, 1, 1, 0, 1, 30, tzinfo=UTC)


def test_settings_from_env_reads_either_asknews_secret_name() -> None:
    s = Settings.from_env(
        {
            "METACULUS_TOKEN": "tok-abcdefgh12345",
            "ASKNEWS_CLIENT_ID": "id",
            "ASKNEWS_SECRET": "sec",
            "AUGUR_MAX_QUESTIONS_PER_RUN": "3",
            "OLLAMA_HOST": "127.0.0.1:11434",
        }
    )
    assert s.has_metaculus_token
    assert s.has_asknews
    assert s.max_questions_per_run == 3
    assert s.chains()["forecast"][0].endpoint.base_url == "http://127.0.0.1:11434"


def test_public_settings_never_carry_secrets() -> None:
    s = Settings.from_env({"METACULUS_TOKEN": "tok-abcdefgh12345"})
    public = s.public_dict()
    assert public["metaculus_token"] is True
    assert "tok-abcdefgh12345" not in repr(public)
    assert "tok-abcdefgh12345" not in repr(s)


def test_invalid_settings_name_the_field() -> None:
    with pytest.raises(ConfigError, match="max_questions_per_run"):
        Settings.from_env({"AUGUR_MAX_QUESTIONS_PER_RUN": "-1"})
