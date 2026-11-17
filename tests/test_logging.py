"""No token or secret may reach a log line."""

from __future__ import annotations

import io
import json

import pytest

from augur.log import Redactor, configure_logging, get_logger

TOKEN = "mtok_9f8e7d6c5b4a39281706f5e4d3c2b1a0"
SECRET = "askn_secret_0123456789abcdef"


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    monkeypatch.setenv("METACULUS_TOKEN", TOKEN)
    monkeypatch.setenv("ASKNEWS_CLIENT_SECRET", SECRET)
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    return stream


def test_secrets_are_redacted_everywhere_in_an_event(captured: io.StringIO) -> None:
    log = get_logger("test").bind(run_id="run_1", question_id=7)
    log.info(
        f"calling with {TOKEN}",
        url=f"https://example.org/?key={SECRET}",
        headers={"Authorization": f"Token {TOKEN}", "Accept": "json"},
        nested=[{"deep": f"x{TOKEN}y"}],
        authorization="Bearer abcdefghijklmnop",
    )
    try:
        raise RuntimeError(f"server said {TOKEN}")
    except RuntimeError:
        log.exception("failed")
    out = captured.getvalue()
    assert TOKEN not in out
    assert SECRET not in out
    assert "abcdefghijklmnop" not in out
    lines = [json.loads(line) for line in out.splitlines()]
    assert lines[0]["run_id"] == "run_1"
    assert lines[0]["question_id"] == 7
    assert lines[0]["headers"]["Accept"] == "json"
    assert "[redacted]" in lines[0]["event"]


def test_auth_scheme_values_are_redacted_without_knowing_the_secret() -> None:
    redactor = Redactor(environ={})
    event = redactor(None, "info", {"event": "header Token zzzzzzzzzzzzzzzz sent"})
    assert "zzzzzzzzzzzzzzzz" not in event["event"]


def test_short_secret_values_do_not_redact_ordinary_text() -> None:
    redactor = Redactor(environ={"METACULUS_TOKEN": "a"})
    assert redactor(None, "info", {"event": "a cat"})["event"] == "a cat"


def test_bad_level_says_what_to_use() -> None:
    with pytest.raises(ValueError, match="DEBUG, INFO or WARNING"):
        configure_logging("LOUD")
