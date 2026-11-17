from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from augur.clock import FixedClock
from augur.ledger import Ledger

T0 = datetime(2026, 11, 17, 18, 0, tzinfo=UTC)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(T0)


@pytest.fixture
def ledger(tmp_path: Path, clock: FixedClock) -> Iterator[Ledger]:
    with Ledger.open(tmp_path / "ledger.sqlite", clock) as led:
        yield led
