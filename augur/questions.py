"""The ``Question`` model, parsed from a Metaculus post.

Binary, multiple choice, numeric and discrete questions are forecast (M3);
date questions and anything else parse so that they can be listed and
skipped with a reason. A numeric or discrete question carries its
:class:`~augur.distributions.Scaling`, which fixes the CDF Metaculus accepts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from augur.clock import ensure_utc, parse_iso
from augur.distributions import Scaling

QuestionType = Literal["binary", "multiple_choice", "numeric", "discrete", "date", "other"]
_KNOWN: frozenset[str] = frozenset({"binary", "multiple_choice", "numeric", "discrete", "date"})
# The types Augur forecasts.
FORECAST_TYPES: frozenset[str] = frozenset({"binary", "multiple_choice", "numeric", "discrete"})
CONTINUOUS_TYPES: frozenset[str] = frozenset({"numeric", "discrete"})


class MyForecast(BaseModel):
    """The account's latest forecast on a question, from ``my_forecasts.latest``."""

    model_config = ConfigDict(frozen=True)

    start_time: datetime | None = None
    end_time: datetime | None = None
    forecast_values: list[float] | None = None


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    post_id: int
    question_id: int
    title: str
    type: QuestionType
    status: str
    url: str
    open_time: datetime
    scheduled_close_time: datetime
    scheduled_resolve_time: datetime | None = None
    description: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    options: list[str] | None = None
    scaling: Scaling | None = None
    scaling_error: str = ""
    unit: str = ""
    my_latest_forecast: MyForecast | None = None

    @field_validator("open_time", "scheduled_close_time", "scheduled_resolve_time")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    def already_forecast(self, at: datetime) -> bool:
        """True when the account holds a standing forecast at ``at``."""
        latest = self.my_latest_forecast
        if latest is None or latest.forecast_values is None:
            return False
        return latest.end_time is None or latest.end_time > at

    def unsupported_reason(self) -> str | None:
        """Why this question cannot be forecast, or ``None`` when it can."""
        if self.type not in FORECAST_TYPES:
            return f"{self.type} questions are not forecast"
        if self.type == "multiple_choice" and not self.options:
            return "multiple choice with no options listed"
        if self.type in CONTINUOUS_TYPES and self.scaling is None:
            return f"no usable scaling ({self.scaling_error or 'missing'})"
        return None

    def is_open(self, at: datetime) -> bool:
        return self.status == "open" and self.open_time <= at < self.scheduled_close_time


def _time(value: Any) -> datetime | None:
    """ISO 8601 text, or Unix seconds (``my_forecasts`` uses epoch floats)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    return parse_iso(value) if isinstance(value, str) and value else None


def question_from_post(post: dict[str, Any], site: str = "https://www.metaculus.com") -> Question:
    """Build a :class:`Question` from a ``/api/posts/{id}/`` payload.

    Group and conditional posts carry no single ``question`` and are refused.
    """
    q = post.get("question")
    if not isinstance(q, dict):
        raise ValueError(f"post {post.get('id')} has no single question (group or conditional)")
    qtype = q.get("type")
    latest_raw = (q.get("my_forecasts") or {}).get("latest")
    latest = None
    if isinstance(latest_raw, dict):
        latest = MyForecast(
            start_time=_time(latest_raw.get("start_time")),
            end_time=_time(latest_raw.get("end_time")),
            forecast_values=latest_raw.get("forecast_values"),
        )
    open_time = _time(q.get("open_time")) or _time(post.get("open_time"))
    close_time = _time(q.get("scheduled_close_time")) or _time(post.get("scheduled_close_time"))
    scaling: Scaling | None = None
    scaling_error = ""
    if qtype in CONTINUOUS_TYPES:
        try:
            scaling = Scaling.from_question(q)
        except (ValueError, TypeError, KeyError) as exc:
            scaling_error = str(exc)
    if open_time is None or close_time is None:
        raise ValueError(f"post {post.get('id')} lacks an open or close time")
    return Question(
        post_id=int(post["id"]),
        question_id=int(q["id"]),
        title=str(q.get("title") or post.get("title") or ""),
        type=qtype if qtype in _KNOWN else "other",
        status=str(q.get("status") or post.get("status") or "unknown"),
        url=f"{site}/questions/{post['id']}/",
        open_time=open_time,
        scheduled_close_time=close_time,
        scheduled_resolve_time=_time(q.get("scheduled_resolve_time")),
        description=str(q.get("description") or ""),
        resolution_criteria=str(q.get("resolution_criteria") or ""),
        fine_print=str(q.get("fine_print") or ""),
        options=q.get("options"),
        scaling=scaling,
        scaling_error=scaling_error,
        unit=str(q.get("unit") or ""),
        my_latest_forecast=latest,
    )
