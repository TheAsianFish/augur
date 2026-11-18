"""Prices and listing dates from OpenRouter's public model list.

``GET https://openrouter.ai/api/v1/models`` needs no key and gives, for every
model it serves, the price per token and ``created`` (when OpenRouter listed
it). Two uses:

* **Prices.** A call is reserved and settled at the listed price, so the cost
  caps bind on what the model really costs. A model with no price on a host
  that bills is not called (:class:`~augur.errors.UnpricedModelError`); it is
  never assumed to cost frontier rates (M1's rule), which would let the caps
  stop a run on a model that costs nothing.
* **Release dates.** A model cannot be listed before it is released, so the
  listing date is an upper bound on the release date: using it in the
  backtest gate ("released before the question opened") can only exclude
  more questions, never admit one the model might remember.

A trimmed copy of the list is committed (``openrouter_models.json``) for
offline runs and tests; a live run fetches the list and falls back to the copy
with a warning.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from augur.clock import parse_iso, to_iso
from augur.errors import ProviderError
from augur.llm.endpoints import Route
from augur.llm.registry import FRONTIER_TIER, bare_model_name, local_release_date
from augur.log import get_logger

log = get_logger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
SNAPSHOT_PATH = Path(__file__).with_name("openrouter_models.json")


class CatalogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    listed_on: date
    usd_per_million_input: float | None
    usd_per_million_output: float | None
    context_length: int | None = None

    @property
    def priced(self) -> bool:
        return self.usd_per_million_input is not None and self.usd_per_million_output is not None


class Price(BaseModel):
    model_config = ConfigDict(frozen=True)

    usd_per_million_input: float
    usd_per_million_output: float
    basis: str

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> float:
        return (
            (input_tokens or 0) * self.usd_per_million_input
            + (output_tokens or 0) * self.usd_per_million_output
        ) / 1_000_000


FREE = Price(usd_per_million_input=0.0, usd_per_million_output=0.0, basis="free endpoint")
LOCAL = Price(usd_per_million_input=0.0, usd_per_million_output=0.0, basis="local")


def _per_million(value: object) -> float | None:
    """OpenRouter prices are strings in USD per token; ``-1`` means "varies"."""
    try:
        per_token = float(str(value))
    except ValueError:
        return None
    if per_token < 0:
        return None
    return round(per_token * 1_000_000, 6)


class Catalog:
    def __init__(
        self,
        entries: dict[str, CatalogEntry],
        fetched_at: datetime,
        origin: Literal["live", "snapshot", "empty"],
    ) -> None:
        self.entries = entries
        self.fetched_at = fetched_at
        self.origin = origin
        self._by_bare: dict[str, list[CatalogEntry]] = {}
        for entry in entries.values():
            self._by_bare.setdefault(bare_model_name(entry.id), []).append(entry)

    @classmethod
    def empty(cls) -> Catalog:
        return cls({}, datetime(1970, 1, 1, tzinfo=UTC), "empty")

    @classmethod
    def from_api(
        cls,
        payload: dict[str, Any],
        fetched_at: datetime,
        origin: Literal["live", "snapshot"] = "live",
    ) -> Catalog:
        entries: dict[str, CatalogEntry] = {}
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ProviderError("OpenRouter's model list has no 'data' array; retry later")
        for row in rows:
            try:
                created = int(row["created"])
                pricing = row.get("pricing") or {}
                entry = CatalogEntry(
                    id=str(row["id"]),
                    listed_on=datetime.fromtimestamp(created, UTC).date(),
                    usd_per_million_input=_per_million(pricing.get("prompt")),
                    usd_per_million_output=_per_million(pricing.get("completion")),
                    context_length=row.get("context_length"),
                )
            except (KeyError, TypeError, ValueError):
                continue  # one odd row must not hide the rest
            entries[entry.id] = entry
        return cls(entries, fetched_at, origin)

    def get(self, model: str) -> CatalogEntry | None:
        return self.entries.get(model)

    def same_model(self, model: str) -> list[CatalogEntry]:
        """Entries whose bare name matches (``groq:llama-x`` -> ``meta-llama/llama-x``)."""
        return list(self._by_bare.get(bare_model_name(model), []))

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "source": OPENROUTER_MODELS_URL,
            "fetched_at": to_iso(self.fetched_at),
            "note": "Trimmed: id, created, per-token prices and context length only.",
            "data": [
                {
                    "id": e.id,
                    "created": int(
                        datetime(
                            e.listed_on.year, e.listed_on.month, e.listed_on.day, tzinfo=UTC
                        ).timestamp()
                    ),
                    "pricing": {
                        "prompt": _token_price(e.usd_per_million_input),
                        "completion": _token_price(e.usd_per_million_output),
                    },
                    "context_length": e.context_length,
                }
                for e in sorted(self.entries.values(), key=lambda e: e.id)
            ],
        }


def snapshot_text(catalog: Catalog) -> str:
    """The snapshot as JSON with one model per line, so a refresh diffs by model."""
    snap = catalog.to_snapshot()
    head = {k: v for k, v in snap.items() if k != "data"}
    rows = ",\n".join(
        "  " + json.dumps(row, sort_keys=True, separators=(",", ":")) for row in snap["data"]
    )
    opening = json.dumps(head, sort_keys=True, indent=1)[:-2]
    return f'{opening},\n "data": [\n{rows}\n ]\n}}\n'


def _token_price(per_million: float | None) -> str:
    if per_million is None:
        return "-1"
    return f"{per_million / 1_000_000:.12f}".rstrip("0").rstrip(".") or "0"


async def fetch_catalog(
    http: httpx.AsyncClient, now: datetime, url: str = OPENROUTER_MODELS_URL
) -> Catalog:
    """The live list; on any failure the committed snapshot, with a warning."""
    try:
        resp = await http.get(url, timeout=30.0)
        resp.raise_for_status()
        return Catalog.from_api(resp.json(), now, "live")
    except (httpx.HTTPError, ValueError, ProviderError) as exc:
        log.warning(
            "model list unavailable; using the committed snapshot",
            error=type(exc).__name__,
            snapshot=SNAPSHOT_PATH.name,
        )
        return load_snapshot()


def load_snapshot(path: Path = SNAPSHOT_PATH) -> Catalog:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("no model list snapshot; every billed model is unpriced", path=str(path))
        return Catalog.empty()
    return Catalog.from_api(payload, parse_iso(payload["fetched_at"]), "snapshot")


def price_for(route: Route, catalog: Catalog) -> Price | None:
    """What a call on ``route`` costs per token, or ``None`` if unknown on a billed host."""
    endpoint = route.endpoint
    if endpoint.is_local:
        return LOCAL
    if endpoint.billing == "free":
        if bare_model_name(route.model) not in FRONTIER_TIER:
            return FREE
        # A frontier model on a "free" host (Gemini's free tier) bills once the
        # key has billing on, so it is priced as OpenRouter lists it and the
        # caps still bind; with no listing it is not called.
        listed = [e for e in catalog.same_model(route.model) if e.priced]
        if not listed:
            return None
        return Price(
            usd_per_million_input=max(e.usd_per_million_input or 0.0 for e in listed),
            usd_per_million_output=max(e.usd_per_million_output or 0.0 for e in listed),
            basis=f"OpenRouter list, same model ({catalog.origin}, {catalog.fetched_at:%Y-%m-%d})",
        )
    if endpoint.billing == "openrouter":
        entry = catalog.get(route.model)
        if entry is not None and entry.priced:
            assert entry.usd_per_million_input is not None
            assert entry.usd_per_million_output is not None
            return Price(
                usd_per_million_input=entry.usd_per_million_input,
                usd_per_million_output=entry.usd_per_million_output,
                basis=f"OpenRouter list ({catalog.origin}, {catalog.fetched_at:%Y-%m-%d})",
            )
    return None


def release_bound(route: Route, catalog: Catalog) -> tuple[date | None, str]:
    """An upper bound on the model's release date, and where it came from.

    Local models carry hand-checked release dates. Hosted ones use the
    OpenRouter listing date: the exact id on OpenRouter, otherwise the latest
    listing of a model with the same bare name (the latest, because a later
    bound only excludes more backtest questions).
    """
    known = local_release_date(route.model)
    if known is not None:
        return known, "release announcement"
    if route.endpoint.billing == "openrouter":
        entry = catalog.get(route.model)
        if entry is not None:
            return entry.listed_on, "OpenRouter listing date"
    same = catalog.same_model(route.model)
    if same:
        return max(e.listed_on for e in same), "OpenRouter listing date (same bare name)"
    return None, "unknown"
