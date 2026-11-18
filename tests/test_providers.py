from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from augur.config import Settings
from augur.errors import (
    ConfigError,
    ProviderError,
    ReplayMissError,
)
from augur.llm.base import LlmRequest, LlmResponse, Message
from augur.llm.factory import build_providers, default_http_client
from augur.llm.ollama import OllamaProvider
from augur.llm.openai_compat import OpenAICompatProvider
from augur.llm.replay import RecordingProvider, ReplayProvider
from augur.ratelimit import RateLimiter

REQ = LlmRequest(
    model="qwen2.5:7b",
    messages=(Message(role="system", content="s"), Message(role="user", content="u")),
    temperature=0.3,
    seed=7,
    max_tokens=50,
)


def client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


async def no_sleep(_: float) -> None:
    return None


class FakeTime:
    """A monotonic clock that only moves when the limiter sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def fast_limiter(time: FakeTime | None = None) -> RateLimiter:
    t = time or FakeTime()
    return RateLimiter(1000, burst=100, max_concurrent=4, monotonic=t.monotonic, sleep=t.sleep)


# Ollama


async def test_ollama_body_and_response() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"message": {"content": "hi"}, "prompt_eval_count": 11, "eval_count": 5},
        )

    p = OllamaProvider(client(httpx.MockTransport(handler)), "http://h:1/")
    out = await p.complete(REQ)
    assert str(seen[0].url) == "http://h:1/api/chat"
    assert json.loads(seen[0].content) == {
        "model": "qwen2.5:7b",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "stream": False,
        "options": {"temperature": 0.3, "seed": 7, "num_predict": 50, "num_ctx": 8192},
    }
    assert out == LlmResponse(
        text="hi", model="qwen2.5:7b", input_tokens=11, output_tokens=5, cost_usd=0.0
    )
    assert p.name == "ollama"


async def test_ollama_omits_seed_when_none() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"message": {"content": "x"}})

    p = OllamaProvider(client(httpx.MockTransport(handler)), "http://h")
    out = await p.complete(REQ.model_copy(update={"seed": None}))
    assert "seed" not in json.loads(seen[0].content)["options"]
    assert out.input_tokens is None


async def test_ollama_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    p = OllamaProvider(client(httpx.MockTransport(handler)), "http://h:1")
    with pytest.raises(ProviderError, match=r"ollama serve.*ollama pull qwen2\.5:7b"):
        await p.complete(REQ)


async def test_ollama_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen2.5:7b' not found"})

    p = OllamaProvider(client(httpx.MockTransport(handler)), "http://h")
    with pytest.raises(ProviderError, match=r"not pulled; run `ollama pull qwen2\.5:7b`"):
        await p.complete(REQ)


# Proxy


# Replay


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.closed = False

    async def complete(self, request: LlmRequest) -> LlmResponse:
        text = request.messages[-1].content.upper()
        return LlmResponse(
            text=text, model=request.model, input_tokens=3, output_tokens=4, cost_usd=0.25
        )

    async def aclose(self) -> None:
        self.closed = True


async def test_record_then_replay(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "rec.jsonl"
    inner = FakeProvider()
    rec = RecordingProvider(inner, path)
    assert rec.name == "fake"
    other = REQ.model_copy(update={"seed": 8})
    r1, r2 = await asyncio.gather(rec.complete(REQ), rec.complete(other))
    await rec.aclose()
    assert inner.closed
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["request_hash"] in {REQ.request_hash(), other.request_hash()}

    replay = ReplayProvider(path)
    assert replay.name == "replay"
    assert await replay.complete(REQ) == r1
    assert await replay.complete(other) == r2
    await replay.aclose()


async def test_replay_last_duplicate_wins(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    rows = [
        {
            "request_hash": REQ.request_hash(),
            "request": REQ.model_dump(mode="json"),
            "response": LlmResponse(text=t, model=REQ.model).model_dump(mode="json"),
        }
        for t in ("first", "second")
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert (await ReplayProvider(path).complete(REQ)).text == "second"


async def test_replay_miss(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ReplayMissError) as info:
        await ReplayProvider(path).complete(REQ)
    msg = str(info.value)
    assert "qwen2.5:7b" in msg
    assert REQ.request_hash()[:12] in msg
    assert str(path) in msg
    assert "AUGUR_RECORD_PATH" in msg


def test_replay_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="AUGUR_RECORD_PATH"):
        ReplayProvider(tmp_path / "nope.jsonl")


# Factory


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})


async def test_factory_selection(tmp_path: Path) -> None:
    http = client(httpx.MockTransport(ok_handler))
    ollama = build_providers(Settings(), http)
    assert list(ollama) == ["ollama"]
    assert isinstance(ollama["ollama"], OllamaProvider)
    settings = Settings(
        forecast_models="openrouter:openai/gpt-5,groq:llama-3.3-70b-versatile",
        research_models="ollama:qwen2.5:7b",
        keys={"OPENROUTER_API_KEY": SecretStr("k")},
    )
    built = build_providers(settings, http)
    assert sorted(built) == ["groq", "ollama", "openrouter"]
    assert isinstance(built["openrouter"], OpenAICompatProvider)
    rec = tmp_path / "r.jsonl"
    wrapped = build_providers(Settings(record_path=rec), http)["ollama"]
    assert isinstance(wrapped, RecordingProvider)
    assert wrapped.name == "ollama"
    rec.write_text("")
    replay = build_providers(
        settings.model_copy(update={"provider": "replay", "replay_path": rec}), http
    )
    assert {type(p) for p in replay.values()} == {ReplayProvider}
    await http.aclose()


async def test_factory_errors() -> None:
    http = client(httpx.MockTransport(ok_handler))
    with pytest.raises(ConfigError, match="AUGUR_REPLAY_PATH"):
        build_providers(Settings(provider="replay"), http)
    await http.aclose()


async def test_default_http_client() -> None:
    http = default_http_client()
    assert http.timeout == httpx.Timeout(600.0, connect=10.0)
    assert http.headers["User-Agent"].startswith("augur-forecast/")
    await http.aclose()
