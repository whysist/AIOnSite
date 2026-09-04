"""Provider tests using httpx.MockTransport -- no network, no API keys."""

from __future__ import annotations

import httpx
import pytest

from src.core.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderMalformedResponseError,
    ProviderTimeoutError,
)
from src.llm.ollama_provider import OllamaProvider
from src.llm.openai_provider import OpenAICompatibleProvider


def _mount(provider, handler):
    provider._client = httpx.AsyncClient(
        base_url=provider._base_url,
        transport=httpx.MockTransport(handler),
        headers={"Content-Type": "application/json"},
    )
    return provider


async def test_openai_compatible_happy_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "hello there"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    llm = _mount(OpenAICompatibleProvider(model="test-model"), handler)
    resp = await llm.generate([{"role": "user", "content": "hi"}])
    assert resp.content == "hello there"
    assert resp.usage.total_tokens == 5
    assert resp.provider == "openai"
    await llm.aclose()


async def test_openai_compatible_http_error_becomes_provider_error():
    def handler(_request):
        return httpx.Response(500, text="boom")

    llm = _mount(OpenAICompatibleProvider(model="m"), handler)
    with pytest.raises(ProviderError):
        await llm.generate([{"role": "user", "content": "hi"}])
    await llm.aclose()


async def test_ollama_happy_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:7b-instruct",
                "message": {"role": "assistant", "content": "pong"},
                "done": True,
                "prompt_eval_count": 4,
                "eval_count": 1,
            },
        )

    llm = _mount(OllamaProvider(model="qwen2.5:7b-instruct"), handler)
    resp = await llm.generate([{"role": "user", "content": "ping"}])
    assert resp.content == "pong"
    assert resp.usage.total_tokens == 5
    await llm.aclose()


async def test_ollama_missing_model_message():
    def handler(_request):
        return httpx.Response(404, text="not found")

    llm = _mount(OllamaProvider(model="ghost"), handler)
    with pytest.raises(ProviderError) as ei:
        await llm.generate([{"role": "user", "content": "x"}])
    assert "ollama pull" in str(ei.value).lower()
    await llm.aclose()


async def test_ollama_connection_refused_raises_connection_error():
    """Server unreachable (e.g. `ollama serve` not running) is distinct
    from a timeout -- it must not be reported as the same exception type
    a slow-but-running model would raise."""

    def handler(_request):
        raise httpx.ConnectError("connection refused", request=None)

    llm = _mount(OllamaProvider(model="m"), handler)
    with pytest.raises(ProviderConnectionError) as ei:
        await llm.generate([{"role": "user", "content": "x"}])
    assert "is it running" in str(ei.value).lower()
    await llm.aclose()


async def test_ollama_slow_response_raises_timeout_not_connection_error():
    """A model still loading (read timeout) must not be reported as
    'is it running?' -- the server IS running, it's just slow."""

    def handler(_request):
        raise httpx.ReadTimeout("timed out", request=None)

    llm = _mount(OllamaProvider(model="m"), handler)
    with pytest.raises(ProviderTimeoutError) as ei:
        await llm.generate([{"role": "user", "content": "x"}])
    message = str(ei.value).lower()
    assert "is it running" not in message
    assert "loading" in message
    await llm.aclose()


async def test_ollama_eventual_success_after_transport_is_fine():
    """Sanity: once the transport actually succeeds, no exception at all."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m", "message": {"role": "assistant", "content": "ok"},
                "done": True, "prompt_eval_count": 1, "eval_count": 1,
            },
        )

    llm = _mount(OllamaProvider(model="m"), handler)
    resp = await llm.generate([{"role": "user", "content": "x"}])
    assert resp.content == "ok"
    await llm.aclose()


async def test_ollama_malformed_json_body_raises_malformed_response_error():
    def handler(_request):
        return httpx.Response(200, content=b"not json {{{")

    llm = _mount(OllamaProvider(model="m"), handler)
    with pytest.raises(ProviderMalformedResponseError):
        await llm.generate([{"role": "user", "content": "x"}])
    await llm.aclose()
