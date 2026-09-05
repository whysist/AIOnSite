"""analyze_image_with_vlm: real Ollama /api/chat wiring, via httpx.MockTransport
-- no network, mirrors tests/test_llm/test_providers_mocked.py's pattern.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from src.core.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ToolExecutionError,
)
from src.vision.vlm import analyze_image_with_vlm


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://localhost:11434", transport=httpx.MockTransport(handler))


async def test_missing_file_raises_tool_execution_error(tmp_path):
    with pytest.raises(ToolExecutionError, match="image not found"):
        await analyze_image_with_vlm(str(tmp_path / "ghost.png"))


async def test_happy_path_sends_base64_image_and_parses_entities(tmp_path):
    image = tmp_path / "schematic.png"
    image.write_bytes(b"not a real png, just bytes for the wire format")

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        import json
        body = json.loads(request.content)
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "model": "moondream:1.8b",
                "message": {
                    "content": "Schematic shows V-101 with gas outlet FV-103 and pump P-101.",
                },
                "done": True,
            },
        )

    client = _client(handler)
    result = await analyze_image_with_vlm(str(image), model="moondream:1.8b", client=client)
    await client.aclose()

    assert result.confidence == 1.0
    assert set(result.detected_entities) == {"V-101", "FV-103", "P-101"}
    assert result.raw_response["model"] == "moondream:1.8b"

    sent = captured["body"]["messages"][0]
    assert sent["images"] == [base64.b64encode(image.read_bytes()).decode("ascii")]


async def test_model_not_installed_returns_actionable_error(tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"x")

    def handler(_request):
        return httpx.Response(404, text="model not found")

    with pytest.raises(ProviderError, match="not installed"):
        await analyze_image_with_vlm(str(image), model="nonexistent-vlm", client=_client(handler))


async def test_server_unreachable_becomes_connection_error(tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"x")

    def handler(_request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProviderConnectionError):
        await analyze_image_with_vlm(str(image), client=_client(handler))


async def test_empty_response_has_zero_confidence(tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"x")

    def handler(_request):
        return httpx.Response(200, json={"model": "moondream:1.8b", "message": {"content": ""}, "done": True})

    result = await analyze_image_with_vlm(str(image), client=_client(handler))
    assert result.confidence == 0.0
    assert result.detected_entities == []
