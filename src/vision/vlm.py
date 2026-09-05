"""Local vision-language model calls (real image understanding, not OCR).

Talks to the same local Ollama server as the text model, over its native
``/api/chat`` endpoint with an ``images`` field (base64), via ``httpx`` --
consistent with the rest of this project's "providers speak the wire
protocol directly, no vendor SDK" decision (see ``src/llm/ollama_provider.py``).

Previously this function was a filename-matching stub that never called a
model at all. It is now a real call, and fails the same way
``OllamaProvider`` does: loudly and specifically (model not installed,
server unreachable, timed out) rather than degrading to a plausible-looking
canned answer.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import httpx

from ..core.config import get_settings
from ..core.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderMalformedResponseError,
    ProviderTimeoutError,
    ToolExecutionError,
)
from .schemas import VisionResult

DEFAULT_PROMPT = (
    "Describe what is shown in this image. If it is an equipment schematic, "
    "diagram, or a scanned inspection note, list every equipment tag "
    "(e.g. V-101, P-101, FV-103) and any readings, findings, or anomalies "
    "you can identify."
)

# Equipment tags in this project's dataset follow a short-letters + digits
# pattern (V-101, P-101, FV-103, HX-101, ...) -- used to pull a structured
# entity list out of the model's free-text description.
_ENTITY_PATTERN = re.compile(r"\b[A-Z]{1,3}-\d{2,4}\b")


async def analyze_image_with_vlm(
    image_path: str,
    prompt: str = DEFAULT_PROMPT,
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    connect_timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
) -> VisionResult:
    """Ask the local vision model to describe *image_path*.

    Raises the same provider-error hierarchy ``OllamaProvider`` uses (not a
    silently-degraded result) on anything that isn't a straightforward
    "the file doesn't exist" -- see the module docstring.
    """
    path = Path(image_path)
    if not path.is_file():
        raise ToolExecutionError(f"image not found: {image_path}")

    settings = get_settings()
    model = model or settings.vlm_model
    base_url = (base_url or settings.ollama_base_url).rstrip("/")
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
    }

    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        base_url=base_url, timeout=httpx.Timeout(timeout, connect=connect_timeout)
    )
    try:
        try:
            resp = await http_client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama at {base_url} did not respond within the configured timeout "
                f"while running the vision model '{model}'. A vision model's first "
                "request after `ollama pull` can be slow while it loads. "
                f"({type(exc).__name__}: {exc})",
                details={"base_url": base_url, "model": model},
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot reach Ollama at {base_url}. Is it running? Try: `ollama serve` "
                f"and `ollama pull {model}`. ({exc})",
                details={"base_url": base_url, "model": model},
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {base_url}: {exc}",
                details={"base_url": base_url, "model": model},
            ) from exc

        if resp.status_code == 404:
            raise ProviderError(
                f"Vision model '{model}' is not installed. Run: ollama pull {model}",
                details={"model": model},
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:400]}",
                details={"status_code": resp.status_code},
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderMalformedResponseError(
                f"Ollama returned a response that was not valid JSON: {exc}",
                details={"body": resp.text[:400]},
            ) from exc
    finally:
        if owns_client:
            await http_client.aclose()

    content = ((data.get("message") or {}).get("content") or "").strip()
    return VisionResult(
        summary=content,
        detected_entities=sorted(set(_ENTITY_PATTERN.findall(content))),
        confidence=1.0 if content else 0.0,
        raw_response={"model": data.get("model") or model, "status": "completed"},
    )
