"""Native Ollama provider (``POST /api/chat``).

Ollama also ships an OpenAI-compatible endpoint, but its native API gives
clearer errors and does not require the ``/v1`` shim, so we target it
directly.  Default endpoint: ``http://localhost:11434``.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..core.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderMalformedResponseError,
    ProviderTimeoutError,
)
from .base import BaseLLM
from .schemas import LLMRequest, LLMResponse, UsageMetadata


class OllamaProvider(BaseLLM):
    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str | None = None,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
    ) -> None:
        super().__init__(model=model, is_local=True)
        self._base_url = base_url.rstrip("/")
        # Read timeout (model generation) and connect timeout (server
        # reachability) are deliberately independent: a slow local model
        # load should not be mistaken for a server that is not running.
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
        )

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        if not request.model:
            raise ProviderError("Ollama requires an explicit model name (e.g. 'qwen2.5:7b-instruct')")

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.wire_messages(),
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.stop:
            payload["options"]["stop"] = request.stop
        if request.json_mode:
            payload["format"] = "json"

        try:
            resp = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            # Covers connect-timeout, read-timeout, write-timeout and
            # pool-timeout alike: the server may simply still be loading the
            # model. This is NOT "is it running?" -- do not say so.
            raise ProviderTimeoutError(
                f"Ollama at {self._base_url} did not respond within the configured "
                "timeout. The model may still be loading (first request after "
                "`ollama pull`/server start can be slow) -- consider raising "
                "llm_read_timeout_seconds if this persists.  "
                f"({type(exc).__name__}: {exc})",
                details={"base_url": self._base_url, "model": request.model},
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                "Cannot reach Ollama at "
                f"{self._base_url}. Is it running?  Try: `ollama serve` and "
                f"`ollama pull {request.model}`.  ({exc})",
                details={"base_url": self._base_url, "model": request.model},
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                "Cannot reach Ollama at "
                f"{self._base_url}. Is it running?  Try: `ollama serve` and "
                f"`ollama pull {request.model}`.  ({exc})",
                details={"base_url": self._base_url, "model": request.model},
            ) from exc

        if resp.status_code == 404:
            raise ProviderError(
                f"Ollama model '{request.model}' is not installed. "
                f"Run: ollama pull {request.model}",
                details={"model": request.model},
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
        content = (data.get("message") or {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        return LLMResponse(
            content=content,
            model=data.get("model") or request.model,
            provider=self.provider_name,
            finish_reason="stop" if data.get("done") else None,
            usage=UsageMetadata(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            raw=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network dependent
            raise ProviderError(f"Could not list Ollama models: {exc}") from exc
        return [m["name"] for m in resp.json().get("models", [])]

    async def aclose(self) -> None:
        await self._client.aclose()
