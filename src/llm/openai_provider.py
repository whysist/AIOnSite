"""OpenAI-compatible chat-completions provider (``/v1/chat/completions``).

Implemented directly on :mod:`httpx` so the project has no hard dependency
on the ``openai`` SDK.  The same class backs OpenAI itself, vLLM, and any
other server that speaks the OpenAI wire format -- only the base URL, key
and ``is_local`` flag differ (see :mod:`src.llm.local_provider` /
:mod:`src.llm.vllm_provider`).
"""

from __future__ import annotations

from typing import Any

import httpx

from ..core.exceptions import ProviderError
from .base import BaseLLM
from .schemas import LLMRequest, LLMResponse, UsageMetadata


class OpenAICompatibleProvider(BaseLLM):
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        is_local: bool = False,
    ) -> None:
        super().__init__(model=model, is_local=is_local)
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=self._headers(),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.wire_messages(),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = request.stop
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.post("/chat/completions", json=payload)
        except httpx.RequestError as exc:
            raise ProviderError(
                f"Cannot reach {self.provider_name} endpoint at {self._base_url}: {exc}",
                details={"base_url": self._base_url},
            ) from exc

        if resp.status_code == 400 and request.json_mode:
            # Some OpenAI-compatible servers reject response_format -- retry once.
            payload.pop("response_format", None)
            resp = await self._client.post("/chat/completions", json=payload)

        if resp.status_code >= 400:
            raise ProviderError(
                f"{self.provider_name} returned HTTP {resp.status_code}: {resp.text[:400]}",
                details={"status_code": resp.status_code},
            )

        data = resp.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(
                f"Unexpected response shape from {self.provider_name}: {data}"
            ) from exc

        usage_raw = data.get("usage") or {}
        return LLMResponse(
            content=content,
            model=data.get("model") or request.model,
            provider=self.provider_name,
            finish_reason=choice.get("finish_reason"),
            usage=UsageMetadata(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
            raw=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/models")
            return resp.status_code < 500
        except httpx.RequestError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


# Backwards-compatible name used by the original skeleton / factory.
OpenAIProvider = OpenAICompatibleProvider
