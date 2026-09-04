"""vLLM provider.

vLLM exposes an OpenAI-compatible server, so this is a thin subclass that
defaults to the vLLM port and is marked on-premise.
"""

from __future__ import annotations

from .openai_provider import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    provider_name = "vllm"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = "local",
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            is_local=True,
        )
