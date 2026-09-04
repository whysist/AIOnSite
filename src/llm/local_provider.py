"""Generic local OpenAI-compatible provider.

Thin subclass of :class:`OpenAICompatibleProvider` that defaults to a
localhost endpoint and is flagged ``is_local=True`` so the router / sovereign
policy treats it as on-premise.
"""

from __future__ import annotations

from .openai_provider import OpenAICompatibleProvider


class LocalProvider(OpenAICompatibleProvider):
    provider_name = "local"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "local",
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
