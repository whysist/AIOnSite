from openai import AsyncOpenAI

from .base import BaseLLM


class LocalProvider(BaseLLM):
    """Adapter for OpenAI-compatible locally hosted models."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "local",
        model: str | None = None,
    ):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    async def generate(self, messages, **kwargs):
        return await self.client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=messages,
            **kwargs,
        )
