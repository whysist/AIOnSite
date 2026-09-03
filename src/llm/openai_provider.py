from openai import AsyncOpenAI

from .base import BaseLLM


class OpenAIProvider(BaseLLM):
    """OpenAI API provider adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    async def generate(self, messages, **kwargs):
        return await self.client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=messages,
            **kwargs,
        )
