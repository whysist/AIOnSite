from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Provider-neutral interface for language models."""

    @abstractmethod
    async def generate(self, messages, **kwargs):
        raise NotImplementedError
