"""LLM factory.

The *only* place that maps a provider identifier to a concrete class.
Agents receive a :class:`BaseLLM` and never import a provider directly.

    settings ──► create_llm(settings) ──► BaseLLM (selected provider)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.exceptions import ConfigurationError, SovereigntyError
from .base import BaseLLM

if TYPE_CHECKING:
    from ..core.config import LLMProvider, Settings


def _normalise(provider: str | LLMProvider) -> str:
    return getattr(provider, "value", str(provider)).lower()


def create_llm(
    settings: Settings | None = None,
    *,
    provider: str | LLMProvider | None = None,
    model: str | None = None,
) -> BaseLLM:
    """Build the configured LLM provider.

    Parameters
    ----------
    settings:
        Application settings.  When ``None`` the cached settings are used.
    provider / model:
        Optional overrides (used by the router to pick a specific model
        for one pipeline node).
    """
    from ..core.config import get_settings

    settings = settings or get_settings()
    name = _normalise(provider or settings.llm_provider)
    model = model or settings.llm_model

    if settings.sovereign_mode and name in {"openai"}:
        raise SovereigntyError(
            f"sovereign_mode is enabled; provider '{name}' is not permitted. "
            "Use 'ollama', 'vllm', 'local' or 'echo'."
        )

    if name == "openai":
        from .openai_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=model,
            timeout=settings.request_timeout_seconds,
            is_local=False,
        )

    if name == "local":
        from .local_provider import LocalProvider

        if not settings.local_llm_base_url:
            raise ConfigurationError("local_llm_base_url must be set for provider 'local'")
        return LocalProvider(
            base_url=settings.local_llm_base_url,
            api_key=settings.local_llm_api_key or "local",
            model=model,
            timeout=settings.request_timeout_seconds,
        )

    if name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=model,
            timeout=settings.request_timeout_seconds,
        )

    if name == "vllm":
        from .vllm_provider import VLLMProvider

        if not settings.vllm_base_url:
            raise ConfigurationError("vllm_base_url must be set for provider 'vllm'")
        return VLLMProvider(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key or "local",
            model=model,
            timeout=settings.request_timeout_seconds,
        )

    if name == "echo":
        from .echo_provider import EchoProvider

        return EchoProvider(model=model)

    raise ConfigurationError(f"Unsupported LLM provider: {name!r}")
