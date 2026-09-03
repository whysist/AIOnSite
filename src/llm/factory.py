def create_llm(provider: str, **kwargs):
    """Create an LLM provider from configuration."""

    provider = provider.lower()

    if provider == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(**kwargs)

    if provider == "local":
        from .local_provider import LocalProvider
        return LocalProvider(**kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")
