"""Application configuration.

Loads validated settings from environment variables / ``.env`` using
``pydantic-settings``.  Agent and pipeline code must depend on this module
rather than reading ``os.environ`` directly.

Backward compatibility: a module-level ``settings`` object is still exported,
but new code should call :func:`get_settings` (cached).
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError


class Environment(str, enum.Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LLMProvider(str, enum.Enum):
    """Supported LLM providers.

    New providers can be added here without touching agent code -- the
    factory (``src/llm/factory.py``) is the only place that maps an enum
    value to a concrete implementation.
    """

    OPENAI = "openai"
    LOCAL = "local"          # generic OpenAI-compatible local server
    OLLAMA = "ollama"        # native Ollama API
    VLLM = "vllm"            # vLLM OpenAI-compatible server
    ECHO = "echo"            # deterministic offline provider (tests / demo)

    @property
    def is_cloud(self) -> bool:
        """Whether traffic for this provider leaves the machine by default."""
        return self is LLMProvider.OPENAI


# Providers that are always acceptable under sovereign / local-only mode.
_LOCAL_PROVIDERS = {
    LLMProvider.LOCAL,
    LLMProvider.OLLAMA,
    LLMProvider.VLLM,
    LLMProvider.ECHO,
}


class Settings(BaseSettings):
    """Validated application settings.

    Values are read (in order of precedence) from real environment
    variables, then ``.env``, then the defaults declared here.
    """

    # --- Application -------------------------------------------------------
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    configs_dir: Path = Path("configs")

    # --- Provider selection ---------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.OPENAI

    # --- OpenAI (cloud) -----------------------------------------------------
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # --- Generic local OpenAI-compatible server ---------------------------
    local_llm_base_url: str | None = "http://localhost:8001/v1"
    local_llm_api_key: str | None = "local"

    # --- Ollama ----------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"

    # --- Local vision-language model (image / scanned-document understanding) --
    # Served by the same local Ollama instance as the text model above --
    # a distinct, smaller model tuned for multimodal input rather than a
    # separate provider, so it stays covered by the same sovereignty checks.
    vlm_model: str = "moondream:1.8b"

    # --- vLLM ----------------------------------------------------------
    vllm_base_url: str | None = "http://localhost:8000/v1"
    vllm_api_key: str | None = "local"

    # --- Model parameters -------------------------------------------------
    llm_model: str | None = None
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2000, gt=0, le=200_000)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    # Separate connect vs read timeout so a slow local model (long read) is
    # not confused with a server that refuses connections (fast connect
    # failure). Read defaults to request_timeout_seconds when unset.
    llm_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    # Default raised from the old blanket 60s: a CPU-only load of a 7B model
    # (no GPU/VRAM) can take several minutes on its first request, and that
    # must not be mistaken for the server being down (see ProviderTimeoutError
    # vs ProviderConnectionError in src/llm/ollama_provider.py).
    llm_read_timeout_seconds: float | None = Field(default=300.0, gt=0)

    # --- Orchestration limits ------------------------------------------
    max_retries: int = Field(default=2, ge=0, le=10)
    max_replans: int = Field(default=1, ge=0, le=5)
    # Must stay >= llm_read_timeout_seconds: this wraps the whole node
    # (including the LLM call) in asyncio.wait_for, so a shorter value here
    # would cancel a slow-loading model before the provider's own read
    # timeout ever gets a chance to matter.
    node_timeout_seconds: float = Field(default=300.0, gt=0)
    plan_retry_backoff_seconds: float = Field(default=0.5, ge=0)
    tool_max_repair_attempts: int = Field(default=2, ge=0, le=10)

    # --- Sovereignty ---------------------------------------------------
    sovereign_mode: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @model_validator(mode="after")
    def _enforce_sovereign_mode(self) -> Settings:
        """A cloud provider must never be selected while sovereign mode is on."""
        if self.sovereign_mode and self.llm_provider not in _LOCAL_PROVIDERS:
            raise ValueError(
                "sovereign_mode is enabled but llm_provider="
                f"'{self.llm_provider.value}' is a cloud provider. "
                "Use one of: " + ", ".join(sorted(p.value for p in _LOCAL_PROVIDERS))
            )
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    def provider_allowed(self, provider: LLMProvider) -> bool:
        """Whether *provider* may be used given the current policy."""
        if self.sovereign_mode:
            return provider in _LOCAL_PROVIDERS
        return True

    def safe_dump(self) -> dict[str, object]:
        """Config snapshot with secrets redacted -- safe for logs / audit."""
        secret_fields = {
            "openai_api_key",
            "local_llm_api_key",
            "vllm_api_key",
        }
        out: dict[str, object] = {}
        for name, value in self.model_dump().items():
            if name in secret_fields and value:
                out[name] = "***redacted***"
            elif isinstance(value, enum.Enum):
                out[name] = value.value
            elif isinstance(value, Path):
                out[name] = str(value)
            else:
                out[name] = value
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings.

    Raises :class:`ConfigurationError` (not a raw ``ValidationError``) so the
    rest of the application has a single error type to catch at startup.
    """
    try:
        return Settings()
    except ValidationError as exc:  # pragma: no cover - exercised via tests
        raise ConfigurationError(f"Invalid configuration:\n{exc}") from exc


def reload_settings() -> Settings:
    """Clear the cache and re-read settings (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()


try:
    # Backward-compatible module-level accessor used by older code / scripts.
    settings = get_settings()
except ConfigurationError:  # pragma: no cover
    # Defer failure to the point of use rather than crashing on import.
    settings = None  # type: ignore[assignment]
