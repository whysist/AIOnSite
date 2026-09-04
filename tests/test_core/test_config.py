import pytest

from src.core.config import Environment, LLMProvider, Settings
from src.core.exceptions import ConfigurationError


def test_defaults(monkeypatch):
    for var in ("ENVIRONMENT", "LLM_PROVIDER", "SOVEREIGN_MODE", "LLM_TEMPERATURE",
                "LLM_MAX_TOKENS", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.environment is Environment.DEVELOPMENT
    assert s.llm_provider is LLMProvider.OPENAI
    assert s.llm_temperature == 0.2
    assert s.llm_max_tokens == 2000
    assert s.sovereign_mode is False


def test_provider_selection_from_env():
    s = Settings(_env_file=None, llm_provider="ollama")
    assert s.llm_provider is LLMProvider.OLLAMA


@pytest.mark.parametrize("bad", [-0.1, 2.5])
def test_temperature_validation(bad):
    with pytest.raises(Exception):
        Settings(_env_file=None, llm_temperature=bad)


@pytest.mark.parametrize("bad", [0, -10])
def test_max_tokens_must_be_positive(bad):
    with pytest.raises(Exception):
        Settings(_env_file=None, llm_max_tokens=bad)


def test_invalid_provider_rejected():
    with pytest.raises(Exception):
        Settings(_env_file=None, llm_provider="nope")


def test_invalid_log_level_rejected():
    with pytest.raises(Exception):
        Settings(_env_file=None, log_level="LOUD")


def test_sovereign_mode_rejects_cloud_provider():
    with pytest.raises(Exception):
        Settings(_env_file=None, sovereign_mode=True, llm_provider="openai")


def test_sovereign_mode_allows_local_providers():
    for prov in ("ollama", "vllm", "local", "echo"):
        s = Settings(_env_file=None, sovereign_mode=True, llm_provider=prov)
        assert s.provider_allowed(LLMProvider(prov))
    s = Settings(_env_file=None, sovereign_mode=True, llm_provider="ollama")
    assert s.provider_allowed(LLMProvider.OPENAI) is False


def test_safe_dump_redacts_secrets():
    s = Settings(_env_file=None, llm_provider="ollama",
                 openai_api_key="sk-secret", local_llm_api_key="tok")
    dumped = s.safe_dump()
    assert dumped["openai_api_key"] == "***redacted***"
    assert dumped["local_llm_api_key"] == "***redacted***"
    assert dumped["llm_provider"] == "ollama"  # enum rendered as its value


def test_get_settings_wraps_validation_error(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "9")
    from src.core import config

    config.get_settings.cache_clear()
    with pytest.raises(ConfigurationError):
        config.get_settings()
    config.get_settings.cache_clear()
