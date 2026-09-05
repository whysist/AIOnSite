import os

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


def test_sovereign_mode_forces_ml_libraries_offline(monkeypatch):
    """Regression test: a live run's logs showed sentence-transformers
    making real HTTPS calls to huggingface.co on every startup, even with
    sovereign_mode on and the model already cached locally -- see the
    docstring on Settings._enforce_offline_ml_libraries."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    Settings(_env_file=None, sovereign_mode=True, llm_provider="ollama")

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_non_sovereign_mode_does_not_force_libraries_offline(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    Settings(_env_file=None, sovereign_mode=False, llm_provider="openai")

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_sovereign_mode_never_clobbers_an_explicit_operator_setting(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    Settings(_env_file=None, sovereign_mode=True, llm_provider="ollama")

    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_get_settings_wraps_validation_error(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "9")
    from src.core import config

    config.get_settings.cache_clear()
    with pytest.raises(ConfigurationError):
        config.get_settings()
    config.get_settings.cache_clear()
