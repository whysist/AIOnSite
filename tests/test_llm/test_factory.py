import pytest

from src.core.config import Settings
from src.core.exceptions import ConfigurationError, SovereigntyError
from src.llm.base import BaseLLM
from src.llm.echo_provider import EchoProvider
from src.llm.factory import create_llm
from src.llm.ollama_provider import OllamaProvider
from src.llm.openai_provider import OpenAICompatibleProvider


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_factory_returns_base_llm_for_each_provider():
    assert isinstance(create_llm(_settings(llm_provider="echo")), EchoProvider)
    assert isinstance(create_llm(_settings(llm_provider="openai")), OpenAICompatibleProvider)
    assert isinstance(create_llm(_settings(llm_provider="ollama")), OllamaProvider)
    for prov in ("echo", "openai", "ollama", "vllm", "local"):
        assert isinstance(create_llm(_settings(llm_provider=prov)), BaseLLM)


def test_factory_unknown_provider():
    s = _settings()
    with pytest.raises(ConfigurationError):
        create_llm(s, provider="banana")


def test_factory_blocks_cloud_under_sovereign_override():
    s = _settings(llm_provider="ollama", sovereign_mode=True)
    with pytest.raises(SovereigntyError):
        create_llm(s, provider="openai")


def test_local_provider_flagged_local():
    assert create_llm(_settings(llm_provider="ollama")).is_local is True
    assert create_llm(_settings(llm_provider="vllm")).is_local is True
    assert create_llm(_settings(llm_provider="openai")).is_local is False


async def test_model_override_applied():
    llm = create_llm(_settings(llm_provider="echo"), model="custom-model")
    assert llm.model == "custom-model"
