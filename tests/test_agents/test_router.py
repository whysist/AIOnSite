import pytest

from src.agents.router import AgentRouter, ModelRouter
from src.core.config import Settings
from src.core.exceptions import SovereigntyError
from src.pipeline.models import PipelineNode


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_agent_router_lookup():
    r = AgentRouter({"a": 1})
    assert r.get_agent("a") == 1
    assert r.get_agent("missing") is None


def test_default_route_uses_configured_provider():
    router = ModelRouter(_settings(llm_provider="ollama"))
    decision = router.route(PipelineNode(name="n"))
    assert decision.provider == "ollama"
    assert decision.is_local is True


def test_node_override_honoured_when_allowed():
    router = ModelRouter(_settings(llm_provider="ollama"))
    decision = router.route(PipelineNode(name="n", provider="vllm", model="foo"))
    assert decision.provider == "vllm"
    assert decision.model == "foo"


def test_sovereign_mode_refuses_cloud_node():
    router = ModelRouter(_settings(llm_provider="ollama", sovereign_mode=True))
    with pytest.raises(SovereigntyError):
        router.route(PipelineNode(name="n", provider="openai"))


def test_confidential_forces_local_when_default_is_cloud():
    router = ModelRouter(_settings(llm_provider="openai"), confidential=True)
    decision = router.route(PipelineNode(name="n"))
    assert decision.is_local is True
    assert decision.provider in {"ollama", "local", "vllm", "echo"}


def test_model_from_config_routing_map():
    cfg = {"ollama": {"model": "qwen2.5:7b-instruct"}}
    router = ModelRouter(_settings(llm_provider="ollama"), model_config=cfg)
    decision = router.route(PipelineNode(name="n"))
    assert decision.model == "qwen2.5:7b-instruct"
