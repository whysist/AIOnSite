
from src.agents.base_agent import LLMAgent
from src.agents.inspection_agent import InspectionAgent
from src.agents.roles import build_agent_library
from src.llm.base import BaseLLM
from src.llm.echo_provider import EchoProvider
from src.llm.schemas import LLMResponse
from src.state.session import create_session
from src.tools.registry import default_registry


async def test_inspection_agent_runs_with_llm():
    state = create_session({"task": "Check V-101 pressure"})
    state.messages.append({"role": "user", "content": "Check V-101 pressure"})
    out = await InspectionAgent(EchoProvider()).run(state)
    assert out.messages[-1]["role"] == "assistant"
    assert out.metadata["inspection"]["ok"] is True


async def test_inspection_agent_without_llm_does_not_fabricate():
    state = create_session({"task": "x"})
    out = await InspectionAgent(None).run(state)
    assert out.metadata["inspection"]["status"] == "no_model"


async def test_llm_agent_uses_tool_when_model_requests_it():
    class ToolThenAnswer(BaseLLM):
        provider_name = "tta"

        def __init__(self):
            super().__init__(model="tta")
            self._calls = 0

        async def _complete(self, request):
            self._calls += 1
            if self._calls == 1:
                return LLMResponse(
                    content='{"tool": "calculator", "arguments": {"expression": "2 + 3"}}',
                    provider="tta",
                )
            return LLMResponse(content="the answer is 5", provider="tta")

    registry = default_registry()
    agent = LLMAgent(name="analyst", system_prompt="s", llm=ToolThenAnswer(),
                     tools=["calculator"])
    result = await agent.execute("add two and three", registry=registry)
    assert result.output == "the answer is 5"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].output["result"] == 5.0


def test_agent_library_has_core_roles():
    lib = build_agent_library(EchoProvider())
    for role in ("researcher", "analyst", "summarizer", "executor"):
        assert role in lib
