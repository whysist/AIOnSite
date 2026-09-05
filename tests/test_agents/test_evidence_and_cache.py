"""Priority 2 (structured evidence propagation) and Priority 5 (execution-
scoped tool-call caching), exercised directly against ``LLMAgent.execute``.
"""

from __future__ import annotations

from src.agents.base_agent import LLMAgent
from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse
from src.pipeline.state import ToolCallCache
from src.state.evidence import Evidence
from src.tools.builtin.db_tool import EquipmentLookupTool
from src.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EquipmentLookupTool())
    return reg


class _ScriptedLLM(BaseLLM):
    provider_name = "scripted"

    def __init__(self, replies):
        super().__init__(model="scripted")
        self._replies = list(replies)
        self.seen_messages: list[list] = []

    async def _complete(self, request):
        self.seen_messages.append(list(request.messages))
        return LLMResponse(content=self._replies.pop(0), model="scripted", provider="scripted")


# ---------------------------------------------------------------------
# TEST 4: structured evidence propagation


async def test_downstream_agent_prompt_includes_upstream_structured_evidence():
    llm = _ScriptedLLM(["done, using the provided evidence"])
    agent = LLMAgent(name="analyst", system_prompt="s", llm=llm, tools=[])
    ev = [
        Evidence(
            source="equipment_lookup", producer_tool="equipment_lookup", producer_node="step_1",
            content={"equipment_id": "P-101", "info": {"type": "pump"}},
        )
    ]
    await agent.execute("Analyse P-101", context={"step_1": "P-101 is a pump."}, evidence=ev)
    user_msg = next(m for m in llm.seen_messages[0] if m.role.value == "user")
    assert "STRUCTURED EVIDENCE" in user_msg.content
    assert ev[0].id in user_msg.content
    assert "P-101" in user_msg.content


async def test_no_evidence_block_rendered_when_no_evidence_available():
    llm = _ScriptedLLM(["ok"])
    agent = LLMAgent(name="analyst", system_prompt="s", llm=llm, tools=[])
    await agent.execute("task", context=None, evidence=None)
    user_msg = next(m for m in llm.seen_messages[0] if m.role.value == "user")
    assert "STRUCTURED EVIDENCE" not in user_msg.content


# ---------------------------------------------------------------------
# TEST 5: duplicate retrieval prevention via execution-scoped cache


async def test_second_agent_reuses_cached_tool_call_instead_of_re_executing():
    cache = ToolCallCache()
    registry = _registry()

    first = LLMAgent(
        name="researcher", system_prompt="s",
        llm=_ScriptedLLM(['{"tool": "equipment_lookup", "arguments": {"equipment_id": "P-101", "query_type": "all"}}', "done"]),
        tools=["equipment_lookup"],
    )
    result1 = await first.execute("look up P-101", registry=registry, tool_cache=cache)
    assert result1.tool_results[0].ok is True
    assert result1.tool_results[0].cached is False

    second = LLMAgent(
        name="analyst", system_prompt="s",
        llm=_ScriptedLLM(['{"tool": "equipment_lookup", "arguments": {"equipment_id": "P-101", "query_type": "all"}}', "done"]),
        tools=["equipment_lookup"],
    )
    result2 = await second.execute("look up P-101 again", registry=registry, tool_cache=cache)
    assert result2.tool_results[0].ok is True
    assert result2.tool_results[0].cached is True
    assert result2.tool_results[0].output == result1.tool_results[0].output


async def test_cache_miss_for_different_arguments():
    cache = ToolCallCache()
    registry = _registry()
    agent = LLMAgent(
        name="researcher", system_prompt="s",
        llm=_ScriptedLLM(['{"tool": "equipment_lookup", "arguments": {"equipment_id": "P-101", "query_type": "info"}}', "done"]),
        tools=["equipment_lookup"],
    )
    await agent.execute("look up P-101 info", registry=registry, tool_cache=cache)

    agent2 = LLMAgent(
        name="analyst", system_prompt="s",
        llm=_ScriptedLLM(['{"tool": "equipment_lookup", "arguments": {"equipment_id": "P-101", "query_type": "all"}}', "done"]),
        tools=["equipment_lookup"],
    )
    result2 = await agent2.execute("look up P-101 all", registry=registry, tool_cache=cache)
    assert result2.tool_results[0].cached is False


async def test_cache_not_consulted_for_non_cacheable_tool():
    """Only tools that explicitly declare ``cacheable = True`` are ever
    served from the cache -- must not silently start caching a tool no one
    asserted was safe to reuse."""
    from src.tools.builtin.calculator import CalculatorTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    cache = ToolCallCache()

    agent1 = LLMAgent(
        name="a", system_prompt="s",
        llm=_ScriptedLLM(['{"tool": "calculator", "arguments": {"expression": "1 + 1"}}', "done"]),
        tools=["calculator"],
    )
    await agent1.execute("t", registry=registry, tool_cache=cache)

    agent2 = LLMAgent(
        name="b", system_prompt="s",
        llm=_ScriptedLLM(['{"tool": "calculator", "arguments": {"expression": "1 + 1"}}', "done"]),
        tools=["calculator"],
    )
    result2 = await agent2.execute("t", registry=registry, tool_cache=cache)
    assert result2.tool_results[0].cached is False
