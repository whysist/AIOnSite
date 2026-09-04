"""Tool schema exposure and the bounded argument-repair loop.

Reproduces the two concrete failures from the V-101 incident report: the
model calling ``read_file`` with ``file_path`` instead of ``relative_path``,
and ``calculate_deviation`` with ``observed_pressure``/``approved_limit``
instead of ``actual``/``limit``. Before the fix, the prompt never contained
real parameter names (only name + free-text description) and a validation
failure was absorbed silently with no consequence for ``AgentResult.ok``.
"""

from __future__ import annotations

from src.agents.base_agent import LLMAgent, _render_tool_schema
from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse
from src.tools.builtin.calculator import DeviationTool
from src.tools.registry import ToolRegistry


class _ScriptedLLM(BaseLLM):
    provider_name = "scripted"

    def __init__(self, replies):
        super().__init__(model="scripted")
        self._replies = list(replies)
        self.seen_messages: list[list] = []

    async def _complete(self, request):
        self.seen_messages.append(list(request.messages))
        return LLMResponse(content=self._replies.pop(0), model="scripted", provider="scripted")


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(DeviationTool())
    return reg


def test_render_tool_schema_includes_real_parameter_names_and_types():
    spec = DeviationTool.spec()
    rendered = _render_tool_schema(spec)
    assert "actual" in rendered
    assert "limit" in rendered
    assert "required" in rendered
    assert "number" in rendered


async def test_prompt_exposes_real_schema_not_just_description():
    llm = _ScriptedLLM(['{"tool": "calculate_deviation", "arguments": {"actual": 120, "limit": 100}}', "done"])
    agent = LLMAgent(name="analyst", system_prompt="s", llm=llm, tools=["calculate_deviation"])
    await agent.execute("compare readings", registry=_registry())
    user_msg = next(m for m in llm.seen_messages[0] if m.role.value == "user")
    assert "actual" in user_msg.content
    assert "limit" in user_msg.content


async def test_wrong_argument_names_trigger_repair_and_succeed():
    """The model first guesses wrong field names (as observed in the
    incident), then, given structured feedback, corrects itself."""
    llm = _ScriptedLLM([
        '{"tool": "calculate_deviation", "arguments": {"observed_pressure": 120, "approved_limit": 100}}',
        '{"tool": "calculate_deviation", "arguments": {"actual": 120, "limit": 100}}',
        "The deviation is 20 bar (20%).",
    ])
    agent = LLMAgent(name="analyst", system_prompt="s", llm=llm, tools=["calculate_deviation"])
    result = await agent.execute("compare readings", registry=_registry())

    assert result.ok is True
    assert len(result.tool_results) == 2
    assert result.tool_results[0].ok is False
    assert result.tool_results[0].error_kind == "validation"
    assert result.tool_results[1].ok is True
    assert result.metadata["unresolved_tools"] == []
    assert len(result.metadata["repair_log"]) == 1
    assert result.metadata["repair_log"][0]["tool"] == "calculate_deviation"
    assert not result.metadata["repair_log"][0].get("exhausted")


async def test_repair_feedback_message_names_required_fields():
    llm = _ScriptedLLM([
        '{"tool": "calculate_deviation", "arguments": {"wrong": 1}}',
        "give up",
    ])
    agent = LLMAgent(
        name="analyst", system_prompt="s", llm=llm,
        tools=["calculate_deviation"], max_repair_attempts=1,
    )
    await agent.execute("compare readings", registry=_registry())
    tool_feedback = next(m for m in llm.seen_messages[1] if m.role.value == "tool")
    assert "actual" in tool_feedback.content
    assert "limit" in tool_feedback.content


async def test_repair_exhaustion_marks_agent_result_not_ok():
    """When the model never corrects itself, the loop must not silently
    let the run 'succeed' -- this is the core fix for the reported bug
    where AgentResult.ok was hardcoded True regardless of tool failures."""
    llm = _ScriptedLLM([
        '{"tool": "calculate_deviation", "arguments": {"observed_pressure": 120}}',
        '{"tool": "calculate_deviation", "arguments": {"observed_pressure": 120}}',
        "I could not run the calculation, but here is a manual estimate: ~20 bar.",
    ])
    agent = LLMAgent(
        name="analyst", system_prompt="s", llm=llm,
        tools=["calculate_deviation"], max_repair_attempts=1,
    )
    result = await agent.execute("compare readings", registry=_registry())

    assert result.ok is False
    assert result.metadata["unresolved_tools"] == ["calculate_deviation"]
    assert any(entry.get("exhausted") for entry in result.metadata["repair_log"])
    assert all(not tr.ok for tr in result.tool_results)


async def test_unknown_tool_name_is_recorded_as_a_not_found_tool_result():
    """Regression test: calling a tool name the registry doesn't have used
    to only produce a conversational nudge ('answer directly') with no
    ToolResult ever created -- so ``error_kind="not_found"`` (declared on
    the model) was unreachable, and the failure was invisible to the audit
    trail, evidence tracking, and ``AgentResult.ok``. The agent must still
    have a genuinely usable tool declared (``calculate_deviation``) so the
    tool-call parser engages at all; the model then hallucinates a
    different, unregistered tool name.
    """
    llm = _ScriptedLLM([
        '{"tool": "search_knowledge_base", "arguments": {"query": "V-101 SOP"}}',
        "No knowledge base is available, so I cannot retrieve the SOP.",
    ])
    agent = LLMAgent(name="researcher", system_prompt="s", llm=llm,
                     tools=["calculate_deviation"])
    result = await agent.execute("find the SOP", registry=_registry())

    assert result.ok is False
    assert result.metadata["unresolved_tools"] == ["search_knowledge_base"]
    assert len(result.tool_results) == 1
    not_found = result.tool_results[0]
    assert not_found.tool == "search_knowledge_base"
    assert not_found.ok is False
    assert not_found.error_kind == "not_found"
