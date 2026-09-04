"""Agent abstractions.

* :class:`BaseAgent` -- the original minimal ``run(state)`` contract, kept
  for backward compatibility (``InspectionAgent`` and early tests use it).
* :class:`Agent` -- the reusable base for the pipeline: carries a name,
  description, system prompt, an injected :class:`BaseLLM`, an optional set
  of allowed tool names and metadata, and exposes :meth:`execute`.
* :class:`LLMAgent` -- a concrete :class:`Agent` that runs a bounded
  tool-use loop against the injected model and returns an
  :class:`AgentResult`.

Agents never construct a provider themselves -- the model is injected.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from ..core.exceptions import AgentExecutionError
from ..core.logging import get_logger
from ..llm.base import BaseLLM
from ..llm.schemas import Message, Role
from ..pipeline.models import AgentResult, ToolResult
from ..tools.registry import ToolRegistry

_log = get_logger("agents")


class BaseAgent(ABC):
    """Legacy minimal interface -- operates in place on a state object."""

    @abstractmethod
    async def run(self, state: Any) -> Any:
        raise NotImplementedError


class Agent:
    """Reusable pipeline agent."""

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        system_prompt: str = "",
        llm: BaseLLM | None = None,
        tools: list[str] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        max_tool_iterations: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tool_iterations = max_tool_iterations
        self.metadata = metadata or {}

    def with_llm(self, llm: BaseLLM) -> Agent:
        """Return a shallow copy bound to *llm* (used by the router)."""
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.llm = llm
        return clone

    async def execute(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> AgentResult:
        raise NotImplementedError


class LLMAgent(Agent):
    """Default agent: system prompt + task + upstream context, bounded tool loop."""

    _TOOL_INSTRUCTIONS = (
        "You may use tools. To call a tool, reply with ONLY a JSON object:\n"
        '{{"tool": "<name>", "arguments": {{...}}}}\n'
        "After you receive the tool result you may call another tool or give "
        "your final answer as plain text. Available tools:\n{tools}"
    )

    async def execute(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> AgentResult:
        if self.llm is None:
            raise AgentExecutionError(f"agent {self.name!r} has no LLM bound")

        started = time.perf_counter()
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))

        usable_tools = [t for t in self.tools if registry and registry.has(t)]
        user_content = _render_task(task, context)
        if usable_tools:
            specs = "\n".join(
                f"- {registry.get(t).name}: {registry.get(t).description}"
                for t in usable_tools
            )
            user_content += "\n\n" + self._TOOL_INSTRUCTIONS.format(tools=specs)
        messages.append(Message(role=Role.USER, content=user_content))

        tool_results: list[ToolResult] = []
        last = None
        for _ in range(self.max_tool_iterations + 1):
            last = await self.llm.generate(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                metadata={"aionsite_kind": self.metadata.get("kind", self.name)},
            )
            call = _parse_tool_call(last.content) if usable_tools else None
            if not call:
                break
            name, arguments = call
            if not registry or not registry.has(name):
                messages.append(Message(role=Role.ASSISTANT, content=last.content))
                messages.append(
                    Message(role=Role.USER, content=f"Tool {name!r} is not available. Answer directly.")
                )
                continue
            result = await registry.call(name, **arguments)
            tool_results.append(result)
            messages.append(Message(role=Role.ASSISTANT, content=last.content))
            messages.append(
                Message(
                    role=Role.TOOL,
                    content=json.dumps(
                        {"ok": result.ok, "output": result.output, "error": result.error},
                        default=str,
                    ),
                    name=name,
                )
            )

        assert last is not None
        return AgentResult(
            agent=self.name,
            ok=True,
            output=last.content.strip(),
            tool_results=tool_results,
            model=last.model,
            provider=last.provider,
            prompt_tokens=last.usage.prompt_tokens,
            completion_tokens=last.usage.completion_tokens,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata={"tool_calls": len(tool_results)},
        )


# ----------------------------------------------------------------------
def _render_task(task: str, context: dict[str, Any] | None) -> str:
    if not context:
        return f"TASK:\n{task}"
    lines = [f"TASK:\n{task}", "", "CONTEXT FROM PREVIOUS STEPS:"]
    for key, value in context.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        lines.append(f"[{key}]\n{text}")
    return "\n".join(lines)


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "tool" in obj:
        return str(obj["tool"]), dict(obj.get("arguments") or {})
    return None
