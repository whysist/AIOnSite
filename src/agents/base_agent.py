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
        max_repair_attempts: int = 2,
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
        # Bounded per-tool retry budget for argument-validation repair (see
        # ``LLMAgent.execute``): distinct from ``max_tool_iterations``, which
        # bounds the whole tool-use loop across all tools.
        self.max_repair_attempts = max_repair_attempts
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

        usable_tools: list[str] = []
        if registry is not None:
            usable_tools = [t for t in self.tools if registry.has(t)]
        user_content = _render_task(task, context)
        if usable_tools and registry is not None:
            specs = "\n".join(
                _render_tool_schema(registry.get(t).spec()) for t in usable_tools
            )
            user_content += "\n\n" + self._TOOL_INSTRUCTIONS.format(tools=specs)
        messages.append(Message(role=Role.USER, content=user_content))

        tool_results: list[ToolResult] = []
        repair_attempts: dict[str, int] = {}
        repair_log: list[dict[str, Any]] = []
        unresolved_tools: set[str] = set()
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
                # Record this as a real (unresolved) ToolResult -- previously
                # it only produced a conversational nudge and never became a
                # ToolResult at all, so ``error_kind="not_found"`` (declared
                # on the model) was unreachable and this failure was invisible
                # to the audit trail, evidence tracking and the verifier.
                result = ToolResult(
                    tool=name, ok=False, error_kind="not_found",
                    error=f"unknown tool: {name!r}", input=arguments,
                )
                tool_results.append(result)
                unresolved_tools.add(name)
                messages.append(Message(role=Role.ASSISTANT, content=last.content))
                messages.append(
                    Message(role=Role.USER, content=f"Tool {name!r} is not available. Answer directly.")
                )
                continue
            result = await registry.call(name, **arguments)
            tool_results.append(result)
            messages.append(Message(role=Role.ASSISTANT, content=last.content))

            if not result.ok and result.error_kind == "validation":
                attempts = repair_attempts.get(name, 0)
                if attempts < self.max_repair_attempts:
                    repair_attempts[name] = attempts + 1
                    repair_log.append(
                        {"tool": name, "attempt": attempts + 1, "error": result.error}
                    )
                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content=_repair_feedback(name, result),
                            name=name,
                        )
                    )
                    continue
                unresolved_tools.add(name)
                repair_log.append(
                    {
                        "tool": name, "attempt": attempts + 1,
                        "error": result.error, "exhausted": True,
                    }
                )
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=(
                            f"Call to {name!r} failed argument validation again: "
                            f"{result.error}. No further repair attempts remain for "
                            "this tool in this step -- proceed without it and state "
                            "plainly that this could not be completed."
                        ),
                        name=name,
                    )
                )
                continue

            if not result.ok:
                # Non-validation failure (the tool itself raised): not a
                # repairable argument-shape problem, so don't spend repair
                # budget on it -- record as unresolved immediately.
                unresolved_tools.add(name)

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
            ok=not unresolved_tools,
            output=last.content.strip(),
            tool_results=tool_results,
            model=last.model,
            provider=last.provider,
            prompt_tokens=last.usage.prompt_tokens,
            completion_tokens=last.usage.completion_tokens,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata={
                "tool_calls": len(tool_results),
                "repair_log": repair_log,
                "unresolved_tools": sorted(unresolved_tools),
            },
        )


# ----------------------------------------------------------------------
def _render_tool_schema(spec: dict[str, Any]) -> str:
    """Render a tool's real parameter contract for the prompt.

    Previously only ``name`` + free-text ``description`` reached the model
    (see the old inline ``f"- {t.name}: {t.description}"`` this replaces) --
    the model had to guess argument names from prose. ``BaseTool.spec()``
    already exposes the exact Pydantic JSON schema; this just formats it
    compactly enough for a small local model's context rather than dumping
    raw JSON Schema.
    """
    input_schema = spec.get("input_schema") or {}
    properties = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])
    lines = [f"- {spec['name']}: {spec['description']}"]
    if not properties:
        lines.append("    arguments: (none)")
    for pname, pschema in properties.items():
        ptype = pschema.get("type", "any")
        if pname in required:
            marker = "required"
        else:
            default = pschema.get("default")
            marker = f"optional, default={default!r}"
        desc = pschema.get("description", "")
        suffix = f" -- {desc}" if desc else ""
        lines.append(f"    - {pname} ({ptype}, {marker}){suffix}")
    return "\n".join(lines)


def _repair_feedback(name: str, result: ToolResult) -> str:
    """Structured correction prompt for a tool call that failed validation.

    Distinct from the generic ``{"ok": false, ...}`` tool message used for
    other failures: this names exactly which arguments are required and
    what was wrong, so the model has a concrete target to correct rather
    than having to re-derive the schema from the error text alone.
    """
    schema = result.expected_schema or {}
    required = schema.get("required") or []
    properties = schema.get("properties") or {}
    lines = [
        f"Your call to {name!r} failed argument validation: {result.error}",
        "Required arguments: " + (", ".join(required) if required else "(none)"),
    ]
    if properties:
        types = {pname: pschema.get("type", "any") for pname, pschema in properties.items()}
        lines.append("Parameter types: " + json.dumps(types))
    lines.append(
        "Call the tool again with corrected arguments as a JSON object "
        '(e.g. {"tool": "' + name + '", "arguments": {...}}).'
    )
    return "\n".join(lines)


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
