"""Concrete role agents used by the pipeline.

Each is an :class:`LLMAgent` with a role-specific system prompt.  The
factory :func:`build_agent_library` returns a name -> agent mapping that
the pipeline executor resolves node ``agent`` fields against.
"""

from __future__ import annotations

from ..llm.base import BaseLLM
from ..pipeline.models import NodeType
from .base_agent import LLMAgent

_RESEARCHER = (
    "You are a research agent in a local, sovereign inspection workbench. "
    "Given a task and any context, extract and organise the relevant facts. "
    "Be concrete. If information is missing, say so explicitly -- never invent "
    "evidence, numbers, or citations."
)

_ANALYST = (
    "You are an analysis agent. Work only from the provided context and facts. "
    "Perform comparisons and reasoning step by step. If a numeric calculation "
    "is needed and a tool is available, call the tool rather than doing mental "
    "arithmetic. State assumptions explicitly."
)

_EXECUTOR = (
    "You are an execution agent. Carry out the concrete step you are given, "
    "using tools when appropriate. Report exactly what was done and the result."
)

_SUMMARIZER = (
    "You are a synthesis agent. Combine the prior step outputs into a single, "
    "well-structured final answer. Do not introduce claims that are not "
    "supported by the context. Keep it concise and decision-ready."
)

_GENERIC = (
    "You are a helpful agent in a local inspection workbench. Complete the task "
    "using only the provided context. Do not fabricate evidence."
)

_PROMPTS: dict[str, str] = {
    "researcher": _RESEARCHER,
    "analyst": _ANALYST,
    "executor": _EXECUTOR,
    "summarizer": _SUMMARIZER,
    "custom": _GENERIC,
}

_DEFAULT_TOOLS: dict[str, list[str]] = {
    "analyst": ["calculator", "calculate_deviation", "json_parse"],
    "executor": ["calculator", "calculate_deviation", "json_parse", "read_file", "text_stats"],
    "researcher": ["read_file", "text_stats", "json_parse"],
}


def build_agent_library(llm: BaseLLM) -> dict[str, LLMAgent]:
    """Return the standard role agents, each bound to *llm*."""
    library: dict[str, LLMAgent] = {}
    for role, prompt in _PROMPTS.items():
        library[role] = LLMAgent(
            name=role,
            description=f"{role} role agent",
            system_prompt=prompt,
            llm=llm,
            tools=_DEFAULT_TOOLS.get(role, []),
            metadata={"kind": role},
        )
    # aliases so planner output using node-type names still resolves.
    # "custom" is always present (it is a key of _PROMPTS above).
    generic_agent = library["custom"]
    for node_type in NodeType:
        library.setdefault(node_type.value, generic_agent)
    return library
