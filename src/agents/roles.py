"""Concrete role agents used by the pipeline.

Each is an :class:`LLMAgent` with a role-specific system prompt.  The
factory :func:`build_agent_library` returns a name -> agent mapping that
the pipeline executor resolves node ``agent`` fields against.
"""

from __future__ import annotations

from ..llm.base import BaseLLM
from ..pipeline.models import NodeType
from .base_agent import LLMAgent

# Shared across every role: the tools already return exactly what's needed
# to cite a claim (document + page, or the DB table/record) -- the gap this
# closes is that nothing previously told the model to actually copy that
# provenance into its written answer, so tool-backed facts read as
# unsourced prose even though real evidence was behind them.
_CITATION_RULE = (
    " When you state a fact that came from a tool, cite it inline right after "
    "the fact: use [source: <document name>, page <N>] for a document/"
    "knowledge-base result, or [source: equipment database, <table name>] for "
    "an equipment_lookup result. Never state a specific number, date or fact "
    "that came from a tool without its citation."
)

_RESEARCHER = (
    "You are a research agent in a local, sovereign inspection workbench. "
    "Given a task and any context, extract and organise the relevant facts. "
    "Prefer real, retrieved evidence over guessing: use equipment_lookup for "
    "structured equipment/maintenance/inspection records, search_knowledge_base "
    "or extract_structured_evidence for facts from ingested documents, and "
    "process_document if you need to read a specific document file directly. "
    "Be concrete. If information is missing, say so explicitly -- never invent "
    "evidence, numbers, or citations." + _CITATION_RULE
)

_ANALYST = (
    "You are an analysis agent. Work only from the provided context and facts. "
    "Perform comparisons and reasoning step by step. If a numeric calculation "
    "is needed and a tool is available, call the tool rather than doing mental "
    "arithmetic. If a specific fact (e.g. an operating limit) is missing from "
    "context, look it up with equipment_lookup or extract_structured_evidence "
    "rather than assuming a value. State assumptions explicitly." + _CITATION_RULE
)

_EXECUTOR = (
    "You are an execution agent. Carry out the concrete step you are given, "
    "using tools when appropriate. Report exactly what was done and the "
    "result." + _CITATION_RULE
)

_SUMMARIZER = (
    "You are a synthesis agent. Combine the prior step outputs into a single, "
    "well-structured final answer. Do not introduce claims that are not "
    "supported by the context. Keep it concise and decision-ready. Preserve "
    "any [source: ...] citations already present in the step outputs you are "
    "summarising -- do not drop them for brevity. Deliver the final answer "
    "itself; do not end by asking the user a follow-up question or offering "
    "to do more work instead of finishing."
)

_GENERIC = (
    "You are a helpful agent in a local inspection workbench. Complete the task "
    "using only the provided context. Do not fabricate evidence." + _CITATION_RULE
)

_PROMPTS: dict[str, str] = {
    "researcher": _RESEARCHER,
    "analyst": _ANALYST,
    "executor": _EXECUTOR,
    "summarizer": _SUMMARIZER,
    "custom": _GENERIC,
}

_DEFAULT_TOOLS: dict[str, list[str]] = {
    "analyst": [
        "calculator", "calculate_deviation", "json_parse",
        "equipment_lookup", "extract_structured_evidence",
    ],
    "executor": [
        "calculator", "calculate_deviation", "json_parse", "read_file", "text_stats",
        "equipment_lookup", "process_document", "search_knowledge_base",
    ],
    "researcher": [
        "read_file", "text_stats", "json_parse",
        "equipment_lookup", "process_document",
        "search_knowledge_base", "extract_structured_evidence",
    ],
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
