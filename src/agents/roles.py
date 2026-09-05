"""Concrete role agents used by the pipeline.

Each is an :class:`LLMAgent` with a role-specific system prompt.  The
factory :func:`build_agent_library` returns a name -> agent mapping that
the pipeline executor resolves node ``agent`` fields against.
"""

from __future__ import annotations

from ..llm.base import BaseLLM
from ..pipeline.models import NodeType
from .base_agent import LLMAgent

_EVIDENCE_REUSE = (
    "Before calling a retrieval tool, check the STRUCTURED EVIDENCE block (if "
    "present) for what earlier steps already retrieved. If it already answers "
    "what you need, use it directly -- do not call the same tool again for the "
    "same equipment/document/query. Only retrieve again when the information "
    "you need is genuinely absent from that evidence, is insufficient for the "
    "question, or requires a different target (a different equipment id, a "
    "different query type, a different document) than what is already there."
)

_RESEARCHER = (
    "You are a research agent in a local, sovereign inspection workbench. "
    "Given a task and any context, extract and organise the relevant facts. "
    "Prefer real, retrieved evidence over guessing: use equipment_lookup for "
    "structured equipment/maintenance/inspection records, search_knowledge_base "
    "or extract_structured_evidence for facts from ingested documents, and "
    "process_document if you need to read a specific document file directly. "
    "Be concrete. If information is missing, say so explicitly -- never invent "
    "evidence, numbers, or citations. " + _EVIDENCE_REUSE
)

_ANALYST = (
    "You are an analysis agent. Work only from the provided context and facts. "
    "Perform comparisons and reasoning step by step. If a numeric calculation "
    "is needed and a tool is available, call the tool rather than doing mental "
    "arithmetic. If a specific fact (e.g. an operating limit) is missing from "
    "both the provided evidence and context, look it up with equipment_lookup "
    "or extract_structured_evidence rather than assuming a value -- but check "
    "the evidence already provided first. State assumptions explicitly. "
    + _EVIDENCE_REUSE
)

_EXECUTOR = (
    "You are an execution agent. Carry out the concrete step you are given, "
    "using tools when appropriate. Report exactly what was done and the result. "
    "If the step asks for a deliverable document (e.g. an approval note, a "
    "findings summary) rather than just an answer in chat, use "
    "generate_word_document to actually produce it -- do not describe what "
    "the document would contain instead of generating it. " + _EVIDENCE_REUSE
)

_SUMMARIZER = (
    "You are a synthesis agent. Combine the prior step outputs into a single, "
    "well-structured final answer. Do not introduce claims that are not "
    "supported by the context. Keep it concise and decision-ready. If the "
    "task asked for the result as a Word document (e.g. an approval note or "
    "report), use generate_word_document with the finalised content -- do not "
    "just describe the document in your text response instead of generating it."
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
    "analyst": [
        "calculator", "calculate_deviation", "json_parse",
        "equipment_lookup", "extract_structured_evidence", "execute_python",
    ],
    "executor": [
        "calculator", "calculate_deviation", "json_parse", "read_file", "text_stats",
        "equipment_lookup", "process_document", "search_knowledge_base", "execute_python",
        "generate_word_document",
    ],
    "researcher": [
        "read_file", "text_stats", "json_parse",
        "equipment_lookup", "process_document",
        "search_knowledge_base", "extract_structured_evidence",
    ],
    "summarizer": ["generate_word_document"],
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
