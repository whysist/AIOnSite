"""Deterministic task classification, run before planning.

This is the fix for the audited gap where every request -- from "tell me
about P-101" to a full multi-source investigation -- went through the
identical Planner (LLM) -> DAG -> multi-agent -> Verifier workflow. That is
not "the multi-agent path is wrong"; it is that nothing ever decided
*whether a task needed it* before paying for it.

Classification here is intentionally narrow and rule-based, not another
LLM call: the categories below are coarse, the equipment-id extraction is a
plain regex, and the routing decision it feeds
(``TaskClassification.requires_planning``) defaults to ``True`` (the
existing, already-correct full path) whenever the rules are not clearly
satisfied. A wrong "this is complex" guess costs a little latency; a wrong
"this is simple" guess would under-serve a real investigation, so the
rules are deliberately conservative in that direction.
"""

from __future__ import annotations

import enum
import re

from pydantic import BaseModel


class TaskComplexity(str, enum.Enum):
    SIMPLE_LOOKUP = "simple_lookup"
    RETRIEVAL_QUERY = "retrieval_query"
    CALCULATION = "calculation"
    DOCUMENT_ANALYSIS = "document_analysis"
    INSPECTION_ANALYSIS = "inspection_analysis"
    MULTI_STEP_INVESTIGATION = "multi_step_investigation"


class TaskClassification(BaseModel):
    complexity: TaskComplexity
    # Whether the full Planner (LLM) -> DAG -> multi-agent -> Verifier path
    # is needed. ``False`` only for a narrowly-scoped, high-confidence
    # direct lookup (see ``classify_task``).
    requires_planning: bool
    reason: str
    equipment_id: str | None = None


_EQUIPMENT_ID = re.compile(r"\b([A-Z]{1,4}-\d{2,5})\b")

_LOOKUP_PHRASES = (
    "tell me about", "what is", "what's", "describe", "give me information",
    "information about", "details about", "who owns", "where is", "info on",
)

# Presence of any of these means the task is more than a plain identity
# lookup -- comparison, computation, historical review, or a decision --
# and must go through full planning regardless of how it is phrased.
_COMPLEXITY_SIGNALS = (
    "compare", "calculate", "compute", "deviation", "recommend",
    "recommendation", "investigate", "investigation", "should", "sop",
    "compliance", "maintenance history", "inspection history", "versus",
    " vs ", "against", "root cause", "risk", "decommission",
    "out of service", "trend", "over time", "history and", "and calculate",
    "and recommend", "audit",
)

_INVESTIGATION_SIGNALS = (
    "investigate", "investigation", "recommend", "recommendation",
    "should", "risk", "decommission", "out of service", "root cause",
)

_MAX_SIMPLE_LOOKUP_WORDS = 25


def _extract_equipment_id(task: str) -> str | None:
    match = _EQUIPMENT_ID.search(task)
    return match.group(1) if match else None


def classify_task(task: str) -> TaskClassification:
    """Classify *task* without calling any model.

    Only ever returns ``requires_planning=False`` for a short, unambiguous
    equipment-identity question (an equipment id present, a lookup phrase
    present, no complexity signal, and under a short word-count cap).
    Everything else -- including anything ambiguous -- keeps the existing
    full Planner/DAG/Verifier path, which was already correct; this
    function only *adds* a cheaper path for the case that clearly doesn't
    need it.
    """
    text = task.strip()
    lowered = text.lower()
    equipment_id = _extract_equipment_id(text)
    word_count = len(lowered.split())
    has_complexity_signal = any(sig in lowered for sig in _COMPLEXITY_SIGNALS)
    has_lookup_phrase = any(p in lowered for p in _LOOKUP_PHRASES)

    if (
        equipment_id
        and has_lookup_phrase
        and not has_complexity_signal
        and word_count <= _MAX_SIMPLE_LOOKUP_WORDS
    ):
        return TaskClassification(
            complexity=TaskComplexity.SIMPLE_LOOKUP,
            requires_planning=False,
            equipment_id=equipment_id,
            reason=(
                "short equipment-identity question with no calculation, "
                "comparison or investigation signal"
            ),
        )

    if equipment_id and has_complexity_signal:
        complexity = (
            TaskComplexity.MULTI_STEP_INVESTIGATION
            if any(sig in lowered for sig in _INVESTIGATION_SIGNALS)
            else TaskComplexity.INSPECTION_ANALYSIS
        )
        return TaskClassification(
            complexity=complexity,
            requires_planning=True,
            equipment_id=equipment_id,
            reason="calculation, comparison or investigation signal present",
        )

    if has_complexity_signal and any(w in lowered for w in ("calculate", "compute", "deviation")):
        return TaskClassification(
            complexity=TaskComplexity.CALCULATION,
            requires_planning=True,
            equipment_id=equipment_id,
            reason="calculation signal present without a specific equipment id",
        )

    if equipment_id:
        return TaskClassification(
            complexity=TaskComplexity.RETRIEVAL_QUERY,
            requires_planning=True,
            equipment_id=equipment_id,
            reason=(
                "equipment referenced but the request does not clearly match "
                "the narrow simple-lookup pattern"
            ),
        )

    return TaskClassification(
        complexity=TaskComplexity.MULTI_STEP_INVESTIGATION,
        requires_planning=True,
        equipment_id=None,
        reason="no equipment id detected; defaulting to full planning",
    )
