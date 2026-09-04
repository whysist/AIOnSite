"""Task requirement tracking.

A :class:`~src.agents.planner.Plan` used to carry only a free-text ``goal``
and steps -- there was nothing to check the final answer against, so a
pipeline that ran to completion and a pipeline that actually satisfied the
user's task were indistinguishable. :class:`TaskRequirement` gives the rest
of the system (executor, verifier, orchestrator) an explicit, individually
trackable checklist derived from the plan, so completeness can be verified
rather than assumed.

:func:`extract_requirements` is deliberately a plain deterministic function,
not another LLM call: task-requirement extraction must be something the rest
of the system can depend on even when the model is unavailable or degraded
(see ``Plan.degraded`` in ``src/agents/planner.py``).
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class RequirementStatus(str, enum.Enum):
    PENDING = "pending"
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class TaskRequirement(BaseModel):
    id: str
    description: str
    required: bool = True
    status: RequirementStatus = RequirementStatus.PENDING
    evidence_ids: list[str] = Field(default_factory=list)
    # The plan step whose successful completion is expected to satisfy this
    # requirement, when known. ``None`` for requirements the planner stated
    # explicitly at the plan level rather than deriving from a single step.
    source_step_id: str | None = None
    failure_reason: str | None = None


def extract_requirements(plan: Any) -> list[TaskRequirement]:
    """Derive a non-empty requirement checklist from a validated plan.

    Prefers ``plan.requirements`` (free-text requirements the planner LLM
    was asked to list); when that is empty -- including for the deterministic
    fallback plan, which never has any -- falls back to one requirement per
    plan step, so every plan has a checklist that verification can check
    against even when the model did not cooperate.

    ``plan`` is accepted structurally (duck-typed) rather than typed as
    ``agents.planner.Plan`` to avoid a circular import between the planner
    and the pipeline packages.
    """
    explicit = list(getattr(plan, "requirements", None) or [])
    if explicit:
        return [
            TaskRequirement(id=f"req_{i}", description=str(desc))
            for i, desc in enumerate(explicit, start=1)
        ]

    requirements: list[TaskRequirement] = []
    for step in getattr(plan, "steps", None) or []:
        step_id = getattr(step, "id", None)
        description = getattr(step, "description", "") or step_id or ""
        if not step_id:
            continue
        requirements.append(
            TaskRequirement(
                id=f"req_{step_id}",
                description=description,
                source_step_id=step_id,
            )
        )
    return requirements
