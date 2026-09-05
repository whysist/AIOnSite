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

from pydantic import BaseModel, Field, model_validator


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
    # The pipeline step(s) whose successful completion + produced evidence
    # are expected to satisfy this requirement -- the backbone of
    # requirement -> step -> evidence traceability. Empty when the planner
    # stated the requirement at the plan level with no single responsible
    # step (in which case verification falls back to a weaker, explicitly
    # secondary text-overlap check -- see ``verification/verifier.py``).
    source_step_ids: list[str] = Field(default_factory=list)
    # Whether satisfying this requirement is expected to leave behind
    # recorded ``Evidence`` (a tool actually retrieved/computed something),
    # as opposed to a purely reasoning/formatting step (e.g. "produce a
    # final recommendation") that has nothing to point at. Defaults to
    # ``True``: most requirements in an inspection workflow are claims
    # about a concrete fact, and treating "no evidence" as fine by default
    # would silently readmit the failure mode this field exists to catch.
    needs_evidence: bool = True
    failure_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_source_step_id(cls, data: Any) -> Any:
        """Accept the old single-step ``source_step_id=...`` constructor
        kwarg (used throughout the existing test suite and any external
        caller) and fold it into ``source_step_ids`` -- extends the schema
        without breaking call sites that predate multi-step requirements.
        """
        if isinstance(data, dict) and "source_step_id" in data and "source_step_ids" not in data:
            data = dict(data)
            legacy = data.pop("source_step_id")
            data["source_step_ids"] = [legacy] if legacy else []
        return data

    @property
    def source_step_id(self) -> str | None:
        """Backward-compatible single-step view: the first responsible step, if any."""
        return self.source_step_ids[0] if self.source_step_ids else None


def extract_requirements(plan: Any) -> list[TaskRequirement]:
    """Derive a non-empty requirement checklist from a validated plan.

    Prefers ``plan.requirements`` (the planner LLM's requirement list,
    each of which should already carry the step id(s) responsible for
    satisfying it -- see ``agents.planner.RequirementSpec``); when that is
    empty -- including for the deterministic fallback plan, which never has
    any -- falls back to one requirement per plan step, so every plan has a
    checklist that verification can check against even when the model did
    not cooperate.

    ``plan`` is accepted structurally (duck-typed) rather than typed as
    ``agents.planner.Plan`` to avoid a circular import between the planner
    and the pipeline packages.
    """
    explicit = list(getattr(plan, "requirements", None) or [])
    if explicit:
        known_step_ids = {
            step_id for s in (getattr(plan, "steps", None) or [])
            if (step_id := getattr(s, "id", None))
        }
        requirements: list[TaskRequirement] = []
        for i, spec in enumerate(explicit, start=1):
            # ``spec`` is normally a ``RequirementSpec`` (description +
            # step_ids + needs_evidence); tolerate a bare string too, for
            # any caller still constructing ``Plan.requirements`` the old
            # way (e.g. directly in a test) rather than via the planner.
            description = getattr(spec, "description", None)
            if description is None:
                description = str(spec)
            raw_step_ids = list(getattr(spec, "step_ids", None) or [])
            # Plan-level validation (see ``agents.planner.Plan``) already
            # rejects unknown step ids coming from the model, but this
            # function must stay safe for hand-built plans too, so filter
            # defensively rather than trusting the input blindly.
            linked = [sid for sid in raw_step_ids if sid in known_step_ids]
            needs_evidence = bool(getattr(spec, "needs_evidence", True))
            requirements.append(
                TaskRequirement(
                    id=f"req_{i}",
                    description=str(description),
                    source_step_ids=linked,
                    needs_evidence=needs_evidence,
                )
            )
        return requirements

    requirements = []
    for step in getattr(plan, "steps", None) or []:
        step_id = getattr(step, "id", None)
        description = getattr(step, "description", "") or step_id or ""
        if not step_id:
            continue
        requirements.append(
            TaskRequirement(
                id=f"req_{step_id}",
                description=description,
                source_step_ids=[step_id],
                # A one-requirement-per-step fallback checklist is a
                # completion signal ("did this step run"), not a claim
                # that the step must have produced retrievable evidence
                # (e.g. a summarizer step never calls a retrieval tool) --
                # so this fallback path intentionally does not require
                # evidence the way an explicit, planner-authored
                # requirement does.
                needs_evidence=False,
            )
        )
    return requirements
