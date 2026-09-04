"""Task completion status -- distinct from :class:`ExecutionStatus`.

``ExecutionStatus`` (``src/pipeline/models.py``) answers "did the DAG finish
running?". It says nothing about whether the requested task was actually
accomplished: a pipeline can run every node to completion while its most
important tool call silently failed. ``TaskStatus`` answers the second
question, computed from :class:`~src.pipeline.requirements.TaskRequirement`
satisfaction and the verifier's verdict, so the two can (and, when evidence
or requirements are missing, should) diverge -- e.g. execution ``COMPLETED``
with task ``INCOMPLETE`` is an expected, valid combination.
"""

from __future__ import annotations

import enum

from ..verification.schemas import VerificationResult
from .models import ExecutionStatus
from .requirements import RequirementStatus, TaskRequirement


class TaskStatus(str, enum.Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"
    FAILED = "failed"


def compute_task_status(
    requirements: list[TaskRequirement],
    verification: VerificationResult | None,
    execution_status: ExecutionStatus,
) -> TaskStatus:
    """Combine requirement satisfaction + verification + execution outcome.

    Order of precedence: a hard execution failure always wins (nothing to
    evaluate); a *required* requirement that was structurally blocked (its
    producing step failed/was skipped) is reported as BLOCKED rather than
    merely INCOMPLETE, since it reflects a missing capability rather than an
    unconvincing answer; a verification failure or any unsatisfied required
    requirement makes the task INCOMPLETE; only when every required
    requirement is satisfied and verification passed (or was not run at all)
    is the task COMPLETED.
    """
    if execution_status is ExecutionStatus.FAILED:
        return TaskStatus.FAILED

    if any(
        r.required and r.status is RequirementStatus.BLOCKED for r in requirements
    ):
        return TaskStatus.BLOCKED

    if verification is not None and not verification.passed:
        return TaskStatus.INCOMPLETE

    unresolved_required = [
        r
        for r in requirements
        if r.required and r.status is not RequirementStatus.SATISFIED
    ]
    if unresolved_required:
        return TaskStatus.INCOMPLETE

    return TaskStatus.COMPLETED
