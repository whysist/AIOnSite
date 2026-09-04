"""TaskStatus is a separate axis from ExecutionStatus -- 'the DAG finished'
must never be conflated with 'the task was accomplished'."""

from src.pipeline.models import ExecutionStatus
from src.pipeline.requirements import RequirementStatus, TaskRequirement
from src.pipeline.task_status import TaskStatus, compute_task_status
from src.verification.schemas import Recommendation, VerificationResult


def _req(status, required=True):
    return TaskRequirement(id="r1", description="d", required=required, status=status)


def test_execution_failed_is_always_task_failed():
    assert compute_task_status([], None, ExecutionStatus.FAILED) is TaskStatus.FAILED


def test_all_requirements_satisfied_and_verification_passed_is_completed():
    reqs = [_req(RequirementStatus.SATISFIED)]
    verification = VerificationResult(passed=True, score=1.0, recommendation=Recommendation.ACCEPT)
    status = compute_task_status(reqs, verification, ExecutionStatus.COMPLETED)
    assert status is TaskStatus.COMPLETED


def test_execution_completed_but_task_incomplete_is_a_valid_divergent_state():
    """The core status-semantics fix: DAG-finished must not imply task-done."""
    reqs = [_req(RequirementStatus.UNSATISFIED)]
    status = compute_task_status(reqs, None, ExecutionStatus.COMPLETED)
    assert status is TaskStatus.INCOMPLETE
    # and this is a materially different signal than execution status:
    assert ExecutionStatus.COMPLETED is not TaskStatus.COMPLETED  # different enums entirely


def test_blocked_requirement_yields_blocked_task_status():
    reqs = [_req(RequirementStatus.BLOCKED)]
    status = compute_task_status(reqs, None, ExecutionStatus.COMPLETED)
    assert status is TaskStatus.BLOCKED


def test_failed_verification_yields_incomplete():
    reqs = [_req(RequirementStatus.SATISFIED)]
    verification = VerificationResult(passed=False, score=0.2, recommendation=Recommendation.REPLAN)
    status = compute_task_status(reqs, verification, ExecutionStatus.COMPLETED)
    assert status is TaskStatus.INCOMPLETE


def test_optional_unsatisfied_requirement_does_not_block_completion():
    reqs = [
        _req(RequirementStatus.SATISFIED),
        _req(RequirementStatus.UNSATISFIED, required=False),
    ]
    status = compute_task_status(reqs, None, ExecutionStatus.COMPLETED)
    assert status is TaskStatus.COMPLETED


def test_no_requirements_and_no_verification_defaults_to_completed():
    status = compute_task_status([], None, ExecutionStatus.COMPLETED)
    assert status is TaskStatus.COMPLETED
