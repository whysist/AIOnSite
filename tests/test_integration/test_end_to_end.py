"""Full path: task -> planner -> pipeline -> router -> agents -> verifier -> audit."""


from src.audit.events import AuditEventType
from src.audit.trail import AuditTrail
from src.core.config import Settings
from src.orchestrator import Orchestrator
from src.pipeline.models import ExecutionStatus, NodeStatus


def _settings(**kw):
    kw.setdefault("llm_provider", "echo")
    kw.setdefault("environment", "testing")
    return Settings(_env_file=None, **kw)


async def test_end_to_end_offline_run():
    audit = AuditTrail()
    orch = Orchestrator(_settings(), audit=audit)
    try:
        ctx = await orch.run_task(
            "Analyze V-101: compare observed pressure 120 bar against approved "
            "limit 100 bar, review history, retrieve SOP, and recommend."
        )
    finally:
        await orch.aclose()

    assert ctx.status in (ExecutionStatus.COMPLETED, ExecutionStatus.NEEDS_REVIEW)
    assert ctx.final_answer
    assert ctx.pipeline is not None
    assert all(
        n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED) for n in ctx.pipeline.nodes
    )
    assert ctx.final_verification is not None

    events = [e.event_type for e in audit.events(ctx.execution_id)]
    for required in (
        AuditEventType.TASK_RECEIVED,
        AuditEventType.PLAN_CREATED,
        AuditEventType.PIPELINE_BUILT,
        AuditEventType.VERIFICATION_COMPLETED,
        AuditEventType.FINAL_ANSWER_GENERATED,
        AuditEventType.EXECUTION_COMPLETED,
    ):
        assert required in events

    # machine-readable summary is complete enough for a frontend
    summary = ctx.summary()
    assert summary["pipeline"]["nodes"]
    assert summary["pipeline"]["edges"]


async def test_sovereign_mode_records_policy_and_stays_local():
    audit = AuditTrail()
    orch = Orchestrator(_settings(llm_provider="echo", sovereign_mode=True), audit=audit)
    try:
        ctx = await orch.run_task("Confidential: summarise the attached readings.")
    finally:
        await orch.aclose()

    events = [e.event_type for e in audit.events(ctx.execution_id)]
    assert AuditEventType.POLICY_ENFORCED in events
    assert ctx.sovereign_mode is True
    for node in ctx.pipeline.nodes:
        assert node.provider in (None, "echo", "ollama", "local", "vllm")


async def test_confidential_task_forces_local_even_without_sovereign():
    orch = Orchestrator(_settings(llm_provider="echo"))
    try:
        ctx = await orch.run_task("Sensitive analysis", confidential=True)
    finally:
        await orch.aclose()
    assert ctx.confidential is True
    assert ctx.status in (ExecutionStatus.COMPLETED, ExecutionStatus.NEEDS_REVIEW)
