"""Full path: task -> planner -> pipeline -> router -> agents -> verifier -> audit."""

import json

from src.audit.events import AuditEventType
from src.audit.trail import AuditTrail
from src.core.config import Settings
from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse
from src.orchestrator import Orchestrator
from src.pipeline.models import ExecutionStatus, NodeStatus
from src.pipeline.task_status import TaskStatus


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


class _ScriptedIncidentLLM(BaseLLM):
    """Reproduces the reported V-101 failure mode deterministically:
    the researcher step always calls ``read_file`` with the wrong argument
    name (``file_path`` instead of ``relative_path``) and never corrects
    itself, while an independent summarizer step still produces a
    confident-sounding answer. Keyed off ``aionsite_kind`` the same way
    ``EchoProvider`` is, but scripted rather than generic so the tool-call
    and repair path is actually exercised (``EchoProvider`` never emits
    tool calls at all)."""

    provider_name = "scripted-incident"

    def __init__(self) -> None:
        super().__init__(model="scripted-incident")

    async def _complete(self, request):
        kind = str(request.metadata.get("aionsite_kind", "")).lower()
        if kind == "plan":
            content = json.dumps({
                "goal": "Analyze V-101 pressure reading against the SOP",
                "steps": [
                    {
                        "id": "step_1", "agent": "researcher",
                        "description": "Retrieve the V-101 inspection package and SOP.",
                        "depends_on": [],
                    },
                    {
                        "id": "step_2", "agent": "summarizer",
                        "description": "Summarise findings into a final recommendation.",
                        "depends_on": [],
                    },
                ],
                "requirements": [
                    "Retrieve the V-101 inspection package",
                    "Retrieve the relevant SOP",
                ],
            })
        elif kind == "researcher":
            # Wrong field name, exactly as in the incident report.
            content = '{"tool": "read_file", "arguments": {"file_path": "V-101_inspection_package.txt"}}'
        else:
            content = (
                "Observed pressure 120 bar exceeds the 100 bar approved limit. "
                "The final verified answer is that the vessel should be accepted."
            )
        return LLMResponse(content=content, model=self.model, provider=self.provider_name)


async def test_v101_style_scenario_cannot_produce_false_completion(monkeypatch):
    """Reproduces the reported incident end-to-end and asserts the
    corrected behaviour: an unresolved required tool failure must never
    coexist with a task reported as COMPLETED / verification reported as
    passed."""
    scripted = _ScriptedIncidentLLM()
    # ModelRouter resolves every node's LLM through this factory function;
    # patching it here (rather than the provider config) exercises the
    # real orchestrator/router/executor/verifier wiring end-to-end while
    # keeping the model output deterministic and offline.
    monkeypatch.setattr("src.agents.router.create_llm", lambda *a, **kw: scripted)

    audit = AuditTrail()
    orch = Orchestrator(_settings(), audit=audit)
    try:
        ctx = await orch.run_task(
            "Analyze the V-101 inspection package: compare the observed operating "
            "pressure of 120 bar against the approved limit of 100 bar, retrieve "
            "the relevant SOP, and produce a final inspection recommendation."
        )
    finally:
        await orch.aclose()

    events = [e.event_type for e in audit.events(ctx.execution_id)]
    assert AuditEventType.TOOL_REPAIR_ATTEMPTED in events
    assert AuditEventType.TOOL_REPAIR_EXHAUSTED in events
    assert AuditEventType.TASK_STATUS_DETERMINED in events

    researcher_node = ctx.pipeline.node("step_1")
    assert researcher_node.status is NodeStatus.FAILED
    assert researcher_node.error is not None

    # The core acceptance criterion: tool failure + missing evidence must
    # never coexist with a completed task / a passed verification.
    assert ctx.task_status is not TaskStatus.COMPLETED
    assert not (
        ctx.final_verification is not None and ctx.final_verification.passed
        and ctx.task_status is TaskStatus.COMPLETED
    )
    if ctx.final_answer is not None:
        assert ctx.final_answer_degraded is True
        assert "INSPECTION STATUS" in ctx.final_answer
        assert "INCOMPLETE" in ctx.final_answer or "BLOCKED" in ctx.final_answer
