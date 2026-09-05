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


class _ScriptedReplanLLM(BaseLLM):
    """Every call returns a valid plan and a valid final answer -- the only
    thing that varies is the verifier's LLM critique, which fails the first
    attempt (recommendation=replan) and passes the second. Used to prove the
    orchestrator's auto-replan loop actually re-plans and re-runs on a
    REPLAN verdict, bounded by ``max_replans``, rather than just reporting
    the verdict and stopping.
    """

    provider_name = "scripted-replan"

    def __init__(self) -> None:
        super().__init__(model="scripted-replan")
        self.verify_calls = 0
        self.plan_calls = 0

    async def _complete(self, request):
        kind = str(request.metadata.get("aionsite_kind", "")).lower()
        if kind == "plan":
            self.plan_calls += 1
            content = json.dumps({
                "goal": "Produce a decisive final answer",
                "steps": [
                    {
                        "id": "step_1", "agent": "summarizer",
                        "description": "Produce the final answer.", "depends_on": [],
                    }
                ],
                "requirements": ["Provide a decisive final answer"],
            })
        elif kind == "verify":
            self.verify_calls += 1
            passed = self.verify_calls > 1
            content = json.dumps({
                "passed": passed,
                "score": 1.0 if passed else 0.2,
                "issues": [] if passed else [
                    {"code": "incomplete", "message": "no decisive answer", "severity": "critical"}
                ],
                "recommendation": "accept" if passed else "replan",
            })
        else:
            content = "Final answer: the vessel is within limits [source: equipment database, table]."
        return LLMResponse(content=content, model=self.model, provider=self.provider_name)


async def test_orchestrator_auto_replans_on_replan_verdict_then_stops(monkeypatch):
    from src.tools.registry import ToolRegistry

    scripted = _ScriptedReplanLLM()
    monkeypatch.setattr("src.agents.router.create_llm", lambda *a, **kw: scripted)

    audit = AuditTrail()
    orch = Orchestrator(
        _settings(max_replans=1), audit=audit, registry=ToolRegistry(),
    )
    try:
        ctx = await orch.run_task("Is V-101 within limits?")
    finally:
        await orch.aclose()

    # One initial plan + one replan = 2 plans; one failing verify + one
    # passing verify = 2 verifications -- proves the loop actually ran
    # twice, not that it merely reported "replan" and stopped.
    assert scripted.plan_calls == 2
    assert scripted.verify_calls == 2
    assert ctx.replans == 1
    assert ctx.final_verification is not None
    assert ctx.final_verification.passed is True

    events = [e.event_type for e in audit.events(ctx.execution_id)]
    assert AuditEventType.REPLAN_TRIGGERED in events
    # the archived first attempt is still visible, not silently discarded
    assert len(ctx.state.scratch.get("attempts", [])) == 1


async def test_orchestrator_stops_replanning_at_max_replans(monkeypatch):
    """A verifier that always says replan must not loop forever -- it stops
    at settings.max_replans and reports whatever the last attempt produced."""
    from src.tools.registry import ToolRegistry

    class _AlwaysReplanLLM(_ScriptedReplanLLM):
        async def _complete(self, request):
            kind = str(request.metadata.get("aionsite_kind", "")).lower()
            if kind == "verify":
                self.verify_calls += 1
                return LLMResponse(
                    content=json.dumps({
                        "passed": False, "score": 0.1,
                        "issues": [{"code": "incomplete", "message": "x", "severity": "critical"}],
                        "recommendation": "replan",
                    }),
                    model=self.model, provider=self.provider_name,
                )
            return await super()._complete(request)

    scripted = _AlwaysReplanLLM()
    monkeypatch.setattr("src.agents.router.create_llm", lambda *a, **kw: scripted)

    orch = Orchestrator(
        _settings(max_replans=2), registry=ToolRegistry(),
    )
    try:
        ctx = await orch.run_task("Is V-101 within limits?")
    finally:
        await orch.aclose()

    # max_replans=2 -> at most 3 total plan/run cycles (1 original + 2 replans)
    assert scripted.plan_calls == 3
    assert ctx.replans == 2
    assert ctx.final_verification is not None
    assert ctx.final_verification.passed is False
