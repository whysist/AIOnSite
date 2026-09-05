"""Full path: task -> planner -> pipeline -> router -> agents -> verifier -> audit."""

import json
import sqlite3
from contextlib import contextmanager

import pytest

from src.audit.events import AuditEventType
from src.audit.trail import AuditTrail
from src.core.config import Settings
from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse
from src.orchestrator import Orchestrator
from src.pipeline.models import ExecutionStatus, NodeStatus
from src.pipeline.task_status import TaskStatus


@pytest.fixture
def equipment_db(tmp_path, monkeypatch):
    """Seed a temp SQLite DB with a P-101 row, same schema as
    ``tests/test_tools/test_equipment_lookup.py`` -- lets integration tests
    exercise the real ``equipment_lookup`` tool deterministically."""
    path = tmp_path / "equipment.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE equipment (
            equipment_id TEXT, name TEXT, type TEXT, service TEXT, status TEXT, area TEXT
        );
        CREATE TABLE operating_parameters (
            equipment_id TEXT, parameter TEXT, normal_min REAL, normal_max REAL,
            recommended_limit REAL, absolute_limit REAL, unit TEXT, revision TEXT
        );
        CREATE TABLE inspection_records (
            inspection_id INTEGER, equipment_id TEXT, date TEXT, parameter TEXT,
            observed_value REAL, unit TEXT, observation TEXT
        );
        CREATE TABLE maintenance_records (
            record_id INTEGER, equipment_id TEXT, date TEXT, issue TEXT,
            action TEXT, status TEXT, notes TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO equipment VALUES (?,?,?,?,?,?)",
        ("P101", "P-101 Feed Pump", "Centrifugal Pump", "Feed", "Active", "Area 2"),
    )
    conn.commit()
    conn.close()

    @contextmanager
    def _fake_connection(db_path=None):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    monkeypatch.setattr("src.database.models.get_connection", _fake_connection)
    return path


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


# ---------------------------------------------------------------------
# Priority 3: intent/complexity classification routing.


class _LookupOnlyLLM(BaseLLM):
    """Fails the test outright if ever asked to plan -- proves the fast
    path genuinely skips the planning LLM call rather than merely hiding
    it."""

    provider_name = "lookup-only"

    def __init__(self) -> None:
        super().__init__(model="lookup-only")

    async def _complete(self, request):
        kind = str(request.metadata.get("aionsite_kind", "")).lower()
        if kind == "plan":
            raise AssertionError(
                "planner LLM was called for a task classified as a simple lookup"
            )
        if kind == "researcher":
            content = '{"tool": "equipment_lookup", "arguments": {"equipment_id": "P-101", "query_type": "info"}}'
        elif kind == "verify":
            content = json.dumps({"passed": True, "score": 1.0, "issues": [], "recommendation": "accept"})
        else:
            content = "P-101 is a centrifugal feed pump, currently active."
        return LLMResponse(content=content, model=self.model, provider=self.provider_name)


async def test_simple_lookup_bypasses_planning_and_completes(monkeypatch, equipment_db):
    """TEST 6: a simple equipment-identity question must not invoke the
    planner LLM at all, and must still complete with a correct evidence-
    linked verification."""
    scripted = _LookupOnlyLLM()
    monkeypatch.setattr("src.agents.router.create_llm", lambda *a, **kw: scripted)

    audit = AuditTrail()
    orch = Orchestrator(_settings(), audit=audit)
    try:
        ctx = await orch.run_task("Tell me about P-101. What is the equipment?")
    finally:
        await orch.aclose()

    assert ctx.metadata["classification"]["complexity"] == "simple_lookup"
    assert ctx.metadata["classification"]["requires_planning"] is False
    # deterministic lookup plan: researcher + summarizer + the appended verifier
    assert [n.id for n in ctx.pipeline.nodes] == ["step_1", "step_2", "verify_final"]
    assert ctx.plan_degraded is False
    assert ctx.final_answer is not None
    assert ctx.task_status is TaskStatus.COMPLETED
    assert ctx.final_verification is not None and ctx.final_verification.passed is True

    events = [e.event_type for e in audit.events(ctx.execution_id)]
    assert AuditEventType.TASK_CLASSIFIED in events
    assert AuditEventType.PLAN_FAILED not in events


async def test_complex_investigation_still_uses_full_planning(monkeypatch, equipment_db):
    """TEST 7: a task with comparison/investigation signals must still go
    through the full Planner -> DAG -> multi-agent -> Verifier path."""
    audit = AuditTrail()
    orch = Orchestrator(_settings(), audit=audit)
    try:
        ctx = await orch.run_task(
            "Analyze P-101: compare observed pressure 120 bar against approved "
            "limit 100 bar, review history, retrieve SOP, and recommend."
        )
    finally:
        await orch.aclose()

    assert ctx.metadata["classification"]["requires_planning"] is True
    assert ctx.status in (ExecutionStatus.COMPLETED, ExecutionStatus.NEEDS_REVIEW)


async def test_sovereign_mode_still_enforced_on_the_simple_lookup_fast_path(monkeypatch, equipment_db):
    """TEST 8: the classification fast path must not create a loophole
    around sovereignty enforcement -- lookup-plan nodes still go through
    the same ``build_pipeline``/``ModelRouter`` policy as any other plan."""
    scripted = _LookupOnlyLLM()
    monkeypatch.setattr("src.agents.router.create_llm", lambda *a, **kw: scripted)

    orch = Orchestrator(_settings(llm_provider="echo", sovereign_mode=True))
    try:
        ctx = await orch.run_task("Tell me about P-101.")
    finally:
        await orch.aclose()

    assert ctx.metadata["classification"]["complexity"] == "simple_lookup"
    for node in ctx.pipeline.nodes:
        assert node.require_local is True
        assert node.provider in (None, "echo", "ollama", "local", "vllm")


# ---------------------------------------------------------------------
# Priority 4/5: duplicate retrieval elimination via evidence reuse + cache.


class _DuplicateRetrievalLLM(BaseLLM):
    """Two independent agents that would, pre-fix, each call
    ``equipment_lookup`` themselves -- both end up issuing the identical
    call, proving the cache (not merely hoped-for prompt discipline)
    prevents the second one from actually re-executing it."""

    provider_name = "dup-retrieval"

    def __init__(self) -> None:
        super().__init__(model="dup-retrieval")
        self._tool_call_counts: dict[str, int] = {}

    async def _complete(self, request):
        kind = str(request.metadata.get("aionsite_kind", "")).lower()
        call = '{"tool": "equipment_lookup", "arguments": {"equipment_id": "P-101", "query_type": "all"}}'
        if kind == "plan":
            return LLMResponse(
                content=json.dumps({
                    "goal": "Describe P-101 fully",
                    "steps": [
                        {"id": "step_1", "agent": "researcher", "description": "Retrieve P-101 data.", "depends_on": []},
                        {"id": "step_2", "agent": "analyst", "description": "Double-check P-101 data.", "depends_on": ["step_1"]},
                        {"id": "step_3", "agent": "summarizer", "description": "Summarise.", "depends_on": ["step_2"]},
                    ],
                    "requirements": [
                        {"description": "Report P-101 equipment info", "step_ids": ["step_1"], "needs_evidence": True},
                    ],
                }),
                model=self.model, provider=self.provider_name,
            )
        if kind in ("researcher", "analyst"):
            calls_so_far = self._tool_call_counts.get(kind, 0)
            self._tool_call_counts[kind] = calls_so_far + 1
            if calls_so_far == 0:
                return LLMResponse(content=call, model=self.model, provider=self.provider_name)
            return LLMResponse(
                content=f"{kind} done using equipment_lookup", model=self.model, provider=self.provider_name,
            )
        if kind == "verify":
            return LLMResponse(
                content=json.dumps({"passed": True, "score": 1.0, "issues": [], "recommendation": "accept"}),
                model=self.model, provider=self.provider_name,
            )
        return LLMResponse(content="P-101 is a pump.", model=self.model, provider=self.provider_name)


async def test_second_agents_duplicate_tool_call_is_served_from_cache(monkeypatch, equipment_db):
    """TEST 5: when a downstream agent nonetheless issues the same
    retrieval call a dependency already made, the execution-scoped tool
    cache serves it instead of hitting the database again, and no
    duplicate Evidence record is created."""
    scripted = _DuplicateRetrievalLLM()
    monkeypatch.setattr("src.agents.router.create_llm", lambda *a, **kw: scripted)

    audit = AuditTrail()
    orch = Orchestrator(_settings(), audit=audit)
    try:
        ctx = await orch.run_task("Tell me everything about P-101.")
    finally:
        await orch.aclose()

    equipment_lookup_calls = [
        tr for tr in ctx.state.tool_results if tr.tool == "equipment_lookup"
    ]
    assert len(equipment_lookup_calls) == 2
    assert equipment_lookup_calls[0].cached is False
    assert equipment_lookup_calls[1].cached is True
    # Only one Evidence record exists for the two (identical) calls.
    equipment_evidence = [e for e in ctx.state.evidence.values() if e.source == "equipment_lookup"]
    assert len(equipment_evidence) == 1

    events = [e.event_type for e in audit.events(ctx.execution_id)]
    assert AuditEventType.TOOL_CACHE_HIT in events
