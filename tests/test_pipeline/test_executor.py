import pytest

from src.agents.base_agent import Agent
from src.agents.router import ModelRouter
from src.audit.trail import AuditTrail
from src.core.config import Settings
from src.core.exceptions import PipelineExecutionError
from src.pipeline.executor import PipelineExecutor
from src.pipeline.models import (
    AgentResult,
    Criticality,
    NodeOutcome,
    NodeStatus,
    NodeType,
    Pipeline,
    PipelineNode,
    RetryPolicy,
    ToolResult,
)
from src.pipeline.state import ExecutionContext
from src.verification.verifier import ResultVerifier


class FakeAgent(Agent):
    def __init__(self, name, *, fail_times=0, record=None):
        super().__init__(name=name)
        self.fail_times = fail_times
        self.calls = 0
        self.record = record if record is not None else []

    def with_llm(self, llm):
        return self

    async def execute(self, task, *, context=None, registry=None, evidence=None, tool_cache=None):
        self.calls += 1
        self.record.append((self.name, sorted((context or {}).keys())))
        if self.calls <= self.fail_times:
            raise RuntimeError(f"{self.name} transient failure #{self.calls}")
        return AgentResult(agent=self.name, ok=True, output=f"{self.name}-ok")


def _executor(library):
    settings = Settings(_env_file=None, llm_provider="echo")
    return PipelineExecutor(
        agent_library=library,
        registry=None,
        router=ModelRouter(settings),
        audit=AuditTrail(),
        verifier=ResultVerifier(None),
        settings=settings,
    ), settings


def _node(nid, agent, deps=(), ntype=NodeType.CUSTOM, retries=0, criticality=Criticality.REQUIRED):
    return PipelineNode(
        id=nid, name=nid, type=ntype, agent=agent, depends_on=list(deps),
        retry_policy=RetryPolicy(max_retries=retries, backoff_seconds=0),
        criticality=criticality,
    )


class ToolFailAgent(Agent):
    """Returns without raising -- ``AgentResult.ok=False`` with an
    unresolved tool failure, simulating a repair-exhausted tool call. Used
    to test that node outcome/status react to ``result.ok``/``tool_results``
    directly, not only to exceptions."""

    def __init__(self, name, *, tool_name="calculate_deviation"):
        super().__init__(name=name)
        self.tool_name = tool_name

    def with_llm(self, llm):
        return self

    async def execute(self, task, *, context=None, registry=None, evidence=None, tool_cache=None):
        tr = ToolResult(tool=self.tool_name, ok=False, error_kind="validation", error="bad args")
        return AgentResult(
            agent=self.name, ok=False, output="could not complete the calculation",
            tool_results=[tr],
            metadata={
                "unresolved_tools": [self.tool_name],
                "repair_log": [
                    {"tool": self.tool_name, "attempt": 1, "error": "bad args", "exhausted": True}
                ],
            },
        )


class ToolOkAgent(Agent):
    """Succeeds with one tool call, for evidence-tracking tests."""

    def __init__(self, name, *, tool_name="read_file", output="file contents"):
        super().__init__(name=name)
        self.tool_name = tool_name
        self.output = output

    def with_llm(self, llm):
        return self

    async def execute(self, task, *, context=None, registry=None, evidence=None, tool_cache=None):
        tr = ToolResult(tool=self.tool_name, ok=True, output=self.output)
        return AgentResult(agent=self.name, ok=True, output="done", tool_results=[tr])


async def test_sequential_pipeline_completes_in_order():
    order = []
    lib = {n: FakeAgent(n, record=order) for n in ("a", "b", "c")}
    pipeline = Pipeline(goal="g", nodes=[
        _node("a", "a"), _node("b", "b", ["a"]), _node("c", "c", ["b"]),
    ])
    ex, _ = _executor(lib)
    ctx = ExecutionContext(task="t")
    await ex.run(ctx, pipeline)
    assert [n.status for n in pipeline.nodes] == [NodeStatus.COMPLETED] * 3
    assert [r[0] for r in order] == ["a", "b", "c"]
    # b and c received upstream context
    assert order[1] == ("b", ["a"])


async def test_parallel_nodes_both_run_and_downstream_sees_both():
    order = []
    lib = {n: FakeAgent(n, record=order) for n in ("root", "l", "r", "join")}
    pipeline = Pipeline(goal="g", nodes=[
        _node("root", "root"),
        _node("l", "l", ["root"]),
        _node("r", "r", ["root"]),
        _node("join", "join", ["l", "r"]),
    ])
    ex, _ = _executor(lib)
    await ex.run(ExecutionContext(task="t"), pipeline)
    assert all(n.status is NodeStatus.COMPLETED for n in pipeline.nodes)
    join_ctx = next(rec for rec in order if rec[0] == "join")
    assert join_ctx[1] == ["l", "r"]


async def test_retry_then_success():
    agent = FakeAgent("flaky", fail_times=2)
    pipeline = Pipeline(goal="g", nodes=[_node("n", "flaky", retries=2)])
    ex, _ = _executor({"flaky": agent})
    ctx = ExecutionContext(task="t")
    await ex.run(ctx, pipeline)
    assert pipeline.nodes[0].status is NodeStatus.COMPLETED
    assert pipeline.nodes[0].attempts == 3


async def test_failure_propagates_and_skips_dependents():
    lib = {"bad": FakeAgent("bad", fail_times=99), "down": FakeAgent("down")}
    pipeline = Pipeline(goal="g", nodes=[
        _node("bad", "bad", retries=1),
        _node("down", "down", ["bad"]),
    ])
    ex, _ = _executor(lib)
    ctx = ExecutionContext(task="t")
    with pytest.raises(PipelineExecutionError):
        await ex.run(ctx, pipeline)
    assert pipeline.node("bad").status is NodeStatus.FAILED
    assert pipeline.node("down").status is NodeStatus.SKIPPED
    assert lib["down"].calls == 0


async def test_verifier_node_produces_verification_result():
    lib = {"a": FakeAgent("a")}
    pipeline = Pipeline(goal="g", nodes=[
        _node("a", "a"),
        _node("v", "verifier", ["a"], ntype=NodeType.VERIFIER),
    ])
    ex, _ = _executor(lib)
    ctx = ExecutionContext(task="t")
    await ex.run(ctx, pipeline)
    assert "v" in ctx.state.verifications
    assert ctx.final_verification is not None


# ---------------------------------------------------------------------
# Criticality gating: a node whose agent returns ok=False (unresolved tool
# failure) without raising an exception must not be silently reported as
# COMPLETED -- this is the direct fix for the reported bug where
# AgentResult.ok was computed but never read by the executor.


async def test_required_tool_failure_blocks_node_and_downstream():
    lib = {"bad": ToolFailAgent("bad"), "down": FakeAgent("down")}
    pipeline = Pipeline(goal="g", nodes=[
        _node("bad", "bad"),  # default criticality=REQUIRED
        _node("down", "down", ["bad"]),
    ])
    ex, _ = _executor(lib)
    ctx = ExecutionContext(task="t")
    with pytest.raises(PipelineExecutionError):
        await ex.run(ctx, pipeline)
    node = pipeline.node("bad")
    assert node.status is NodeStatus.FAILED
    assert node.outcome is NodeOutcome.FAILED
    assert pipeline.node("down").status is NodeStatus.SKIPPED
    assert lib["down"].calls == 0


async def test_optional_tool_failure_degrades_but_allows_continuation():
    lib = {"opt": ToolFailAgent("opt"), "down": FakeAgent("down")}
    pipeline = Pipeline(goal="g", nodes=[
        _node("opt", "opt", criticality=Criticality.OPTIONAL),
        _node("down", "down", ["opt"]),
    ])
    ex, _ = _executor(lib)
    ctx = ExecutionContext(task="t")
    await ex.run(ctx, pipeline)
    node = pipeline.node("opt")
    assert node.status is NodeStatus.COMPLETED
    assert node.outcome is NodeOutcome.DEGRADED
    assert pipeline.node("down").status is NodeStatus.COMPLETED
    assert lib["down"].calls == 1


async def test_critical_tool_failure_marks_outcome_blocked():
    lib = {"crit": ToolFailAgent("crit")}
    pipeline = Pipeline(goal="g", nodes=[_node("crit", "crit", criticality=Criticality.CRITICAL)])
    ex, _ = _executor(lib)
    ctx = ExecutionContext(task="t")
    with pytest.raises(PipelineExecutionError):
        await ex.run(ctx, pipeline)
    assert pipeline.node("crit").outcome is NodeOutcome.BLOCKED
    assert pipeline.node("crit").status is NodeStatus.FAILED


async def test_agent_result_ok_true_and_no_failures_is_satisfied():
    lib = {"a": FakeAgent("a")}
    pipeline = Pipeline(goal="g", nodes=[_node("a", "a")])
    ex, _ = _executor(lib)
    ctx = ExecutionContext(task="t")
    await ex.run(ctx, pipeline)
    assert pipeline.node("a").outcome is NodeOutcome.SATISFIED


class EvidenceRecordingAgent(Agent):
    """Records whatever ``evidence``/``tool_cache`` the executor passed in."""

    def __init__(self, name):
        super().__init__(name=name)
        self.seen_evidence = None
        self.seen_tool_cache = None

    def with_llm(self, llm):
        return self

    async def execute(self, task, *, context=None, registry=None, evidence=None, tool_cache=None):
        self.seen_evidence = evidence
        self.seen_tool_cache = tool_cache
        return AgentResult(agent=self.name, ok=True, output=f"{self.name}-ok")


async def test_executor_passes_dependency_evidence_and_shared_cache_to_agents():
    upstream = ToolOkAgent("r")
    downstream = EvidenceRecordingAgent("d")
    lib = {"r": upstream, "d": downstream}
    pipeline = Pipeline(goal="g", nodes=[_node("r", "r"), _node("d", "d", ["r"])])
    ex, _ = _executor(lib)
    ctx = ExecutionContext(task="t")
    await ex.run(ctx, pipeline)

    assert downstream.seen_evidence is not None
    assert len(downstream.seen_evidence) == 1
    assert downstream.seen_evidence[0].producer_node == "r"
    # Same ToolCallCache instance for the whole execution -- shared state,
    # not a fresh cache per node.
    assert downstream.seen_tool_cache is ctx.state.tool_cache


async def test_successful_tool_call_is_recorded_as_evidence():
    lib = {"r": ToolOkAgent("r")}
    pipeline = Pipeline(goal="g", nodes=[_node("r", "r")])
    ex, _ = _executor(lib)
    ctx = ExecutionContext(task="t")
    await ex.run(ctx, pipeline)
    assert len(ctx.state.evidence) == 1
    ev = next(iter(ctx.state.evidence.values()))
    assert ev.producer_node == "r"
    assert ev.producer_tool == "read_file"
    assert ev.content == "file contents"
    # the ToolResult links back to the evidence it produced
    assert ctx.state.node_results["r"].tool_results[0].evidence_id == ev.id
