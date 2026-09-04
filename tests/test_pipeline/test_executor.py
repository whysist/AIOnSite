import pytest

from src.agents.base_agent import Agent
from src.agents.router import ModelRouter
from src.audit.trail import AuditTrail
from src.core.config import Settings
from src.core.exceptions import PipelineExecutionError
from src.pipeline.executor import PipelineExecutor
from src.pipeline.models import (
    NodeStatus,
    NodeType,
    Pipeline,
    PipelineNode,
    RetryPolicy,
)
from src.pipeline.models import AgentResult
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

    async def execute(self, task, *, context=None, registry=None):
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


def _node(nid, agent, deps=(), ntype=NodeType.CUSTOM, retries=0):
    return PipelineNode(
        id=nid, name=nid, type=ntype, agent=agent, depends_on=list(deps),
        retry_policy=RetryPolicy(max_retries=retries, backoff_seconds=0),
    )


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
