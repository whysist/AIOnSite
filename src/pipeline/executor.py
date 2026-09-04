"""Pipeline execution engine.

Executes the DAG layer by layer; nodes within a layer run concurrently
(``asyncio.gather``).  Per node: dependency-gated, timed, retried within
its :class:`RetryPolicy`, and fully audited.  A node whose dependency
failed or was skipped is itself skipped -- errors propagate, they do not
crash the run.  Retries are bounded; there is no unbounded loop.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from ..agents.base_agent import Agent
from ..agents.router import ModelRouter
from ..audit.events import AuditEventType
from ..audit.trail import AuditTrail
from ..core.config import Settings, get_settings
from ..core.exceptions import PipelineExecutionError
from ..core.logging import bind_execution, get_logger
from ..state.evidence import Evidence
from ..tools.registry import ToolRegistry
from ..verification.schemas import Recommendation, VerificationResult
from ..verification.verifier import ResultVerifier
from .graph import PipelineGraph
from .models import (
    AgentResult,
    Criticality,
    NodeOutcome,
    NodeStatus,
    NodeType,
    Pipeline,
    PipelineNode,
)
from .requirements import RequirementStatus
from .state import ExecutionContext

_log = get_logger("pipeline.executor")


class PipelineExecutor:
    def __init__(
        self,
        *,
        agent_library: Mapping[str, Agent],
        registry: ToolRegistry,
        router: ModelRouter,
        audit: AuditTrail,
        verifier: ResultVerifier,
        settings: Settings | None = None,
    ) -> None:
        # The executor only reads from the library, so a read-only
        # ``Mapping`` (covariant in its value type) is the right contract:
        # it accepts ``dict[str, LLMAgent]`` as well as ``dict[str, Agent]``.
        self._library: Mapping[str, Agent] = agent_library
        self._registry = registry
        self._router = router
        self._audit = audit
        self._verifier = verifier
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    async def run(self, ctx: ExecutionContext, pipeline: Pipeline) -> ExecutionContext:
        bind_execution(execution_id=ctx.execution_id)
        graph = PipelineGraph(pipeline)
        layers = graph.execution_layers()

        self._audit.record(
            ctx.execution_id, AuditEventType.PIPELINE_BUILT, component="executor",
            message=f"{len(pipeline.nodes)} nodes in {len(layers)} layers",
            metadata={"layers": layers},
        )

        for layer in layers:
            await asyncio.gather(
                *(self._run_node(ctx, pipeline, pipeline.node(nid)) for nid in layer)
            )

        self._finalise(ctx, pipeline)
        return ctx

    # ------------------------------------------------------------------
    async def _run_node(
        self, ctx: ExecutionContext, pipeline: Pipeline, node: PipelineNode
    ) -> None:
        # dependency gate
        for dep in node.depends_on:
            if pipeline.node(dep).status in (
                NodeStatus.FAILED, NodeStatus.SKIPPED, NodeStatus.CANCELLED
            ):
                node.status = NodeStatus.SKIPPED
                node.error = f"upstream node {dep!r} did not complete"
                self._audit.record(
                    ctx.execution_id, AuditEventType.NODE_SKIPPED, component="executor",
                    status="skipped", message=node.error, metadata={"node": node.id},
                )
                return

        node.status = NodeStatus.RUNNING
        node.started_at = time.time()
        self._audit.record(
            ctx.execution_id, AuditEventType.NODE_STARTED, component="executor",
            message=node.description, metadata={"node": node.id, "agent": node.agent},
        )

        context = ctx.state.outputs_for(node.depends_on)

        if node.type is NodeType.VERIFIER:
            await self._run_verifier_node(ctx, pipeline, node, context)
            return

        agent = self._library.get(node.agent) or self._library.get("custom")
        if agent is None:
            node.status = NodeStatus.FAILED
            node.error = f"no agent registered for {node.agent!r}"
            node.finished_at = time.time()
            self._audit.record(
                ctx.execution_id, AuditEventType.NODE_FAILED, component="executor",
                status="failed", message=node.error, metadata={"node": node.id},
            )
            return

        llm, decision = self._router.get_llm(node)
        node.provider, node.model = decision.provider, decision.model
        bound = agent.with_llm(llm)

        last_error: str | None = None
        for attempt in range(1, node.retry_policy.max_retries + 2):
            node.attempts = attempt
            try:
                result: AgentResult = await asyncio.wait_for(
                    bound.execute(
                        f"{node.description}\n\n(Overall goal: {pipeline.goal})",
                        context=context,
                        registry=self._registry,
                    ),
                    timeout=node.timeout_seconds or self._settings.node_timeout_seconds,
                )
                result.node_id = node.id
                node.result = result
                node.finished_at = time.time()
                ctx.state.record(node.id, result)

                for tr in result.tool_results:
                    self._audit.record(
                        ctx.execution_id,
                        AuditEventType.TOOL_CALLED if tr.ok else AuditEventType.TOOL_FAILED,
                        component="tool", status="ok" if tr.ok else "failed",
                        message=tr.tool,
                        metadata={"node": node.id, "input": tr.input, "error": tr.error},
                    )
                    # Any successful tool call becomes first-class Evidence,
                    # generically (not tied to a specific tool implementation)
                    # so downstream nodes and the verifier can see *what was
                    # actually retrieved* instead of only free text.
                    if tr.ok:
                        ev = Evidence(
                            source=tr.tool, content=tr.output, reference=tr.call_id,
                            producer_tool=tr.tool, producer_node=node.id,
                        )
                        tr.evidence_id = ev.id
                        ctx.state.record_evidence(ev)
                for entry in result.metadata.get("repair_log") or []:
                    event = (
                        AuditEventType.TOOL_REPAIR_EXHAUSTED if entry.get("exhausted")
                        else AuditEventType.TOOL_REPAIR_ATTEMPTED
                    )
                    self._audit.record(
                        ctx.execution_id, event, component="tool",
                        status="failed" if entry.get("exhausted") else "retrying",
                        message=entry["tool"],
                        metadata={
                            "node": node.id, "attempt": entry["attempt"],
                            "error": entry.get("error"),
                        },
                    )

                unresolved_tools: set[str] = set(result.metadata.get("unresolved_tools") or [])
                if unresolved_tools and node.criticality in (Criticality.REQUIRED, Criticality.CRITICAL):
                    node.status = NodeStatus.FAILED
                    node.outcome = (
                        NodeOutcome.BLOCKED if node.criticality is Criticality.CRITICAL
                        else NodeOutcome.FAILED
                    )
                    node.error = (
                        f"{node.criticality.value} tool(s) failed after repair: "
                        + ", ".join(sorted(unresolved_tools))
                    )
                    ctx.state.errors.append({"node": node.id, "error": node.error})
                    self._audit.record(
                        ctx.execution_id,
                        AuditEventType.NODE_BLOCKED if node.outcome is NodeOutcome.BLOCKED
                        else AuditEventType.NODE_FAILED,
                        component="executor", status="failed", message=node.error,
                        metadata={"node": node.id, "unresolved_tools": sorted(unresolved_tools)},
                    )
                    return

                node.status = NodeStatus.COMPLETED
                if unresolved_tools:
                    node.outcome = NodeOutcome.DEGRADED
                    self._audit.record(
                        ctx.execution_id, AuditEventType.NODE_DEGRADED, component="executor",
                        status="degraded",
                        message=f"optional tool(s) failed: {', '.join(sorted(unresolved_tools))}",
                        metadata={"node": node.id, "unresolved_tools": sorted(unresolved_tools)},
                    )
                else:
                    node.outcome = NodeOutcome.SATISFIED

                self._audit.record(
                    ctx.execution_id, AuditEventType.NODE_COMPLETED, component="executor",
                    message=f"attempt {attempt}",
                    metadata={
                        "node": node.id, "provider": node.provider, "model": node.model,
                        "duration_ms": node.duration_ms, "tool_calls": len(result.tool_results),
                        "outcome": node.outcome.value,
                    },
                )
                return
            except (TimeoutError, Exception) as exc:  # noqa: BLE001
                last_error = (
                    f"timeout after {node.timeout_seconds}s"
                    if isinstance(exc, asyncio.TimeoutError)
                    else f"{type(exc).__name__}: {exc}"
                )
                if attempt <= node.retry_policy.max_retries:
                    node.status = NodeStatus.RETRYING
                    self._audit.record(
                        ctx.execution_id, AuditEventType.RETRY_TRIGGERED, component="executor",
                        status="retrying", message=last_error,
                        metadata={"node": node.id, "attempt": attempt},
                    )
                    await asyncio.sleep(node.retry_policy.delay_for(attempt))
                    continue
                break

        node.status = NodeStatus.FAILED
        node.error = last_error
        node.finished_at = time.time()
        ctx.state.errors.append({"node": node.id, "error": last_error})
        self._audit.record(
            ctx.execution_id, AuditEventType.NODE_FAILED, component="executor",
            status="failed", message=last_error or "unknown error",
            metadata={"node": node.id, "attempts": node.attempts},
        )

    # ------------------------------------------------------------------
    async def _run_verifier_node(
        self,
        ctx: ExecutionContext,
        pipeline: Pipeline,
        node: PipelineNode,
        context: dict[str, Any],
    ) -> None:
        combined = "\n\n".join(f"[{k}]\n{v}" for k, v in context.items()) or ""
        node_outcomes = {
            n.id: (n.outcome.value if n.outcome else n.status.value) for n in pipeline.nodes
        }
        try:
            verdict = await asyncio.wait_for(
                self._verifier.verify(
                    ctx.task, combined,
                    require_evidence=bool(ctx.requirements),
                    requirements=ctx.requirements,
                    evidence=list(ctx.state.evidence.values()),
                    tool_failures=ctx.state.unresolved_tool_failures,
                    node_outcomes=node_outcomes,
                ),
                timeout=node.timeout_seconds or self._settings.node_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            verdict = VerificationResult(
                passed=False, score=0.0, recommendation=Recommendation.ESCALATE,
                checker="error", metadata={"error": f"{type(exc).__name__}: {exc}"},
            )

        ctx.state.verifications[node.id] = verdict
        if verdict.requirement_statuses:
            by_id = {r.id: r for r in ctx.requirements}
            for rid, status_value in verdict.requirement_statuses.items():
                req = by_id.get(rid)
                if req is None:
                    continue
                try:
                    req.status = RequirementStatus(status_value)
                except ValueError:
                    continue
                if req.status is not RequirementStatus.SATISFIED:
                    missing = next(
                        (m for m in verdict.missing_requirements if m.requirement_id == rid),
                        None,
                    )
                    req.failure_reason = missing.reason if missing else req.failure_reason
        result = AgentResult(
            agent="verifier", node_id=node.id, ok=verdict.passed,
            output=("verification passed" if verdict.passed else "verification failed"),
            structured=verdict.model_dump(),
        )
        node.result = result
        node.status = NodeStatus.COMPLETED
        node.finished_at = time.time()
        ctx.state.record(node.id, result)
        self._audit.record(
            ctx.execution_id, AuditEventType.VERIFICATION_COMPLETED, component="verifier",
            status="ok" if verdict.passed else "failed",
            message=f"score={verdict.score} recommendation={verdict.recommendation.value}",
            metadata={"node": node.id, "issues": [i.model_dump() for i in verdict.issues]},
        )

    # ------------------------------------------------------------------
    def _finalise(self, ctx: ExecutionContext, pipeline: Pipeline) -> None:
        answer_nodes = [
            n for n in pipeline.nodes
            if n.type is not NodeType.VERIFIER and n.status is NodeStatus.COMPLETED
        ]
        summarizers = [n for n in answer_nodes if n.type is NodeType.SUMMARIZER]
        chosen = (summarizers or answer_nodes)[-1] if (summarizers or answer_nodes) else None
        if chosen and chosen.result:
            ctx.final_answer = chosen.result.output

        verifier_nodes = [n for n in pipeline.nodes if n.type is NodeType.VERIFIER]
        if verifier_nodes and verifier_nodes[-1].id in ctx.state.verifications:
            ctx.final_verification = ctx.state.verifications[verifier_nodes[-1].id]

        failed = [n.id for n in pipeline.nodes if n.status is NodeStatus.FAILED]
        if failed and ctx.final_answer is None:
            raise PipelineExecutionError(
                "all pipeline branches failed", details={"failed_nodes": failed}
            )
