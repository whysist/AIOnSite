"""Strongly-typed data models for the agent pipeline (DAG).

These models are the contract between the planner, the pipeline builder,
the execution engine, the verifier, the audit layer and the API/frontend.
They are pure data -- no behaviour that imports agents or providers -- so
every other module can depend on them without creating cycles.
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class NodeType(str, enum.Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    SUMMARIZER = "summarizer"
    CUSTOM = "custom"


class NodeStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ExecutionStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    CANCELLED = "cancelled"


class Criticality(str, enum.Enum):
    """How much a node's (and its tools') failure should matter.

    ``OPTIONAL``  -- failure degrades the node but never blocks the task.
    ``REQUIRED``  -- unresolved failure fails the node and blocks dependents
                     (the default: most steps in a plan matter).
    ``CRITICAL``  -- unresolved failure blocks the node and is surfaced as a
                     structural block on the whole task, not just a normal
                     node failure.
    """

    OPTIONAL = "optional"
    REQUIRED = "required"
    CRITICAL = "critical"


class NodeOutcome(str, enum.Enum):
    """Whether a node achieved its objective -- distinct from :class:`NodeStatus`.

    A node can reach ``NodeStatus.COMPLETED`` (the coroutine returned without
    raising) while its outcome is ``DEGRADED`` (an optional tool failed) --
    these two axes are tracked separately on purpose so "the code ran" is
    never silently reported as "the objective was met".
    """

    SATISFIED = "satisfied"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    FAILED = "failed"


class RetryPolicy(BaseModel):
    max_retries: int = Field(default=1, ge=0, le=10)
    backoff_seconds: float = Field(default=0.5, ge=0)

    def delay_for(self, attempt: int) -> float:
        return self.backoff_seconds * max(1, attempt)


class ToolResult(BaseModel):
    tool: str
    call_id: str = Field(default_factory=lambda: _new_id("tool"))
    ok: bool = True
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    error: str | None = None
    # ``error_kind`` distinguishes *why* a call failed so callers can decide
    # whether it is repairable: "validation" (bad arguments -- feed the
    # schema back to the model and retry) vs "execution" (the tool itself
    # raised) vs "not_found" (unknown tool name). ``None`` on success.
    error_kind: Literal["validation", "execution", "not_found"] | None = None
    # Populated only on a validation failure: the tool's JSON schema, so a
    # repair prompt can tell the model exactly what shape is expected.
    expected_schema: dict[str, Any] | None = None
    # Set once this call's output has been recorded as first-class Evidence
    # (see ``src/state/evidence.py``) -- links the call back to that record.
    evidence_id: str | None = None
    # True when this result was served from the execution-scoped tool-call
    # cache (see ``src/pipeline/state.py::ToolCallCache``) instead of
    # actually invoking the tool -- kept visible on the record itself so
    # provenance/audit never has to guess whether a call really happened.
    cached: bool = False
    duration_ms: float | None = None
    started_at: float = Field(default_factory=time.time)


class AgentResult(BaseModel):
    """Outcome of a single agent invocation on a pipeline node."""

    agent: str
    node_id: str | None = None
    run_id: str = Field(default_factory=lambda: _new_id("run"))
    ok: bool = True
    output: str = ""
    structured: dict[str, Any] | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineNode(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("node"))
    name: str
    type: NodeType = NodeType.CUSTOM
    description: str = ""
    agent: str = ""                       # agent key resolved by the executor
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    # Optional per-node provider/model override honoured by the router.
    provider: str | None = None
    model: str | None = None
    require_local: bool = False
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_seconds: float | None = None
    # How much this node's (and, by default, its tools') failure matters.
    # See :class:`Criticality`. Defaults to REQUIRED: most plan steps exist
    # because they matter, so silence-by-default would hide real gaps.
    criticality: Criticality = Criticality.REQUIRED
    # runtime fields ------------------------------------------------------
    status: NodeStatus = NodeStatus.PENDING
    outcome: NodeOutcome | None = None
    attempts: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    result: AgentResult | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return None


class PipelineEdge(BaseModel):
    source: str
    target: str
    # optional predicate name for conditional edges (evaluated by executor)
    condition: str | None = None


class Pipeline(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("pipeline"))
    goal: str = ""
    nodes: list[PipelineNode] = Field(default_factory=list)
    edges: list[PipelineEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _check_integrity(self) -> Pipeline:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate node ids in pipeline")
        id_set = set(ids)
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in id_set:
                    raise ValueError(f"node {node.id!r} depends on unknown node {dep!r}")
                if dep == node.id:
                    raise ValueError(f"node {node.id!r} depends on itself")
        for edge in self.edges:
            if edge.source not in id_set or edge.target not in id_set:
                raise ValueError(f"edge {edge.source}->{edge.target} references unknown node")
        return self

    def node(self, node_id: str) -> PipelineNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def as_graph_dict(self) -> dict[str, Any]:
        """Machine-readable view for the frontend pipeline visualiser."""
        return {
            "id": self.id,
            "goal": self.goal,
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.type.value,
                    "description": n.description,
                    "agent": n.agent,
                    "depends_on": n.depends_on,
                    "tools": n.tools,
                    "provider": n.provider,
                    "model": n.model,
                    "status": n.status.value,
                    "criticality": n.criticality.value,
                    "outcome": n.outcome.value if n.outcome else None,
                    "attempts": n.attempts,
                    "duration_ms": n.duration_ms,
                    "error": n.error,
                    "result": n.result.model_dump() if n.result else None,
                }
                for n in self.nodes
            ],
            "edges": [e.model_dump() for e in self.edges],
            "metadata": self.metadata,
        }
