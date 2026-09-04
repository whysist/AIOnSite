"""Shared, explicit execution state for one pipeline run.

There is no global mutable state in the orchestrator -- everything a run
needs is carried on an :class:`ExecutionContext` instance that is created
per request and threaded through the executor, agents and verifier.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from ..verification.schemas import VerificationResult
from .models import AgentResult, ExecutionStatus, Pipeline, ToolResult


class PipelineState(BaseModel):
    """Accumulated results, keyed by node id."""

    node_results: dict[str, AgentResult] = Field(default_factory=dict)
    tool_results: list[ToolResult] = Field(default_factory=list)
    verifications: dict[str, VerificationResult] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    scratch: dict[str, Any] = Field(default_factory=dict)

    def record(self, node_id: str, result: AgentResult) -> None:
        self.node_results[node_id] = result
        self.tool_results.extend(result.tool_results)

    def outputs_for(self, node_ids: list[str]) -> dict[str, str]:
        return {
            nid: self.node_results[nid].output
            for nid in node_ids
            if nid in self.node_results
        }


class ExecutionContext(BaseModel):
    """Everything one run of the orchestrator needs and produces."""

    execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:16]}")
    task: str = ""
    created_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    status: ExecutionStatus = ExecutionStatus.PENDING

    sovereign_mode: bool = False
    confidential: bool = False

    plan: dict[str, Any] | None = None
    pipeline: Pipeline | None = None
    state: PipelineState = Field(default_factory=PipelineState)

    final_answer: str | None = None
    final_verification: VerificationResult | None = None
    replans: int = 0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    def mark_finished(self, status: ExecutionStatus) -> None:
        self.status = status
        self.finished_at = time.time()

    @property
    def duration_ms(self) -> float | None:
        if self.finished_at:
            return (self.finished_at - self.created_at) * 1000
        return None

    def summary(self) -> dict[str, Any]:
        """Machine-readable snapshot for the API / frontend."""
        return {
            "execution_id": self.execution_id,
            "task": self.task,
            "status": self.status.value,
            "sovereign_mode": self.sovereign_mode,
            "confidential": self.confidential,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "replans": self.replans,
            "error": self.error,
            "final_answer": self.final_answer,
            "final_verification": (
                self.final_verification.model_dump() if self.final_verification else None
            ),
            "pipeline": self.pipeline.as_graph_dict() if self.pipeline else None,
            "node_results": {
                nid: r.model_dump() for nid, r in self.state.node_results.items()
            },
            "verifications": {
                nid: v.model_dump() for nid, v in self.state.verifications.items()
            },
        }
