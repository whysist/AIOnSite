"""Audit event types and the immutable event record."""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class AuditEventType(str, enum.Enum):
    TASK_RECEIVED = "task_received"
    POLICY_ENFORCED = "policy_enforced"
    PLAN_CREATED = "plan_created"
    PLAN_FAILED = "plan_failed"
    PIPELINE_BUILT = "pipeline_built"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_DEGRADED = "node_degraded"
    NODE_BLOCKED = "node_blocked"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"
    TOOL_CALLED = "tool_called"
    TOOL_FAILED = "tool_failed"
    TOOL_REPAIR_ATTEMPTED = "tool_repair_attempted"
    TOOL_REPAIR_EXHAUSTED = "tool_repair_exhausted"
    VERIFICATION_COMPLETED = "verification_completed"
    RETRY_TRIGGERED = "retry_triggered"
    REPLAN_TRIGGERED = "replan_triggered"
    TASK_STATUS_DETERMINED = "task_status_determined"
    FINAL_ANSWER_GENERATED = "final_answer_generated"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"


class AuditEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    execution_id: str
    event_type: AuditEventType
    component: str
    status: str = "ok"
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}
