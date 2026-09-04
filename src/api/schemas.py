"""Request / response models for the HTTP API.

Kept separate from the orchestration models so the public contract can
evolve independently of internal types.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=20_000)
    confidential: bool = False


class TaskCreatedResponse(BaseModel):
    execution_id: str
    status: str
    final_answer: str | None = None
    verification: dict[str, Any] | None = None
    error: str | None = None


class ExecutionResponse(BaseModel):
    execution_id: str
    status: str
    task: str
    sovereign_mode: bool
    confidential: bool
    duration_ms: float | None = None
    final_answer: str | None = None
    final_verification: dict[str, Any] | None = None
    error: str | None = None
    node_results: dict[str, Any] = Field(default_factory=dict)


class PipelineResponse(BaseModel):
    execution_id: str
    status: str
    pipeline: dict[str, Any] | None = None


class AuditResponse(BaseModel):
    execution_id: str
    events: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    environment: str
    llm_provider: str
    sovereign_mode: bool
    tools: list[str]


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
