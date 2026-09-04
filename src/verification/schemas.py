"""Structured verification results."""

from __future__ import annotations

import enum
import time
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, enum.Enum):
    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class Recommendation(str, enum.Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    CORRECT = "correct"
    REPLAN = "replan"
    ESCALATE = "escalate"
    REJECT = "reject"


class VerificationIssue(BaseModel):
    code: str
    message: str
    severity: Severity = Severity.MINOR
    evidence: str | None = None


class MissingRequirement(BaseModel):
    """A required :class:`~src.pipeline.requirements.TaskRequirement` the
    verifier could not confirm was satisfied."""

    requirement_id: str
    description: str
    reason: str = "no supporting evidence or completed step found"


class MissingEvidence(BaseModel):
    """A gap between "evidence was required" and "evidence was retrieved"."""

    description: str
    expected_source_type: str | None = None


class VerificationResult(BaseModel):
    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation: Recommendation = Recommendation.ACCEPT
    issues: list[VerificationIssue] = Field(default_factory=list)
    # Completeness/grounding gaps -- kept distinct from ``issues`` (which are
    # plausibility signals about the answer's text) because these are about
    # whether the *task* was actually done, not whether the answer reads
    # well. Either list being non-empty forces ``passed = False``.
    missing_requirements: list[MissingRequirement] = Field(default_factory=list)
    missing_evidence: list[MissingEvidence] = Field(default_factory=list)
    # requirement id -> RequirementStatus value, for the caller to apply back
    # onto its own ``TaskRequirement`` records without this module depending
    # on the pipeline package.
    requirement_statuses: dict[str, str] = Field(default_factory=dict)
    checked_at: float = Field(default_factory=time.time)
    checker: str = "rule-based"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_blocking_issue(self) -> bool:
        return any(i.severity in (Severity.MAJOR, Severity.CRITICAL) for i in self.issues)
