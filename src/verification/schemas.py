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


class VerificationResult(BaseModel):
    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation: Recommendation = Recommendation.ACCEPT
    issues: list[VerificationIssue] = Field(default_factory=list)
    checked_at: float = Field(default_factory=time.time)
    checker: str = "rule-based"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_blocking_issue(self) -> bool:
        return any(i.severity in (Severity.MAJOR, Severity.CRITICAL) for i in self.issues)
