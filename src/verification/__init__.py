"""Verification layer: evidence / consistency / correctness checks."""

from .schemas import (
    Recommendation,
    Severity,
    VerificationIssue,
    VerificationResult,
)
from .verifier import ResultVerifier

__all__ = [
    "Recommendation",
    "Severity",
    "VerificationIssue",
    "VerificationResult",
    "ResultVerifier",
]
